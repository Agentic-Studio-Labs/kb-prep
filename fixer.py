"""LLM-powered auto-fix engine.

Applies corrections to documents based on issues found by the scorer.
Original files are never modified — fixed versions go to an output directory.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic

from config import Config
from models import (
    FixAction,
    FixReport,
    Issue,
    Paragraph,
    ParsedDocument,
    ScoreCard,
    Severity,
)
from parser import paragraphs_to_markdown
from prompts import (
    FIX_DANGLING_REFERENCES,
    FIX_GENERIC_HEADING,
    FIX_LONG_PARAGRAPH,
    FIX_UNDEFINED_ACRONYM,
    GENERATE_FILENAME,
)


class DocumentFixer:
    """Apply LLM-powered fixes to documents.

    Optionally accepts a KnowledgeGraph for cross-document
    reference resolution (e.g. resolving "see Unit 2" by
    pulling actual content from Unit 2's document).
    """

    def __init__(self, config: Config, graph=None):
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key required for auto-fix. Set ANTHROPIC_API_KEY or use --llm-key.")
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = config.llm_model
        self.max_tokens = config.llm_max_tokens
        self.output_dir = config.output_dir
        self.graph = graph  # Optional KnowledgeGraph for cross-doc resolution
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def fix(self, doc: ParsedDocument, scorecard: ScoreCard) -> FixReport:
        """Apply all applicable fixes and write corrected document.

        Fixes are applied in a safe order:
        1. In-place text edits (dangling refs, headings, acronyms) — don't change list structure
        2. Paragraph splits — applied in reverse index order to preserve positions
        3. Filename generation — independent of paragraph list
        """
        actions: list[FixAction] = []
        # Work on a copy of paragraphs
        paragraphs = [
            Paragraph(text=p.text, level=p.level, style=p.style, index=p.index)
            for p in doc.paragraphs
        ]

        # Group issues by category
        issues_by_cat: dict[str, list[Issue]] = {}
        for issue in scorecard.all_issues:
            issues_by_cat.setdefault(issue.category, []).append(issue)

        # Phase 1: In-place text edits (safe — don't change list structure)
        para_index = self._build_para_index(paragraphs)

        # Fix dangling references
        for issue in issues_by_cat.get("self_containment", []):
            if issue.location is not None:
                action = await self._fix_dangling_reference(paragraphs, issue.location, _index=para_index)
                if action:
                    actions.append(action)

        # Fix generic headings
        for issue in issues_by_cat.get("heading_quality", []):
            if issue.location is not None and "Generic heading" in issue.message:
                action = await self._fix_generic_heading(paragraphs, issue.location, _index=para_index)
                if action:
                    actions.append(action)

        # Fix undefined acronyms (use working text, not original)
        working_full_text = "\n\n".join(p.text for p in paragraphs if p.text.strip())
        for issue in issues_by_cat.get("acronym_definitions", []):
            match = re.search(r'"(\w+)"', issue.message)
            if match:
                acronym = match.group(1)
                action = await self._fix_acronym(paragraphs, acronym, working_full_text)
                if action:
                    actions.append(action)

        # Phase 2: Structural changes (paragraph splits) — apply in reverse
        # index order so earlier indices stay valid as we insert
        split_issues = [
            issue for issue in issues_by_cat.get("paragraph_length", [])
            if issue.location is not None and "Very long" in issue.message
        ]
        # Process in reverse index order so splits at higher indices don't
        # shift positions of earlier paragraphs. The para_index is NOT rebuilt
        # between iterations — reverse ordering makes this safe.
        split_issues.sort(key=lambda i: i.location, reverse=True)
        para_index = self._build_para_index(paragraphs)  # rebuild after phase 1

        for issue in split_issues:
            action = await self._fix_long_paragraph(paragraphs, issue.location, _index=para_index)
            if action:
                actions.append(action)

        # Phase 3: Generate better filename if needed
        new_filename = None
        if issues_by_cat.get("filename_quality"):
            new_filename = await self._generate_filename(doc)

        # Write fixed document as Markdown
        output_path = self._write_fixed(doc, paragraphs, new_filename)

        return FixReport(
            source_path=doc.metadata.file_path,
            output_path=output_path,
            actions=actions,
            new_filename=new_filename,
        )

    # ------------------------------------------------------------------
    # Individual fix methods
    # ------------------------------------------------------------------

    async def _fix_dangling_reference(
        self, paragraphs: list[Paragraph], para_idx: int,
        _index: dict[int, Paragraph] | None = None,
    ) -> Optional[FixAction]:
        """Rewrite a paragraph to remove dangling references.

        If a knowledge graph is available, enriches the surrounding context
        with cross-document content for better reference resolution.
        """
        para = self._find_paragraph(paragraphs, para_idx, _index=_index)
        if not para:
            return None

        # Gather surrounding context (2 paragraphs before and after)
        context_paras = []
        for p in paragraphs:
            if abs(p.index - para_idx) <= 2 and p.index != para_idx:
                context_paras.append(p.text)

        # Enrich with graph context if available
        if self.graph and not self.graph.is_empty:
            # Extract potential entity references from the paragraph
            graph_context = self._get_graph_context_for_paragraph(para.text)
            if graph_context:
                context_paras.append(
                    f"\n[Related content from other documents:\n{graph_context}]"
                )

        surrounding = "\n\n".join(context_paras)

        prompt = FIX_DANGLING_REFERENCES.format(
            surrounding_context=surrounding,
            paragraph_text=para.text,
        )

        fixed_text = await self._call_llm(prompt)
        if fixed_text and fixed_text != para.text:
            original = para.text
            para.text = fixed_text
            return FixAction(
                category="self_containment",
                original_text=original,
                fixed_text=fixed_text,
                paragraph_index=para_idx,
                description="Rewrote paragraph to be self-contained",
            )
        return None

    async def _fix_generic_heading(
        self, paragraphs: list[Paragraph], para_idx: int,
        _index: dict[int, Paragraph] | None = None,
    ) -> Optional[FixAction]:
        """Replace a generic heading with a descriptive one."""
        heading = self._find_paragraph(paragraphs, para_idx, _index=_index)
        if not heading:
            return None

        # Get content below this heading (up to next heading or 3 paragraphs)
        content_below = []
        collecting = False
        for p in paragraphs:
            if p.index == para_idx:
                collecting = True
                continue
            if collecting:
                if p.is_heading:
                    break
                content_below.append(p.text)
                if len(content_below) >= 3:
                    break

        if not content_below:
            return None

        prompt = FIX_GENERIC_HEADING.format(
            heading_text=heading.text,
            content_below="\n\n".join(content_below),
        )

        new_heading = await self._call_llm(prompt)
        if new_heading and new_heading != heading.text:
            original = heading.text
            # Strip leading markdown # and whitespace, but only from edges
            heading.text = re.sub(r"^#+\s*", "", new_heading.strip()).strip()
            return FixAction(
                category="heading_quality",
                original_text=original,
                fixed_text=heading.text,
                paragraph_index=para_idx,
                description="Replaced generic heading with descriptive one",
            )
        return None

    async def _fix_long_paragraph(
        self, paragraphs: list[Paragraph], para_idx: int,
        _index: dict[int, Paragraph] | None = None,
    ) -> Optional[FixAction]:
        """Split an overly long paragraph into focused sub-paragraphs."""
        para = self._find_paragraph(paragraphs, para_idx, _index=_index)
        if not para:
            return None

        prompt = FIX_LONG_PARAGRAPH.format(
            word_count=para.word_count,
            paragraph_text=para.text,
        )

        result = await self._call_llm(prompt)
        if not result:
            return None

        # Split the result into paragraphs
        new_paras = [p.strip() for p in result.split("\n\n") if p.strip()]
        if len(new_paras) <= 1:
            return None

        # Replace the original paragraph with the splits
        original = para.text
        insert_idx = paragraphs.index(para)
        paragraphs.remove(para)

        for i, text in enumerate(new_paras):
            new_p = Paragraph(
                text=text,
                level=0,
                style="Normal",
                index=para_idx + i,
            )
            paragraphs.insert(insert_idx + i, new_p)

        return FixAction(
            category="paragraph_length",
            original_text=original,
            fixed_text="\n\n".join(new_paras),
            paragraph_index=para_idx,
            description=f"Split into {len(new_paras)} focused paragraphs",
        )

    async def _fix_acronym(
        self, paragraphs: list[Paragraph], acronym: str, full_text: str
    ) -> Optional[FixAction]:
        """Define an undefined acronym on its first use."""
        # Get context around first occurrence
        context_start = full_text.find(acronym)
        if context_start == -1:
            return None
        context = full_text[max(0, context_start - 200):context_start + 200]

        prompt = FIX_UNDEFINED_ACRONYM.format(
            acronym=acronym,
            context=context,
        )

        expansion = await self._call_llm(prompt)
        if not expansion or expansion == "UNKNOWN":
            return None

        # Find the first paragraph containing the acronym and add definition
        for para in paragraphs:
            if acronym in para.text:
                original = para.text
                # Replace first occurrence with "ACRONYM (Full Form)"
                para.text = para.text.replace(
                    acronym, f"{acronym} ({expansion})", 1
                )
                return FixAction(
                    category="acronym_definitions",
                    original_text=original,
                    fixed_text=para.text,
                    paragraph_index=para.index,
                    description=f"Defined {acronym} as {expansion}",
                )
        return None

    async def _generate_filename(self, doc: ParsedDocument) -> Optional[str]:
        """Generate a descriptive filename."""
        # Build a quick summary from headings and first paragraph
        summary_parts = []
        for h in doc.headings[:3]:
            summary_parts.append(h.text)
        for p in doc.body_paragraphs[:2]:
            summary_parts.append(p.text[:200])
        summary = "\n".join(summary_parts)

        prompt = GENERATE_FILENAME.format(
            summary=summary,
            current_filename=doc.metadata.stem,
        )

        result = await self._call_llm(prompt)
        if result:
            # Sanitize
            clean = re.sub(r"[^a-z0-9-]", "", result.lower().replace(" ", "-"))
            return clean or None
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_para_index(paragraphs: list[Paragraph]) -> dict[int, Paragraph]:
        return {p.index: p for p in paragraphs}

    @staticmethod
    def _find_paragraph(
        paragraphs: list[Paragraph], index: int,
        _index: dict[int, Paragraph] | None = None,
    ) -> Optional[Paragraph]:
        if _index is not None:
            return _index.get(index)
        for p in paragraphs:
            if p.index == index:
                return p
        return None

    def _get_graph_context_for_paragraph(self, text: str) -> str:
        """Query the knowledge graph for entities mentioned in text.

        Returns a string of related content from other documents,
        used to enrich the context for dangling reference resolution.
        """
        if not self.graph:
            return ""

        context_parts = []
        # Look up key terms from the paragraph in the graph
        # Extract potential entity references (capitalized phrases, quoted terms)
        candidates = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
        candidates += re.findall(r"(?:Unit|Lesson|Chapter|Section|Module)\s+\d+", text, re.IGNORECASE)

        seen = set()
        for candidate in candidates:
            if candidate.lower() in seen:
                continue
            seen.add(candidate.lower())

            related = self.graph.get_related_content(candidate, max_hops=1)
            for item in related:
                if item["description"] and item["depth"] <= 1:
                    context_parts.append(
                        f"- {item['entity']} ({item['type']}, from {item['source_file']}): "
                        f"{item['description']}"
                    )
                    if len(context_parts) >= 5:
                        break
            if len(context_parts) >= 5:
                break

        return "\n".join(context_parts)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Make a single LLM call with retry on rate limits."""
        max_retries = 3
        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text.strip()
                except Exception as e:
                    error_str = str(e).lower()
                    if attempt < max_retries - 1 and ("429" in error_str or "529" in error_str or "rate" in error_str or "overloaded" in error_str):
                        wait = 2 ** (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    return None
        return None

    def _write_fixed(
        self,
        doc: ParsedDocument,
        paragraphs: list[Paragraph],
        new_filename: Optional[str],
    ) -> str:
        """Write the fixed document as Markdown."""
        os.makedirs(self.output_dir, exist_ok=True)

        content = paragraphs_to_markdown(paragraphs)

        # Determine filename
        stem = new_filename or doc.metadata.stem
        output_path = os.path.join(self.output_dir, f"{stem}.md")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
