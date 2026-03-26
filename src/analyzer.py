"""LLM-powered content analysis with knowledge graph extraction.

Uses Claude to understand document content, extract topics, domains,
audience, and other metadata — plus entities and relationships that
form a cross-document knowledge graph. The graph is used downstream
by the fixer (cross-doc reference resolution), recommender (graph-based
folder clustering), and scorer (knowledge completeness).
"""

import asyncio
import json
from typing import Optional

from anthropic import AsyncAnthropic

from .config import Config
from .graph_builder import KnowledgeGraph
from .models import ContentAnalysis, Entity, ParsedDocument, Relationship
from .prompts import ANALYZE_DOCUMENT


def extract_json(text: str) -> Optional[dict]:
    """Safely extract a JSON object from LLM response text.

    Handles: raw JSON, markdown code blocks, JSON embedded in text.
    Uses escape-aware brace counting for robust extraction.
    """
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a balanced JSON object using brace counting,
    # skipping over braces inside quoted strings.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    next_start = text.find("{", start + 1)
                    if next_start != -1:
                        start = next_start
                        depth = 0
                        continue
                    return None

    return None


def _analysis_is_low_confidence(doc: ParsedDocument, analysis: ContentAnalysis) -> bool:
    """Check whether an LLM analysis is too sparse to trust for graph building.

    Returns True if the analysis should be excluded from the knowledge graph.
    The analysis object is still kept — topics, summary, and other metadata
    may be useful even when entity extraction failed or was too sparse.

    Heuristics:
    - Failed analyses (error in summary) are always low confidence
    - Documents with 200+ words should yield at least 2 entities
    - Entities with zero relationships suggest the LLM didn't find structure
    - All entities having the same type suggests defaulting behavior
    """
    # Failed analysis
    if analysis.summary.startswith("Analysis failed"):
        return True

    # No entities at all
    if not analysis.entities:
        return True

    # Document is substantial but LLM found almost nothing
    word_count = len(doc.full_text.split())
    if word_count >= 200 and len(analysis.entities) < 2:
        return True

    # Entities exist but no relationships — isolated noise
    if len(analysis.entities) >= 3 and not analysis.relationships:
        return True

    # Every entity has the same type — likely defaulting
    if len(analysis.entities) >= 3:
        types = {e.entity_type for e in analysis.entities}
        if len(types) == 1:
            return True

    return False


class ContentAnalyzer:
    """Analyze document content using an LLM and build a knowledge graph."""

    def __init__(self, config: Config):
        if not config.anthropic_api_key:
            raise ValueError("Anthropic API key required for content analysis. Set ANTHROPIC_API_KEY or use --llm-key.")
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = config.llm_model
        self.max_tokens = config.llm_max_tokens
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def analyze(self, doc: ParsedDocument) -> ContentAnalysis:
        """Analyze a single document and return structured metadata with graph data."""
        # Truncate to avoid token limits (keep first ~8000 words)
        text = doc.full_text
        words = text.split()
        if len(words) > 8000:
            text = " ".join(words[:8000]) + "\n\n[Document truncated for analysis...]"

        prompt = ANALYZE_DOCUMENT.format(document_text=text)

        async with self._semaphore:
            response = await self._call_with_retry(prompt)

        # Parse JSON response
        response_text = response.content[0].text.strip()
        data = extract_json(response_text)

        if data is None:
            return ContentAnalysis(summary="Analysis failed — could not parse LLM response.")

        source_file = doc.metadata.filename

        # Parse entities
        entities = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and "name" in e:
                entities.append(
                    Entity(
                        name=e["name"],
                        entity_type=e.get("type", "concept"),
                        source_file=source_file,
                        description=e.get("description", ""),
                    )
                )

        # Parse relationships
        relationships = []
        for r in data.get("relationships", []):
            if isinstance(r, dict) and "source" in r and "target" in r:
                relationships.append(
                    Relationship(
                        source=r["source"],
                        target=r["target"],
                        rel_type=r.get("type", "related_to"),
                        source_file=source_file,
                        context=r.get("context", ""),
                    )
                )

        return ContentAnalysis(
            domain=data.get("domain", ""),
            topics=data.get("topics", []),
            audience=data.get("audience", ""),
            content_type=data.get("content_type", ""),
            key_concepts=data.get("key_concepts", []),
            suggested_tags=data.get("suggested_tags", []),
            summary=data.get("summary", ""),
            entities=entities,
            relationships=relationships,
        )

    async def analyze_and_build_graph(self, docs: list[ParsedDocument]) -> tuple[list[ContentAnalysis], KnowledgeGraph]:
        """Analyze all documents and build a shared knowledge graph.

        This is the primary entry point for the pipeline. Returns both
        the per-document analyses and a merged knowledge graph.

        A lightweight confidence check filters out low-quality analyses
        before merging into the graph. If the LLM returned suspiciously
        few entities for a document's size, the analysis is still kept
        (topics, summary, etc. may be useful) but its entities and
        relationships are not added to the graph.
        """
        tasks = [self.analyze(doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        graph = KnowledgeGraph()
        analyses = []
        skipped = 0
        for doc, result in zip(docs, results):
            if isinstance(result, Exception):
                analysis = ContentAnalysis(summary=f"Analysis failed: {str(result)}")
            else:
                analysis = result
            analyses.append(analysis)

            # Confidence gate: skip graph merge for low-quality analyses
            if _analysis_is_low_confidence(doc, analysis):
                skipped += 1
                continue

            graph.add_analysis(doc, analysis)

        if skipped > 0:
            import logging

            logging.getLogger(__name__).info(f"Skipped {skipped}/{len(docs)} low-confidence analyses from graph merge")

        return analyses, graph

    async def _call_with_retry(self, prompt: str, max_retries: int = 3):
        """Make an LLM call with retry on rate limits."""
        for attempt in range(max_retries):
            try:
                return await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1 and (
                    "429" in error_str or "529" in error_str or "rate" in error_str or "overloaded" in error_str
                ):
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                raise

    # Keep backward compat
    async def analyze_batch(self, docs: list[ParsedDocument]) -> list[ContentAnalysis]:
        """Analyze multiple documents concurrently (without graph)."""
        analyses, _ = await self.analyze_and_build_graph(docs)
        return analyses
