# Auto Markdown Reports & LLM Concurrency — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all CLI commands auto-generate timestamped markdown reports, and parallelize LLM calls using asyncio with a configurable concurrency limit.

**Architecture:** Two independent features sharing a Config change. Reports are composable section writers assembled per-command. Concurrency uses `AsyncAnthropic` + `asyncio.Semaphore` in analyzer and fixer, with `asyncio.run()` at CLI entry points. The recommender also becomes async since it makes LLM calls.

**Tech Stack:** Python asyncio, anthropic.AsyncAnthropic, click, existing test infrastructure (pytest).

**Design doc:** `docs/plans/2026-03-01-reports-and-concurrency-design.md`

---

### Task 1: Add `concurrency` field to Config

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py` (create)

**Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import os
from config import Config


def test_config_defaults():
    """Config has expected default values."""
    cfg = Config()
    assert cfg.concurrency == 5
    assert cfg.llm_model == "claude-sonnet-4-20250514"
    assert cfg.output_dir == "./fixed"


def test_config_with_overrides():
    """with_overrides sets concurrency."""
    cfg = Config().with_overrides(concurrency=10)
    assert cfg.concurrency == 10


def test_config_invalid_override_raises():
    """with_overrides rejects unknown fields."""
    import pytest
    with pytest.raises(ValueError, match="Invalid config fields"):
        Config().with_overrides(fake_field="nope")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `Config` has no `concurrency` field

**Step 3: Write minimal implementation**

In `config.py`, add to the `Config` dataclass after `min_score_for_upload`:

```python
    concurrency: int = 5  # Max parallel LLM calls
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: 3 PASS

**Step 5: Commit**

```
feat(config): add concurrency field with default of 5
```

---

### Task 2: Refactor report into composable section writers

**Files:**
- Modify: `cli.py`
- Test: `tests/test_report.py` (create)

**Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ScoreCard, ScoringResult, Issue, Severity, Readiness


def _make_card(filename="test.docx", score=85.0, issues=None):
    """Helper to build a ScoreCard for testing."""
    card = ScoreCard(file_path=f"/tmp/{filename}")
    card.results = [
        ScoringResult(
            category="structure",
            label="Document Structure",
            score=score,
            weight=0.15,
            issues=issues or [],
        )
    ]
    card.overall_score = score
    return card


def test_report_header():
    """_report_header returns markdown header with timestamp and file count."""
    from cli import _report_header
    lines = _report_header(command="score", file_count=5)
    text = "\n".join(lines)
    assert "# anam-prep" in text
    assert "score" in text.lower()
    assert "5" in text


def test_report_scores():
    """_report_scores returns markdown table rows."""
    from cli import _report_scores
    cards = [_make_card("a.docx", 90), _make_card("b.docx", 60)]
    lines = _report_scores(cards, detail=False)
    text = "\n".join(lines)
    assert "| a.docx" in text
    assert "| b.docx" in text
    assert "90" in text
    assert "Average score" in text


def test_report_scores_detail():
    """_report_scores with detail includes issue breakdown."""
    from cli import _report_scores
    issue = Issue(severity=Severity.WARNING, category="structure", message="Bad heading")
    cards = [_make_card("a.docx", 70, issues=[issue])]
    lines = _report_scores(cards, detail=True)
    text = "\n".join(lines)
    assert "Bad heading" in text


def test_generate_report_path():
    """_generate_report_path returns timestamped filename."""
    from cli import _generate_report_path
    path = _generate_report_path("analyze")
    assert path.startswith("anam-prep-analyze-")
    assert path.endswith(".md")
    # Verify timestamp format YYYYMMDD-HHMMSS
    parts = path.replace("anam-prep-analyze-", "").replace(".md", "")
    assert len(parts) == 15  # YYYYMMDD-HHMMSS
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: FAIL — functions not defined

**Step 3: Implement the section writers**

In `cli.py`, replace the existing `_write_report` function with these composable section writers. Add them in the "Display helpers" section:

```python
def _generate_report_path(command: str) -> str:
    """Generate a timestamped report filename."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"anam-prep-{command}-{ts}.md"


def _report_header(command: str, file_count: int) -> list[str]:
    """Report header with command name, timestamp, and file count."""
    from datetime import datetime
    lines = [
        f"# anam-prep {command} Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Command:** `anam-prep {command}`",
        f"**Files analyzed:** {file_count}",
        "",
    ]
    return lines


def _report_scores(cards: list[ScoreCard], detail: bool) -> list[str]:
    """Score summary table and optional issue detail."""
    lines = [
        "## Scores",
        "",
        "| File | Score | Readiness | Issues |",
        "|------|------:|:---------:|-------:|",
    ]
    for card in cards:
        filename = Path(card.file_path).name
        lines.append(
            f"| {filename} | {card.overall_score:.0f} | {card.readiness.value} | {len(card.all_issues)} |"
        )
    lines.append("")
    avg_score = sum(c.overall_score for c in cards) / len(cards) if cards else 0
    lines.append(f"**Average score:** {avg_score:.1f}")
    lines.append("")

    if detail:
        lines.append("## Issues Detail")
        lines.append("")
        for card in cards:
            if not card.all_issues:
                continue
            filename = Path(card.file_path).name
            lines.append(f"### {filename}")
            lines.append("")
            for issue in card.all_issues:
                sev = issue.severity.value.upper()
                lines.append(f"- **[{sev}]** `{issue.category}`: {issue.message}")
                if issue.fix:
                    lines.append(f"  - Fix: {issue.fix}")
            lines.append("")

    return lines


def _report_analyses(docs, analyses) -> list[str]:
    """Per-file content analysis section."""
    lines = ["## Content Analysis", ""]
    for doc, analysis in zip(docs, analyses):
        filename = doc.metadata.filename
        lines.append(f"### {filename}")
        lines.append("")
        if analysis.domain:
            lines.append(f"- **Domain:** {analysis.domain}")
        if analysis.topics:
            lines.append(f"- **Topics:** {', '.join(analysis.topics)}")
        if analysis.audience:
            lines.append(f"- **Audience:** {analysis.audience}")
        if analysis.content_type:
            lines.append(f"- **Type:** {analysis.content_type}")
        if analysis.key_concepts:
            lines.append(f"- **Concepts:** {', '.join(analysis.key_concepts)}")
        if analysis.summary:
            lines.append("")
            lines.append(f"> {analysis.summary}")
        if analysis.entities:
            lines.append("")
            lines.append(f"**Entities:** {', '.join(e.name for e in analysis.entities[:10])}")
        lines.append("")
    return lines


def _report_graph(graph) -> list[str]:
    """Knowledge graph summary section."""
    if not graph or graph.is_empty:
        return []
    summary = graph.summarize()
    lines = [
        "## Knowledge Graph",
        "",
        f"- **Entities:** {summary.total_entities}",
        f"- **Relationships:** {summary.total_relationships}",
        f"- **Cross-document edges:** {summary.cross_document_edges}",
        f"- **Topic clusters:** {len(summary.clusters)}",
        "",
    ]
    if summary.entity_types:
        lines.append("**Entity types:**")
        lines.append("")
        for etype, count in sorted(summary.entity_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {etype}: {count}")
        lines.append("")
    if summary.orphan_references:
        lines.append(f"**Orphan references:** {', '.join(summary.orphan_references)}")
        lines.append("")
    return lines


def _report_recommendations(recommendation) -> list[str]:
    """Folder recommendation section."""
    from recommender import format_folder_tree
    if not recommendation:
        return []
    lines = [
        "## Recommended Folder Structure",
        "",
        "```",
        format_folder_tree(recommendation.root),
        "```",
        "",
    ]
    if recommendation.file_assignments:
        lines.append("### File Assignments")
        lines.append("")
        lines.append("| File | Folder |")
        lines.append("|------|--------|")
        for filename, folder in recommendation.file_assignments.items():
            lines.append(f"| {filename} | {folder} |")
        lines.append("")
    return lines


def _report_fixes(fix_reports: list) -> list[str]:
    """Fix actions summary section."""
    if not fix_reports:
        return []
    lines = ["## Fixes Applied", ""]
    for report in fix_reports:
        filename = Path(report.source_path).name
        lines.append(f"### {filename}")
        lines.append(f"- **Output:** `{report.output_path}`")
        if report.new_filename:
            lines.append(f"- **Renamed to:** {report.new_filename}")
        if report.actions:
            lines.append(f"- **Actions:** {len(report.actions)}")
            for action in report.actions:
                lines.append(f"  - `{action.category}`: {action.description}")
        lines.append("")
    return lines


def _report_uploads(upload_report) -> list[str]:
    """Upload results section."""
    if not upload_report:
        return []
    lines = [
        "## Upload Results",
        "",
        f"- **Folders created:** {len(upload_report.folders_created)}",
        f"- **Files uploaded:** {len(upload_report.successful)}",
        f"- **Failed:** {len(upload_report.failed)}",
        "",
    ]
    if upload_report.successful:
        lines.append("### Successful")
        lines.append("")
        lines.append("| File | Document ID | Folder |")
        lines.append("|------|------------|--------|")
        for r in upload_report.successful:
            filename = Path(r.file_path).name
            lines.append(f"| {filename} | {r.document_id} | {r.folder_name} |")
        lines.append("")
    if upload_report.failed:
        lines.append("### Failed")
        lines.append("")
        for r in upload_report.failed:
            filename = Path(r.file_path).name
            lines.append(f"- **{filename}**: {r.error}")
        lines.append("")
    return lines


def _write_report_file(report_path: str, sections: list[list[str]]) -> None:
    """Assemble sections and write the report file."""
    lines = []
    for section in sections:
        lines.extend(section)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: 4 PASS

**Step 5: Commit**

```
refactor(cli): extract composable report section writers
```

---

### Task 3: Wire auto-report into `score` command

**Files:**
- Modify: `cli.py` (score command)
- Test: `tests/test_report.py` (extend)

**Step 1: Write the failing test**

Add to `tests/test_report.py`:

```python
import os
import tempfile
from unittest.mock import patch
from click.testing import CliRunner


def test_score_generates_report(tmp_path):
    """score command auto-generates a markdown report file."""
    from cli import cli
    # We need a real DOCX file — reuse test helper from test_scoring
    from tests.test_scoring import _create_test_docx

    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)

    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        result = runner.invoke(cli, ["score", test_file])
        assert result.exit_code == 0
        # Find the generated report
        reports = [f for f in os.listdir(td) if f.startswith("anam-prep-score-")]
        assert len(reports) == 1, f"Expected 1 report, found: {reports}"
        content = open(reports[0]).read()
        assert "# anam-prep score Report" in content
        assert "test.docx" in content


def test_score_no_report_flag(tmp_path):
    """--no-report suppresses report generation."""
    from cli import cli
    from tests.test_scoring import _create_test_docx

    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)

    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        result = runner.invoke(cli, ["score", "--no-report", test_file])
        assert result.exit_code == 0
        reports = [f for f in os.listdir(td) if f.startswith("anam-prep-")]
        assert len(reports) == 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report.py::test_score_generates_report -v`
Expected: FAIL — no report generated

**Step 3: Modify `score` command**

In `cli.py`, update the `score` command:
1. Replace `--report` with `--no-report` flag
2. After printing console output, auto-write report

```python
@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--detail", is_flag=True, help="Show per-issue details")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
def score(path: str, detail: bool, json_out: bool, no_report: bool, exclude: tuple):
    """Analyze and score documents for RAG readiness (no LLM required)."""
    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    console.print(f"\nFound [cyan]{len(files)}[/cyan] file(s) to analyze.\n")

    parser = DocumentParser()
    scorer = QualityScorer()
    cards: list[ScoreCard] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scoring documents...", total=len(files))
        for file_path in files:
            try:
                doc = parser.parse(file_path)
                card = scorer.score(doc)
                cards.append(card)
            except Exception as e:
                console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            progress.advance(task)

    if json_out:
        _print_json(cards)
    else:
        _print_score_table(cards, detail)

    # Auto-generate report
    if not no_report and cards:
        report_path = _generate_report_path("score")
        _write_report_file(report_path, [
            _report_header("score", len(cards)),
            _report_scores(cards, detail),
        ])
        console.print(f"\n[green]Report:[/green] {report_path}")
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat(cli): auto-generate markdown report on score command
```

---

### Task 4: Wire auto-report into `analyze` command

**Files:**
- Modify: `cli.py` (analyze command)
- Test: manual verification (LLM-dependent)

**Step 1: Update `analyze` command**

Replace the `--report` option with `--no-report`. Remove the old `if report:` block. Add auto-report at the end:

```python
@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--detail", is_flag=True, help="Show per-issue details")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
def analyze(path: str, llm_key: str, model: str, detail: bool, no_report: bool, exclude: tuple):
    # ... (existing parsing/analysis/scoring/display code stays the same,
    #      except remove the old `if report:` block and `--report` option) ...

    # Auto-generate report
    if not no_report:
        report_path = _generate_report_path("analyze")
        _write_report_file(report_path, [
            _report_header("analyze", len(docs)),
            _report_scores(cards, detail),
            _report_analyses(docs, analyses),
            _report_graph(graph),
            _report_recommendations(recommendation),
        ])
        console.print(f"\n[green]Report:[/green] {report_path}")
```

**Step 2: Also remove the old `_write_report` function** (now replaced by composable sections)

**Step 3: Commit**

```
feat(cli): auto-generate markdown report on analyze command

Replaces --report opt-in flag with --no-report opt-out.
```

---

### Task 5: Wire auto-report into `fix` command

**Files:**
- Modify: `cli.py` (fix command)

**Step 1: Update `fix` command**

Add `--no-report` flag. Collect `FixReport` objects during the fix loop. Write report at the end:

```python
@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--output", "-o", default="./fixed", help="Output directory for fixed files")
@click.option("--min-score", default=0.0, help="Skip files scoring at or above this value")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
def fix(path: str, llm_key: str, model: str, output: str, min_score: float, no_report: bool, exclude: tuple):
    # ... (existing code, but collect fix_reports) ...

    fix_reports = []  # Add this before the progress loop

    # Inside the fix loop, after `report = fixer_inst.fix(doc, card)`:
    fix_reports.append(report)

    # At end of command:
    if not no_report:
        report_path = _generate_report_path("fix")
        _write_report_file(report_path, [
            _report_header("fix", len(docs)),
            _report_scores(cards, detail=False),
            _report_fixes(fix_reports),
        ])
        console.print(f"\n[green]Report:[/green] {report_path}")
```

**Step 2: Commit**

```
feat(cli): auto-generate markdown report on fix command
```

---

### Task 6: Wire auto-report into `upload` command

**Files:**
- Modify: `cli.py` (upload command)

**Step 1: Update `upload` command**

Add `--no-report` flag. Collect fix_reports during fix step. Write comprehensive report after upload:

```python
# Add --no-report flag to upload command decorator
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")

# In the function signature, add: no_report: bool

# Collect fix_reports in the fix loop (same as Task 5)
fix_reports = []  # Before fix loop
# After `report = fixer_inst.fix(doc, card):` add: fix_reports.append(report)

# At the very end of the upload command (after Step 6 knowledge tool):
if not no_report:
    report_path = _generate_report_path("upload")
    _write_report_file(report_path, [
        _report_header("upload", len(docs)),
        _report_scores(cards, detail=False),
        _report_fixes(fix_reports),
        _report_recommendations(recommendation),
        _report_uploads(report),  # `report` is the UploadReport from upload_batch
    ])
    console.print(f"\n[green]Report:[/green] {report_path}")
```

Note: There's a variable name collision — the upload command uses `report` for `UploadReport`. Rename it to `upload_report` to avoid clashing with `fix_reports`.

**Step 2: Run existing tests to make sure nothing is broken**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```
feat(cli): auto-generate markdown report on upload command
```

---

### Task 7: Convert `ContentAnalyzer` to async

**Files:**
- Modify: `analyzer.py`
- Test: `tests/test_async_analyzer.py` (create)

**Step 1: Write the failing test**

Create `tests/test_async_analyzer.py`:

```python
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import ParsedDocument, DocumentMetadata, Paragraph


def _make_doc(filename="test.docx", text="This is test content about math."):
    return ParsedDocument(
        metadata=DocumentMetadata(file_path=f"/tmp/{filename}", file_type="docx"),
        paragraphs=[Paragraph(text=text, level=0, style="Normal", index=0)],
    )


def _mock_llm_response(json_str: str):
    """Create a mock Anthropic message response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json_str)]
    return msg


def test_analyze_is_async():
    """ContentAnalyzer.analyze is a coroutine."""
    import inspect
    from analyzer import ContentAnalyzer
    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze)


def test_analyze_and_build_graph_is_async():
    """analyze_and_build_graph is a coroutine."""
    import inspect
    from analyzer import ContentAnalyzer
    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze_and_build_graph)


def test_concurrent_analysis():
    """analyze_and_build_graph processes multiple docs concurrently."""
    config = Config(anthropic_api_key="test-key", concurrency=2)

    json_response = '{"domain":"education","topics":["math"],"audience":"students","content_type":"lesson","key_concepts":["fractions"],"suggested_tags":["math"],"summary":"A lesson.","entities":[],"relationships":[]}'

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.messages.create = AsyncMock(
            return_value=_mock_llm_response(json_response)
        )

        analyzer = ContentAnalyzer.__new__(ContentAnalyzer)
        analyzer.client = mock_instance
        analyzer.model = config.llm_model
        analyzer.max_tokens = config.llm_max_tokens
        analyzer._semaphore = asyncio.Semaphore(config.concurrency)

        docs = [_make_doc(f"doc{i}.docx") for i in range(4)]
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 4
        assert all(a.domain == "education" for a in analyses)
        # Verify all 4 calls were made
        assert mock_instance.messages.create.call_count == 4
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_async_analyzer.py -v`
Expected: FAIL — `analyze` is not async yet

**Step 3: Rewrite `analyzer.py` to async**

```python
"""LLM-powered content analysis with knowledge graph extraction.

Uses Claude to understand document content, extract topics, domains,
audience, and other metadata — plus entities and relationships that
form a cross-document knowledge graph. The graph is used downstream
by the fixer (cross-doc reference resolution), recommender (graph-based
folder clustering), and scorer (knowledge completeness).
"""

import asyncio
import json
import re
from typing import Optional

from anthropic import AsyncAnthropic

from config import Config
from graph_builder import KnowledgeGraph
from models import ContentAnalysis, Entity, ParsedDocument, Relationship
from prompts import ANALYZE_DOCUMENT


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
        text = doc.full_text
        words = text.split()
        if len(words) > 8000:
            text = " ".join(words[:8000]) + "\n\n[Document truncated for analysis...]"

        prompt = ANALYZE_DOCUMENT.format(document_text=text)

        async with self._semaphore:
            response = await self._call_with_retry(prompt)

        response_text = response.content[0].text.strip()
        data = self._extract_json(response_text)

        if data is None:
            return ContentAnalysis(summary="Analysis failed — could not parse LLM response.")

        source_file = doc.metadata.filename

        entities = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and "name" in e:
                entities.append(Entity(
                    name=e["name"],
                    entity_type=e.get("type", "concept"),
                    source_file=source_file,
                    description=e.get("description", ""),
                ))

        relationships = []
        for r in data.get("relationships", []):
            if isinstance(r, dict) and "source" in r and "target" in r:
                relationships.append(Relationship(
                    source=r["source"],
                    target=r["target"],
                    rel_type=r.get("type", "related_to"),
                    source_file=source_file,
                    context=r.get("context", ""),
                ))

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

    async def analyze_and_build_graph(
        self, docs: list[ParsedDocument]
    ) -> tuple[list[ContentAnalysis], KnowledgeGraph]:
        """Analyze all documents concurrently and build a shared knowledge graph."""
        # Launch all analyses concurrently (semaphore limits parallelism)
        tasks = [self.analyze(doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        graph = KnowledgeGraph()
        analyses = []

        for doc, result in zip(docs, results):
            if isinstance(result, Exception):
                analysis = ContentAnalysis(summary=f"Analysis failed: {result}")
            else:
                analysis = result
            analyses.append(analysis)
            graph.add_analysis(doc, analysis)

        return analyses, graph

    async def _call_with_retry(self, prompt: str, max_retries: int = 3):
        """Make an async LLM call with retry on rate limits."""
        for attempt in range(max_retries):
            try:
                return await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1 and ("429" in error_str or "529" in error_str or "rate" in error_str or "overloaded" in error_str):
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                raise

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Safely extract a JSON object from LLM response text."""
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

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
                    candidate = text[start:i + 1]
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

    # Keep backward compat
    async def analyze_batch(self, docs: list[ParsedDocument]) -> list[ContentAnalysis]:
        """Analyze multiple documents concurrently (without graph)."""
        analyses, _ = await self.analyze_and_build_graph(docs)
        return analyses
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_async_analyzer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat(analyzer): convert to async with concurrent LLM calls

Uses AsyncAnthropic and asyncio.Semaphore for parallel analysis.
```

---

### Task 8: Convert `DocumentFixer` to async

**Files:**
- Modify: `fixer.py`
- Test: `tests/test_async_fixer.py` (create)

**Step 1: Write the failing test**

Create `tests/test_async_fixer.py`:

```python
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import (
    ParsedDocument, DocumentMetadata, Paragraph,
    ScoreCard, ScoringResult, Issue, Severity,
)


def test_fix_is_async():
    """DocumentFixer.fix is a coroutine."""
    import inspect
    from fixer import DocumentFixer
    assert inspect.iscoroutinefunction(DocumentFixer.fix)


def test_call_llm_is_async():
    """DocumentFixer._call_llm is a coroutine."""
    import inspect
    from fixer import DocumentFixer
    assert inspect.iscoroutinefunction(DocumentFixer._call_llm)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_async_fixer.py -v`
Expected: FAIL — methods not async

**Step 3: Rewrite `fixer.py` to async**

Key changes:
- `from anthropic import AsyncAnthropic`
- `self.client = AsyncAnthropic(...)` in `__init__`, add `self._semaphore = asyncio.Semaphore(config.concurrency)`
- `_call_llm` becomes `async` with `await self.client.messages.create(...)` and `await asyncio.sleep(wait)`
- `fix` becomes `async`, all `_fix_*` methods become `async`
- All calls to `self._call_llm(...)` become `await self._call_llm(...)`
- The semaphore wraps `_call_llm` internally

The structure of `fix()` stays the same — phases are ordered, but each LLM call within a phase goes through the semaphore.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_async_fixer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat(fixer): convert to async with concurrent LLM calls

Uses AsyncAnthropic and asyncio.Semaphore for parallel fixes.
```

---

### Task 9: Convert `FolderRecommender` LLM calls to async

**Files:**
- Modify: `recommender.py`

**Step 1: Update recommender to async**

The recommender makes LLM calls in `_graph_llm_recommend` and `_llm_recommend`. These need to become async too.

Key changes:
- `from anthropic import AsyncAnthropic`
- `self._client = AsyncAnthropic(...)` in `__init__`
- `recommend` becomes `async`
- `_graph_llm_recommend` and `_llm_recommend` become `async`
- `self._client.messages.create(...)` calls become `await self._client.messages.create(...)`
- Heuristic methods stay sync (no LLM calls) but wrap in async signature for uniform interface

**Step 2: Run existing tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```
feat(recommender): convert LLM calls to async
```

---

### Task 10: Update CLI commands to use asyncio.run()

**Files:**
- Modify: `cli.py`

**Step 1: Update `analyze` command**

The key change: wrap the async calls in `asyncio.run()`. Since Click doesn't natively support async, the cleanest pattern is an inner async function:

```python
@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--detail", is_flag=True, help="Show per-issue details")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--concurrency", default=5, type=int, help="Max parallel LLM calls (default: 5)")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
def analyze(path, llm_key, model, detail, no_report, concurrency, exclude):
    """Score documents + LLM content analysis and folder recommendation."""
    import asyncio
    from analyzer import ContentAnalyzer
    from recommender import FolderRecommender, format_folder_tree

    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    config = Config.from_env().with_overrides(
        anthropic_api_key=llm_key,
        llm_model=model,
        concurrency=concurrency,
    )

    parser = DocumentParser()
    analyzer_inst = ContentAnalyzer(config)

    docs = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        parse_task = progress.add_task("Parsing documents...", total=len(files))
        for file_path in files:
            try:
                docs.append(parser.parse(file_path))
            except Exception as e:
                console.print(f"[red]Error: {file_path}: {e}[/red]")
            progress.advance(parse_task)

        progress.add_task("Analyzing content & building knowledge graph...", total=None)
        analyses, graph = asyncio.run(analyzer_inst.analyze_and_build_graph(docs))

    # Rest of the command stays the same (scoring, display, report)...
    # BUT recommender.recommend is now async:
    recommender = FolderRecommender(config, graph=graph)
    recommendation = asyncio.run(recommender.recommend(docs, analyses))
    # ... display and report code ...
```

**Step 2: Apply same pattern to `fix` and `upload`**

- `fix`: `asyncio.run(analyzer_inst.analyze_and_build_graph(docs))`, and wrap the fixer loop in an async function that uses `asyncio.gather` with semaphore for cross-document parallelism
- `upload`: Same async wrapping for analysis, fixing, and recommendation steps

**Step 3: Add `--concurrency` flag to `fix` and `upload` commands**

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat(cli): wire async LLM calls with --concurrency flag

analyze, fix, and upload commands now run LLM calls concurrently
using asyncio.run() with configurable parallelism (default: 5).
```

---

### Task 11: Integration test with mock LLM

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import ParsedDocument, DocumentMetadata, Paragraph


MOCK_ANALYSIS_JSON = '{"domain":"education","topics":["math"],"audience":"students","content_type":"lesson","key_concepts":["fractions"],"suggested_tags":["math"],"summary":"A math lesson.","entities":[{"name":"Fractions","type":"concept","description":"Number parts"}],"relationships":[]}'


def _mock_response(text):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_docs(n=3):
    return [
        ParsedDocument(
            metadata=DocumentMetadata(file_path=f"/tmp/doc{i}.docx", file_type="docx"),
            paragraphs=[Paragraph(text=f"Content about math topic {i}.", level=0, style="Normal", index=0)],
        )
        for i in range(n)
    ]


def test_full_analyze_pipeline_concurrent():
    """analyze_and_build_graph runs concurrently and builds correct graph."""
    from analyzer import ContentAnalyzer

    config = Config(anthropic_api_key="test-key", concurrency=2)

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock = MockClient.return_value
        mock.messages.create = AsyncMock(return_value=_mock_response(MOCK_ANALYSIS_JSON))

        analyzer = ContentAnalyzer(config)
        docs = _make_docs(5)
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 5
        assert not graph.is_empty
        assert mock.messages.create.call_count == 5
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```
test: add integration test for concurrent analysis pipeline
```

---

### Task 12: Update docstring in cli.py header

**Files:**
- Modify: `cli.py` (top-level docstring)

**Step 1: Update the module docstring to reflect new defaults**

```python
"""anam-prep — Document preparation and upload CLI for anam.ai RAG.

Usage:
    anam-prep score   <path>                          Score documents for RAG readiness
    anam-prep analyze <path> --llm-key KEY            Score + LLM content analysis
    anam-prep fix     <path> --llm-key KEY            Score + analyze + auto-fix issues
    anam-prep upload  <path> --api-key KEY            Full pipeline: fix → recommend → upload

All commands auto-generate a timestamped Markdown report (suppress with --no-report).
LLM commands support --concurrency N (default: 5) for parallel API calls.
"""
```

**Step 2: Commit**

```
docs: update CLI docstring with report and concurrency info
```

---

## Summary

| Task | Description | Est. |
|------|-------------|------|
| 1 | Config: add `concurrency` field | 2 min |
| 2 | Refactor report into composable sections | 10 min |
| 3 | Wire auto-report into `score` | 5 min |
| 4 | Wire auto-report into `analyze` | 5 min |
| 5 | Wire auto-report into `fix` | 5 min |
| 6 | Wire auto-report into `upload` | 5 min |
| 7 | Convert `ContentAnalyzer` to async | 10 min |
| 8 | Convert `DocumentFixer` to async | 10 min |
| 9 | Convert `FolderRecommender` to async | 5 min |
| 10 | Wire asyncio.run() + --concurrency in CLI | 10 min |
| 11 | Integration test with mock LLM | 5 min |
| 12 | Update docstring | 1 min |

**Dependencies:** Tasks 1-6 (reports) are independent of Tasks 7-12 (concurrency). Within each group, tasks are sequential.
