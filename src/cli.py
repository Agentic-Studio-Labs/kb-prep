#!/usr/bin/env python3
"""IngestGate — Retrieval Quality Gate for RAG pipelines.

Usage:
    ingestgate score   <path>                          Score documents for retrieval health
    ingestgate analyze <path> --llm-key KEY            Analyze, benchmark, and emit decision support
    ingestgate fix     <path> --llm-key KEY            Analyze + auto-fix issues before ingestion

All commands auto-generate a timestamped Markdown report (suppress with --no-report).
LLM commands support --concurrency N (default: 5) for parallel API calls.
"""

import asyncio
import os
import re
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Config
from .corpus_analyzer import build_corpus_analysis
from .models import ChunkSet, ScoreCard, Severity
from .parser import DocumentParser, discover_files
from .scorer import QualityScorer, generate_split_recommendations

console = Console()


_BENCHMARK_QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "your",
    "about",
    "over",
    "under",
    "when",
    "where",
    "what",
    "which",
}
_BENCHMARK_QUERY_NOTE = "query_source: heading+tfidf deterministic"


def _extract_query_terms(text: str) -> list[str]:
    """Tokenize text into stable lowercase query terms."""
    return [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text)
        if token.lower() not in _BENCHMARK_QUERY_STOPWORDS
    ]


def _build_benchmark_query(doc, corpus_analysis, max_terms: int = 6) -> str:
    """Build a deterministic benchmark query from headings + top TF-IDF terms."""
    terms: list[str] = []
    seen: set[str] = set()

    for heading in doc.headings:
        for token in _extract_query_terms(heading.text):
            if token not in seen:
                terms.append(token)
                seen.add(token)
            if len(terms) >= max_terms:
                return " ".join(terms)

    if corpus_analysis and hasattr(corpus_analysis, "tfidf_matrix") and hasattr(corpus_analysis, "feature_names"):
        try:
            doc_idx = corpus_analysis.doc_labels.index(doc.metadata.filename)
            row = corpus_analysis.tfidf_matrix[doc_idx]
            if row.nnz > 0:
                ranked = sorted(zip(row.indices.tolist(), row.data.tolist()), key=lambda x: x[1], reverse=True)
                for term_idx, _score in ranked:
                    token = corpus_analysis.feature_names[term_idx].lower()
                    if token not in seen and token not in _BENCHMARK_QUERY_STOPWORDS:
                        terms.append(token)
                        seen.add(token)
                    if len(terms) >= max_terms:
                        break
        except Exception:
            pass

    if not terms:
        stem_tokens = _extract_query_terms(doc.metadata.stem)
        if stem_tokens:
            return " ".join(stem_tokens[:max_terms])
        return doc.metadata.filename

    return " ".join(terms[:max_terms])


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="ingestgate")
def cli():
    """Run the Retrieval Quality Gate before ingestion."""
    pass


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--detail", is_flag=True, help="Show per-issue details")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
def score(path: str, detail: bool, json_out: bool, exclude: tuple, no_report: bool):
    """Analyze and score documents for RAG readiness (no LLM required)."""
    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    console.print(f"\nFound [cyan]{len(files)}[/cyan] file(s) to analyze.\n")

    parser = DocumentParser()
    docs = []
    cards: list[ScoreCard] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Parsing documents...", total=len(files))
        for file_path in files:
            try:
                doc = parser.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            progress.advance(task)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing corpus...", total=None)
        corpus_analysis = build_corpus_analysis(docs)

    scorer = QualityScorer(
        corpus_analysis=corpus_analysis,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scoring documents...", total=len(docs))
        cards = []
        for doc in docs:
            cards.append(scorer.score(doc))
            progress.advance(task)

    if json_out:
        _print_json(cards)
    else:
        _print_score_table(cards, detail)

    # Auto-generate report
    if not no_report and cards:
        report_path = _generate_report_path("score")
        _write_report_file(
            report_path,
            [
                _report_header("score", len(cards)),
                _report_scores(cards, detail),
            ],
        )
        console.print(f"\n[green]Report:[/green] {report_path}")


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--detail", is_flag=True, help="Show per-issue details")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--concurrency", default=5, type=int, help="Max parallel LLM calls (default: 5)")
@click.option("--no-export-meta", is_flag=True, help="Skip writing metadata export files")
@click.option("--json-output", "json_out", is_flag=True, help="Output manifest JSON to stdout")
@click.option("--export-chunks/--no-export-chunks", default=True, help="Write per-document .chunks.json files")
@click.option("--chunk-size", default=220, type=int, help="Target words per chunk")
@click.option("--chunk-overlap", default=40, type=int, help="Overlapping words between chunks")
@click.option("--run-benchmark", is_flag=True, help="Run chunk-level retrieval benchmark")
def analyze(
    path: str,
    llm_key: str,
    model: str,
    detail: bool,
    exclude: tuple,
    no_report: bool,
    concurrency: int,
    no_export_meta: bool,
    json_out: bool,
    export_chunks: bool,
    chunk_size: int,
    chunk_overlap: int,
    run_benchmark: bool,
):
    """Score documents + LLM content analysis."""
    from .analyzer import ContentAnalyzer
    from .benchmark import benchmark_chunk_retrieval
    from .chunker import DocumentChunker
    from .cleaner import DocumentCleaner
    from .export import build_manifest_data, write_chunk_sidecar, write_manifest, write_sidecar

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
    cleaner = DocumentCleaner()

    docs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Parsing documents...", total=len(files))
        for file_path in files:
            try:
                doc = parser.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Error: {file_path}: {e}[/red]")
            progress.advance(parse_task)

        cleaned_docs = []
        for doc in docs:
            cleaned_docs.append(
                doc.__class__(
                    metadata=doc.metadata,
                    paragraphs=cleaner.clean_document(doc.paragraphs),
                    heading_tree=doc.heading_tree,
                )
            )
        docs = cleaned_docs

        progress.add_task("Analyzing content & building knowledge graph...", total=None)
        analyses, graph = asyncio.run(analyzer_inst.analyze_and_build_graph(docs))

    # Score with graph and corpus context
    corpus_analysis = build_corpus_analysis(docs)
    scorer = QualityScorer(
        graph=graph,
        corpus_analysis=corpus_analysis,
    )
    cards = [scorer.score(doc) for doc in docs]
    chunker = DocumentChunker(target_words=chunk_size, overlap_words=chunk_overlap)
    chunk_sets = [chunker.chunk_document(doc) for doc in docs]
    split_recommendations = generate_split_recommendations(docs, cards, corpus_analysis=corpus_analysis)
    benchmarks = []

    if run_benchmark:
        all_chunks: list[str] = []
        chunk_source_files: list[str] = []
        for chunk_set in chunk_sets:
            for chunk in chunk_set.chunks:
                all_chunks.append(chunk.text)
                chunk_source_files.append(chunk.source_file)

        if all_chunks:
            queries = []
            gold_sets = []
            for doc in docs:
                query = _build_benchmark_query(doc, corpus_analysis)
                if not query:
                    continue
                gold = {i for i, source in enumerate(chunk_source_files) if source == doc.metadata.filename}
                if not gold:
                    continue
                queries.append(query)
                gold_sets.append(gold)
            benchmarks = benchmark_chunk_retrieval(queries, gold_sets, all_chunks, top_k=5)
            for benchmark in benchmarks:
                if _BENCHMARK_QUERY_NOTE not in benchmark.notes:
                    benchmark.notes.append(_BENCHMARK_QUERY_NOTE)

    # Show scores
    _print_score_table(cards, detail)

    # Show analysis results
    console.print("\n")
    for doc, analysis in zip(docs, analyses):
        _print_analysis(doc.metadata.filename, analysis)

    _print_graph_summary(graph)

    # Auto-generate report
    if not no_report:
        report_path = _generate_report_path("analyze")
        _write_report_file(
            report_path,
            [
                _report_header(
                    "analyze",
                    len(docs),
                    settings={
                        "model": config.llm_model,
                        "concurrency": concurrency,
                        "chunk-size": chunk_size,
                        "chunk-overlap": chunk_overlap,
                        "run-benchmark": run_benchmark,
                        "export-meta": not no_export_meta,
                        "export-chunks": export_chunks,
                    },
                ),
                _report_scores(cards, detail),
                _report_analyses(docs, analyses),
                _report_graph(graph),
            ],
        )
        console.print(f"\n[green]Report:[/green] {report_path}")

    # Export metadata
    if json_out:
        import json as _json

        data = build_manifest_data(
            docs,
            analyses,
            cards,
            corpus_analysis,
            graph,
            chunk_sets=chunk_sets,
            benchmarks=benchmarks,
            split_recommendations=split_recommendations,
        )
        print(_json.dumps(data))
    elif not no_export_meta:
        meta_dir = os.path.join(path, ".ingestgate")
        if export_chunks:
            for chunk_set in chunk_sets:
                write_chunk_sidecar(meta_dir, chunk_set)
        for doc, analysis, card in zip(docs, analyses, cards):
            filename = doc.metadata.filename
            doc_metrics = corpus_analysis.doc_metrics.get(filename)
            write_sidecar(meta_dir, doc.metadata.stem, doc, analysis, card, doc_metrics)
        manifest_path = write_manifest(
            meta_dir,
            docs,
            analyses,
            cards,
            corpus_analysis,
            graph,
            chunk_sets=chunk_sets,
            benchmarks=benchmarks,
            split_recommendations=split_recommendations,
        )
        console.print(f"[green]Metadata:[/green] {manifest_path}")


# ---------------------------------------------------------------------------
# fix command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--output", "-o", default=None, help="Output directory (default: ingestgate-files-{timestamp}/)")
@click.option("--fix-below", default=0.0, help="Only fix documents scoring below this threshold (e.g. --fix-below 70)")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--concurrency", default=5, type=int, help="Max parallel LLM calls (default: 5)")
@click.option("--no-export-meta", is_flag=True, help="Skip writing .meta.json sidecar files and manifest.json")
@click.option("--chunk-size", default=220, type=int, help="Target words per chunk")
@click.option("--chunk-overlap", default=40, type=int, help="Overlapping words between chunks")
def fix(
    path: str,
    llm_key: str,
    model: str,
    output: str,
    fix_below: float,
    exclude: tuple,
    no_report: bool,
    concurrency: int,
    no_export_meta: bool,
    chunk_size: int,
    chunk_overlap: int,
):
    """Score + auto-fix issues, output improved Markdown files."""
    from datetime import datetime

    from .analyzer import ContentAnalyzer
    from .chunker import DocumentChunker
    from .cleaner import DocumentCleaner
    from .export import write_chunk_sidecar, write_manifest, write_sidecar
    from .fixer import DocumentFixer
    from .parser import to_markdown

    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    # Generate timestamped output directory if not specified
    if output is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"ingestgate-files-{ts}"

    config = Config.from_env().with_overrides(
        anthropic_api_key=llm_key,
        llm_model=model,
        output_dir=output,
        concurrency=concurrency,
    )

    parser = DocumentParser()
    analyzer_inst = ContentAnalyzer(config)
    cleaner = DocumentCleaner()

    # Parse all documents first
    docs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing documents...", total=len(files))
        for file_path in files:
            try:
                doc = parser.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            progress.advance(task)

    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
            doc.__class__(
                metadata=doc.metadata,
                paragraphs=cleaner.clean_document(doc.paragraphs),
                heading_tree=doc.heading_tree,
            )
        )
    docs = cleaned_docs

    # Analyze and build knowledge graph
    console.print("[dim]Building knowledge graph...[/dim]")
    analyses, graph = asyncio.run(analyzer_inst.analyze_and_build_graph(docs))

    # Score with graph and corpus context, then fix with graph context
    corpus_analysis = build_corpus_analysis(docs)
    scorer = QualityScorer(
        graph=graph,
        corpus_analysis=corpus_analysis,
    )
    cards = [scorer.score(doc) for doc in docs]
    chunker = DocumentChunker(target_words=chunk_size, overlap_words=chunk_overlap)
    chunk_sets = [chunker.chunk_document(doc) for doc in docs]
    split_recommendations = generate_split_recommendations(docs, cards, corpus_analysis=corpus_analysis)
    chunk_sets_by_file = {cs.source_file: cs for cs in chunk_sets}
    fixer_inst = DocumentFixer(config, graph=graph)
    fix_reports = []

    # Gather docs that need fixing
    docs_to_fix = []
    for doc, card in zip(docs, cards):
        if fix_below > 0 and card.overall_score >= fix_below:
            console.print(f"[dim]Skipping {doc.metadata.filename} (score: {card.overall_score:.0f})[/dim]")
            continue
        if not card.all_issues:
            console.print(f"[green]✓ {doc.metadata.filename} — no issues to fix[/green]")
            continue
        docs_to_fix.append((doc, card))

    if docs_to_fix:

        async def _fix_all():
            tasks = [fixer_inst.fix(doc, card) for doc, card in docs_to_fix]
            return await asyncio.gather(*tasks, return_exceptions=True)

        console.print(f"[dim]Fixing {len(docs_to_fix)} document(s) concurrently...[/dim]")
        results = asyncio.run(_fix_all())

        for (doc, card), result in zip(docs_to_fix, results):
            if isinstance(result, Exception):
                console.print(f"[red]Error fixing {doc.metadata.file_path}: {result}[/red]")
            else:
                fix_reports.append(result)
                console.print(
                    f"[green]✓ {doc.metadata.filename}[/green] → "
                    f"[cyan]{result.output_path}[/cyan] "
                    f"({len(result.actions)} fixes applied)"
                )
                for action in result.actions:
                    console.print(f"  [dim]• {action.description}[/dim]")

    _print_graph_summary(graph)

    # --- Write fixed files flat to output directory ---
    fixed_name_map: dict[str, str] = {}
    for report in fix_reports:
        orig_name = Path(report.source_path).name
        fixed_stem = Path(report.output_path).stem
        fixed_name_map[orig_name] = fixed_stem

    os.makedirs(output, exist_ok=True)
    _analysis_map = {doc.metadata.filename: analysis for doc, analysis in zip(docs, analyses)}
    _card_map = {doc.metadata.filename: card for doc, card in zip(docs, cards)}

    for doc in docs:
        filename = doc.metadata.filename
        fixed_stem = fixed_name_map.get(filename, doc.metadata.stem)
        md_name = f"{fixed_stem}.md"
        target_path = os.path.join(output, md_name)

        flat_path = os.path.join(output, md_name)
        if not os.path.exists(flat_path):
            md_content = to_markdown(doc)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        if not no_export_meta:
            doc_metrics = corpus_analysis.doc_metrics.get(filename)
            write_sidecar(output, fixed_stem, doc, _analysis_map[filename], _card_map[filename], doc_metrics)
            original_cs = chunk_sets_by_file.get(filename)
            if original_cs:
                sidecar_cs = ChunkSet(
                    document_id=fixed_stem,
                    source_file=original_cs.source_file,
                    chunks=original_cs.chunks,
                )
                write_chunk_sidecar(output, sidecar_cs)

    if not no_export_meta:
        manifest_path = write_manifest(
            output,
            docs,
            analyses,
            cards,
            corpus_analysis,
            graph,
            chunk_sets=chunk_sets,
            benchmarks=[],
            split_recommendations=split_recommendations,
        )
        console.print(f"[green]Metadata:[/green] {manifest_path} + per-doc sidecars")

    console.print(f"\nFixed files written to [cyan]{output}/[/cyan]")

    # --- Report inside output folder ---
    if not no_report:
        report_name = f"ingestgate-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        report_path = os.path.join(output, report_name)
        _write_report_file(
            report_path,
            [
                _report_header(
                    "fix",
                    len(docs),
                    settings={
                        "model": config.llm_model,
                        "concurrency": concurrency,
                        "chunk-size": chunk_size,
                        "chunk-overlap": chunk_overlap,
                        "fix-below": fix_below if fix_below > 0 else None,
                        "export-meta": not no_export_meta,
                    },
                ),
                _report_scores(cards, detail=False),
                _report_fixes(fix_reports),
            ],
        )
        console.print(f"[green]Report:[/green] {report_path}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_score_table(cards: list[ScoreCard], detail: bool):
    """Print a summary table of all scored documents."""
    table = Table(title="IngestGate Scores")
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Score", justify="right")
    table.add_column("Readiness", justify="center")
    table.add_column("Issues", justify="right")
    table.add_column("Size", justify="right")

    for card in cards:
        # Color-code the score
        score = card.overall_score
        if score >= 85:
            score_str = f"[green]{score:.0f}[/green]"
        elif score >= 70:
            score_str = f"[yellow]{score:.0f}[/yellow]"
        elif score >= 50:
            score_str = f"[dark_orange]{score:.0f}[/dark_orange]"
        else:
            score_str = f"[red]{score:.0f}[/red]"

        readiness = card.readiness.value
        readiness_colors = {
            "EXCELLENT": "green",
            "GOOD": "yellow",
            "FAIR": "dark_orange",
            "POOR": "red",
        }
        readiness_str = (
            f"[{readiness_colors.get(readiness, 'white')}]{readiness}[/{readiness_colors.get(readiness, 'white')}]"
        )

        # File size
        size_result = next((r for r in card.results if r.category == "file_size"), None)
        size_str = ""
        if size_result and size_result.issues:
            size_str = f"[red]{size_result.issues[0].message.split('—')[0].strip()}[/red]"

        filename = Path(card.file_path).name
        issue_count = len(card.all_issues)

        table.add_row(filename, score_str, readiness_str, str(issue_count), size_str)

    console.print(table)

    if detail:
        for card in cards:
            if card.all_issues:
                console.print(f"\n[bold]{Path(card.file_path).name}[/bold] — Issues:")

                # Group by category
                by_cat: dict[str, list] = {}
                for issue in card.all_issues:
                    by_cat.setdefault(issue.category, []).append(issue)

                for cat, issues in by_cat.items():
                    # Find the scoring result for label
                    result = next((r for r in card.results if r.category == cat), None)
                    label = result.label if result else cat
                    cat_score = result.score if result else 0
                    severity_color = _severity_color(issues[0].severity)

                    console.print(f"\n  [{severity_color}]{label}[/{severity_color}] (score: {cat_score:.0f})")
                    for issue in issues[:5]:  # Limit display
                        console.print(f"    • {issue.message}")
                        if issue.fix:
                            console.print(f"      [dim]Fix: {issue.fix}[/dim]")
                    if len(issues) > 5:
                        console.print(f"    [dim]... and {len(issues) - 5} more[/dim]")


def _print_graph_summary(graph):
    """Print knowledge graph summary statistics."""
    if not graph or graph.is_empty:
        return

    summary = graph.summarize()
    parts = [
        f"[bold]Entities:[/bold] {summary.total_entities}",
        f"[bold]Relationships:[/bold] {summary.total_relationships}",
        f"[bold]Cross-document edges:[/bold] {summary.cross_document_edges}",
        f"[bold]Topic clusters:[/bold] {len(summary.clusters)}",
    ]

    if summary.entity_types:
        type_str = ", ".join(f"{t}: {c}" for t, c in sorted(summary.entity_types.items(), key=lambda x: -x[1])[:6])
        parts.append(f"[bold]Entity types:[/bold] {type_str}")

    if summary.orphan_references:
        parts.append(f"[bold]Orphan references:[/bold] {', '.join(summary.orphan_references[:5])}")
        if len(summary.orphan_references) > 5:
            parts[-1] += f" (+{len(summary.orphan_references) - 5} more)"

    console.print(Panel("\n".join(parts), title="Knowledge Graph", border_style="magenta"))


def _print_analysis(filename: str, analysis):
    """Print LLM content analysis for a document."""

    panel_content = []
    if analysis.domain:
        panel_content.append(f"[bold]Domain:[/bold] {analysis.domain}")
    if analysis.topics:
        panel_content.append(f"[bold]Topics:[/bold] {', '.join(analysis.topics)}")
    if analysis.audience:
        panel_content.append(f"[bold]Audience:[/bold] {analysis.audience}")
    if analysis.content_type:
        panel_content.append(f"[bold]Type:[/bold] {analysis.content_type}")
    if analysis.key_concepts:
        panel_content.append(f"[bold]Concepts:[/bold] {', '.join(analysis.key_concepts[:8])}")
    if analysis.summary:
        panel_content.append(f"\n{analysis.summary}")

    console.print(Panel("\n".join(panel_content), title=filename, border_style="blue"))


def _print_json(cards: list[ScoreCard]):
    """Output scoring results as JSON."""
    import json

    output = []
    for card in cards:
        output.append(
            {
                "file": card.file_path,
                "score": round(card.overall_score, 1),
                "readiness": card.readiness.value,
                "categories": {
                    r.category: {
                        "score": round(r.score, 1),
                        "issues": len(r.issues),
                    }
                    for r in card.results
                },
                "issues": [
                    {
                        "severity": i.severity.value,
                        "category": i.category,
                        "message": i.message,
                        "fix": i.fix,
                    }
                    for i in card.all_issues
                ],
            }
        )
    console.print_json(json.dumps(output, indent=2))


def _severity_color(severity: Severity) -> str:
    return {
        Severity.CRITICAL: "red",
        Severity.WARNING: "yellow",
        Severity.INFO: "dim",
    }.get(severity, "white")


# ---------------------------------------------------------------------------
# Composable report section writers (Task 2)
# ---------------------------------------------------------------------------


def _generate_report_path(command: str) -> str:
    """Return a timestamped report filename: ingestgate-{command}-{YYYYMMDD-HHMMSS}.md"""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"ingestgate-{command}-{ts}.md"


def _report_header(command: str, file_count: int, settings: dict | None = None) -> list[str]:
    """Markdown header with command name, timestamp, file count, and run settings."""
    from datetime import datetime

    lines: list[str] = []
    lines.append(f"# IngestGate {command} Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Files:** {file_count}")
    if settings:
        parts = [f"{k}={v}" for k, v in settings.items() if v is not None]
        if parts:
            lines.append(f"**Settings:** {', '.join(parts)}")
    lines.append("")
    return lines


def _report_scores(cards: list[ScoreCard], detail: bool) -> list[str]:
    """Score summary table + optional issue detail."""
    lines: list[str] = []
    lines.append("## Scores")
    lines.append("")
    lines.append("| File | Score | Readiness | Issues |")
    lines.append("|------|------:|:---------:|-------:|")
    for card in cards:
        filename = Path(card.file_path).name
        lines.append(f"| {filename} | {card.overall_score:.0f} | {card.readiness.value} | {len(card.all_issues)} |")
    lines.append("")

    avg_score = sum(c.overall_score for c in cards) / len(cards) if cards else 0
    lines.append(f"**Average score:** {avg_score:.1f}")
    lines.append("")

    parse_fidelity_warning_files: list[str] = []
    parse_fidelity_note_files: list[str] = []
    for card in cards:
        for issue in card.all_issues:
            if "Low parse fidelity" not in issue.message:
                continue
            filename = Path(card.file_path).name
            if issue.severity.value in {"warning", "critical"}:
                parse_fidelity_warning_files.append(filename)
            else:
                parse_fidelity_note_files.append(filename)
    if parse_fidelity_warning_files:
        lines.append(f"**Parse fidelity warnings:** {len(parse_fidelity_warning_files)} file(s)")
        lines.append("")
        for filename in parse_fidelity_warning_files:
            lines.append(f"- {filename}")
        lines.append("")
    if parse_fidelity_note_files:
        lines.append(f"**Parse fidelity notes (expected sparse templates): {len(parse_fidelity_note_files)} file(s)**")
        lines.append("")
        for filename in parse_fidelity_note_files:
            lines.append(f"- {filename}")
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
    lines: list[str] = []
    lines.append("## Content Analysis")
    lines.append("")
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
    lines: list[str] = []
    if not graph or graph.is_empty:
        return lines

    summary = graph.summarize()
    lines.append("## Knowledge Graph")
    lines.append("")
    lines.append(f"- **Entities:** {summary.total_entities}")
    lines.append(f"- **Relationships:** {summary.total_relationships}")
    lines.append(f"- **Cross-document edges:** {summary.cross_document_edges}")
    lines.append(f"- **Topic clusters:** {len(summary.clusters)}")
    lines.append("")
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


def _report_fixes(fix_reports: list) -> list[str]:
    """Fix actions summary section."""
    lines: list[str] = []
    if not fix_reports:
        return lines

    lines.append("## Fixes Applied")
    lines.append("")
    for report in fix_reports:
        filename = Path(report.source_path).name
        lines.append(f"### {filename}")
        lines.append("")
        lines.append(f"- **Output:** {report.output_path}")
        lines.append(f"- **Actions:** {len(report.actions)}")
        if report.new_filename:
            lines.append(f"- **Renamed to:** {report.new_filename}")
        if report.new_files:
            lines.append(f"- **Split into:** {', '.join(report.new_files)}")
        lines.append("")
        for action in report.actions:
            lines.append(f"- `{action.category}`: {action.description}")
        lines.append("")
    return lines


def _write_report_file(report_path: str, sections: list[list[str]]) -> None:
    """Assemble sections and write the report file."""
    all_lines: list[str] = []
    for section in sections:
        all_lines.extend(section)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
