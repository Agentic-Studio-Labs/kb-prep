#!/usr/bin/env python3
"""kb-prep — Document preparation and upload CLI for anam.ai RAG.

Usage:
    kb-prep score   <path>                          Score documents for RAG readiness
    kb-prep analyze <path> --llm-key KEY            Score + LLM content analysis
    kb-prep fix     <path> --llm-key KEY            Score + analyze + auto-fix issues
    kb-prep upload  <path> --api-key KEY            Full pipeline: fix → recommend → upload

All commands auto-generate a timestamped Markdown report (suppress with --no-report).
LLM commands support --concurrency N (default: 5) for parallel API calls.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Config
from .corpus_analyzer import build_corpus_analysis
from .models import ScoreCard, Severity
from .parser import DocumentParser, discover_files
from .scorer import QualityScorer

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="kb-prep")
def cli():
    """Prepare and upload documents to anam.ai knowledge base."""
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
@click.option(
    "--folder-hints", default=None, type=click.Path(exists=True), help="File with domain-specific folder guidance"
)
def analyze(
    path: str,
    llm_key: str,
    model: str,
    detail: bool,
    exclude: tuple,
    no_report: bool,
    concurrency: int,
    folder_hints: str,
):
    """Score documents + LLM content analysis and folder recommendation."""
    from .analyzer import ContentAnalyzer
    from .recommender import FolderRecommender, format_folder_tree

    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    hints_text = Path(folder_hints).read_text().strip() if folder_hints else ""
    config = Config.from_env().with_overrides(
        anthropic_api_key=llm_key,
        llm_model=model,
        concurrency=concurrency,
        folder_hints=hints_text or None,
    )

    parser = DocumentParser()
    analyzer_inst = ContentAnalyzer(config)

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

        progress.add_task("Analyzing content & building knowledge graph...", total=None)
        analyses, graph = asyncio.run(analyzer_inst.analyze_and_build_graph(docs))

    # Score with graph and corpus context
    corpus_analysis = build_corpus_analysis(docs)
    scorer = QualityScorer(
        graph=graph,
        corpus_analysis=corpus_analysis,
    )
    cards = [scorer.score(doc) for doc in docs]

    # Show scores
    _print_score_table(cards, detail)

    # Show analysis results
    console.print("\n")
    for doc, analysis in zip(docs, analyses):
        _print_analysis(doc.metadata.filename, analysis)

    # Show knowledge graph summary
    _print_graph_summary(graph)

    # Show folder recommendation (graph-aware)
    console.print("\n")
    recommender = FolderRecommender(config, graph=graph)
    recommendation = asyncio.run(recommender.recommend(docs, analyses))
    tree_str = format_folder_tree(recommendation.root)
    console.print(Panel(tree_str, title="Recommended Folder Structure", border_style="green"))

    if recommendation.file_assignments:
        assign_table = Table(title="File → Folder Assignments")
        assign_table.add_column("File", style="cyan")
        assign_table.add_column("Folder", style="green")
        for filename, folder in recommendation.file_assignments.items():
            assign_table.add_row(filename, folder)
        console.print(assign_table)

    # Validate folder assignments
    sil_score = 0.0
    misplaced: list[tuple[str, float]] = []
    if hasattr(corpus_analysis, "similarity_matrix") and corpus_analysis.similarity_matrix.size > 0:
        sil_score, misplaced = recommender.validate_assignments(
            recommendation.file_assignments,
            corpus_analysis.similarity_matrix,
            corpus_analysis.doc_labels,
        )
        if sil_score != 0:
            console.print(f"\n[dim]Folder coherence (silhouette): {sil_score:.2f}[/dim]")
        if misplaced:
            console.print(f"[yellow]Warning: {len(misplaced)} document(s) may be misplaced:[/yellow]")
            for filename, score in misplaced:
                console.print(f"  [yellow]{filename} (silhouette: {score:.2f})[/yellow]")

    # Auto-generate report
    if not no_report:
        report_path = _generate_report_path("analyze")
        _write_report_file(
            report_path,
            [
                _report_header("analyze", len(docs)),
                _report_scores(cards, detail),
                _report_analyses(docs, analyses),
                _report_graph(graph),
                _report_recommendations(recommendation, sil_score if sil_score != 0 else None, misplaced or None),
            ],
        )
        console.print(f"\n[green]Report:[/green] {report_path}")


# ---------------------------------------------------------------------------
# fix command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default=None, help="LLM model override")
@click.option("--output", "-o", default=None, help="Output directory (default: rag-files-{timestamp}/)")
@click.option("--fix-below", default=0.0, help="Only fix documents scoring below this threshold (e.g. --fix-below 70)")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--concurrency", default=5, type=int, help="Max parallel LLM calls (default: 5)")
@click.option(
    "--folder-hints", default=None, type=click.Path(exists=True), help="File with domain-specific folder guidance"
)
def fix(
    path: str,
    llm_key: str,
    model: str,
    output: str,
    fix_below: float,
    exclude: tuple,
    no_report: bool,
    concurrency: int,
    folder_hints: str,
):
    """Score + auto-fix issues, output improved Markdown files."""
    from datetime import datetime

    from .analyzer import ContentAnalyzer
    from .fixer import DocumentFixer
    from .parser import to_markdown
    from .recommender import FolderRecommender, format_folder_tree

    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    # Generate timestamped output directory if not specified
    if output is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"rag-files-{ts}"

    hints_text = Path(folder_hints).read_text().strip() if folder_hints else ""

    config = Config.from_env().with_overrides(
        anthropic_api_key=llm_key,
        llm_model=model,
        output_dir=output,
        concurrency=concurrency,
        folder_hints=hints_text or None,
    )

    parser = DocumentParser()
    analyzer_inst = ContentAnalyzer(config)

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

    # --- Recommender step: get folder assignments ---
    console.print("[dim]Recommending folder structure...[/dim]")
    recommender = FolderRecommender(config, graph=graph)
    recommendation = asyncio.run(recommender.recommend(docs, analyses))

    # Build map of original filename → fixed filename (stem only, no .md)
    fixed_name_map: dict[str, str] = {}
    for report in fix_reports:
        orig_name = Path(report.source_path).name
        fixed_stem = Path(report.output_path).stem
        fixed_name_map[orig_name] = fixed_stem

    # --- Organize files into subfolders ---
    # LLMs sometimes normalize smart quotes (U+2018/2019) to ASCII (U+0027)
    # in JSON keys, causing lookup mismatches against disk filenames.
    # Build a normalized lookup to handle this.
    _norm_assignments = _normalize_quote_keys(recommendation.file_assignments)

    for doc in docs:
        filename = doc.metadata.filename
        # Determine the .md name for this file in the output
        fixed_stem = fixed_name_map.get(filename, doc.metadata.stem)
        md_name = f"{fixed_stem}.md"

        # Determine target subfolder from recommender
        folder = _norm_assignments.get(_normalize_quotes(filename), "General")
        target_dir = os.path.join(output, folder)
        os.makedirs(target_dir, exist_ok=True)

        flat_path = os.path.join(output, md_name)
        target_path = os.path.join(target_dir, md_name)

        try:
            # Fixed file exists — move it into subfolder
            shutil.move(flat_path, target_path)
        except FileNotFoundError:
            # Not fixed — convert original to markdown and write
            md_content = to_markdown(doc)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(md_content)

    # --- Console output: folder tree + assignments ---
    tree_str = format_folder_tree(recommendation.root)
    console.print(Panel(tree_str, title="Folder Structure", border_style="green"))

    if recommendation.file_assignments:
        assign_table = Table(title="File → Folder Assignments")
        assign_table.add_column("File", style="cyan")
        assign_table.add_column("Folder", style="green")
        for fname, folder in recommendation.file_assignments.items():
            assign_table.add_row(fname, folder)
        console.print(assign_table)

    # Validate folder assignments
    fix_sil_score = 0.0
    fix_misplaced: list[tuple[str, float]] = []
    if hasattr(corpus_analysis, "similarity_matrix") and corpus_analysis.similarity_matrix.size > 0:
        fix_sil_score, fix_misplaced = recommender.validate_assignments(
            recommendation.file_assignments,
            corpus_analysis.similarity_matrix,
            corpus_analysis.doc_labels,
        )
        if fix_sil_score != 0:
            console.print(f"\n[dim]Folder coherence (silhouette): {fix_sil_score:.2f}[/dim]")
        if fix_misplaced:
            console.print(f"[yellow]Warning: {len(fix_misplaced)} document(s) may be misplaced:[/yellow]")
            for filename, score in fix_misplaced:
                console.print(f"  [yellow]{filename} (silhouette: {score:.2f})[/yellow]")

    console.print(f"\nOrganized files written to [cyan]{output}/[/cyan]")

    # --- Report inside output folder ---
    if not no_report:
        report_name = f"kb-prep-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        report_path = os.path.join(output, report_name)
        _write_report_file(
            report_path,
            [
                _report_header("fix", len(docs)),
                _report_scores(cards, detail=False),
                _report_fixes(fix_reports),
                _report_recommendations(
                    recommendation, fix_sil_score if fix_sil_score != 0 else None, fix_misplaced or None
                ),
            ],
        )
        console.print(f"[green]Report:[/green] {report_path}")


# ---------------------------------------------------------------------------
# upload command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--api-key", envvar="ANAM_API_KEY", required=True, help="anam.ai API key")
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", default=None, help="Anthropic API key (for analysis/fix)")
@click.option("--model", default=None, help="LLM model override")
@click.option("--no-fix", is_flag=True, help="Skip auto-fix, upload originals")
@click.option("--use-existing-folders", is_flag=True, help="Use subfolder layout on disk instead of recommender")
@click.option("--dry-run", is_flag=True, help="Show plan without uploading")
@click.option("--persona-id", default=None, help="Attach knowledge tool to this persona")
@click.option("--tool-name", default=None, help="Name for the knowledge tool")
@click.option("--tool-description", default=None, help="Description for when the LLM should search")
@click.option("--exclude", multiple=True, help="Exclude files containing this substring (repeatable)")
@click.option("--no-report", is_flag=True, help="Suppress markdown report generation")
@click.option("--concurrency", default=5, type=int, help="Max parallel LLM calls (default: 5)")
@click.option(
    "--folder-hints", default=None, type=click.Path(exists=True), help="File with domain-specific folder guidance"
)
def upload(
    path: str,
    api_key: str,
    llm_key: str,
    model: str,
    no_fix: bool,
    use_existing_folders: bool,
    dry_run: bool,
    persona_id: str,
    tool_name: str,
    tool_description: str,
    exclude: tuple,
    no_report: bool,
    concurrency: int,
    folder_hints: str,
):
    """Full pipeline: score → analyze → fix → recommend folders → upload."""
    from .analyzer import ContentAnalyzer
    from .anam_client import AnamClient
    from .fixer import DocumentFixer
    from .recommender import FolderRecommender, format_folder_tree

    hints_text = Path(folder_hints).read_text().strip() if folder_hints else ""
    files = discover_files(path, exclude_patterns=list(exclude) if exclude else None)
    if not files:
        console.print("[red]No supported files found.[/red]")
        raise SystemExit(1)

    config = Config.from_env().with_overrides(
        anam_api_key=api_key,
        anthropic_api_key=llm_key,
        llm_model=model,
        concurrency=concurrency,
        folder_hints=hints_text or None,
    )

    parser = DocumentParser()
    docs = []

    # Step 1: Parse
    console.print("\n[bold]Step 1: Parse[/bold]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Parsing...", total=len(files))
        for file_path in files:
            try:
                doc = parser.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Error: {file_path}: {e}[/red]")
            progress.advance(task)

    # Step 2: Analyze + build knowledge graph (if LLM key available)
    graph = None
    analyses = []
    if llm_key:
        console.print("\n[bold]Step 2: LLM Analysis & Knowledge Graph[/bold]")
        analyzer_inst = ContentAnalyzer(config)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            progress.add_task("Analyzing and building graph...", total=None)
            analyses, graph = asyncio.run(analyzer_inst.analyze_and_build_graph(docs))
        _print_graph_summary(graph)
    else:
        console.print("\n[dim]Step 2: Skipped (no --llm-key provided)[/dim]")
        from .models import ContentAnalysis

        analyses = [ContentAnalysis() for _ in docs]

    # Score with graph and corpus context
    corpus_analysis = build_corpus_analysis(docs)
    scorer = QualityScorer(
        graph=graph,
        corpus_analysis=corpus_analysis,
    )
    cards = [scorer.score(doc) for doc in docs]
    _print_score_table(cards, detail=False)

    # Step 3: Auto-fix (if LLM key available and not --no-fix)
    upload_files = files  # Default: upload originals
    fixed_name_map: dict[str, str] = {}  # original_filename → fixed_filename
    fix_reports = []
    if llm_key and not no_fix:
        console.print("\n[bold]Step 3: Auto-Fix[/bold]")
        fixer_inst = DocumentFixer(config, graph=graph)
        fixed_files = []

        # Separate docs with issues from those without
        docs_to_fix = []
        no_fix_indices = []
        for i, (doc, card) in enumerate(zip(docs, cards)):
            if card.all_issues:
                docs_to_fix.append((i, doc, card))
            else:
                no_fix_indices.append(i)
                fixed_files.append((i, doc.metadata.file_path))

        if docs_to_fix:

            async def _fix_all():
                tasks = [fixer_inst.fix(doc, card) for _, doc, card in docs_to_fix]
                return await asyncio.gather(*tasks, return_exceptions=True)

            console.print(f"  [dim]Fixing {len(docs_to_fix)} document(s) concurrently...[/dim]")
            results = asyncio.run(_fix_all())

            for (idx, doc, card), result in zip(docs_to_fix, results):
                if isinstance(result, Exception):
                    console.print(f"  [red]✗[/red] {doc.metadata.filename}: {result}")
                    fixed_files.append((idx, doc.metadata.file_path))
                else:
                    fix_reports.append(result)
                    fixed_files.append((idx, result.output_path))
                    fixed_name_map[doc.metadata.filename] = Path(result.output_path).name
                    console.print(
                        f"  [green]✓[/green] {doc.metadata.filename} → "
                        f"{result.output_path} ({len(result.actions)} fixes)"
                    )

        # Sort by original index to preserve order
        fixed_files.sort(key=lambda x: x[0])
        upload_files = [f for _, f in fixed_files]
    else:
        console.print("\n[dim]Step 3: Skipped (--no-fix or no --llm-key)[/dim]")

    # Step 4: Folder structure
    console.print("\n[bold]Step 4: Folder Structure[/bold]")
    if use_existing_folders:
        recommendation = _recommendation_from_disk(path, upload_files)
        console.print("  [dim]Using existing subfolder layout from disk[/dim]")
    else:
        recommender = FolderRecommender(config if llm_key else None, graph=graph)
        recommendation = asyncio.run(recommender.recommend(docs, analyses))

    tree_str = format_folder_tree(recommendation.root)
    console.print(Panel(tree_str, title="Folder Structure", border_style="green"))

    # Remap file_assignments to use fixed filenames when files were renamed.
    # Normalize quote characters (LLMs convert smart quotes to ASCII).
    upload_assignments = _normalize_quote_keys(recommendation.file_assignments)
    if not use_existing_folders and fixed_name_map:
        for orig_name, fixed_name in fixed_name_map.items():
            norm_orig = _normalize_quotes(orig_name)
            if norm_orig in upload_assignments and fixed_name != orig_name:
                upload_assignments[_normalize_quotes(fixed_name)] = upload_assignments.pop(norm_orig)

    if upload_assignments:
        assign_table = Table(title="File → Folder Assignments")
        assign_table.add_column("File", style="cyan")
        assign_table.add_column("Folder", style="green")
        for filename, folder in upload_assignments.items():
            assign_table.add_row(filename, folder)
        console.print(assign_table)

    if dry_run:
        console.print("\n[yellow]Dry run complete. No files were uploaded.[/yellow]")
        return

    # Step 5: Upload
    console.print("\n[bold]Step 5: Upload to anam.ai[/bold]")
    anam = AnamClient(config)

    # Create folders
    console.print("  Creating folders...")
    folder_map = anam.create_folder_tree(recommendation)
    for path_name, folder_id in folder_map.items():
        console.print(f"  [green]✓[/green] {path_name} → {folder_id}")

    # Upload files
    upload_report = anam.upload_batch(
        files=upload_files,
        folder_map=folder_map,
        file_assignments=upload_assignments,
        progress_callback=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
    )

    # Summary
    console.print("\n[bold]Upload Complete[/bold]")
    console.print(f"  Folders created: {len(upload_report.folders_created)}")
    console.print(f"  Files uploaded:  {len(upload_report.successful)}")
    console.print(f"  Failed:          {len(upload_report.failed)}")

    for r in upload_report.failed:
        console.print(f"  [red]✗ {r.file_path}: {r.error}[/red]")

    # Step 6: Create knowledge tool (optional)
    if persona_id and folder_map:
        all_folder_ids = list(folder_map.values())
        tool_id = anam.create_knowledge_tool(
            name=tool_name or "Document Search",
            description=tool_description or "Search uploaded documents to answer questions accurately.",
            folder_ids=all_folder_ids,
        )
        console.print(f"\n  [green]Knowledge tool created:[/green] {tool_id}")
        console.print(f"  Linked to {len(all_folder_ids)} folder(s)")
        console.print(f"  [dim]Attach to persona {persona_id} in anam.ai settings[/dim]")

    # Auto-generate report
    if not no_report:
        report_path = _generate_report_path("upload")
        _write_report_file(
            report_path,
            [
                _report_header("upload", len(docs)),
                _report_scores(cards, detail=False),
                _report_fixes(fix_reports),
                _report_recommendations(recommendation),
                _report_uploads(upload_report),
            ],
        )
        console.print(f"\n[green]Report:[/green] {report_path}")


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------

# Smart/curly quotes that LLMs commonly normalize to ASCII equivalents.
_QUOTE_TABLE = str.maketrans(
    {
        "\u2018": "'",  # LEFT SINGLE QUOTATION MARK → apostrophe
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK → apostrophe
        "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK → double quote
        "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK → double quote
    }
)


def _normalize_quotes(s: str) -> str:
    """Replace typographic/smart quotes with ASCII equivalents."""
    return s.translate(_QUOTE_TABLE)


def _normalize_quote_keys(d: dict[str, str]) -> dict[str, str]:
    """Return a copy of dict with keys normalized to ASCII quotes."""
    return {_normalize_quotes(k): v for k, v in d.items()}


def _recommendation_from_disk(base_path: str, files: list[str]) -> "FolderRecommendation":
    """Build a FolderRecommendation from existing subfolder layout on disk."""
    from .models import FolderNode, FolderRecommendation

    base = Path(base_path).resolve()
    root = FolderNode(name="Knowledge Base", description="Root knowledge base container")
    assignments: dict[str, str] = {}
    folder_nodes: dict[str, FolderNode] = {}

    for file_path in files:
        fp = Path(file_path).resolve()
        rel_parent = fp.parent.relative_to(base)
        filename = fp.name

        if str(rel_parent) == ".":
            folder_name = "General"
        else:
            folder_name = str(rel_parent)

        assignments[filename] = folder_name

        if folder_name not in folder_nodes:
            node = FolderNode(name=folder_name, description=f"Files from {folder_name}/")
            folder_nodes[folder_name] = node
            root.children.append(node)

        folder_nodes[folder_name].document_files.append(filename)

    return FolderRecommendation(root=root, file_assignments=assignments)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_score_table(cards: list[ScoreCard], detail: bool):
    """Print a summary table of all scored documents."""
    table = Table(title="RAG Readiness Scores")
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

                    console.print(
                        f"\n  [{_severity_color(issues[0].severity)}]{label}[/{_severity_color(issues[0].severity)}] (score: {cat_score:.0f})"
                    )
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
    """Return a timestamped report filename: kb-prep-{command}-{YYYYMMDD-HHMMSS}.md"""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"kb-prep-{command}-{ts}.md"


def _report_header(command: str, file_count: int) -> list[str]:
    """Markdown header with command name, timestamp, and file count."""
    from datetime import datetime

    lines: list[str] = []
    lines.append(f"# kb-prep {command} Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Files:** {file_count}")
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


def _report_recommendations(
    recommendation,
    silhouette_score: Optional[float] = None,
    misplaced: Optional[list[tuple[str, float]]] = None,
) -> list[str]:
    """Folder structure + file assignments section."""
    from .recommender import format_folder_tree

    lines: list[str] = []
    if not recommendation:
        return lines

    lines.append("## Recommended Folder Structure")
    lines.append("")
    lines.append("```")
    lines.append(format_folder_tree(recommendation.root))
    lines.append("```")
    lines.append("")
    if recommendation.file_assignments:
        lines.append("### File Assignments")
        lines.append("")
        lines.append("| File | Folder |")
        lines.append("|------|--------|")
        for filename, folder in recommendation.file_assignments.items():
            lines.append(f"| {filename} | {folder} |")
        lines.append("")
    if silhouette_score is not None:
        lines.append(f"**Folder coherence (silhouette):** {silhouette_score:.2f}")
        lines.append("")
    if misplaced:
        lines.append("**Potentially misplaced documents:**")
        lines.append("")
        for filename, score in misplaced:
            lines.append(f"- {filename} (silhouette: {score:.2f})")
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


def _report_uploads(upload_report) -> list[str]:
    """Upload results table section."""
    lines: list[str] = []
    if not upload_report:
        return lines

    lines.append("## Upload Results")
    lines.append("")
    lines.append(f"- **Folders created:** {len(upload_report.folders_created)}")
    lines.append(f"- **Files uploaded:** {len(upload_report.successful)}")
    lines.append(f"- **Failed:** {len(upload_report.failed)}")
    lines.append("")

    if upload_report.results:
        lines.append("| File | Folder | Status |")
        lines.append("|------|--------|--------|")
        for r in upload_report.results:
            filename = Path(r.file_path).name
            status = r.status.value
            error_note = f" ({r.error})" if r.error else ""
            lines.append(f"| {filename} | {r.folder_name} | {status}{error_note} |")
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
