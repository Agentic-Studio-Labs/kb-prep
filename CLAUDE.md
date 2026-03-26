# CLAUDE.md

## What This Project Does

ragprep is a CLI tool that prepares documents for RAG. It parses DOCX/PDF/TXT/MD files, scores them for retrieval readiness, and optionally auto-fixes issues with an LLM. Works with any RAG pipeline.

## Commands

```bash
python -m src.cli score <path>                          # Heuristic + corpus scoring (no LLM)
python -m src.cli analyze <path> --llm-key KEY          # + LLM analysis + graph/folder enrichment
python -m src.cli analyze <path> --llm-key KEY --chunk-size 220 --chunk-overlap 40 --run-benchmark
python -m src.cli fix <path> --llm-key KEY              # + auto-fix, output Markdown + sidecars
```

## Running Tests

```bash
source .venv/bin/activate    # venv required
python3 -m pytest tests/ -v  # 95 tests
python3 -m pytest test-data/layer5_rag_quality/ -v
ruff check .                 # lint
```

## Environment Variables

- `ANTHROPIC_API_KEY` — required for analyze/fix

## Architecture

```
Parse → Corpus Analyzer (TF-IDF) → Score → [Analyze (LLM)] → [Fix (LLM)]
```

**Pipeline stages:**
- `src/parser.py` — DOCX/PDF/TXT/MD → ParsedDocument (paragraphs + heading tree)
- `src/corpus_analyzer.py` — TF-IDF matrix, similarity, entropy, coherence, retrieval-aware score
- `src/scorer.py` — 9 criteria weighted to 1.0 (+ 1 graph-powered when LLM runs)
- `src/analyzer.py` — LLM content analysis, entity/relationship extraction
- `src/graph_builder.py` — networkx DiGraph, entity resolution (char n-gram TF-IDF cosine), spectral clustering, PageRank
- `src/fixer.py` — LLM-powered fixes (dangling refs, headings, paragraphs, acronyms, filenames)
- `src/recommender.py` — folder structure (graph + LLM → heuristic fallback), silhouette validation
- `src/chunker.py` — structure-aware chunking (heading-preserving with overlap)
- `src/benchmark.py` — chunk retrieval metrics (Recall@5, MRR, nDCG@5)
- `src/cleaner.py` — deterministic cleanup before scoring/chunking
## Key Files

| File | Lines | What it does |
|------|-------|-------------|
| `src/cli.py` | ~1000 | Click CLI, Rich output, report generation |
| `corpus_analyzer.py` | ~300 | TF-IDF, entropy, coherence, TextTiling, BM25 self-retrieval |
| `scorer.py` | ~860 | 10 scoring criteria with weighted scoring model |
| `graph_builder.py` | ~350 | Entity graph, cosine resolution, spectral clustering, PageRank |
| `src/models.py` | ~330 | All dataclasses (ParsedDocument, ScoreCard, Entity, etc.) |

## Scoring Weights

| Criterion | Weight | Method |
|-----------|--------|--------|
| Self-Containment | 20% | Regex pattern matching for dangling references |
| Retrieval-Aware | 20% | Self-retrieval rate from synthetic queries (BM25+) |
| Heading Quality | 15% | Hierarchy checks + generic heading detection |
| Paragraph Length | 10% | Word count thresholds (15-300 words) |
| File Focus | 10% | Vocabulary entropy over TF-IDF term weights |
| Filename Quality | 10% | Regex patterns for generic names |
| Structure | 10% | 4-point checklist (headings, body, sections, paragraphs) |
| Acronym Definitions | 5% | Uppercase pattern detection + definition search |
| Knowledge Completeness | 5%* | Orphan refs + cross-doc connectivity (*with graph) |

When the graph is available, weights are auto-scaled proportionally to keep the total at 1.0.

## Conventions

- **Python 3.10+**, type hints on all function signatures
- **ruff** for linting (line-length 120, rules: E, F, W, I)
- **pytest** for testing (testpaths: tests/)
- Async with `asyncio.Semaphore` for LLM call concurrency
- `AsyncAnthropic` for Claude API (async)
- All LLM responses parsed from free-text JSON via `extract_json()` in src/analyzer.py
- Reports auto-generated as timestamped Markdown (suppress with `--no-report`)
- Analyze metadata writes to `.ragprep/` (manifest + sidecars)

## Data Flow

Documents flow through the pipeline as `ParsedDocument` objects (defined in `src/models.py`). Key properties:
- `doc.paragraphs` — list of `Paragraph(text, level, style, index)`
- `doc.headings` — paragraphs where `level > 0`
- `doc.body_paragraphs` — paragraphs where `level == 0`
- `doc.full_text` — concatenated paragraph text
- `doc.metadata.filename` — used as key in corpus_analysis.doc_metrics

## Testing Patterns

- Synthetic DOCX created via `python-docx` in test fixtures (`_create_test_docx()`)
- `_make_doc(filename, text)` helper for quick ParsedDocument creation
- LLM calls mocked with `unittest.mock.AsyncMock` on `AsyncAnthropic`
- Graph tests build entities/relationships directly via `_add_entity()`/`_add_relationship()`

## Things to Know

- `corpus_analyzer.build_corpus_analysis()` must receive ALL docs at once (TF-IDF needs the full corpus)
- The `fixed/` directory is gitignored — it contains auto-generated output
- `rag-files-*/` directories are timestamped output from `fix` runs
- Louvain clustering (entity graph) is seeded with `seed=42` for determinism
- Spectral clustering (document similarity) uses `random_state=42`
