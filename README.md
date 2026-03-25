# kb-prep

Document preparation pipeline for RAG systems. Scores, analyzes, and fixes documents before they reach your vector database — so retrieval actually works.

Most RAG failures aren't embedding problems or chunk size problems. They're **document problems**: dangling references that make paragraphs meaningless in isolation, buried content that no query can find, headings that don't match the vocabulary users search with. kb-prep catches these issues before upload, not after your users complain.

**What it does:**

- **Scores** documents across 10 criteria including a novel retrieval-aware metric that tests whether each document can actually be found via search
- **Analyzes** content with an LLM to extract entities and relationships, building a knowledge graph across your entire corpus
- **Fixes** issues automatically — rewrites dangling references, splits long paragraphs, replaces generic headings, defines acronyms
- **Organizes** documents into folders using spectral clustering and graph-based community detection

Supports DOCX, PDF, TXT, and Markdown. Works with any vector database (Pinecone, Weaviate, Qdrant, Chroma, etc.) or RAG framework (LlamaIndex, LangChain, etc.). Includes optional direct upload to [anam.ai](https://anam.ai).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

## Quick Start

```bash
# Score documents (no LLM, no API keys)
python -m src.cli score ./my-docs/
python -m src.cli score ./my-docs/ --detail        # show every issue
python -m src.cli score ./my-docs/ --json-output   # machine-readable

# Analyze with LLM (topics, knowledge graph, folder recommendations)
python -m src.cli analyze ./my-docs/ --llm-key $ANTHROPIC_API_KEY

# Auto-fix issues and output improved Markdown
python -m src.cli fix ./my-docs/ --llm-key $ANTHROPIC_API_KEY --output ./fixed/
```

Set environment variables to avoid passing keys every time:

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
```

## How Scoring Works

Every command runs the scoring pipeline. It combines heuristic checks with corpus-level TF-IDF analysis — no LLM needed.

### Pipeline

```
Parse (DOCX/PDF/TXT/MD)
  │
  ├─ Corpus Analyzer ── TF-IDF matrix, document similarity, per-doc metrics
  │
  └─ Scorer ── 8 heuristic criteria + 1 retrieval-aware + 1 graph-powered
```

The **corpus analyzer** (`src/corpus_analyzer.py`) computes a TF-IDF matrix across all documents in one pass, then derives per-document metrics: topic entropy, heading-content coherence, readability grade, topic boundaries, and a self-retrieval score. These feed into the scorer alongside the existing heuristic checks.

### Scoring Criteria

| Criterion              | Weight | What It Checks |
| ---------------------- | ------ | -------------- |
| Self-Containment       | 20%    | Dangling references ("as mentioned above", "see section X") that break paragraph independence |
| Retrieval-Aware        | 20%    | Can the document be found by BM25 queries about its own content? Generates synthetic queries from top TF-IDF terms and measures self-retrieval rate |
| Heading Quality        | 15%    | Hierarchy gaps, generic headings ("Content", "Notes"), heading density |
| Paragraph Length       | 15%    | Too short (<15 words) or too long (>300 words) |
| File Focus             | 10%    | Shannon entropy over TF-IDF topic distribution — flags sprawling multi-topic documents |
| Filename Quality       | 10%    | Generic names ("doc-v2.docx"), too short, no word separators |
| Structure Completeness | 10%    | Presence of headings, substantive body text, multiple sections |
| Acronym Definitions    | 5%     | Uppercase acronyms used repeatedly without "(definition)" nearby |
| Knowledge Completeness | 5%*    | Orphan references, isolated documents (*graph-powered, only with LLM analysis) |
| File Size              | info   | Warns at 25MB, blocks at 50MB |

**Readiness levels:** EXCELLENT (85+), GOOD (70-84), FAIR (50-69), POOR (<50)

### Retrieval-Aware Scoring

The most distinctive criterion. For each document, the scorer:

1. Extracts the top TF-IDF terms (the document's most characteristic vocabulary)
2. Generates synthetic queries from 3-term combinations
3. Runs each query against the full corpus using BM25
4. Measures what percentage of queries about this document actually find it in the top 5 results

A document that scores 80-100% is well-structured for retrieval. A document scoring below 40% has structural problems — buried content, generic vocabulary, or misleading headings — that make it hard to find via search.

### Corpus Analysis Metrics

The corpus analyzer also computes these metrics (available in the analysis output, not used as scoring criteria):

- **Readability grade** — Flesch-Kincaid grade level
- **Information density** — TF-IDF magnitude per section
- **Topic boundaries** — TextTiling-detected topic shifts within documents
- **Document similarity matrix** — cosine similarity between all document pairs

## Auto-Fix (LLM-Powered)

When you run `fix` with `--llm-key`, targeted prompts are sent to Claude to fix each detected issue:

| Issue               | Fix Applied                                             |
| ------------------- | ------------------------------------------------------- |
| Dangling references | Rewrites paragraph to include referenced context inline |
| Generic headings    | Generates descriptive heading from paragraph content    |
| Long paragraphs     | Splits into 2-4 focused sub-paragraphs                  |
| Undefined acronyms  | Inserts "(Full Name)" after first occurrence            |
| Generic filename    | Generates descriptive filename from content             |

Originals are never modified. Fixed files are written as clean Markdown to the output directory.

## Knowledge Graph

When LLM analysis runs (`analyze` or `fix` with `--llm-key`), an in-memory knowledge graph is built across all documents automatically.

The LLM extracts **entities** and **relationships** from each document. Entities are merged into a shared [networkx](https://networkx.org/) directed graph using TF-IDF cosine similarity on character n-grams — this handles morphological variation ("Budget" matches "Budgeting"), word reordering, and typos.

| Entity types                                          | Relationship types                                      |
| ----------------------------------------------------- | ------------------------------------------------------- |
| concept, skill, lesson, resource, assessment, process | prerequisite, related_to, part_of, assesses, influences |

### Graph analysis

- **Spectral clustering** — deterministic document grouping using the eigengap heuristic on the TF-IDF similarity matrix
- **PageRank** — ranks entities by structural importance for folder naming
- **Betweenness centrality** — identifies bridge entities connecting topic clusters
- **Bipartite projection** — document-document similarity via shared entities (blended with TF-IDF similarity)
- **Folder coherence validation** — silhouette analysis scores whether folder assignments actually group similar documents

### Downstream consumers

- **Scorer** — orphan references and cross-document connectivity (Knowledge Completeness criterion)
- **Fixer** — cross-document context for resolving dangling references ("see Unit 2" gets actual Unit 2 content)
- **Recommender** — graph clusters + PageRank for folder naming, silhouette validation for assignment quality

## Folder Recommendations

The tool proposes a folder structure using a 4-tier priority: graph clusters + LLM naming (best), LLM-only, graph-only, or heuristic fallback. The path hierarchy is encoded into folder names:

```
Engineering - API Design
Engineering - Architecture
Product - Requirements
Product - User Research
Onboarding
```

## RAG Evaluation

Evaluate document quality by testing retrieval + answer generation against questions with known ground truth.

```bash
python eval/run_eval.py rag-files-*/ --llm-key $ANTHROPIC_API_KEY
```

**Metrics:** Retrieval Hit Rate, Context Precision, Faithfulness, Answer Correctness.

## Supported File Types

| Format | Parsing                                                |
| ------ | ------------------------------------------------------ |
| .docx  | Full (headings, paragraphs, metadata)                  |
| .pdf   | Full (font-based heading detection, paragraph merging) |
| .md    | Full (Markdown heading syntax)                         |
| .txt   | Basic (paragraph splitting)                            |

## Project Structure

```
kb-prep/
├── src/                         # Source package
│   ├── cli.py                   # CLI entry point (Click) — score, analyze, fix, upload
│   ├── corpus_analyzer.py       # TF-IDF matrix, entropy, coherence, retrieval-aware scoring
│   ├── scorer.py                # Heuristic + corpus-powered scoring criteria
│   ├── parser.py                # DOCX/PDF/TXT/MD parsing + Markdown conversion
│   ├── analyzer.py              # LLM content analysis (topics, entities, relationships)
│   ├── graph_builder.py         # Knowledge graph (networkx) + spectral clustering + PageRank
│   ├── fixer.py                 # LLM auto-fix engine (graph-aware)
│   ├── recommender.py           # Folder recommendation + silhouette validation
│   ├── anam_client.py           # Upload client (see anam.ai section below)
│   ├── prompts.py               # LLM prompt templates
│   ├── config.py                # Settings and API key management
│   └── models.py                # All dataclasses
├── tests/
│   ├── test_corpus_analyzer.py  # TF-IDF, entropy, coherence, retrieval tests
│   ├── test_scoring.py          # Scoring criteria validation
│   ├── test_graph.py            # Entity resolution, clustering, PageRank
│   ├── test_integration.py      # End-to-end pipeline tests
│   ├── test_async_analyzer.py   # Async LLM analysis tests
│   ├── test_async_fixer.py      # Async fixer tests
│   ├── test_config.py           # Configuration tests
│   └── test_report.py           # Report generation tests
├── eval/
│   ├── run_eval.py              # RAG evaluation (BM25 + vector search)
│   └── test-questions.json
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Requirements

- `python-docx` — DOCX parsing
- `PyMuPDF` — PDF parsing
- `click` — CLI framework
- `rich` — terminal formatting
- `requests` — HTTP client
- `anthropic` — Claude API (only needed for analyze/fix/LLM features)
- `networkx` — knowledge graph
- `numpy` — numerical computation
- `scipy` — signal processing (TextTiling), sparse matrices
- `scikit-learn` — TF-IDF vectorization, spectral clustering, cosine similarity

## TODO

- **Structured LLM output** — replace JSON-in-markdown prompts with tool_use for reliable extraction
- **Incremental analysis** — cache per-file analysis results so re-runs only process changed files
- **Relationship deduplication** — merge duplicate edges and track edge weight/frequency
- **Configurable thresholds** — entropy thresholds are now in corpus_analyzer; still need to expose scoring weights and cluster resolution as CLI flags or config
- **Export graph** — write the knowledge graph as GraphML, JSON, or DOT

---

## anam.ai Integration

kb-prep includes built-in upload support for [anam.ai](https://anam.ai) knowledge bases. This is optional — all other features work without it.

### Setup

```bash
export ANAM_API_KEY=your-anam-key
```

### Upload

```bash
# Full pipeline: fix + recommend folders + upload
python -m src.cli upload ./my-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY

# Dry run (preview without uploading)
python -m src.cli upload ./my-docs/ --api-key $ANAM_API_KEY --llm-key $ANTHROPIC_API_KEY --dry-run

# Upload and attach to a persona
python -m src.cli upload ./my-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY \
  --persona-id your-persona-id \
  --tool-name "Doc Search" \
  --tool-description "Search documents to answer questions"
```

### How it works

anam.ai uses direct multipart upload with flat folders (hierarchy encoded in names):

1. **Create folder** — POST to `/v1/knowledge/groups`
2. **Upload file** — multipart POST to `/v1/knowledge/groups/{id}/documents`

Documents transition from `PROCESSING` → `READY` (~30 seconds). When `--persona-id` is provided, a knowledge tool (type `SERVER_RAG`) is created linking all folders to the persona.

The eval script also supports anam.ai vector search for retrieval evaluation:

```bash
python eval/run_eval.py rag-files-*/ --llm-key $ANTHROPIC_API_KEY --anam-key $ANAM_API_KEY
```

**File size limits:** Warns at 25MB, blocks at 50MB.
