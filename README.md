# 3kb-prep

Prepare, score, fix, and upload documents to [anam.ai](https://anam.ai) knowledge base for RAG-powered AI personas.

Takes unstructured documents (DOCX, PDF, TXT, Markdown), scores them for RAG readiness using heuristic analysis, builds an in-memory knowledge graph for cross-document understanding, optionally auto-fixes issues with an LLM, recommends a folder structure, and uploads to anam.ai's knowledge base API.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Quick Start

### Score documents (no LLM, free)

```bash
python cli.py score ./my-docs/
python cli.py score ./my-docs/ --detail        # show every issue
python cli.py score ./my-docs/ --json-output   # machine-readable
```

### Analyze with LLM (extracts topics, builds knowledge graph)

```bash
python cli.py analyze ./my-docs/ --llm-key $ANTHROPIC_API_KEY
```

This parses every document, runs LLM content analysis, builds a knowledge graph across all documents, scores with graph-aware completeness checks, and recommends a folder structure based on detected topic clusters.

### Auto-fix issues and output improved files

```bash
python cli.py fix ./my-docs/ --llm-key $ANTHROPIC_API_KEY --output ./fixed/
```

### Full pipeline: parse → analyze → graph → score → fix → recommend → upload

```bash
python cli.py upload ./my-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY
```

### Dry run (preview without uploading)

```bash
python cli.py upload ./my-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY \
  --dry-run
```

### Upload and attach to a persona

```bash
python cli.py upload ./my-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY \
  --persona-id your-persona-id \
  --tool-name "Lesson Search" \
  --tool-description "Search lesson plans to answer student and teacher questions"
```

## Environment Variables

Instead of passing keys as flags every time:

```bash
export ANAM_API_KEY=your-anam-key
export ANTHROPIC_API_KEY=your-anthropic-key
```

## How Scoring Works

The scorer runs 8 heuristic checks (no LLM needed) plus 1 graph-powered check when LLM analysis is available.


| Criterion              | Weight | What It Checks                                                                                                                                |
| ---------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Self-Containment       | 25%    | Dangling references ("as mentioned above", "see section X") that break paragraph independence                                                 |
| Heading Quality        | 20%    | Hierarchy gaps, generic headings ("Content", "Notes"), heading density                                                                        |
| Paragraph Length       | 15%    | Too short (<15 words, no context) or too long (>300 words, diluted)                                                                           |
| File Focus             | 15%    | Topic coherence — flags sprawling multi-topic documents                                                                                       |
| Filename Quality       | 10%    | Generic names ("doc-v2.docx"), too short, no word separators                                                                                  |
| Structure Completeness | 10%    | Presence of headings, substantive body text, multiple sections                                                                                |
| Acronym Definitions    | 5%     | Uppercase acronyms used repeatedly without "(definition)" nearby                                                                              |
| Knowledge Completeness | 5%     | Orphan references to undefined entities, isolated documents with no cross-document connections (graph-powered, only when LLM analysis is run) |
| File Size              | info   | Warns at 25MB, blocks at 50MB (anam.ai limit)                                                                                                 |


**Readiness levels:** EXCELLENT (85+), GOOD (70-84), FAIR (50-69), POOR (<50)

## Auto-Fix (LLM-Powered)

When you run `fix` or `upload` with `--llm-key`, the tool sends targeted prompts to Claude to fix each detected issue:


| Issue               | Fix Applied                                             |
| ------------------- | ------------------------------------------------------- |
| Dangling references | Rewrites paragraph to include referenced context inline |
| Generic headings    | Generates descriptive heading from paragraph content    |
| Long paragraphs     | Splits into 2-4 focused sub-paragraphs                  |
| Undefined acronyms  | Inserts "(Full Name)" after first occurrence            |
| Generic filename    | Generates descriptive filename from content             |


Originals are never modified. Fixed files are written as clean Markdown to the output directory.

## Knowledge Graph

When LLM analysis runs (`analyze`, `fix`, or `upload` with `--llm-key`), the tool automatically builds an in-memory knowledge graph across all documents. This is not a separate step or flag — it happens as part of standard analysis and improves every downstream stage.

### How it works

The LLM extracts **entities** and **relationships** from each document. Only explicitly extracted entities are added to the graph — topics and key concepts from the analysis metadata are kept separate to avoid over-connecting unrelated documents through generic terms.

Entities are merged into a shared [networkx](https://networkx.org/) directed graph using two-tier deduplication:

1. **Exact match** — O(1) lookup via a normalized (lowercase, stripped) name index
2. **Substring fallback** — for names **8+ characters** only, checks if one name contains another (e.g., "Financial Planning" matches "Financial Planning Basics"). Short names like "Goal" or "Credit" are excluded to prevent false merges.


| Entity types                                          | Relationship types                                      |
| ----------------------------------------------------- | ------------------------------------------------------- |
| concept, skill, lesson, resource, assessment, process | prerequisite, related_to, part_of, assesses, influences |


### Downstream consumers

- **Scorer** — detects orphan references (entities mentioned but never defined in any document) and isolated documents with no cross-document co**Fixer** — enriches dangling reference resolution with cross-document context. When a paragraph says "see Unit 2", the graph provides the actual content from Unit 2's document to inline.
- **Recommender** — uses [Louvain community detection](https://en.wikipedia.org/wiki/Louvain_method) to cluster related documents into folders, then uses the LLM to generate descriptive folder names. Louvain finds densely-connected communities even when they share weak bridges — unlike connected components, which would merge everything reachable into a single cluster. Fnnections, surfaced as the Knowledge Completeness criterion.
- alls back gracefully if the graph is empty.

### Example output

The graph is lightweight (in-memory, no database) and adds negligible overhead since the LLM is already analyzing each document. A summary panel is printed after analysis:

```
╭────────────────────────────── Knowledge Graph ───────────────────────────────╮
│ Entities: 520                                                                │
│ Relationships: 614                                                           │
│ Cross-document edges: 236                                                    │
│ Topic clusters: 43                                                           │
│ Entity types: concept: 181, skill: 97, lesson: 57, resource: 53,            │
│ assessment: 41, process: 32                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Folder Recommendations

The tool proposes a folder structure using a 4-tier priority: graph clusters + LLM naming (best), LLM-only, graph-only, or heuristic fallback. Since anam.ai uses a flat folder structure (no nesting), the path hierarchy is encoded into folder names. For example:

```
📁 Curriculum - Grade 3 Math
📁 Curriculum - Grade 4 Science
📁 Teacher Resources - Lesson Plans
📁 Teacher Resources - Rubrics
📁 Parent Guides
```

With `upload`, these folders are created automatically via the anam.ai API, and each document is uploaded to its assigned folder.

## How anam.ai Upload Works

anam.ai uses a direct multipart upload:

1. **Create folder** — POST to `/v1/knowledge/groups` with name and description
2. **Upload file** — multipart POST to `/v1/knowledge/groups/{id}/documents`

Documents transition from `PROCESSING` → `READY` (typically ~30 seconds). The tool handles folder creation and file upload automatically.

When `--persona-id` is provided, a **knowledge tool** (type `SERVER_RAG`) is created that links all uploaded folders to the persona. The tool's description tells the avatar's LLM when to search the knowledge base.

## Project Structure

```
kb-prep/
├── cli.py              # CLI entry point (Click)
├── parser.py           # DOCX/PDF/TXT/MD parsing + Markdown conversion
├── scorer.py           # 8 heuristic + 1 graph-powered scoring criteria
├── analyzer.py         # LLM content analysis (topics, domain, entities)
├── graph_builder.py    # In-memory knowledge graph (networkx)
├── fixer.py            # LLM auto-fix engine (graph-aware)
├── recommender.py      # Folder structure recommendation (graph-aware)
├── anam_client.py      # anam.ai REST API client
├── prompts.py          # LLM prompt templates
├── config.py           # Settings and API key management
├── models.py           # All dataclasses (incl. Entity, Relationship)
├── requirements.txt
├── eval/
│   ├── run_eval.py               # RAG evaluation script (BM25 + anam vector search)
│   └── test-questions.json        # Test questions with ground truth
└── tests/
    └── test_scoring.py # Scoring validation with synthetic docs
```

## RAG Evaluation

Evaluate document quality by testing retrieval + answer generation against a set of questions with known ground truth.

### Local BM25 retrieval (no API needed)

```bash
python eval/run_eval.py rag-files-20260302-094034/ --llm-key $ANTHROPIC_API_KEY
```

### anam.ai vector search (uses live KB)

```bash
python eval/run_eval.py rag-files-20260302-094034/ \
  --llm-key $ANTHROPIC_API_KEY \
  --anam-key $ANAM_API_KEY
```

With `--anam-key`, the eval searches all anam.ai knowledge base folders using vector similarity instead of local BM25, merging results across folders and ranking by score. The first run logs the response schema at INFO level so you can verify field mapping.

**Metrics:** Retrieval Hit Rate, Context Precision, Faithfulness, Answer Correctness.

## Supported File Types


| Format | Parsing                                                | Upload to anam.ai |
| ------ | ------------------------------------------------------ | ----------------- |
| .docx  | Full (headings, paragraphs, metadata)                  | Yes               |
| .pdf   | Full (font-based heading detection, paragraph merging) | Yes               |
| .md    | Full (Markdown heading syntax)                         | Yes               |
| .txt   | Basic (paragraph splitting)                            | Yes               |


## Requirements

- `python-docx` — DOCX parsing
- `PyMuPDF` — PDF parsing
- `click` — CLI framework
- `rich` — terminal formatting
- `requests` — HTTP client for anam.ai API
- `anthropic` — Claude API (only needed for analyze/fix/LLM features)
- `networkx` — in-memory knowledge graph (used automatically during LLM analysis)

## TODO

- **Structured LLM output** — replace JSON-in-markdown prompts with tool_use / structured output for reliable entity and relationship extraction
- **Incremental analysis** — cache per-file analysis results so re-runs only process changed files
- **Relationship deduplication** — merge duplicate edges (same source, target, type) and track edge weight/frequency
- **Configurable thresholds** — expose scoring weights, fuzzy match length, and cluster resolution as CLI flags or config
- **Export graph** — add `cli.py graph export` command to write the knowledge graph as GraphML, JSON, or DOT for external visualization

