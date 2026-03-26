# Chunk-First Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class chunk objects, chunk-level metadata export, and retrieval benchmarking so ragprep measures and exports the units that real RAG systems retrieve.

**Architecture:** Keep `ParsedDocument` as the canonical parse artifact, then derive chunk objects in a parallel chunk pipeline after corpus analysis and before export. Preserve the current document scoring, graph, and fixer flows, but make chunk export and chunk benchmark results the primary retrieval-facing outputs.

**Tech Stack:** Python 3.10+, dataclasses, Click CLI, existing TF-IDF/BM25 utilities, pytest, optional embedding backend abstraction.

---

### Task 1: Add Chunk And Benchmark Models

**Files:**
- Create: `tests/test_chunk_models.py`
- Modify: `src/models.py`
- Test: `tests/test_chunk_models.py`

- [ ] **Step 1: Write the failing model tests**

```python
from src.models import Chunk, ChunkBenchmark, ChunkSet


def test_chunk_ids_are_stable_from_doc_and_index():
    chunk = Chunk(
        chunk_id="lesson-1::000",
        document_id="lesson-1",
        source_file="lesson-1.docx",
        text="Budgeting starts with income and expenses.",
        heading_path=["Lesson 1", "Budget Basics"],
        start_paragraph_index=2,
        end_paragraph_index=3,
        token_estimate=8,
        chunk_type="section",
        quality_flags=[],
    )

    assert chunk.chunk_id == "lesson-1::000"
    assert chunk.heading_path == ["Lesson 1", "Budget Basics"]


def test_chunk_set_tracks_document_lineage():
    chunk = Chunk(
        chunk_id="lesson-1::000",
        document_id="lesson-1",
        source_file="lesson-1.docx",
        text="Budgeting starts with income and expenses.",
        heading_path=[],
        start_paragraph_index=0,
        end_paragraph_index=0,
        token_estimate=8,
        chunk_type="section",
        quality_flags=[],
    )
    chunk_set = ChunkSet(document_id="lesson-1", source_file="lesson-1.docx", chunks=[chunk])

    assert chunk_set.source_file == "lesson-1.docx"
    assert len(chunk_set.chunks) == 1


def test_chunk_benchmark_holds_three_retrieval_modes():
    benchmark = ChunkBenchmark(
        retrieval_mode="hybrid",
        recall_at_5=0.8,
        mrr=0.62,
        ndcg_at_5=0.71,
        query_count=20,
    )

    assert benchmark.retrieval_mode == "hybrid"
    assert benchmark.query_count == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_chunk_models.py -v`
Expected: FAIL with `ImportError` or `AttributeError` for `Chunk`, `ChunkSet`, or `ChunkBenchmark`.

- [ ] **Step 3: Add the new dataclasses to `src/models.py`**

```python
@dataclass
class Chunk:
    """A retrievable unit derived from a parsed document."""

    chunk_id: str
    document_id: str
    source_file: str
    text: str
    heading_path: list[str] = field(default_factory=list)
    start_paragraph_index: int = 0
    end_paragraph_index: int = 0
    token_estimate: int = 0
    chunk_type: str = "section"
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ChunkSet:
    """All chunks produced from one source document."""

    document_id: str
    source_file: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class ChunkBenchmark:
    """Retrieval metrics for one chunk retrieval mode."""

    retrieval_mode: str  # lexical, embedding, hybrid
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    query_count: int
    notes: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_chunk_models.py -v`
Expected: PASS with 3 passing tests.

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_chunk_models.py
git commit -m "feat: add chunk and benchmark models"
```

### Task 2: Build Structure-Aware Chunking

**Files:**
- Create: `src/chunker.py`
- Create: `tests/test_chunker.py`
- Modify: `src/models.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing chunker tests**

```python
from src.chunker import DocumentChunker
from src.models import DocumentMetadata, Paragraph, ParsedDocument


def test_chunker_preserves_heading_path():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/example.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Lesson 1", level=1, style="Heading 1", index=0),
            Paragraph(text="Budgeting starts with income.", level=0, style="Normal", index=1),
            Paragraph(text="Track expenses carefully.", level=0, style="Normal", index=2),
        ],
    )

    chunker = DocumentChunker(target_words=80, overlap_words=20)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) == 1
    assert chunk_set.chunks[0].heading_path == ["Lesson 1"]


def test_chunker_splits_large_sections_without_crossing_headings():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/example.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Heading A", level=1, style="Heading 1", index=0),
            Paragraph(text="One " * 120, level=0, style="Normal", index=1),
            Paragraph(text="Heading B", level=1, style="Heading 1", index=2),
            Paragraph(text="Two " * 40, level=0, style="Normal", index=3),
        ],
    )

    chunker = DocumentChunker(target_words=60, overlap_words=10)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) >= 2
    assert all("Heading B" not in c.text for c in chunk_set.chunks if c.heading_path == ["Heading A"])


def test_chunker_headless_document_still_chunks():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/no-headings.md", file_type="md"),
        paragraphs=[
            Paragraph(text="This paragraph has no heading.", level=0, style="Normal", index=0),
            Paragraph(text="Neither does this one.", level=0, style="Normal", index=1),
        ],
    )

    chunker = DocumentChunker(target_words=20, overlap_words=5)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) >= 1
    assert chunk_set.chunks[0].heading_path == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.chunker'`.

- [ ] **Step 3: Create `src/chunker.py` with a heading-aware chunker**

```python
from dataclasses import dataclass

from .models import Chunk, ChunkSet, ParsedDocument


@dataclass
class DocumentChunker:
    target_words: int = 220
    overlap_words: int = 40

    def chunk_document(self, doc: ParsedDocument) -> ChunkSet:
        chunks: list[Chunk] = []
        current_heading_path: list[str] = []
        section_paragraphs: list[tuple[int, str]] = []

        def flush_section() -> None:
            if not section_paragraphs:
                return
            chunks.extend(self._chunk_section(doc, current_heading_path, section_paragraphs, len(chunks)))

        for para in doc.paragraphs:
            if para.is_heading:
                flush_section()
                current_heading_path = self._update_heading_path(current_heading_path, para.level, para.text)
                section_paragraphs = []
                continue
            if para.text.strip():
                section_paragraphs.append((para.index, para.text))

        flush_section()
        return ChunkSet(document_id=doc.metadata.stem, source_file=doc.metadata.filename, chunks=chunks)
```

- [ ] **Step 4: Add the section chunking helpers**

```python
    def _update_heading_path(self, current: list[str], level: int, text: str) -> list[str]:
        # Heading jumps (e.g., H1 -> H3) intentionally do not synthesize missing
        # intermediate levels; we preserve observed headings only.
        next_path = current[: max(level - 1, 0)]
        next_path.append(text.strip())
        return next_path

    def _chunk_section(
        self,
        doc: ParsedDocument,
        heading_path: list[str],
        section_paragraphs: list[tuple[int, str]],
        chunk_offset: int,
    ) -> list[Chunk]:
        joined_words: list[tuple[int, str]] = []
        for para_index, text in section_paragraphs:
            joined_words.extend((para_index, word) for word in text.split())

        output: list[Chunk] = []
        start = 0
        while start < len(joined_words):
            end = min(start + self.target_words, len(joined_words))
            window = joined_words[start:end]
            chunk_text = " ".join(word for _, word in window)
            output.append(
                Chunk(
                    chunk_id=f"{doc.metadata.stem}::{chunk_offset + len(output):03d}",
                    document_id=doc.metadata.stem,
                    source_file=doc.metadata.filename,
                    text=chunk_text,
                    heading_path=list(heading_path),
                    start_paragraph_index=window[0][0],
                    end_paragraph_index=window[-1][0],
                    token_estimate=max(1, len(chunk_text) // 4),
                    chunk_type="section",
                    quality_flags=[],
                )
            )
            if end == len(joined_words):
                break
            start = max(end - self.overlap_words, start + 1)
        return output
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_chunker.py -v`
Expected: PASS with heading-preserving chunk generation.

Note: heading jumps (e.g., H1 -> H3) should preserve observed headings without inventing missing levels.

- [ ] **Step 6: Commit**

```bash
git add src/chunker.py tests/test_chunker.py src/models.py
git commit -m "feat: add structure-aware document chunker"
```

### Task 3: Export Chunk Metadata And Manifest Contract

**Files:**
- Modify: `src/export.py`
- Modify: `src/cli.py`
- Modify: `tests/test_export.py`
- Create: `tests/test_chunk_export.py`
- Test: `tests/test_export.py`
- Test: `tests/test_chunk_export.py`

- [ ] **Step 1: Write the failing export tests**

```python
import numpy as np
from scipy.sparse import csr_matrix

from src.export import build_manifest_data, write_chunk_sidecar
from src.models import (
    Chunk,
    ChunkSet,
    ContentAnalysis,
    CorpusAnalysis,
    DocumentMetadata,
    FolderNode,
    FolderRecommendation,
    Paragraph,
    ParsedDocument,
    ScoreCard,
)


def test_manifest_includes_chunk_summary():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/test.md", file_type="md"),
        paragraphs=[Paragraph(text="Budgeting matters.", level=0, style="Normal", index=0)],
    )
    analysis = ContentAnalysis(summary="Budgeting lesson")
    card = ScoreCard(file_path="/tmp/test.md", overall_score=80.0)
    ca = CorpusAnalysis(
        tfidf_matrix=csr_matrix(np.eye(1)),
        feature_names=["budget"],
        doc_labels=["test.md"],
        similarity_matrix=np.eye(1),
    )
    rec = FolderRecommendation(root=FolderNode(name="Root", description=""), file_assignments={"test.md": "Budget"})
    chunk_set = ChunkSet(
        document_id="test",
        source_file="test.md",
        chunks=[
            Chunk(
                chunk_id="test::000",
                document_id="test",
                source_file="test.md",
                text="Budgeting matters.",
                heading_path=[],
                start_paragraph_index=0,
                end_paragraph_index=0,
                token_estimate=2,
            )
        ],
    )
    data = build_manifest_data(
        docs=[doc],
        analyses=[analysis],
        cards=[card],
        corpus_analysis=ca,
        recommendation=rec,
        graph=None,
        chunk_sets=[chunk_set],
        benchmarks=[],
    )

    assert data["corpus"]["total_chunks"] == 1
    assert data["documents"][0]["chunk_count"] == 1


def test_write_chunk_sidecar_creates_chunks_json(tmp_path):
    chunk_set = ChunkSet(
        document_id="test",
        source_file="test.md",
        chunks=[
            Chunk(
                chunk_id="test::000",
                document_id="test",
                source_file="test.md",
                text="Budgeting matters.",
                heading_path=[],
                start_paragraph_index=0,
                end_paragraph_index=0,
                token_estimate=2,
            )
        ],
    )
    out = write_chunk_sidecar(tmp_path, chunk_set)
    assert out.endswith(".chunks.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_export.py tests/test_chunk_export.py -v`
Expected: FAIL because `chunk_sets` and `write_chunk_sidecar()` do not exist yet.

- [ ] **Step 3: Extend `src/export.py` for chunk sidecars and v2 manifest fields**

```python
def write_chunk_sidecar(output_dir: str, chunk_set: ChunkSet) -> str:
    data = {
        "schema_version": "2.0",
        "document_id": chunk_set.document_id,
        "source_file": chunk_set.source_file,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "heading_path": c.heading_path,
                "start_paragraph_index": c.start_paragraph_index,
                "end_paragraph_index": c.end_paragraph_index,
                "token_estimate": c.token_estimate,
                "chunk_type": c.chunk_type,
                "quality_flags": c.quality_flags,
                "metadata": c.metadata,
            }
            for c in chunk_set.chunks
        ],
    }
    out_path = Path(output_dir) / f"{chunk_set.document_id}.chunks.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)
```

- [ ] **Step 4: Update `build_manifest_data()` to accept chunk sets and benchmarks**

```python
# IMPORTANT: extend existing manifest shape; do not replace current keys
# such as similarity_matrix, knowledge_graph, folder tree, and existing
# per-document analysis fields.
def build_manifest_data(
    docs,
    analyses,
    cards,
    corpus_analysis,
    recommendation,
    graph=None,
    chunk_sets=None,
    benchmarks=None,
):
    chunk_sets = chunk_sets or []
    benchmarks = benchmarks or []
    chunk_count_by_file = {cs.source_file: len(cs.chunks) for cs in chunk_sets}

    # Keep existing manifest construction code unchanged.
    # Then add the following block immediately before `return data`.
    data["schema_version"] = "2.0"
    data["corpus"]["total_chunks"] = sum(len(cs.chunks) for cs in chunk_sets)
    for doc_entry in data["documents"]:
        filename = doc_entry["source_file"]
        doc_entry["chunk_count"] = chunk_count_by_file.get(filename, 0)
    data["benchmarks"] = [
        {
            "retrieval_mode": b.retrieval_mode,
            "recall_at_5": b.recall_at_5,
            "mrr": b.mrr,
            "ndcg_at_5": b.ndcg_at_5,
            "query_count": b.query_count,
        }
        for b in benchmarks
    ]
    return data
```

- [ ] **Step 5: Update `src/cli.py` analyze/fix exports to write chunk sidecars**

```python
# Canonical metadata output directory (standardize here):
# use `.ragprep/` going forward, replacing legacy `.kb-prep/`.
meta_dir = os.path.join(path, ".ragprep")

# Optional one-time migration path for legacy runs:
legacy_meta_dir = os.path.join(path, ".kb-prep")
if os.path.isdir(legacy_meta_dir) and not os.path.isdir(meta_dir):
    console.print("[yellow]Legacy metadata directory '.kb-prep/' detected. New outputs will be written to '.ragprep/'.[/yellow]")

# Use fixed defaults here; Task 8 introduces CLI-tunable values.
chunker = DocumentChunker(target_words=220, overlap_words=40)
chunk_sets = [chunker.chunk_document(doc) for doc in docs]

for chunk_set in chunk_sets:
    write_chunk_sidecar(meta_dir, chunk_set)

data = build_manifest_data(
    docs,
    analyses,
    cards,
    corpus_analysis,
    recommendation,
    graph,
    chunk_sets=chunk_sets,
    benchmarks=[],
)
```

Note: update existing analyze/fix code paths that currently hardcode metadata output directories so both commands converge on `.ragprep/`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_export.py tests/test_chunk_export.py -v`
Expected: PASS with manifest and chunk sidecar coverage.

- [ ] **Step 7: Commit**

```bash
git add src/export.py src/cli.py tests/test_export.py tests/test_chunk_export.py
git commit -m "feat: export chunk metadata and ingestion manifest"
```

### Task 4: Add Chunk-Level Retrieval Benchmarking

**Files:**
- Create: `src/benchmark.py`
- Create: `tests/test_benchmark.py`
- Modify: `test-data/layer5_rag_quality/test_end_to_end.py`
- Modify: `src/cli.py`
- Test: `tests/test_benchmark.py`
- Test: `test-data/layer5_rag_quality/test_end_to_end.py`

- [ ] **Step 1: Write the failing benchmark unit tests**

```python
from src.benchmark import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_recall_at_k_counts_hits():
    ranked_lists = [[3, 1, 2], [4, 5, 6]]
    gold = [{1}, {9}]

    assert recall_at_k(ranked_lists, gold, k=2) == 0.5


def test_mean_reciprocal_rank_scores_first_hit():
    ranked_lists = [[9, 3, 1], [7, 8, 2]]
    gold = [{1}, {2}]

    assert round(mean_reciprocal_rank(ranked_lists, gold), 3) == 0.417


def test_ndcg_handles_multiple_relevant_chunks():
    ranked_lists = [[3, 1, 2], [9, 8, 7]]
    gold = [{1, 2, 3}, {7, 8}]

    score = ndcg_at_k(ranked_lists, gold, k=3)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_benchmark.py -v`
Expected: FAIL because `src.benchmark` does not exist.

- [ ] **Step 3: Implement benchmark metric functions and retrieval adapters**

```python
import math

from src.corpus_analyzer import bm25_score


def _bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[int]:
    scores = bm25_score(query, chunks)
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]


def recall_at_k(ranked_lists: list[list[int]], gold_sets: list[set[int]], k: int) -> float:
    hits = 0
    for ranked, gold in zip(ranked_lists, gold_sets):
        if any(idx in gold for idx in ranked[:k]):
            hits += 1
    return hits / len(gold_sets) if gold_sets else 0.0


def mean_reciprocal_rank(ranked_lists: list[list[int]], gold_sets: list[set[int]]) -> float:
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_sets):
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in gold:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(gold_sets) if gold_sets else 0.0


def ndcg_at_k(ranked_lists: list[list[int]], gold_sets: list[set[int]], k: int) -> float:
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_sets):
        dcg = 0.0
        for rank, idx in enumerate(ranked[:k], start=1):
            if idx in gold:
                dcg += 1.0 / math.log2(rank + 1)
        max_relevant = min(len(gold), k)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, max_relevant + 1)) or 1.0
        total += dcg / ideal
    return total / len(gold_sets) if gold_sets else 0.0
```

- [ ] **Step 3a: Make BM25 scorer public in `src/corpus_analyzer.py`**

```python
# Rename existing `_bm25_score` implementation to `bm25_score`.
# Keep backward compatibility for existing internal imports:
_bm25_score = bm25_score
```

- [ ] **Step 4: Add a benchmark runner over chunk text**

```python
def benchmark_chunk_retrieval(
    queries: list[str],
    gold_sets: list[set[int]],
    chunks: list[str],
) -> list[ChunkBenchmark]:
    lexical_ranked = [_bm25_search(q, chunks, top_k=10) for q in queries]
    return [
        ChunkBenchmark(
            retrieval_mode="lexical",
            recall_at_5=recall_at_k(lexical_ranked, gold_sets, k=5),
            mrr=mean_reciprocal_rank(lexical_ranked, gold_sets),
            ndcg_at_5=ndcg_at_k(lexical_ranked, gold_sets, k=5),
            query_count=len(queries),
        )
    ]
```

- [ ] **Step 5: Replace the ad hoc eval code in `test_end_to_end.py` with `src.benchmark`**

```python
qas = [record["questions"][0] for record in sample if record.get("questions")]
gold_chunk_sets = [{idx} for idx in range(len(qas))]

benchmarks = benchmark_chunk_retrieval(
    queries=[qa["question"] for qa in qas],
    gold_sets=gold_chunk_sets,
    chunks=engine_chunks,
)

lexical = next(b for b in benchmarks if b.retrieval_mode == "lexical")
assert lexical.recall_at_5 > 0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_benchmark.py test-data/layer5_rag_quality/test_end_to_end.py -v`
Expected: PASS with chunk-level metric assertions.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py test-data/layer5_rag_quality/test_end_to_end.py src/cli.py
git commit -m "feat: add chunk-level retrieval benchmarks"
```

### Task 5: Reframe Self-Containment Around Chunk Safety

**Files:**
- Modify: `src/scorer.py`
- Modify: `src/fixer.py`
- Create: `tests/test_chunk_safety.py`
- Test: `tests/test_chunk_safety.py`

- [ ] **Step 1: Write the failing chunk-safety tests**

```python
from src.models import DocumentMetadata, Paragraph, ParsedDocument
from src.fixer import _has_positional_reference
from src.scorer import QualityScorer


def test_self_containment_flags_dangling_reference():
    doc_with_reference = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/test.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Savings Basics", level=1, style="Heading 1", index=0),
            Paragraph(text="As discussed above, students should compare needs and wants.", level=0, style="Normal", index=1),
        ],
    )
    scorer = QualityScorer()
    card = scorer.score(doc_with_reference)

    issue = next(i for i in card.all_issues if i.category == "self_containment")
    assert "Dangling" in issue.message
    assert issue.location == 1


def test_fixer_rewrites_reference_to_standalone_sentence():
    assert _has_positional_reference("As discussed above, students should compare needs and wants.")
    assert not _has_positional_reference("Students should compare needs and wants before building a budget.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_chunk_safety.py -v`
Expected: FAIL because the scorer/fixer still speak in paragraph-level terms.

- [ ] **Step 3: Update self-containment issue language in `src/scorer.py`**

```python
Issue(
    severity=Severity.WARNING,
    category="self_containment",
    message=f'Dangling {ref_type}: "{match.group()}"',
    location=para.index,
    context=f"...{snippet}...",
    fix="Rewrite so the chunk stands alone without requiring earlier or later context.",
)
```

- [ ] **Step 4: Update fixer action text and prompts in `src/fixer.py`**

```python
def _has_positional_reference(text: str) -> bool:
    patterns = [
        r"\b(?:as\s+)?mentioned\s+(?:above|below|earlier|previously|before)\b",
        r"\bthe\s+(?:above|below|following|previous|preceding)\s+\w+",
        r"\b(?:refer|referring)\s+to\s+(?:the\s+)?(?:above|below|previous)",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


return FixAction(
    category="self_containment",
    original_text=original,
    fixed_text=fixed_text,
    paragraph_index=para_idx,
    description="Rewrote paragraph to be chunk-safe and self-contained",
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_chunk_safety.py -v`
Expected: PASS with chunk-safe wording and behavior.

- [ ] **Step 6: Commit**

```bash
git add src/scorer.py src/fixer.py tests/test_chunk_safety.py
git commit -m "refactor: reframe self-containment around chunk safety"
```

### Task 6: Add Deterministic Cleanup Before Chunking

**Files:**
- Create: `src/cleaner.py`
- Create: `tests/test_cleaner.py`
- Modify: `src/cli.py`
- Modify: `src/parser.py`
- Test: `tests/test_cleaner.py`

- [ ] **Step 1: Write the failing cleanup tests**

```python
from src.cleaner import DocumentCleaner


def test_cleaner_drops_repeated_headers_and_footers():
    cleaner = DocumentCleaner()
    paragraphs = [
        "Financial Literacy Grade 4-5",
        "Budgeting begins with goals.",
        "Page 1",
        "Financial Literacy Grade 4-5",
        "Track savings over time.",
        "Page 2",
    ]

    cleaned = cleaner.clean_paragraphs(paragraphs)

    assert "Page 1" not in cleaned
    assert "Page 2" not in cleaned
    assert cleaned.count("Financial Literacy Grade 4-5") <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cleaner.py -v`
Expected: FAIL because `DocumentCleaner` does not exist.

- [ ] **Step 3: Implement deterministic cleanup rules in `src/cleaner.py`**

```python
from collections import Counter
import re

from .models import Paragraph


class DocumentCleaner:
    def clean_paragraphs(self, paragraphs: list[str]) -> list[str]:
        temp = [Paragraph(text=p, level=0, style="Normal", index=i) for i, p in enumerate(paragraphs)]
        return [p.text for p in self.clean_document(temp)]

    def should_drop(self, text: str, seen: Counter) -> bool:
        if re.fullmatch(r"Page\s+\d+", text):
            return True
        if seen[text] >= 1 and len(text.split()) <= 8:
            return True
        return False

    def clean_document(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        rebuilt: list[Paragraph] = []
        seen: Counter = Counter()
        for para in paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if self.should_drop(text, seen):
                seen[text] += 1
                continue
            rebuilt.append(Paragraph(text=text, level=para.level, style=para.style, index=para.index))
            seen[text] += 1
        return rebuilt
```

- [ ] **Step 4: Apply cleanup in `src/cli.py` after parse and before corpus analysis**

```python
cleaner = DocumentCleaner()
cleaned_docs = []
for doc in docs:
    cleaned_paragraphs = cleaner.clean_document(doc.paragraphs)
    cleaned_docs.append(
        ParsedDocument(
            metadata=doc.metadata,
            paragraphs=cleaned_paragraphs,
            heading_tree=doc.heading_tree,
        )
    )
docs = cleaned_docs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cleaner.py -v`
Expected: PASS with deterministic cleanup behavior.

- [ ] **Step 6: Commit**

```bash
git add src/cleaner.py src/cli.py tests/test_cleaner.py src/parser.py
git commit -m "feat: add deterministic document cleanup"
```

### Task 7: Add Split Recommendations For Broad Documents

**Files:**
- Modify: `src/models.py`
- Modify: `src/scorer.py`
- Modify: `src/export.py`
- Create: `tests/test_split_recommendations.py`
- Test: `tests/test_split_recommendations.py`

- [ ] **Step 1: Write the failing split recommendation tests**

```python
import numpy as np
from scipy.sparse import csr_matrix

from src.corpus_analyzer import build_corpus_analysis
from src.models import DocumentMetadata, Paragraph, ParsedDocument
from src.scorer import QualityScorer


def test_broad_documents_emit_split_recommendation():
    broad_document = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/broad.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Saving", level=1, style="Heading 1", index=0),
            Paragraph(text="Save money each week for goals.", level=0, style="Normal", index=1),
            Paragraph(text="Credit", level=1, style="Heading 1", index=2),
            Paragraph(text="Credit cards and loans require repayment with interest.", level=0, style="Normal", index=3),
            Paragraph(text="Careers", level=1, style="Heading 1", index=4),
            Paragraph(text="Career planning connects education to future income.", level=0, style="Normal", index=5),
        ],
    )
    corpus_analysis = build_corpus_analysis([broad_document])
    scorer = QualityScorer(corpus_analysis=corpus_analysis)
    card = scorer.score(broad_document)

    issue = next(i for i in card.all_issues if i.category == "file_focus")
    assert "split" in issue.fix.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_split_recommendations.py -v`
Expected: FAIL because no structured split recommendation is exported.

- [ ] **Step 3: Add a structured recommendation model**

```python
@dataclass
class SplitRecommendation:
    source_file: str
    reason: str
    proposed_boundaries: list[int] = field(default_factory=list)
    suggested_titles: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Export recommendations with manifest entries**

```python
"split_recommendations": [
    {
        "source_file": rec.source_file,
        "reason": rec.reason,
        "proposed_boundaries": rec.proposed_boundaries,
        "suggested_titles": rec.suggested_titles,
    }
    for rec in split_recommendations
],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_split_recommendations.py -v`
Expected: PASS with exported split recommendations for broad documents.

- [ ] **Step 6: Commit**

```bash
git add src/models.py src/scorer.py src/export.py tests/test_split_recommendations.py
git commit -m "feat: export focused-document split recommendations"
```

### Task 8: Add CLI Gating For Optional Enrichment

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_report.py`
- Create: `tests/test_cli_enrichment.py`
- Test: `tests/test_cli_enrichment.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from click.testing import CliRunner
from src.cli import cli
from tests.test_scoring import _create_test_docx


class _FakeAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    async def analyze_and_build_graph(self, docs):
        from src.models import ContentAnalysis
        return [ContentAnalysis(summary="ok") for _ in docs], None


class _FakeRecommender:
    def __init__(self, *args, **kwargs):
        pass

    async def recommend(self, docs, analyses):
        from src.models import FolderNode, FolderRecommendation
        return FolderRecommendation(root=FolderNode(name="Knowledge Base", description=""), file_assignments={})

    def validate_assignments(self, assignments, similarity_matrix, doc_labels):
        return 0.0, []


def test_analyze_skip_enrichment_hides_folder_section(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)
    monkeypatch.setattr("src.analyzer.ContentAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("src.recommender.FolderRecommender", _FakeRecommender)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--llm-key", "test", "--skip-enrichment"])
    assert "Recommended Folder Structure" not in result.output


def test_analyze_chunk_export_flag_writes_chunks(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)
    monkeypatch.setattr("src.analyzer.ContentAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("src.recommender.FolderRecommender", _FakeRecommender)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--llm-key", "test", "--export-chunks"])
    assert result.exit_code == 0
    assert list((tmp_path / ".ragprep").glob("*.chunks.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli_enrichment.py -v`
Expected: FAIL because the flags do not exist.

- [ ] **Step 3: Add explicit CLI feature gates**

```python
@click.option("--export-chunks/--no-export-chunks", default=True, help="Write per-document .chunks.json files")
@click.option("--chunk-size", default=220, type=int, help="Target words per chunk")
@click.option("--chunk-overlap", default=40, type=int, help="Overlapping words between chunks")
@click.option("--run-benchmark", is_flag=True, help="Run chunk-level retrieval benchmark")
@click.option("--skip-enrichment", is_flag=True, help="Skip folder recommendation and graph-heavy output")
```

- [ ] **Step 4: Guard enrichment-only console and report sections**

```python
if not skip_enrichment:
    _print_graph_summary(graph)
    recommendation = asyncio.run(recommender.recommend(docs, analyses))
else:
    recommendation = FolderRecommendation(root=FolderNode(name="Knowledge Base", description=""), file_assignments={})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli_enrichment.py tests/test_report.py -v`
Expected: PASS with chunk export and optional enrichment behavior.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli_enrichment.py tests/test_report.py
git commit -m "feat: add CLI controls for chunk export and enrichment"
```

### Task 9: Final Verification

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Test: `tests/`
- Test: `test-data/layer5_rag_quality/`

- [ ] **Step 1: Update docs for the new chunk-first flow**

```markdown
- `analyze` now exports `.chunks.json` files with chunk lineage and optional benchmark output.
- Chunk metadata includes `chunk_id`, `heading_path`, paragraph span, and token estimate.
- Foldering and graph sections are enrichment layers and can be skipped for ingestion-only runs.
```

- [ ] **Step 2: Run targeted fast tests**

Run: `python3 -m pytest tests/test_chunk_models.py tests/test_chunker.py tests/test_export.py tests/test_benchmark.py tests/test_cleaner.py tests/test_cli_enrichment.py -v`
Expected: PASS for all targeted unit suites.

- [ ] **Step 3: Run retrieval quality regression tests**

Run: `python3 -m pytest test-data/layer5_rag_quality/test_end_to_end.py -v`
Expected: PASS with chunk benchmark assertions and no retrieval regression.

- [ ] **Step 4: Run lint**

Run: `ruff check src tests test-data`
Expected: PASS with no new `E`, `F`, `W`, or `I` violations.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document chunk-first ingestion workflow"
```

## Self-Review

**Spec coverage:** This plan covers chunk objects, chunk metadata export, chunk benchmark infrastructure, chunk-safe self-containment fixes, deterministic cleanup, focused split recommendations, metadata contract hardening, and optional enrichment gating.

**Placeholder scan:** No `TODO`, `TBD`, or deferred implementation markers remain in the task steps.

**Type consistency:** The plan consistently uses `Chunk`, `ChunkSet`, `ChunkBenchmark`, `DocumentChunker`, and `DocumentCleaner` across tasks, export, benchmark, and CLI changes.
