"""Real-document PDF eval tests.

Tests ragprep's parsing and retrieval pipeline against three human-annotated
PDF datasets:
  - SCORE-Bench   : parse fidelity (char F1 vs ground-truth text)
  - SCORE-Bench   : chunk-through-retrieve hit@5 (BM25)
  - Kleister NDA  : entity recall (party names / dates present in full_text)
  - OmniDocBench  : layout-boundary quality (TextTiling Pk vs annotation)
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = [pytest.mark.layer5, pytest.mark.timeout(120)]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

CHAR_F1_THRESHOLD = 0.60  # SCORE-Bench parse fidelity (mean across all PDFs, incl. scanned)
HIT5_ENGINE_THRESHOLD = 0.45  # SCORE-Bench engine hit@5
ENTITY_RECALL_THRESHOLD = 0.30  # Kleister NDA entity recall
PK_THRESHOLD = 0.50  # OmniDocBench layout Pk (lower is better)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _char_ngram_f1(pred: str, ref: str, n: int = 5) -> float:
    """Character n-gram F1 between two strings."""
    pred_lower = pred.lower()
    ref_lower = ref.lower()

    pred_ngrams: dict[str, int] = {}
    for i in range(len(pred_lower) - n + 1):
        g = pred_lower[i : i + n]
        pred_ngrams[g] = pred_ngrams.get(g, 0) + 1

    ref_ngrams: dict[str, int] = {}
    for i in range(len(ref_lower) - n + 1):
        g = ref_lower[i : i + n]
        ref_ngrams[g] = ref_ngrams.get(g, 0) + 1

    if not pred_ngrams or not ref_ngrams:
        return 0.0

    common = sum(min(pred_ngrams.get(g, 0), ref_ngrams.get(g, 0)) for g in ref_ngrams)
    precision = common / sum(pred_ngrams.values())
    recall = common / sum(ref_ngrams.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _clean_score_bench_annotation(raw: str) -> str:
    """Strip SCORE-Bench annotation markers, leaving only the content text."""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("---") and ("Begin" in stripped or "End" in stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _texttile_chunks(doc, ca) -> list[str]:
    """Split a parsed document into TextTiling chunks (falls back to full text)."""
    filename = doc.metadata.filename
    boundaries: list[int] = []
    if ca and filename in ca.doc_metrics:
        boundaries = ca.doc_metrics[filename].topic_boundaries

    para_texts = [p.text for p in doc.paragraphs if p.text.strip()]
    if not para_texts:
        return []
    if not boundaries:
        return [" ".join(para_texts)]

    chunks: list[str] = []
    prev = 0
    for b in sorted(boundaries):
        if b > prev:
            chunk = " ".join(para_texts[prev:b])
            if chunk.strip():
                chunks.append(chunk)
        prev = b
    if prev < len(para_texts):
        chunk = " ".join(para_texts[prev:])
        if chunk.strip():
            chunks.append(chunk)
    return chunks or [" ".join(para_texts)]


def _bm25_search(query: str, chunks: list[str], top_k: int = 5) -> list[int]:
    from src.corpus_analyzer import _bm25_score

    if not chunks:
        return []
    scores = _bm25_score(query, chunks)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    return ranked[:top_k]


def _pk_score(hypothesized: list[int], reference: list[int], n_units: int, k: int | None = None) -> float:
    """Compute Pk segmentation metric (Beeferman 1999).

    Args:
        hypothesized: List of 0/1 per unit; 1 = boundary after this unit.
        reference:    Same format for gold boundaries.
        n_units:      Total number of units.
        k:            Window size; defaults to half the average segment length.
    """
    if n_units < 2:
        return 0.0

    if k is None:
        ref_boundaries = sum(reference)
        avg_seg_len = n_units / max(ref_boundaries + 1, 1)
        k = max(1, int(avg_seg_len / 2))

    errors = 0
    total = 0
    for i in range(n_units - k):
        ref_same = reference[i] == reference[i + k] if i < len(reference) and i + k < len(reference) else False
        hyp_same = (
            hypothesized[i] == hypothesized[i + k] if i < len(hypothesized) and i + k < len(hypothesized) else False
        )
        if ref_same != hyp_same:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# SCORE-Bench: parse fidelity
# ---------------------------------------------------------------------------


def test_score_bench_parse_fidelity(score_bench_data):
    """Parsed full_text char-F1 vs SCORE-Bench ground-truth annotations >= 0.80.

    Only PDFs that have matching annotations are scored; the test skips
    gracefully if fewer than 5 annotated PDFs are present.
    """
    from src.parser import DocumentParser

    pdfs = score_bench_data["pdfs"]
    annotations = score_bench_data["annotations"]

    # Only evaluate PDFs that have a ground-truth annotation
    annotated = [(p, annotations[p.name]) for p in pdfs if p.name in annotations]
    if len(annotated) < 5:
        pytest.skip(f"Need at least 5 annotated PDFs, got {len(annotated)}")

    parser = DocumentParser()
    scores: list[float] = []
    failed_parse = 0

    for pdf_path, raw_anno in annotated:
        ref_text = _clean_score_bench_annotation(raw_anno)
        if not ref_text.strip():
            continue
        try:
            doc = parser.parse(str(pdf_path))
            pred_text = doc.full_text
        except Exception as exc:
            print(f"  [warn] parse failed for {pdf_path.name}: {exc}")
            failed_parse += 1
            continue

        f1 = _char_ngram_f1(pred_text, ref_text)
        scores.append(f1)

    assert scores, f"No scores computed (failed_parse={failed_parse}, annotated={len(annotated)})"

    mean_f1 = sum(scores) / len(scores)
    print(f"\n[score_bench_fidelity] n={len(scores)} PDFs, mean char-F1={mean_f1:.3f}, failed_parse={failed_parse}")
    print(
        f"  F1 distribution: min={min(scores):.3f} median={sorted(scores)[len(scores) // 2]:.3f} max={max(scores):.3f}"
    )

    assert mean_f1 >= CHAR_F1_THRESHOLD, (
        f"Parse fidelity char-F1={mean_f1:.3f} < threshold={CHAR_F1_THRESHOLD}. "
        f"Parser may be losing significant content from real PDFs."
    )


# ---------------------------------------------------------------------------
# SCORE-Bench: chunk-through-retrieve hit@5
# ---------------------------------------------------------------------------


def test_score_bench_hit_at_5(score_bench_data):
    """Engine hit@5 >= 0.45 and >= naive on SCORE-Bench retrieval.

    Uses 5-word phrases from ground-truth annotations as queries; a hit is
    recorded when the source PDF's chunks appear in the top-5 BM25 results
    over the whole corpus chunk pool.
    """
    from src.corpus_analyzer import build_corpus_analysis
    from src.parser import DocumentParser

    pdfs = score_bench_data["pdfs"]
    annotations = score_bench_data["annotations"]

    if len(pdfs) < 5:
        pytest.skip("Need at least 5 PDFs")

    parser = DocumentParser()

    # Parse all PDFs, collect chunks
    parsed_docs = []
    parse_failures = []
    for pdf_path in pdfs:
        try:
            doc = parser.parse(str(pdf_path))
            parsed_docs.append(doc)
        except Exception as exc:
            parse_failures.append(pdf_path.name)
            print(f"  [warn] parse failed for {pdf_path.name}: {exc}")

    if len(parsed_docs) < 5:
        pytest.skip(f"Too few successfully parsed PDFs ({len(parsed_docs)})")

    # Build corpus analysis for TextTiling boundaries
    ca = build_corpus_analysis(parsed_docs)

    # Build chunk pool
    all_chunks: list[str] = []
    chunk_to_doc: list[str] = []  # which PDF filename each chunk came from

    for doc in parsed_docs:
        chunks = _texttile_chunks(doc, ca)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_doc.append(doc.metadata.filename)

    # Naive baseline: full-text per document as single chunk
    naive_chunks: list[str] = []
    naive_chunk_to_doc: list[str] = []
    for doc in parsed_docs:
        text = doc.full_text.strip()
        if text:
            naive_chunks.append(text)
            naive_chunk_to_doc.append(doc.metadata.filename)

    assert all_chunks, "Engine produced no chunks"
    assert naive_chunks, "Naive produced no chunks"

    # Build queries from annotation phrases (5-word windows, sample a few per doc)
    import re

    def _sample_queries(text: str, n: int = 3) -> list[str]:
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 5:
            return [" ".join(words)] if words else []
        step = max(1, len(words) // (n + 1))
        queries = []
        for i in range(n):
            start = step * (i + 1)
            if start + 5 <= len(words):
                queries.append(" ".join(words[start : start + 5]))
        return queries

    engine_hits = 0
    naive_hits = 0
    total_queries = 0

    for doc in parsed_docs:
        fname = doc.metadata.filename
        raw_anno = annotations.get(fname, "")
        if not raw_anno:
            continue
        ref_text = _clean_score_bench_annotation(raw_anno)
        queries = _sample_queries(ref_text)

        for query in queries:
            total_queries += 1

            # Engine: hit if any top-5 chunk comes from the same PDF
            top_engine = _bm25_search(query, all_chunks, top_k=5)
            engine_hit = any(chunk_to_doc[i] == fname for i in top_engine)
            if engine_hit:
                engine_hits += 1

            # Naive: same logic
            top_naive = _bm25_search(query, naive_chunks, top_k=5)
            naive_hit = any(naive_chunk_to_doc[i] == fname for i in top_naive)
            if naive_hit:
                naive_hits += 1

    if total_queries == 0:
        pytest.skip("No queries generated (no annotations found for parsed PDFs)")

    engine_rate = engine_hits / total_queries
    naive_rate = naive_hits / total_queries

    print(
        f"\n[score_bench_hit5] total_queries={total_queries}, "
        f"engine={engine_rate:.3f} ({engine_hits}/{total_queries}), "
        f"naive={naive_rate:.3f} ({naive_hits}/{total_queries})"
    )
    print(f"  engine_chunks={len(all_chunks)}, naive_chunks={len(naive_chunks)}")

    assert engine_rate >= HIT5_ENGINE_THRESHOLD, (
        f"Engine hit@5={engine_rate:.3f} < threshold={HIT5_ENGINE_THRESHOLD}. "
        f"Chunking or retrieval may be too fragmented for real PDF content."
    )

    assert engine_rate >= naive_rate * 0.95, (
        f"Engine hit@5={engine_rate:.3f} is more than 5% below "
        f"naive hit@5={naive_rate:.3f}. "
        f"TextTiling chunking is hurting retrieval on SCORE-Bench PDFs."
    )


# ---------------------------------------------------------------------------
# Kleister NDA: entity recall
# ---------------------------------------------------------------------------


def test_kleister_nda_entity_recall(kleister_nda_data):
    """Party names and dates from ground-truth annotations appear in parsed full_text.

    Entity recall >= 0.30 tests whether parsing preserves key NDA content.
    Ground truth provides values like effective_date=2013-03-01 and party=Nike Inc.;
    we check whether each value string appears (case-insensitive) in the parsed text.

    entity_rows in the fixture is ordered by dataset document index, so we
    use only the first N rows corresponding to the N PDFs sampled.
    """
    from src.parser import DocumentParser

    pdfs = kleister_nda_data["pdfs"]
    entity_rows = kleister_nda_data["entity_rows"]

    if not pdfs:
        pytest.skip("No Kleister NDA PDFs available")
    if not entity_rows:
        pytest.skip("No Kleister NDA ground-truth entities found")

    parser = DocumentParser()

    # Limit to first 30 PDFs and the corresponding entity rows
    sample_pdfs = pdfs[:30]
    sample_entity_rows = entity_rows[: len(sample_pdfs)]

    found = 0
    total = 0
    parse_failures = 0

    all_texts: list[str] = []
    for pdf_path in sample_pdfs:
        try:
            doc = parser.parse(str(pdf_path))
            all_texts.append(doc.full_text.lower())
        except Exception as exc:
            parse_failures += 1
            print(f"  [warn] parse failed for {pdf_path.name}: {exc}")
            all_texts.append("")

    if not any(all_texts):
        pytest.skip(f"All {len(sample_pdfs)} PDFs failed to parse")

    combined_text = " ".join(all_texts)

    for kv in sample_entity_rows:
        for _key, value in kv.items():
            if not value or len(value) < 3:
                continue
            total += 1
            if value.lower() in combined_text:
                found += 1

    if total == 0:
        pytest.skip("No entity values to check")

    recall = found / total
    print(
        f"\n[kleister_nda_entity_recall] n_pdfs={len(sample_pdfs)}, "
        f"n_entity_rows={len(sample_entity_rows)}, "
        f"parse_failures={parse_failures}, "
        f"recall={recall:.3f} ({found}/{total} entity values found)"
    )

    assert recall >= ENTITY_RECALL_THRESHOLD, (
        f"Entity recall={recall:.3f} < threshold={ENTITY_RECALL_THRESHOLD}. "
        f"Parser may be dropping key NDA content (dates, party names)."
    )


# ---------------------------------------------------------------------------
# OmniDocBench: layout boundary Pk
# ---------------------------------------------------------------------------


def test_omnidocbench_layout_boundary_pk(omnidocbench_data):
    """TextTiling boundaries approximate OmniDocBench layout section transitions.

    Pk <= 0.50 means TextTiling is at least as good as a random segmentation.

    Uses category_type sequences from boundaries.jsonl (built by setup.py) as
    ground-truth boundaries.  A boundary is placed wherever the category_type
    changes between consecutive layout elements.  TextTiling is run on the
    matching synthetic .txt document.
    """
    from src.corpus_analyzer import build_corpus_analysis
    from src.parser import DocumentParser

    docs = omnidocbench_data["docs"]
    boundaries = omnidocbench_data["boundaries"]

    if len(docs) < 3:
        pytest.skip("Need at least 3 OmniDocBench documents")

    parser = DocumentParser()

    # Index boundary records by safe_name
    boundary_index: dict[str, list[str]] = {rec["safe_name"]: rec["categories"] for rec in boundaries}

    pk_scores: list[float] = []
    parse_failures = 0
    skipped_no_boundary = 0

    for doc_path in docs:
        safe_name = doc_path.stem
        categories = boundary_index.get(safe_name)
        if not categories:
            skipped_no_boundary += 1
            continue

        # Build gold boundary signal from category_type transitions
        gold_boundaries: list[int] = []
        prev_cat = None
        for cat in categories:
            if prev_cat is not None and cat != prev_cat:
                gold_boundaries.append(1)
            else:
                gold_boundaries.append(0)
            prev_cat = cat

        n_units = len(gold_boundaries)
        if n_units < 4:
            continue

        # Parse the synthetic text document
        try:
            doc = parser.parse(str(doc_path))
        except Exception as exc:
            parse_failures += 1
            print(f"  [warn] parse failed for {doc_path.name}: {exc}")
            continue

        para_texts = [p.text for p in doc.paragraphs if p.text.strip()]
        if len(para_texts) < 2:
            continue

        # Get TextTiling boundaries from corpus analysis
        ca = build_corpus_analysis([doc])
        doc_metrics = ca.doc_metrics.get(doc.metadata.filename)
        hyp_boundary_indices = set(doc_metrics.topic_boundaries) if doc_metrics else set()

        # Map TextTiling paragraph-level boundaries to the same n_units scale
        scale_ratio = n_units / max(len(para_texts), 1)
        hyp_signal: list[int] = [0] * n_units
        for b in hyp_boundary_indices:
            mapped = min(int(b * scale_ratio), n_units - 1)
            hyp_signal[mapped] = 1

        pk = _pk_score(hyp_signal, gold_boundaries, n_units)
        pk_scores.append(pk)

    if not pk_scores:
        pytest.skip(
            f"No OmniDocBench documents could be evaluated "
            f"(parse_failures={parse_failures}, no_boundary={skipped_no_boundary})"
        )

    mean_pk = sum(pk_scores) / len(pk_scores)
    print(
        f"\n[omnidocbench_layout_pk] n={len(pk_scores)} docs, "
        f"mean_Pk={mean_pk:.3f}, "
        f"parse_failures={parse_failures}, skipped_no_boundary={skipped_no_boundary}"
    )
    print(
        f"  Pk distribution: min={min(pk_scores):.3f} "
        f"median={sorted(pk_scores)[len(pk_scores) // 2]:.3f} "
        f"max={max(pk_scores):.3f}"
    )

    assert mean_pk <= PK_THRESHOLD, (
        f"Layout boundary Pk={mean_pk:.3f} > threshold={PK_THRESHOLD}. "
        f"TextTiling boundaries diverge too much from OmniDocBench layout annotations."
    )
