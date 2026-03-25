"""Task 9 — BM25+ Retrieval validation."""

import pytest

pytestmark = [pytest.mark.layer4, pytest.mark.timeout(120)]


def test_task09_bm25_returns_results(engine):
    """BM25 should return ranked results for a simple query."""
    docs = [
        "Adding fractions requires finding common denominators.",
        "Insurance protects against unexpected financial loss.",
        "A budget tracks household income and expenses.",
    ]
    results = engine.bm25_search("fractions common denominator", docs, top_k=3)
    assert len(results) > 0, "BM25 should return results"
    assert results[0] == 0, "First doc should rank highest for fraction query"


def test_task09_bm25_relevance_ordering(engine, squad_data):
    """Documents matching the query should rank higher than unrelated documents."""
    if len(squad_data) < 50:
        pytest.skip("Need at least 50 SQuAD paragraphs")

    # Pick a question and its source paragraph
    sample = squad_data[:50]
    texts = [r["text"] for r in sample]

    correct_retrievals = 0
    total = 0
    for r in sample[:20]:
        if not r.get("questions"):
            continue
        query = r["questions"][0]["question"]
        doc_idx = texts.index(r["text"])

        results = engine.bm25_search(query, texts, top_k=5)
        if doc_idx in results:
            correct_retrievals += 1
        total += 1

    if total > 0:
        hit_rate = correct_retrievals / total
        assert hit_rate > 0.3, f"BM25 hit@5 on SQuAD = {hit_rate:.2f}, expected > 0.3"


def test_task09_bm25_scores_nonnegative(engine):
    """All BM25 scores should be non-negative."""
    docs = ["The cat sat on the mat.", "Dogs run in the park.", "Birds fly in the sky."]
    scores = engine.bm25_scores("cat mat", docs)
    assert all(s >= 0 for s in scores), f"BM25 scores should be non-negative: {scores}"
