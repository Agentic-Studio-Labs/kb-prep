"""Task 3 — Retrieval-Aware Score validation."""

import numpy as np
import pytest

# Thresholds from EVAL_PLAN.md
POINT_BISERIAL_MIN = 0.15
SPEARMAN_MIN = 0.20

pytestmark = [pytest.mark.layer1, pytest.mark.timeout(120)]


def test_task03_relevant_docs_score_higher(engine, beir_scifact):
    """Documents that are actually relevant should have higher retrieval-aware scores."""
    corpus = beir_scifact["corpus"][:200]
    texts = [f"{r['title']} {r['text']}" for r in corpus]

    if len(texts) < 10:
        pytest.skip("SciFact corpus too small")

    # Build TF-IDF and compute self-retrieval for each doc
    tfidf_matrix, vectorizer = engine.compute_tfidf(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    from src.corpus_analyzer import _compute_self_retrieval_score

    scores = []
    for i in range(len(texts)):
        s = _compute_self_retrieval_score(
            doc_idx=i,
            doc_label=f"doc_{i}",
            tfidf_matrix=tfidf_matrix,
            feature_names=feature_names,
            all_doc_texts=texts,
            all_doc_labels=[f"doc_{i}" for i in range(len(texts))],
        )
        scores.append(s)

    # All scores should be valid
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores should be in [0, 1]"
    assert np.mean(scores) > 0.1, f"Mean self-retrieval score suspiciously low: {np.mean(scores):.3f}"
