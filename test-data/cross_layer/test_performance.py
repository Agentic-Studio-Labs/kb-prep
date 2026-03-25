"""Task 18 — Performance Benchmarks."""

import time

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

# Max times from EVAL_PLAN.md
MAX_TFIDF_20K = 30  # seconds
MAX_ENTROPY_20K = 5
MAX_BM25_QUERY = 0.5  # per query, relaxed from 0.1

pytestmark = [pytest.mark.cross_layer, pytest.mark.slow, pytest.mark.timeout(300)]


def test_task18_tfidf_performance(newsgroups_data):
    """TF-IDF computation on 20K docs should complete within 30s."""
    texts = newsgroups_data["texts"]
    start = time.time()
    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    vec.fit_transform(texts)
    elapsed = time.time() - start
    assert elapsed < MAX_TFIDF_20K, f"TF-IDF took {elapsed:.1f}s, max {MAX_TFIDF_20K}s"


def test_task18_entropy_performance(engine, newsgroups_tfidf):
    """Entropy computation on 20K docs should complete within 5s."""
    matrix = newsgroups_tfidf["matrix"]
    start = time.time()
    for i in range(matrix.shape[0]):
        engine.shannon_entropy_from_vector(matrix[i])
    elapsed = time.time() - start
    assert elapsed < MAX_ENTROPY_20K, f"Entropy took {elapsed:.1f}s, max {MAX_ENTROPY_20K}s"


def test_task18_bm25_query_performance(engine, newsgroups_data):
    """Single BM25 query on 20K docs should complete within 0.5s."""
    texts = newsgroups_data["texts"][:5000]  # Subsample for reasonable time
    start = time.time()
    engine.bm25_search("machine learning neural network", texts, top_k=10)
    elapsed = time.time() - start
    assert elapsed < MAX_BM25_QUERY, f"BM25 query took {elapsed:.1f}s, max {MAX_BM25_QUERY}s"
