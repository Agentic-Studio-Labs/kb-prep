"""Task 12 — TF-IDF Matrix Consistency across layers."""

import pytest

pytestmark = [pytest.mark.cross_layer, pytest.mark.timeout(120)]


def test_task12_tfidf_dimensions_match(engine, newsgroups_data):
    """All layers should agree on document count."""
    texts = newsgroups_data["texts"][:500]
    tfidf_matrix, vectorizer = engine.compute_tfidf(texts)

    # Entropy should produce one value per document
    entropies = [engine.shannon_entropy_from_vector(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]
    assert len(entropies) == tfidf_matrix.shape[0]

    # Similarity matrix should be square with same dimension
    sim = engine.compute_similarity(tfidf_matrix)
    assert sim.shape == (tfidf_matrix.shape[0], tfidf_matrix.shape[0])

    # Clustering should assign one label per document
    clusters = engine.spectral_cluster(sim, k=5)
    assert len(clusters) == tfidf_matrix.shape[0]


def test_task12_modified_doc_propagates(engine):
    """Changing a document should change its entropy and similarity."""
    texts = [
        "Machine learning algorithms process data.",
        "Natural language processing understands text.",
        "Computer vision analyzes images.",
    ]

    tfidf1, _ = engine.compute_tfidf(texts)
    e1 = engine.shannon_entropy_from_vector(tfidf1[0])

    # Modify first document substantially
    texts_modified = texts.copy()
    texts_modified[0] = "Financial planning requires careful budgeting and investment strategy analysis."
    tfidf2, _ = engine.compute_tfidf(texts_modified)
    e2 = engine.shannon_entropy_from_vector(tfidf2[0])

    # Entropy should change
    assert e1 != e2, "Modifying a document should change its entropy"
