"""Tests for corpus analysis engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CorpusAnalysis, DocMetrics


def test_doc_metrics_defaults():
    """DocMetrics has expected fields with correct types."""
    dm = DocMetrics(
        entropy=0.5,
        coherence=0.8,
        readability_grade=5.2,
        info_density=[0.3, 0.5, 0.7],
        topic_boundaries=[3, 7],
        self_retrieval_score=0.85,
    )
    assert 0.0 <= dm.entropy <= 1.0
    assert 0.0 <= dm.coherence <= 1.0
    assert dm.readability_grade == 5.2
    assert len(dm.topic_boundaries) == 2
    assert dm.self_retrieval_score == 0.85


def test_corpus_analysis_structure():
    """CorpusAnalysis holds matrix + metrics dict."""
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(np.array([[1, 0], [0, 1]]))
    sim = np.eye(2)
    metrics = {"doc1.md": DocMetrics(0.5, 0.8, 5.2, [], [], 0.85)}
    ca = CorpusAnalysis(
        tfidf_matrix=matrix,
        feature_names=["term_a", "term_b"],
        doc_labels=["doc1.md", "doc2.md"],
        similarity_matrix=sim,
        doc_metrics=metrics,
    )
    assert ca.tfidf_matrix.shape == (2, 2)
    assert ca.similarity_matrix.shape == (2, 2)
    assert "doc1.md" in ca.doc_metrics


from models import DocumentMetadata, Paragraph, ParsedDocument


def _make_doc(filename: str, text: str) -> ParsedDocument:
    """Create a ParsedDocument from raw text."""
    paragraphs = [
        Paragraph(text=p.strip(), level=0, style="Normal", index=i)
        for i, p in enumerate(text.split("\n\n"))
        if p.strip()
    ]
    ext = Path(filename).suffix.lstrip(".") or "md"
    return ParsedDocument(
        metadata=DocumentMetadata(
            file_path=f"/tmp/{filename}",
            file_type=ext,
            file_size_bytes=len(text),
        ),
        paragraphs=paragraphs,
    )


def test_build_tfidf_matrix_shape():
    """TF-IDF matrix has correct shape (n_docs, n_terms)."""
    from corpus_analyzer import build_corpus_analysis

    docs = [
        _make_doc("fractions.md", "Adding fractions requires common denominators."),
        _make_doc("budgeting.md", "A budget tracks income and expenses."),
        _make_doc("insurance.md", "Insurance protects against financial risk."),
    ]
    ca = build_corpus_analysis(docs)
    assert ca.tfidf_matrix.shape[0] == 3
    assert ca.tfidf_matrix.shape[1] > 0
    assert len(ca.feature_names) == ca.tfidf_matrix.shape[1]
    assert len(ca.doc_labels) == 3


def test_similarity_matrix_properties():
    """Similarity matrix is square, symmetric, diagonal is ~1.0."""
    from corpus_analyzer import build_corpus_analysis

    docs = [
        _make_doc("a.md", "Fractions and decimals are important math topics."),
        _make_doc("b.md", "Fractions can be added by finding common denominators."),
        _make_doc("c.md", "Insurance protects against unexpected financial loss."),
    ]
    ca = build_corpus_analysis(docs)
    sim = ca.similarity_matrix

    assert sim.shape == (3, 3)
    np.testing.assert_array_almost_equal(sim, sim.T)
    for i in range(3):
        assert sim[i, i] == pytest.approx(1.0, abs=0.01)
    assert sim[0, 1] > sim[0, 2]
