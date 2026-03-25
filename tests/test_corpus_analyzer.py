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


# ---------------------------------------------------------------------------
# Task 4: Readability Scoring (Flesch-Kincaid)
# ---------------------------------------------------------------------------


def test_readability_grade_range():
    from corpus_analyzer import _compute_readability

    simple = "The cat sat on the mat. The dog ran in the park. We like to play."
    grade = _compute_readability(simple)
    assert 0.0 <= grade <= 6.0, f"Simple text should be low grade, got {grade}"
    complex_text = (
        "The juxtaposition of macroeconomic indicators necessitates a comprehensive "
        "analysis of the underlying fiscal determinants and their ramifications on "
        "aggregate demand within the broader socioeconomic framework."
    )
    complex_grade = _compute_readability(complex_text)
    assert complex_grade > grade


def test_syllable_counting():
    from corpus_analyzer import _count_syllables

    assert _count_syllables("cat") == 1
    assert _count_syllables("money") == 2
    assert _count_syllables("insurance") == 3
    assert _count_syllables("the") >= 1


# ---------------------------------------------------------------------------
# Task 5: TF-IDF Coherence (Heading vs Content)
# ---------------------------------------------------------------------------


def test_coherence_high_for_matching_heading():
    from corpus_analyzer import _compute_coherence

    heading_text = "Adding Fractions with Unlike Denominators"
    content_below = (
        "To add fractions with unlike denominators, first find a common denominator. "
        "Convert each fraction, then add the numerators."
    )
    coherence = _compute_coherence([(heading_text, content_below)])
    assert coherence > 0.3


def test_coherence_low_for_mismatched_heading():
    from corpus_analyzer import _compute_coherence

    heading_text = "Content"
    content_below = "Insurance protects families against financial loss from unexpected events."
    coherence = _compute_coherence([(heading_text, content_below)])
    assert coherence < 0.2


# ---------------------------------------------------------------------------
# Task 6: TextTiling Topic Boundaries
# ---------------------------------------------------------------------------


def test_texttiling_finds_topic_shift():
    from corpus_analyzer import _compute_topic_boundaries

    paragraphs = [
        "Adding fractions requires finding a common denominator.",
        "To convert fractions, multiply numerator and denominator.",
        "Practice adding fractions with unlike denominators.",
        "Insurance protects against unexpected financial loss.",
        "Life insurance provides coverage for beneficiaries.",
        "Health insurance covers medical expenses and prescriptions.",
    ]
    boundaries = _compute_topic_boundaries(paragraphs, block_size=1)
    assert len(boundaries) >= 1
    assert any(2 <= b <= 4 for b in boundaries)


def test_texttiling_no_boundary_in_coherent_text():
    from corpus_analyzer import _compute_topic_boundaries

    paragraphs = [
        "Fractions represent parts of a whole number.",
        "Adding fractions requires common denominators.",
        "To find the common denominator, use the least common multiple.",
        "Once denominators match, add the numerators directly.",
    ]
    boundaries = _compute_topic_boundaries(paragraphs, block_size=1)
    assert len(boundaries) <= 1


# ---------------------------------------------------------------------------
# Task 7: Retrieval-Aware Scoring
# ---------------------------------------------------------------------------


def test_retrieval_aware_score_distinct_docs():
    from corpus_analyzer import build_corpus_analysis

    docs = [
        _make_doc(
            "fractions.md",
            (
                "Adding fractions requires finding a common denominator. "
                "The least common multiple helps convert fractions. "
                "Numerators are added once denominators match. "
                "Simplify the resulting fraction to lowest terms."
            ),
        ),
        _make_doc(
            "insurance.md",
            (
                "Insurance protects families against unexpected financial loss. "
                "Health insurance covers medical expenses. "
                "Life insurance provides beneficiary coverage. "
                "Premiums are paid monthly or annually."
            ),
        ),
        _make_doc(
            "budgeting.md",
            (
                "A budget tracks income and expenses each month. "
                "Savings goals help families plan for the future. "
                "Emergency funds cover unexpected costs. "
                "Tracking spending reveals patterns."
            ),
        ),
    ]
    ca = build_corpus_analysis(docs)
    for label in ca.doc_labels:
        score = ca.doc_metrics[label].self_retrieval_score
        assert score > 0.3, f"{label} self-retrieval too low: {score}"


# ---------------------------------------------------------------------------
# Task 12: Information-Dense Overlap + Info Density
# ---------------------------------------------------------------------------


def test_select_overlap_sentences_prefers_informative():
    from corpus_analyzer import select_overlap_sentences

    sentences = [
        "SMART goals have five specific components for success.",
        "Review answers.",
        "The teacher should provide additional materials.",
        "Compound interest calculations require understanding of exponential growth.",
    ]
    selected = select_overlap_sentences(sentences, budget=30)
    selected_text = " ".join(selected)
    assert "Review answers" not in selected_text or "SMART" in selected_text


def test_info_density_computed():
    from corpus_analyzer import build_corpus_analysis

    doc = _make_doc(
        "test.md",
        "First section about fractions and denominators.\n\nSecond section about insurance and risk management.",
    )
    ca = build_corpus_analysis([doc])
    metrics = ca.doc_metrics.get("test.md")
    assert metrics is not None
    assert isinstance(metrics.info_density, list)


# ---------------------------------------------------------------------------
# Task 14: BM25+ with Rocchio Query Expansion
# ---------------------------------------------------------------------------


def test_rocchio_expands_query():
    from corpus_analyzer import rocchio_expand_query

    corpus = [
        "Adding fractions with common denominators and simplifying results",
        "A budget tracks household income expenses and savings goals",
        "Insurance protects against financial risk and unexpected loss",
    ]
    expanded = rocchio_expand_query("money management", corpus, top_k=1, n_expand=3)
    assert len(expanded.split()) > len("money management".split())
