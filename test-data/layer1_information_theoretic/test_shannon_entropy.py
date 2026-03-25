"""Task 1 — Shannon Entropy validation against real corpora."""

import numpy as np
import pytest

# Thresholds from EVAL_PLAN.md
CUAD_LT_NEWSGROUPS = True  # Contracts < newsgroup discussions
CV_MIN, CV_MAX = 0.05, 2.0  # Coefficient of variation bounds

pytestmark = [pytest.mark.layer1, pytest.mark.timeout(120)]


def test_task01_entropy_no_nan(engine, newsgroups_tfidf):
    """No NaN or Inf values in entropy output."""
    matrix = newsgroups_tfidf["matrix"]
    for i in range(min(500, matrix.shape[0])):
        e = engine.shannon_entropy_from_vector(matrix[i])
        assert not np.isnan(e), f"NaN entropy at doc {i}"
        assert not np.isinf(e), f"Inf entropy at doc {i}"
        assert 0.0 <= e <= 1.0, f"Entropy {e} out of [0,1] at doc {i}"


def test_task01_entropy_ordering(engine, newsgroups_tfidf, cuad_data):
    """Contracts (CUAD) should have lower mean entropy than newsgroup posts."""
    # Newsgroups entropy from pre-computed TF-IDF
    ng_matrix = newsgroups_tfidf["matrix"]
    ng_sample = [engine.shannon_entropy_from_vector(ng_matrix[i]) for i in range(min(1000, ng_matrix.shape[0]))]

    # CUAD entropy — compute from raw text
    cuad_texts = [r["text"] for r in cuad_data[:500]]
    cuad_entropies = [engine.shannon_entropy(t) for t in cuad_texts]

    ng_mean = np.mean(ng_sample)
    cuad_mean = np.mean(cuad_entropies)
    assert cuad_mean < ng_mean, (
        f"Contracts (mean={cuad_mean:.4f}) should have lower entropy than newsgroups (mean={ng_mean:.4f})"
    )


def test_task01_entropy_distribution(engine, newsgroups_tfidf):
    """Entropy distribution should not be degenerate (CV between 0.05 and 2.0)."""
    matrix = newsgroups_tfidf["matrix"]
    entropies = [engine.shannon_entropy_from_vector(matrix[i]) for i in range(min(2000, matrix.shape[0]))]
    std = np.std(entropies)
    mean = np.mean(entropies)
    if mean > 0:
        cv = std / mean
        assert CV_MIN < cv < CV_MAX, f"Coefficient of variation {cv:.3f} outside [{CV_MIN}, {CV_MAX}]"


def test_task01_entropy_empty_doc(engine):
    """Empty document should return entropy = 0."""
    assert engine.shannon_entropy("") == 0.0


def test_task01_entropy_single_word(engine):
    """Single word should return entropy = 0."""
    assert engine.shannon_entropy("word") == 0.0
