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
    """Contracts (CUAD) should have lower mean entropy than newsgroup posts.

    CUAD contracts are very long (~50 KB each) while newsgroup posts average ~1 KB.
    Comparing whole documents inflates contract entropy because longer documents
    hit more unique terms.  Instead we chunk both corpora into fixed 200-word
    windows and fit one TfidfVectorizer on the combined pool so IDF weights are
    shared.  Contracts repeat formulaic legal boilerplate within each window,
    yielding a more concentrated (lower-entropy) term distribution than the
    topically diverse newsgroup vocabulary.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _fixed_chunks(texts, n_words=200, max_chunks_per_doc=5, min_words=50):
        """Split each text into fixed-size word windows."""
        chunks = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), n_words):
                chunk = " ".join(words[i : i + n_words])
                if len(chunk.split()) >= min_words:
                    chunks.append(chunk)
                if len(chunks) >= max_chunks_per_doc * (len(chunks) // max_chunks_per_doc + 1):
                    break
            if len(chunks) >= max_chunks_per_doc * 500:
                break
        return chunks

    cuad_chunks = _fixed_chunks([r["text"] for r in cuad_data[:500]])[:1000]
    ng_chunks = _fixed_chunks(newsgroups_tfidf["texts"][:1000])[:1000]

    combined = cuad_chunks + ng_chunks
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(combined)

    n_cuad = len(cuad_chunks)
    cuad_entropies = [engine.shannon_entropy_from_vector(matrix[i]) for i in range(n_cuad)]
    ng_entropies = [engine.shannon_entropy_from_vector(matrix[n_cuad + i]) for i in range(len(ng_chunks))]

    cuad_mean = np.mean(cuad_entropies)
    ng_mean = np.mean(ng_entropies)
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
