"""Task 5 — Deterministic Spectral Clustering validation."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

# Thresholds from EVAL_PLAN.md
NMI_MIN = 0.35
ARI_MIN = 0.15

pytestmark = [pytest.mark.layer2, pytest.mark.timeout(120)]


def test_task05_spectral_deterministic(engine, newsgroups_tfidf):
    """Same input must produce same clusters every time."""
    sim = cosine_similarity(newsgroups_tfidf["matrix"][:500])
    results = [engine.spectral_cluster(sim, k=10) for _ in range(5)]
    for r in results[1:]:
        assert np.array_equal(r, results[0]), "Spectral clustering must be deterministic"


def test_task05_spectral_nmi(engine, newsgroups_tfidf):
    """NMI on 20 Newsgroups should meet threshold."""
    # Use a subsample for speed
    n = min(2000, newsgroups_tfidf["matrix"].shape[0])
    matrix = newsgroups_tfidf["matrix"][:n]
    labels = newsgroups_tfidf["labels"][:n]
    sim = cosine_similarity(matrix)

    k = len(set(labels))
    clusters = engine.spectral_cluster(sim, k=k)
    nmi = normalized_mutual_info_score(labels, clusters)
    assert nmi >= NMI_MIN, f"NMI={nmi:.3f}, expected >= {NMI_MIN}"


def test_task05_spectral_ari(engine, newsgroups_tfidf):
    """ARI on 20 Newsgroups should meet threshold."""
    n = min(2000, newsgroups_tfidf["matrix"].shape[0])
    matrix = newsgroups_tfidf["matrix"][:n]
    labels = newsgroups_tfidf["labels"][:n]
    sim = cosine_similarity(matrix)

    k = len(set(labels))
    clusters = engine.spectral_cluster(sim, k=k)
    ari = adjusted_rand_score(labels, clusters)
    assert ari >= ARI_MIN, f"ARI={ari:.3f}, expected >= {ARI_MIN}"
