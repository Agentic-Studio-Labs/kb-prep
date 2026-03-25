"""Task 11 — Silhouette-Based Folder Validation."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

pytestmark = [pytest.mark.layer4, pytest.mark.timeout(120)]


def test_task11_true_better_than_random(engine, newsgroups_tfidf):
    """True labels should produce higher silhouette than random labels."""
    n = min(1000, newsgroups_tfidf["matrix"].shape[0])
    sim = cosine_similarity(newsgroups_tfidf["matrix"][:n])
    dist = 1 - np.clip(sim, 0, 1)
    true_labels = newsgroups_tfidf["labels"][:n]

    np.random.seed(42)
    random_labels = np.random.randint(0, 20, size=n)

    true_sil = engine.silhouette_score(dist, true_labels)
    random_sil = engine.silhouette_score(dist, random_labels)

    assert true_sil > random_sil, f"True grouping ({true_sil:.3f}) should outscore random ({random_sil:.3f})"


def test_task11_silhouette_valid_range(engine, newsgroups_tfidf):
    """Silhouette values should be in [-1, 1]."""
    n = min(500, newsgroups_tfidf["matrix"].shape[0])
    sim = cosine_similarity(newsgroups_tfidf["matrix"][:n])
    dist = 1 - np.clip(sim, 0, 1)
    labels = newsgroups_tfidf["labels"][:n]

    sil = engine.silhouette_score(dist, labels)
    assert -1.0 <= sil <= 1.0, f"Silhouette {sil} outside [-1, 1]"
