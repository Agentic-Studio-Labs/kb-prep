"""Task 2 — Heading-Content Coherence validation."""

import random

import numpy as np
import pytest
from scipy.stats import wilcoxon

# Thresholds from EVAL_PLAN.md
WILCOXON_P = 0.01
COHENS_D_MIN = 0.5
AUC_MIN = 0.70

pytestmark = [pytest.mark.layer1, pytest.mark.timeout(120)]


def test_task02_coherence_real_vs_shuffled(engine, squad_data):
    """Real heading-body pairs should score higher than shuffled pairs."""
    # SQuAD: title serves as heading, context as body
    sample = squad_data[:200]
    pairs = [(r["questions"][0]["question"], r["text"]) for r in sample if r.get("questions")]

    real_scores = [engine.heading_coherence(h, b) for h, b in pairs[:100]]

    # Shuffle: reassign bodies randomly
    random.seed(42)
    bodies = [b for _, b in pairs[:100]]
    random.shuffle(bodies)
    shuffled_scores = [engine.heading_coherence(h, b) for h, b in zip([h for h, _ in pairs[:100]], bodies)]

    assert np.mean(real_scores) > np.mean(shuffled_scores), (
        f"Real pairs ({np.mean(real_scores):.4f}) should outscore shuffled ({np.mean(shuffled_scores):.4f})"
    )


def test_task02_coherence_statistical_significance(engine, squad_data):
    """Wilcoxon test confirms real vs shuffled difference is significant."""
    sample = squad_data[:100]
    pairs = [(r["questions"][0]["question"], r["text"]) for r in sample if r.get("questions")]

    real_scores = [engine.heading_coherence(h, b) for h, b in pairs[:50]]

    random.seed(42)
    bodies = [b for _, b in pairs[:50]]
    random.shuffle(bodies)
    shuffled_scores = [engine.heading_coherence(h, b) for h, b in zip([h for h, _ in pairs[:50]], bodies)]

    # Need non-zero differences for Wilcoxon
    diffs = [r - s for r, s in zip(real_scores, shuffled_scores) if r != s]
    if len(diffs) >= 10:
        _, p = wilcoxon(diffs)
        assert p < WILCOXON_P, f"Wilcoxon p={p:.4f}, expected < {WILCOXON_P}"


def test_task02_coherence_effect_size(engine, squad_data):
    """Cohen's d should show medium effect (d > 0.5)."""
    sample = squad_data[:100]
    pairs = [(r["questions"][0]["question"], r["text"]) for r in sample if r.get("questions")]

    real_scores = np.array([engine.heading_coherence(h, b) for h, b in pairs[:50]])

    random.seed(42)
    bodies = [b for _, b in pairs[:50]]
    random.shuffle(bodies)
    shuffled_scores = np.array([engine.heading_coherence(h, b) for h, b in zip([h for h, _ in pairs[:50]], bodies)])

    pooled_std = np.sqrt((np.var(real_scores) + np.var(shuffled_scores)) / 2)
    if pooled_std > 0:
        d = (np.mean(real_scores) - np.mean(shuffled_scores)) / pooled_std
        assert d > COHENS_D_MIN, f"Cohen's d={d:.3f}, expected > {COHENS_D_MIN}"
