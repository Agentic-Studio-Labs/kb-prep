"""Task 7 — TextTiling Topic Boundary Detection."""

import numpy as np
import pytest

# Thresholds from EVAL_PLAN.md
PK_MAX = 0.44  # At least as good as original TextTiling

pytestmark = [pytest.mark.layer3, pytest.mark.timeout(120)]


def _pk_metric(true_boundaries, predicted_boundaries, n_sentences, window_size=None):
    """Compute Pk metric for segmentation evaluation."""
    if window_size is None:
        # Standard: half the mean segment length
        if len(true_boundaries) > 0:
            mean_seg = n_sentences / (len(true_boundaries) + 1)
            window_size = max(1, int(mean_seg / 2))
        else:
            window_size = max(1, n_sentences // 4)

    # Build segment label arrays
    true_labels = np.zeros(n_sentences, dtype=int)
    seg = 0
    for b in sorted(true_boundaries):
        if b < n_sentences:
            true_labels[b:] = seg + 1
            seg += 1

    pred_labels = np.zeros(n_sentences, dtype=int)
    seg = 0
    for b in sorted(predicted_boundaries):
        if b < n_sentences:
            pred_labels[b:] = seg + 1
            seg += 1

    # Pk: probability that two sentences separated by window_size
    # are incorrectly classified as same/different segment
    errors = 0
    total = 0
    for i in range(n_sentences - window_size):
        j = i + window_size
        true_same = true_labels[i] == true_labels[j]
        pred_same = pred_labels[i] == pred_labels[j]
        if true_same != pred_same:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


def test_task07_texttiling_synthetic():
    """TextTiling should find boundaries in a document with clear topic shifts."""
    from src.corpus_analyzer import _compute_topic_boundaries

    # Two distinct topics joined together
    topic_a = [
        "Machine learning algorithms process large datasets efficiently.",
        "Neural networks learn hierarchical representations of data.",
        "Gradient descent optimizes the loss function iteratively.",
        "Backpropagation computes gradients through the network layers.",
    ]
    topic_b = [
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "Napoleon Bonaparte rose to power during the revolutionary period.",
        "The Declaration of the Rights of Man established fundamental freedoms.",
        "The Reign of Terror saw thousands executed by guillotine.",
    ]

    paragraphs = topic_a + topic_b
    boundaries = _compute_topic_boundaries(paragraphs, block_size=1)
    # Should find a boundary near index 4 (between topic_a and topic_b)
    assert len(boundaries) >= 1, f"Should find at least 1 boundary, got {boundaries}"
    assert any(3 <= b <= 5 for b in boundaries), f"Boundary should be near index 4, got {boundaries}"


def test_task07_texttiling_no_empty_chunks(engine):
    """TextTiling should never produce empty chunks."""
    paragraphs = [
        "First paragraph about topic one.",
        "Second paragraph continuing topic one.",
        "Third paragraph on a completely different topic.",
        "Fourth paragraph about the new topic.",
    ]
    boundaries = engine.texttile(paragraphs, block_size=1)
    # Verify no adjacent boundaries (which would create empty chunks)
    for i in range(len(boundaries) - 1):
        assert boundaries[i + 1] > boundaries[i] + 1, "Adjacent boundaries create empty chunks"


def test_task07_texttiling_preserves_content(engine):
    """All text must be accounted for — nothing dropped."""
    paragraphs = [f"Paragraph {i} with some content about topic {i % 3}." for i in range(20)]
    boundaries = engine.texttile(paragraphs, block_size=2)
    # All boundaries should be valid indices
    for b in boundaries:
        assert 0 < b < len(paragraphs), f"Boundary {b} out of range [1, {len(paragraphs) - 1}]"


def test_task07_texttiling_choi(engine, choi_data):
    """Pk on Choi dataset should be ≤ 0.44 (original TextTiling baseline)."""
    if not choi_data:
        pytest.skip("Choi dataset not available")

    pk_scores = []
    for record in choi_data[:100]:  # Sample for speed
        segments = record["segments"]
        true_boundaries = record["boundaries"]

        # Flatten segments into paragraphs (split on newlines)
        paragraphs = []
        for seg in segments:
            paras = [p.strip() for p in seg.split("\n") if p.strip()]
            paragraphs.extend(paras)

        if len(paragraphs) < 4:
            continue

        predicted = engine.texttile(paragraphs, block_size=2)
        pk = _pk_metric(true_boundaries, predicted, len(paragraphs))
        pk_scores.append(pk)

    if pk_scores:
        mean_pk = np.mean(pk_scores)
        assert mean_pk <= PK_MAX, f"Mean Pk={mean_pk:.3f}, expected ≤ {PK_MAX}"
