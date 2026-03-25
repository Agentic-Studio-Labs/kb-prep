"""Task 17 — Edge Cases."""

import numpy as np
import pytest

pytestmark = [pytest.mark.cross_layer, pytest.mark.timeout(120)]


@pytest.mark.parametrize(
    "input_doc,expected_entropy",
    [
        ("", 0.0),
        ("word", 0.0),
        ("the the the the", 0.0),  # all stopwords
    ],
)
def test_task17_entropy_edge_cases(engine, input_doc, expected_entropy):
    """Edge case inputs should produce valid entropy."""
    result = engine.shannon_entropy(input_doc)
    assert result == pytest.approx(expected_entropy, abs=0.01)
    assert not np.isnan(result)


def test_task17_empty_overlap(engine):
    """Empty sentence list should return empty overlap."""
    result = engine.select_overlap([], budget=100)
    assert result == []


def test_task17_single_word_texttiling(engine):
    """Single paragraph should produce no boundaries."""
    result = engine.texttile(["Just one paragraph."])
    assert result == []


def test_task17_bm25_empty_query(engine):
    """Empty query should not crash."""
    docs = ["Some document text."]
    scores = engine.bm25_scores("", docs)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_task17_unicode(engine):
    """Unicode content should not crash."""
    engine.shannon_entropy("🎉🎊🎈 emoji content")
    engine.shannon_entropy("中文内容 Chinese text mixed with English")
    engine.shannon_entropy("مرحبا RTL text")
    # No assertion needed — just verifying no crash
