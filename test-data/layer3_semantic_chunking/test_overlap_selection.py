"""Task 8 — Information-Dense Overlap Selection."""

import numpy as np
import pytest

pytestmark = [pytest.mark.layer3, pytest.mark.timeout(120)]


def test_task08_overlap_tfidf_density(engine):
    """Overlap tokens should have higher mean TF-IDF weight than average chunk tokens."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = [
        "SMART goals have five specific components for student success.",
        "Teachers should review homework assignments regularly.",
        "Review the answers.",
        "Compound interest calculations require understanding of exponential growth rates.",
        "The bell rang for lunch.",
        "Fibonacci sequences appear throughout natural biological systems.",
    ]

    selected = engine.select_overlap(sentences, budget=40)
    not_selected = [s for s in sentences if s not in selected]

    # TF-IDF density of selected should exceed not-selected
    if selected and not_selected:
        vec = TfidfVectorizer(stop_words="english")
        all_vecs = vec.fit_transform(sentences)

        selected_indices = [sentences.index(s) for s in selected]
        other_indices = [i for i in range(len(sentences)) if i not in selected_indices]

        selected_density = np.mean([all_vecs[i].power(2).sum() for i in selected_indices])
        other_density = np.mean([all_vecs[i].power(2).sum() for i in other_indices])

        assert selected_density >= other_density, (
            f"Selected density ({selected_density:.4f}) should exceed other ({other_density:.4f})"
        )


def test_task08_overlap_respects_budget(engine):
    """Selected overlap should stay within the word budget."""
    sentences = [f"This is sentence number {i} with several words in it." for i in range(20)]
    budget = 30
    selected = engine.select_overlap(sentences, budget=budget)
    total_words = sum(len(s.split()) for s in selected)
    # Allow some slack (first sentence always included even if over budget)
    assert total_words <= budget + 15, f"Overlap has {total_words} words, budget was {budget}"


def test_task08_overlap_preserves_order(engine):
    """Selected sentences should maintain original order."""
    sentences = [
        "Alpha sentence comes first.",
        "Beta sentence is second.",
        "Gamma sentence is third.",
        "Delta sentence is fourth.",
        "Epsilon sentence is fifth.",
    ]
    selected = engine.select_overlap(sentences, budget=50)
    # Check that selected items appear in the same relative order
    indices = [sentences.index(s) for s in selected]
    assert indices == sorted(indices), "Selected sentences should maintain original order"
