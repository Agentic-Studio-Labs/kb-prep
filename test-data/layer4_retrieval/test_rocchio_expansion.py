"""Task 10 — Rocchio Query Expansion."""

import pytest

pytestmark = [pytest.mark.layer4, pytest.mark.timeout(120)]


def test_task10_rocchio_expands_query(engine):
    """Expanded query should contain more terms than original."""
    corpus = [
        "Machine learning uses neural networks for classification.",
        "Deep learning models require large training datasets.",
        "Natural language processing handles text understanding.",
    ]
    expanded = engine.rocchio_expand("classification models", corpus, top_k=2, n_expand=3)
    assert len(expanded.split()) > len("classification models".split()), (
        f"Expanded query should be longer: '{expanded}'"
    )


def test_task10_rocchio_not_degenerate(engine):
    """Expanded query should not be empty or excessively long."""
    corpus = [f"Document {i} about topic {i % 5} with various content." for i in range(20)]
    original = "topic content"
    expanded = engine.rocchio_expand(original, corpus)
    assert len(expanded.strip()) > 0, "Expanded query should not be empty"
    assert len(expanded.split()) <= 5 * len(original.split()) + 10, (
        f"Expanded query too long ({len(expanded.split())} words)"
    )


def test_task10_rocchio_adds_relevant_terms(engine):
    """Expansion should add domain-relevant terms from the corpus."""
    corpus = [
        "Adding fractions with common denominators and simplifying results",
        "Multiplying fractions by finding the product of numerators and denominators",
        "Dividing fractions using the reciprocal method",
    ]
    expanded = engine.rocchio_expand("math operations", corpus, top_k=2, n_expand=3)
    expanded_terms = set(expanded.lower().split())
    # Should pick up fraction-related terms
    fraction_terms = {"fractions", "denominators", "numerators", "reciprocal", "simplifying"}
    overlap = expanded_terms & fraction_terms
    assert len(overlap) > 0, f"Expansion should add domain terms. Got: {expanded_terms}"
