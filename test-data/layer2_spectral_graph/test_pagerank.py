"""Task 6 — PageRank validation."""

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

# Thresholds from EVAL_PLAN.md
SPEARMAN_DEGREE_MIN = 0.70
SPEARMAN_DOCFREQ_MIN = 0.50

pytestmark = [pytest.mark.layer2, pytest.mark.timeout(120)]


def test_task06_pagerank_fb15k(engine, fb15k237_data):
    """PageRank should correlate with in-degree on FB15k-237."""
    # Build graph from triples (sample for speed)
    triples = fb15k237_data[:50000]
    G = nx.DiGraph()
    for head, rel, tail in triples:
        G.add_edge(head, tail, relation=rel)

    pr = engine.pagerank(G)

    # Compute in-degree for nodes with PageRank
    in_degrees = {node: G.in_degree(node) for node in pr}

    nodes = list(pr.keys())
    pr_values = [pr[n] for n in nodes]
    degree_values = [in_degrees[n] for n in nodes]

    rho, p = spearmanr(pr_values, degree_values)
    assert rho > SPEARMAN_DEGREE_MIN, f"PageRank-degree correlation ρ={rho:.3f}, expected > {SPEARMAN_DEGREE_MIN}"


def test_task06_pagerank_convergence(engine, fb15k237_data):
    """PageRank values should sum to ~1.0 and contain no NaN."""
    triples = fb15k237_data[:10000]
    G = nx.DiGraph()
    for head, rel, tail in triples:
        G.add_edge(head, tail)

    pr = engine.pagerank(G)

    assert len(pr) > 0, "PageRank returned empty results"
    assert all(not np.isnan(v) for v in pr.values()), "PageRank contains NaN"
    assert all(v >= 0 for v in pr.values()), "PageRank contains negative values"
    assert abs(sum(pr.values()) - 1.0) < 1e-4, f"PageRank sum = {sum(pr.values()):.6f}, expected ~1.0"


def test_task06_pagerank_no_negative(engine, fb15k237_data):
    """No negative PageRank values."""
    triples = fb15k237_data[:5000]
    G = nx.DiGraph()
    for head, rel, tail in triples:
        G.add_edge(head, tail)
    pr = engine.pagerank(G)
    assert all(v >= 0 for v in pr.values())
