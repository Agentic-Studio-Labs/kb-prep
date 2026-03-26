"""Chunk-level retrieval benchmarks."""

import math
from typing import Callable, Optional

from .corpus_analyzer import bm25_score
from .models import ChunkBenchmark


def _bm25_search(query: str, chunks: list[str], top_k: int = 10) -> list[int]:
    """Rank chunk indices by BM25+ score."""
    scores = bm25_score(query, chunks)
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]


def recall_at_k(ranked_lists: list[list[int]], gold_sets: list[set[int]], k: int) -> float:
    """Fraction of queries with at least one relevant item in top-k."""
    if not gold_sets:
        return 0.0
    hits = 0
    for ranked, gold in zip(ranked_lists, gold_sets):
        if any(idx in gold for idx in ranked[:k]):
            hits += 1
    return hits / len(gold_sets)


def mean_reciprocal_rank(ranked_lists: list[list[int]], gold_sets: list[set[int]]) -> float:
    """Mean reciprocal rank over query results."""
    if not gold_sets:
        return 0.0
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_sets):
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in gold:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(gold_sets)


def ndcg_at_k(ranked_lists: list[list[int]], gold_sets: list[set[int]], k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    if not gold_sets:
        return 0.0
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_sets):
        dcg = 0.0
        for rank, idx in enumerate(ranked[:k], start=1):
            if idx in gold:
                dcg += 1.0 / math.log2(rank + 1)
        max_relevant = min(len(gold), k)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, max_relevant + 1)) or 1.0
        total += dcg / ideal
    return total / len(gold_sets)


def _to_benchmark(mode: str, ranked_lists: list[list[int]], gold_sets: list[set[int]], k: int) -> ChunkBenchmark:
    return ChunkBenchmark(
        retrieval_mode=mode,
        recall_at_5=recall_at_k(ranked_lists, gold_sets, k=k),
        mrr=mean_reciprocal_rank(ranked_lists, gold_sets),
        ndcg_at_5=ndcg_at_k(ranked_lists, gold_sets, k=k),
        query_count=len(gold_sets),
    )


def benchmark_chunk_retrieval(
    queries: list[str],
    gold_sets: list[set[int]],
    chunks: list[str],
    top_k: int = 5,
    embedding_ranker: Optional[Callable[[str, list[str], int], list[int]]] = None,
    hybrid_ranker: Optional[Callable[[str, list[str], int], list[int]]] = None,
) -> list[ChunkBenchmark]:
    """Benchmark lexical retrieval plus optional embedding/hybrid rankers."""
    if not queries or not chunks:
        return []

    results: list[ChunkBenchmark] = []

    lexical_ranked = [_bm25_search(query, chunks, top_k=max(top_k, 10)) for query in queries]
    results.append(_to_benchmark("lexical", lexical_ranked, gold_sets, k=top_k))

    if embedding_ranker is not None:
        embedding_ranked = [embedding_ranker(query, chunks, max(top_k, 10)) for query in queries]
        results.append(_to_benchmark("embedding", embedding_ranked, gold_sets, k=top_k))

    if hybrid_ranker is not None:
        hybrid_ranked = [hybrid_ranker(query, chunks, max(top_k, 10)) for query in queries]
        results.append(_to_benchmark("hybrid", hybrid_ranked, gold_sets, k=top_k))

    return results
