from src.benchmark import benchmark_chunk_retrieval, mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_recall_at_k_counts_hits():
    ranked_lists = [[3, 1, 2], [4, 5, 6]]
    gold = [{1}, {9}]

    assert recall_at_k(ranked_lists, gold, k=2) == 0.5


def test_mean_reciprocal_rank_scores_first_hit():
    ranked_lists = [[9, 3, 1], [7, 8, 2]]
    gold = [{1}, {2}]

    assert round(mean_reciprocal_rank(ranked_lists, gold), 3) == 0.333


def test_ndcg_handles_multiple_relevant_chunks():
    ranked_lists = [[3, 1, 2], [9, 8, 7]]
    gold = [{1, 2, 3}, {7, 8}]

    score = ndcg_at_k(ranked_lists, gold, k=3)
    assert 0.0 <= score <= 1.0


def test_benchmark_chunk_retrieval_returns_lexical_result():
    queries = ["saving goal", "credit card"]
    chunks = [
        "Saving goals help students plan spending and income.",
        "Credit cards can accrue interest if balances are not paid.",
        "Plants grow in sunlight and soil.",
    ]
    gold_sets = [{0}, {1}]

    results = benchmark_chunk_retrieval(queries, gold_sets, chunks, top_k=5)

    assert len(results) == 1
    assert results[0].retrieval_mode == "lexical"
    assert results[0].query_count == 2
