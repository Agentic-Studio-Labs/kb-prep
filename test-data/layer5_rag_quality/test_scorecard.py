"""Task 16 — Preprocessing Impact Scorecard."""

from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).parent.parent / "reports"

pytestmark = [pytest.mark.layer5, pytest.mark.timeout(120)]


def test_task16_scorecard_generation(engine, squad_data):
    """Generate and verify the preprocessing impact scorecard."""
    if len(squad_data) < 20:
        pytest.skip("Need SQuAD data")

    from layer5_rag_quality.naive_baseline import NaiveBM25

    sample = squad_data[:30]
    texts = [r["text"] for r in sample]
    naive = NaiveBM25(texts)

    engine_hit5 = 0
    naive_hit5 = 0
    total = 0

    for r in sample:
        if not r.get("questions"):
            continue
        query = r["questions"][0]["question"]
        doc_idx = texts.index(r["text"])

        if doc_idx in engine.bm25_search(query, texts, top_k=5):
            engine_hit5 += 1
        if doc_idx in naive.search(query, top_k=5):
            naive_hit5 += 1
        total += 1

    if total == 0:
        pytest.skip("No questions available")

    e_rate = engine_hit5 / total
    n_rate = naive_hit5 / total

    # Write scorecard
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scorecard = REPORTS_DIR / "scorecard.txt"
    with open(scorecard, "w") as f:
        f.write("PREPROCESSING IMPACT SCORECARD\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Metric':<25} {'Engine':>8} {'Naive':>8} {'Delta':>10}\n")
        f.write("-" * 50 + "\n")
        delta = ((e_rate - n_rate) / n_rate * 100) if n_rate > 0 else 0
        f.write(f"{'Hit@5 (SQuAD)':<25} {e_rate:>8.3f} {n_rate:>8.3f} {delta:>+9.1f}%\n")
        f.write("=" * 50 + "\n")

    assert scorecard.exists(), "Scorecard should be written"
    print(f"\nScorecard written to {scorecard}")
    print(scorecard.read_text())
