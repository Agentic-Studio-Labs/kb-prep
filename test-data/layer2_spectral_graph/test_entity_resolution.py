"""Task 4 — Cosine-Similarity Entity Resolution against Leipzig benchmarks."""

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

# Thresholds from EVAL_PLAN.md (TF-IDF baseline, deliberately lower than deep learning SOTA)
F1_DBLP_ACM = 0.80
F1_DBLP_SCHOLAR = 0.70
F1_ABT_BUY = 0.30
F1_AMAZON_GOOGLE = 0.30

pytestmark = [pytest.mark.layer2, pytest.mark.timeout(120)]


def _load_leipzig_dataset(ds_dir: Path):
    """Load a Leipzig ER dataset: find source files and mapping."""
    csv_files = sorted(ds_dir.rglob("*.csv"))
    if len(csv_files) < 3:
        return None, None, None

    # Heuristic: file with "mapping" or "match" in name is the ground truth
    mapping_file = None
    source_files = []
    for f in csv_files:
        name_lower = f.stem.lower()
        if "mapping" in name_lower or "match" in name_lower or "perfect" in name_lower:
            mapping_file = f
        else:
            source_files.append(f)

    if not mapping_file or len(source_files) < 2:
        return None, None, None

    # Load sources
    def load_source(path):
        records = {}
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("id", row.get("ID", ""))
                    # Concatenate all text fields
                    text = " ".join(str(v) for k, v in row.items() if k.lower() != "id")
                    records[rid] = text
        except Exception:
            pass
        return records

    source_a = load_source(source_files[0])
    source_b = load_source(source_files[1])

    # Load mapping
    matches = set()
    try:
        with open(mapping_file, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    matches.add((row[0].strip(), row[1].strip()))
    except Exception:
        pass

    return source_a, source_b, matches


def _evaluate_er(engine, source_a, source_b, true_matches):
    """Sweep thresholds and find best F1."""
    # Build pairs (sample if too many)
    pairs = []
    labels = []
    a_ids = list(source_a.keys())[:200]
    b_ids = list(source_b.keys())[:200]

    for a_id in a_ids:
        for b_id in b_ids[:50]:  # Limit cross-product
            pairs.append((source_a[a_id], source_b[b_id]))
            labels.append(1 if (a_id, b_id) in true_matches else 0)

    if not pairs or sum(labels) == 0:
        return 0.0, 0.0

    similarities = engine.entity_cosine_similarity(pairs)

    best_f1 = 0.0
    best_thresh = 0.0
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = [1 if s >= thresh else 0 for s in similarities]
        if sum(preds) > 0:
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

    return best_f1, best_thresh


@pytest.mark.parametrize(
    "ds_name,min_f1",
    [
        ("DBLP-ACM", F1_DBLP_ACM),
        ("DBLP-Scholar", F1_DBLP_SCHOLAR),
        ("Abt-Buy", F1_ABT_BUY),
        ("Amazon-Google", F1_AMAZON_GOOGLE),
    ],
)
def test_task04_entity_resolution(engine, leipzig_er_data, ds_name, min_f1):
    """Entity resolution F1 on Leipzig benchmark."""
    if ds_name not in leipzig_er_data:
        pytest.skip(f"{ds_name} not downloaded")

    source_a, source_b, true_matches = _load_leipzig_dataset(leipzig_er_data[ds_name])
    if source_a is None:
        pytest.skip(f"Could not parse {ds_name} dataset")

    best_f1, best_thresh = _evaluate_er(engine, source_a, source_b, true_matches)
    assert best_f1 >= min_f1, f"{ds_name}: F1={best_f1:.3f} at threshold={best_thresh:.2f}, expected >= {min_f1}"
