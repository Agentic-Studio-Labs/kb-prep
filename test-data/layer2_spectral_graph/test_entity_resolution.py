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


def _normalize_id_key(fieldnames):
    """Return the ID field name, handling BOM and quoted variants."""
    for name in fieldnames or []:
        clean = name.strip().lstrip("\ufeff").strip('"').lower()
        if clean == "id":
            return name
    return None


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

    # Load sources — handle BOM-prefixed ID column names
    def load_source(path):
        records = {}
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                id_key = _normalize_id_key(reader.fieldnames)
                for row in reader:
                    rid = row.get(id_key, "") if id_key else ""
                    # Concatenate all text fields
                    text = " ".join(str(v) for k, v in row.items() if k != id_key)
                    records[str(rid).strip()] = text
        except Exception:
            pass
        return records

    # Load mapping using DictReader to skip the header row and read column names
    matches = set()
    col_a = col_b = None
    try:
        with open(mapping_file, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if len(cols) >= 2:
                col_a, col_b = cols[0], cols[1]
            for row in reader:
                if col_a and col_b:
                    matches.add((row[col_a].strip(), row[col_b].strip()))
    except Exception:
        pass

    # Build id→source mapping to match source files to the right mapping column
    sources = {f.stem: load_source(f) for f in source_files}

    # Identify which source file corresponds to col_a vs col_b by checking overlap
    file_a, file_b = source_files[0], source_files[1]
    sample_a_keys = list(sources[file_a.stem].keys())[:20]
    col_a_ids = {m[0] for m in matches}
    overlap_a = sum(1 for k in sample_a_keys if k in col_a_ids)
    overlap_b = sum(1 for k in sample_a_keys if k in {m[1] for m in matches})

    if overlap_b > overlap_a:
        # file_a IDs match col_b in the mapping — swap so (source_a, source_b) = (col_a, col_b)
        source_a = sources[file_b.stem]
        source_b = sources[file_a.stem]
    else:
        source_a = sources[file_a.stem]
        source_b = sources[file_b.stem]

    return source_a, source_b, matches


def _evaluate_er(engine, source_a, source_b, true_matches):
    """Sweep thresholds and find best F1."""
    # Build a balanced sample: select a_ids that have at least one true match,
    # then include the corresponding true-match b_ids plus some non-matching b_ids.
    # A naive prefix-slice can miss all positives when source files aren't
    # sorted consistently with the mapping (e.g. DBLP-Scholar).
    a_id_to_matches = {}
    for a_id, b_id in true_matches:
        if a_id in source_a and b_id in source_b:
            a_id_to_matches.setdefault(a_id, []).append(b_id)

    sampled_a = list(a_id_to_matches.keys())[:100]
    # Collect the true-match b_ids for those a_ids
    positive_b_ids = {b for a in sampled_a for b in a_id_to_matches[a]}
    # Add some non-matching b_ids for negatives (up to 3× positives)
    all_b_ids = list(source_b.keys())
    negative_b_ids = [b for b in all_b_ids if b not in positive_b_ids][: len(positive_b_ids) * 3]
    candidate_b_ids = list(positive_b_ids) + negative_b_ids

    pairs = []
    labels = []
    for a_id in sampled_a:
        for b_id in candidate_b_ids[:50]:  # Limit cross-product per a_id
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
