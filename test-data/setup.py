#!/usr/bin/env python3
"""Download and prepare all test corpora for the eval suite.

Run: python test-data/setup.py
Idempotent — skips datasets already cached.
"""

import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

CORPORA_DIR = Path(__file__).parent / "corpora"
MANIFEST_PATH = CORPORA_DIR / "manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _is_ready(name: str, manifest: dict) -> bool:
    return name in manifest and manifest[name].get("status") == "ready"


# ---------------------------------------------------------------------------
# Dataset setup functions
# ---------------------------------------------------------------------------


def setup_squad(manifest: dict):
    """SQuAD 1.1 validation — unique context paragraphs with questions."""
    name = "squad"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from datasets import load_dataset

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("squad", split="validation")
    # Extract unique contexts with their questions
    contexts: dict[str, list[dict]] = {}
    for row in ds:
        ctx = row["context"]
        if ctx not in contexts:
            contexts[ctx] = []
        contexts[ctx].append(
            {
                "question": row["question"],
                "answers": row["answers"]["text"],
            }
        )

    records = []
    for i, (ctx, qas) in enumerate(contexts.items()):
        records.append({"id": f"squad_{i}", "text": ctx, "questions": qas})

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} unique paragraphs")


def setup_beir(manifest: dict):
    """BEIR datasets — SciFact, NFCorpus, TREC-COVID."""
    from datasets import load_dataset

    for dataset_name in ["scifact", "nfcorpus", "trec-covid"]:
        key = f"beir_{dataset_name}"
        if _is_ready(key, manifest):
            print(f"  {key}: already cached, skipping")
            continue

        out_dir = CORPORA_DIR / "beir" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Corpus
        try:
            corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
        except Exception:
            corpus_ds = load_dataset(f"BeIR/{dataset_name}", split="corpus")

        corpus_file = out_dir / "corpus.jsonl"
        with open(corpus_file, "w") as f:
            for row in corpus_ds:
                f.write(
                    json.dumps({"_id": row.get("_id", ""), "title": row.get("title", ""), "text": row.get("text", "")})
                    + "\n"
                )

        # Queries
        try:
            queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
            queries_file = out_dir / "queries.jsonl"
            with open(queries_file, "w") as f:
                for row in queries_ds:
                    f.write(json.dumps({"_id": row.get("_id", ""), "text": row.get("text", "")}) + "\n")
        except Exception as e:
            print(f"    Warning: could not load queries for {dataset_name}: {e}")

        manifest[key] = {"status": "ready", "rows": len(corpus_ds), "date": datetime.now().isoformat()}
        print(f"  {key}: {len(corpus_ds)} documents")


def setup_cuad(manifest: dict):
    """CUAD — contract understanding dataset with clause-level QA."""
    name = "cuad"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from datasets import load_dataset

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("theatticusproject/cuad", split="test")
    records = []
    seen_contexts = set()
    for row in ds:
        ctx = row["context"]
        if ctx not in seen_contexts:
            seen_contexts.add(ctx)
            records.append(
                {
                    "id": f"cuad_{len(records)}",
                    "text": ctx,
                    "title": row.get("title", ""),
                }
            )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} unique contracts")


def setup_hotpotqa(manifest: dict):
    """HotpotQA distractor split — sample 500 for multi-hop QA evaluation."""
    name = "hotpotqa"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from datasets import load_dataset

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    # Sample 500
    ds_sample = ds.select(range(min(500, len(ds))))

    records = []
    for row in ds_sample:
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "supporting_facts": {
                    "title": row["supporting_facts"]["title"],
                    "sent_id": row["supporting_facts"]["sent_id"],
                },
                "context": {
                    "title": row["context"]["title"],
                    "sentences": row["context"]["sentences"],
                },
            }
        )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} questions")


def setup_newsgroups(manifest: dict):
    """20 Newsgroups — all documents with headers/footers/quotes removed."""
    name = "newsgroups"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from sklearn.datasets import fetch_20newsgroups

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    records = []
    for i, (text, label) in enumerate(zip(data.data, data.target)):
        records.append(
            {
                "id": f"ng_{i}",
                "text": text,
                "label": int(label),
                "category": data.target_names[label],
            }
        )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} documents across {len(data.target_names)} categories")


def setup_choi(manifest: dict):
    """Choi text segmentation dataset — documents with known boundaries."""
    name = "choi"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo (or download just the data files)
    repo_dir = out_dir / "repo"
    if not repo_dir.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/koomri/text-segmentation.git", str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except Exception as e:
            print(f"    Warning: could not clone Choi repo: {e}")
            manifest[name] = {"status": "failed", "error": str(e)}
            return

    # Parse .ref files
    data_dir = repo_dir / "data" / "choi"
    if not data_dir.exists():
        # Try alternative paths
        for candidate in [repo_dir / "data", repo_dir]:
            ref_files = list(candidate.rglob("*.ref"))
            if ref_files:
                data_dir = candidate
                break

    ref_files = list(data_dir.rglob("*.ref")) if data_dir.exists() else []
    records = []
    for ref_file in sorted(ref_files):
        text = ref_file.read_text(encoding="utf-8", errors="replace")
        segments = text.split("==========")
        segments = [s.strip() for s in segments if s.strip()]
        if segments:
            # Boundaries are at the split points
            boundaries = []
            offset = 0
            for seg in segments[:-1]:
                offset += len(seg.split("\n"))
                boundaries.append(offset)
            records.append(
                {
                    "id": ref_file.stem,
                    "segments": segments,
                    "boundaries": boundaries,
                }
            )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} segmented documents")


def setup_fb15k237(manifest: dict):
    """FB15k-237 knowledge graph — triple files for PageRank evaluation."""
    name = "fb15k237"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import requests

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try downloading from a mirror/alternative source
    urls = [
        "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt",
        "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/valid.txt",
        "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/test.txt",
    ]

    total_triples = 0
    for url in urls:
        filename = url.split("/")[-1]
        out_file = out_dir / filename
        if out_file.exists():
            continue
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_file.write_text(resp.text)
            total_triples += len(resp.text.strip().split("\n"))
        except Exception as e:
            print(f"    Warning: could not download {filename}: {e}")

    manifest[name] = {"status": "ready", "triples": total_triples, "date": datetime.now().isoformat()}
    print(f"  {name}: {total_triples} triples")


def setup_sts(manifest: dict):
    """STS Benchmark — sentence pairs with similarity scores."""
    name = "sts"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from datasets import load_dataset

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("sentence-transformers/stsb", split="test")
    records = []
    for row in ds:
        records.append(
            {
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": row["score"],
            }
        )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} sentence pairs")


def setup_arxiv_sample(manifest: dict):
    """arXiv ML papers — sample 200 for multi-domain testing."""
    name = "arxiv_sample"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    from datasets import load_dataset

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    ds_sample = ds.select(range(min(200, len(ds))))

    records = []
    for i, row in enumerate(ds_sample):
        records.append(
            {
                "id": f"arxiv_{i}",
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }
        )

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest[name] = {"status": "ready", "rows": len(records), "date": datetime.now().isoformat()}
    print(f"  {name}: {len(records)} papers")


def setup_leipzig_er(manifest: dict):
    """Leipzig entity resolution benchmarks — Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-Scholar."""
    name = "leipzig_er"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import requests

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # These datasets are available from various mirrors
    datasets_info = {
        "Abt-Buy": "https://dbs.uni-leipzig.de/file/Abt-Buy.zip",
        "Amazon-Google": "https://dbs.uni-leipzig.de/file/Amazon-Google.zip",
        "DBLP-ACM": "https://dbs.uni-leipzig.de/file/DBLP-ACM.zip",
        "DBLP-Scholar": "https://dbs.uni-leipzig.de/file/DBLP-Scholar.zip",
    }

    total = 0
    for ds_name, url in datasets_info.items():
        ds_dir = out_dir / ds_name
        if ds_dir.exists() and any(ds_dir.glob("*.csv")):
            continue
        ds_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            zip_path = out_dir / f"{ds_name}.zip"
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ds_dir)
            zip_path.unlink()
            total += 1
            print(f"    {ds_name}: downloaded")
        except Exception as e:
            print(f"    Warning: could not download {ds_name}: {e}")

    manifest[name] = {
        "status": "ready" if total > 0 else "partial",
        "datasets": total,
        "date": datetime.now().isoformat(),
    }
    print(f"  {name}: {total} datasets downloaded")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_SETUP_FNS = [
    ("SQuAD 1.1", setup_squad),
    ("BEIR (SciFact, NFCorpus, TREC-COVID)", setup_beir),
    ("CUAD", setup_cuad),
    ("HotpotQA", setup_hotpotqa),
    ("20 Newsgroups", setup_newsgroups),
    ("Choi Segmentation", setup_choi),
    ("FB15k-237", setup_fb15k237),
    ("STS Benchmark", setup_sts),
    ("arXiv Sample", setup_arxiv_sample),
    ("Leipzig ER", setup_leipzig_er),
]


if __name__ == "__main__":
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    print("Setting up test corpora...\n")
    for label, fn in ALL_SETUP_FNS:
        print(f"[{label}]")
        try:
            fn(manifest)
        except Exception as e:
            print(f"  ERROR: {e}")
            manifest[label] = {"status": "failed", "error": str(e)}
        _save_manifest(manifest)

    print(f"\nManifest written to {MANIFEST_PATH}")
    ready = sum(1 for v in manifest.values() if v.get("status") == "ready")
    print(f"Ready: {ready}/{len(ALL_SETUP_FNS)} datasets")
