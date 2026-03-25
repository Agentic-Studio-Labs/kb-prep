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
    """BEIR datasets — SciFact, NFCorpus, TREC-COVID.

    Uses BeIR/{name}-generated-queries which contains corpus text + queries
    in a single split (the old BeIR/{name} dataset scripts are deprecated).
    """
    from datasets import load_dataset

    for dataset_name in ["scifact", "nfcorpus", "trec-covid"]:
        key = f"beir_{dataset_name}"
        if _is_ready(key, manifest):
            print(f"  {key}: already cached, skipping")
            continue

        out_dir = CORPORA_DIR / "beir" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load via generated-queries variant (has corpus text + query per row)
        ds = load_dataset(f"BeIR/{dataset_name}-generated-queries", split="train")

        # Extract unique corpus documents
        seen_ids = set()
        corpus_file = out_dir / "corpus.jsonl"
        queries_file = out_dir / "queries.jsonl"
        with open(corpus_file, "w") as cf, open(queries_file, "w") as qf:
            for row in ds:
                doc_id = row.get("_id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    cf.write(
                        json.dumps(
                            {
                                "_id": doc_id,
                                "title": row.get("title", ""),
                                "text": row.get("text", ""),
                            }
                        )
                        + "\n"
                    )
                # Each row also has a generated query
                query = row.get("query", "")
                if query:
                    qf.write(
                        json.dumps(
                            {
                                "_id": doc_id,
                                "text": query,
                            }
                        )
                        + "\n"
                    )

        manifest[key] = {"status": "ready", "rows": len(seen_ids), "date": datetime.now().isoformat()}
        print(f"  {key}: {len(seen_ids)} documents, {len(ds)} queries")


def setup_cuad(manifest: dict):
    """CUAD — contract understanding dataset with clause-level QA.

    Downloads the SQuAD-format JSON directly from HuggingFace since the
    HF datasets library can't load the restructured repo (PDF-based).
    """
    name = "cuad"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import requests

    out_dir = CORPORA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    records = []
    seen_contexts = set()
    for article in data.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            ctx = para.get("context", "")
            if ctx and ctx not in seen_contexts:
                seen_contexts.add(ctx)
                records.append(
                    {
                        "id": f"cuad_{len(records)}",
                        "text": ctx,
                        "title": title,
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
    """Leipzig entity resolution benchmarks — Abt-Buy, Amazon-GoogleProducts, DBLP-ACM, DBLP-Scholar."""
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
        "Amazon-GoogleProducts": "https://dbs.uni-leipzig.de/file/Amazon-GoogleProducts.zip",
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


def setup_score_bench(manifest: dict):
    """SCORE-Bench — 224 real-world PDFs with text annotations."""
    name = "score_bench"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import requests

    out_dir = CORPORA_DIR / name
    pdf_dir = out_dir / "pdfs"
    anno_dir = out_dir / "annotations"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/datasets/unstructuredio/SCORE-Bench/resolve/main"

    # List PDF files from src/
    api_resp = requests.get(
        "https://huggingface.co/api/datasets/unstructuredio/SCORE-Bench/tree/main/src",
        timeout=30,
    )
    api_resp.raise_for_status()
    pdf_files = [f["path"] for f in api_resp.json() if f["path"].endswith(".pdf")]

    # Sample 30 for manageable download
    pdf_files = pdf_files[:30]

    downloaded = 0
    for pdf_path in pdf_files:
        filename = pdf_path.split("/")[-1]
        local_pdf = pdf_dir / filename
        if local_pdf.exists():
            downloaded += 1
            continue

        # Download PDF
        try:
            resp = requests.get(f"{base}/{pdf_path}", timeout=60)
            resp.raise_for_status()
            local_pdf.write_bytes(resp.content)
        except Exception as e:
            print(f"    Warning: failed to download {filename}: {e}")
            continue

        # Download matching annotation
        anno_name = f"{filename}__uns-plaintext-v1.0.0__0x0001__0.txt"
        try:
            resp = requests.get(f"{base}/content-gt/{anno_name}", timeout=30)
            if resp.status_code == 200:
                (anno_dir / anno_name).write_text(resp.text)
        except Exception:
            pass

        downloaded += 1

    manifest[name] = {"status": "ready", "pdfs": downloaded, "date": datetime.now().isoformat()}
    print(f"  {name}: {downloaded} PDFs downloaded")


def setup_omnidocbench(manifest: dict):
    """OmniDocBench — synthetic text documents built from layout annotations.

    The HF repo contains page images (PNG) and a JSON annotation file with
    per-element text and category_type.  Since DocumentParser cannot handle
    PNGs, we synthesise a .txt document per source PDF stem by concatenating
    the annotation text blocks in reading order.  The annotation metadata
    (category_type sequence) is preserved in a separate JSONL file for use
    by the boundary Pk test.
    """
    name = "omnidocbench"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import re

    import requests

    out_dir = CORPORA_DIR / name
    txt_dir = out_dir / "txts"
    txt_dir.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main"

    # Download annotations JSON
    anno_file = out_dir / "annotations.json"
    if not anno_file.exists():
        resp = requests.get(f"{base}/OmniDocBench.json", timeout=120)
        resp.raise_for_status()
        anno_file.write_bytes(resp.content)

    annotations = json.loads(anno_file.read_text())

    # Group annotation pages by source document stem.
    # image_path patterns:
    #   {stem}.pdf_{page}.jpg  or  {stem}_page_{page}.jpg/png
    def _doc_stem(image_path: str) -> str:
        m = re.match(r"^(.+\.pdf)_\d+\.", image_path)
        if m:
            return m.group(1)
        m2 = re.match(r"^(.+?)_page_\d+\.", image_path)
        if m2:
            return m2.group(1) + ".pdf"
        return image_path

    doc_pages: dict[str, list[dict]] = {}
    for anno in annotations:
        image_path = anno.get("page_info", {}).get("image_path", "")
        if not image_path:
            continue
        stem = _doc_stem(image_path)
        doc_pages.setdefault(stem, []).append(anno)

    # Filter to docs with Latin text content
    latin_docs = []
    for stem, pages in doc_pages.items():
        for page in pages:
            for det in page.get("layout_dets", []):
                text = det.get("text", "")
                if text and any(c.isascii() and c.isalpha() for c in text):
                    latin_docs.append(stem)
                    break
            else:
                continue
            break

    # Sample up to 30 documents
    latin_docs = list(dict.fromkeys(latin_docs))[:30]  # deduplicate, preserve order

    boundary_records = []
    built = 0
    for stem in latin_docs:
        safe_name = re.sub(r"[^\w.-]", "_", stem)
        txt_path = txt_dir / f"{safe_name}.txt"

        pages = sorted(doc_pages[stem], key=lambda a: a.get("page_info", {}).get("page_no") or 0)

        lines: list[str] = []
        category_seq: list[str] = []
        for page in pages:
            dets = sorted(page.get("layout_dets", []), key=lambda d: d.get("order") or 0)
            for det in dets:
                text = det.get("text", "").strip()
                cat = det.get("category_type", "text")
                if not text:
                    continue
                lines.append(text)
                category_seq.append(cat)

        if not lines:
            continue

        if not txt_path.exists():
            txt_path.write_text("\n\n".join(lines), encoding="utf-8")

        boundary_records.append({"stem": stem, "safe_name": safe_name, "categories": category_seq})
        built += 1

    # Always (re)write boundary metadata so it stays in sync with txt files
    boundary_file = out_dir / "boundaries.jsonl"
    with open(boundary_file, "w") as f:
        for rec in boundary_records:
            f.write(json.dumps(rec) + "\n")

    manifest[name] = {"status": "ready", "docs": built, "date": datetime.now().isoformat()}
    print(f"  {name}: {built} synthetic documents built from annotations")


def setup_kleister_nda(manifest: dict):
    """Kleister NDA — 540 real NDA PDFs with entity annotations."""
    name = "kleister_nda"
    if _is_ready(name, manifest):
        print(f"  {name}: already cached, skipping")
        return
    import requests

    out_dir = CORPORA_DIR / name
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    gh_base = "https://raw.githubusercontent.com/applicaai/kleister-nda/master"
    gh_api = "https://api.github.com/repos/applicaai/kleister-nda/contents"

    # Download ground truth TSV
    tsv_file = out_dir / "in-header.tsv"
    if not tsv_file.exists():
        resp = requests.get(f"{gh_base}/in-header.tsv", timeout=30)
        resp.raise_for_status()
        tsv_file.write_text(resp.text)

    # Download train and dev-0 ground truth splits
    for split in ["train", "dev-0"]:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["expected.tsv", "in.tsv"]:
            local = split_dir / fname
            if not local.exists():
                try:
                    resp = requests.get(f"{gh_base}/{split}/{fname}", timeout=30)
                    if resp.status_code == 200:
                        local.write_text(resp.text)
                except Exception:
                    pass

    # List and download PDFs from documents/
    try:
        resp = requests.get(f"{gh_api}/documents", timeout=30)
        resp.raise_for_status()
        pdf_entries = [f for f in resp.json() if f["name"].endswith(".pdf")]
    except Exception as e:
        print(f"    Warning: could not list documents: {e}")
        pdf_entries = []

    # Sample 50 PDFs (full set is 540, too many for quick eval)
    pdf_entries = pdf_entries[:50]

    downloaded = 0
    for entry in pdf_entries:
        filename = entry["name"]
        local_pdf = pdf_dir / filename
        if local_pdf.exists():
            downloaded += 1
            continue
        try:
            resp = requests.get(entry["download_url"], timeout=60)
            resp.raise_for_status()
            local_pdf.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            print(f"    Warning: failed to download {filename}: {e}")

    manifest[name] = {"status": "ready", "pdfs": downloaded, "date": datetime.now().isoformat()}
    print(f"  {name}: {downloaded} PDFs downloaded")


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
    ("SCORE-Bench", setup_score_bench),
    ("OmniDocBench", setup_omnidocbench),
    ("Kleister NDA", setup_kleister_nda),
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

    # Prune stale manifest entries from previous runs
    valid_keys = set()
    for label, fn in ALL_SETUP_FNS:
        # Each setup function uses its own key (fn.__doc__ names it, or we check the manifest)
        valid_keys.add(label)
    # Also keep keys written by the functions themselves (e.g. "beir_scifact")
    fn_keys = {k for k in manifest if any(k.startswith(label.split()[0].lower()) for label, _ in ALL_SETUP_FNS)}
    valid_keys.update(fn_keys)
    stale = [k for k in manifest if k not in valid_keys]
    for k in stale:
        del manifest[k]
    if stale:
        _save_manifest(manifest)

    # Summary
    print(f"\nManifest written to {MANIFEST_PATH}")
    ready = sum(1 for v in manifest.values() if v.get("status") == "ready")
    total = len(manifest)
    print(f"Ready: {ready}/{total} datasets\n")

    # File breakdown
    counts: dict[str, int] = {}
    total_bytes = 0
    for path in CORPORA_DIR.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            ext = path.suffix.lower() or "(no ext)"
            counts[ext] = counts.get(ext, 0) + 1
            total_bytes += path.stat().st_size

    print("Files downloaded:")
    for ext, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:5d} {ext}")
    mb = total_bytes / (1024 * 1024)
    print(f"  Total: {sum(counts.values())} files, {mb:.0f} MB")
