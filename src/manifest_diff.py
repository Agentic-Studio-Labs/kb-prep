"""Generate before/after markdown diffs from two IngestGate manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _doc_key(source_file: str) -> str:
    """Normalize docs across extensions (.pdf/.docx -> .md)."""
    return Path(source_file).stem


def _sum_issue_counts(doc: dict) -> int:
    counts = doc.get("issues_by_severity", {})
    return int(counts.get("critical", 0)) + int(counts.get("warning", 0)) + int(counts.get("info", 0))


def _format_gate_dist(corpus: dict) -> str:
    dist = corpus.get("gate_decision_distribution", {})
    if not dist:
        return "n/a"
    ordered = sorted(dist.items(), key=lambda item: item[0])
    return ", ".join(f"{k}: {v}" for k, v in ordered)


def _benchmark_row(manifest: dict) -> dict:
    rows = manifest.get("benchmarks", [])
    if not rows:
        return {}
    return rows[0]


def build_diff_markdown(before_manifest: dict, after_manifest: dict, run_folder: str = "") -> str:
    before_docs = {_doc_key(d["source_file"]): d for d in before_manifest.get("documents", [])}
    after_docs = {_doc_key(d["source_file"]): d for d in after_manifest.get("documents", [])}
    common_keys = sorted(set(before_docs) & set(after_docs))

    before_corpus = before_manifest.get("corpus", {})
    after_corpus = after_manifest.get("corpus", {})
    before_bench = _benchmark_row(before_manifest)
    after_bench = _benchmark_row(after_manifest)

    lines: list[str] = []
    lines.append("# Before vs After Diff")
    lines.append("")
    if run_folder:
        lines.append(f"Run folder: `{run_folder}`")
        lines.append("")
    lines.append("Compared artifacts:")
    lines.append("- `01-before-manifest.json`")
    lines.append("- `03-after-clean-manifest.json`")
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("|---|---:|---:|---:|")
    b_docs = int(before_corpus.get("total_documents", 0))
    a_docs = int(after_corpus.get("total_documents", 0))
    b_avg = float(before_corpus.get("avg_score", 0.0))
    a_avg = float(after_corpus.get("avg_score", 0.0))
    lines.append(f"| Total documents | {b_docs} | {a_docs} | {a_docs - b_docs:+d} |")
    lines.append(f"| Average score | {b_avg:.1f} | {a_avg:.1f} | {a_avg - b_avg:+.1f} |")
    lines.append(
        f"| Gate distribution | {_format_gate_dist(before_corpus)} | {_format_gate_dist(after_corpus)} | n/a |"
    )
    if before_bench and after_bench:
        for metric in ("recall_at_5", "mrr", "ndcg_at_5", "query_count"):
            b_val = before_bench.get(metric, 0.0)
            a_val = after_bench.get(metric, 0.0)
            if metric == "query_count":
                lines.append(f"| {metric} (lexical) | {int(b_val)} | {int(a_val)} | {int(a_val) - int(b_val):+d} |")
            else:
                delta = float(a_val) - float(b_val)
                lines.append(f"| {metric} (lexical) | {float(b_val):.3f} | {float(a_val):.3f} | {delta:+.3f} |")
    lines.append("")
    lines.append("## Document-Level Delta")
    lines.append("")
    lines.append("| Document stem | Before | After | Delta | Gate before -> after | Issues before -> after |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for key in common_keys:
        b_doc = before_docs[key]
        a_doc = after_docs[key]
        b_score = float(b_doc.get("overall_score", 0.0))
        a_score = float(a_doc.get("overall_score", 0.0))
        b_gate = b_doc.get("gate_decision", "")
        a_gate = a_doc.get("gate_decision", "")
        b_issues = _sum_issue_counts(b_doc)
        a_issues = _sum_issue_counts(a_doc)
        lines.append(
            f"| {key} | {b_score:.1f} | {a_score:.1f} | {a_score - b_score:+.1f} | "
            f"{b_gate} -> {a_gate} | {b_issues} -> {a_issues} |"
        )

    blocked_after = [
        after_docs[k] for k in common_keys if after_docs[k].get("gate_decision") == "REMEDIATION_RECOMMENDED"
    ]
    if blocked_after:
        blocked = blocked_after[0]
        key = _doc_key(blocked["source_file"])
        b_doc = before_docs[key]
        a_doc = after_docs[key]
        lines.append("")
        lines.append("## Why A Doc Is Still Blocked")
        lines.append("")
        lines.append("Remaining `REMEDIATION_RECOMMENDED` document:")
        lines.append(f"- `{a_doc['source_file']}`")
        lines.append("")
        lines.append("Notable criterion changes:")
        b_criteria = b_doc.get("criteria_scores", {})
        a_criteria = a_doc.get("criteria_scores", {})
        for criterion in sorted(set(b_criteria) | set(a_criteria)):
            b_score = b_criteria.get(criterion, {}).get("score")
            a_score = a_criteria.get(criterion, {}).get("score")
            if isinstance(b_score, (int, float)) and isinstance(a_score, (int, float)):
                delta = a_score - b_score
                if abs(delta) >= 5.0:
                    lines.append(f"- `{criterion}`: {b_score:.1f} -> {a_score:.1f} ({delta:+.1f})")
        lines.append("")
        lines.append("Criteria still limiting gate decision:")
        for criterion in sorted(a_criteria):
            score = a_criteria[criterion].get("score")
            if isinstance(score, (int, float)) and score < 70:
                lines.append(f"- `{criterion}`: {score:.1f}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This comparison matches documents by filename stem so extension changes don't break diffing.")
    lines.append("- Recommended comparison dashboard: `03-after-clean-web-report.html`.")
    lines.append("")
    return "\n".join(lines)


def write_diff_markdown(before_manifest_path: str, after_manifest_path: str, output_path: str) -> str:
    before_path = Path(before_manifest_path)
    after_path = Path(after_manifest_path)
    out_path = Path(output_path)
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    run_folder = str(out_path.parent)
    markdown = build_diff_markdown(before, after, run_folder=run_folder)
    out_path.write_text(markdown, encoding="utf-8")
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate before/after markdown diff from two manifests.")
    parser.add_argument("before_manifest", help="Path to baseline manifest JSON")
    parser.add_argument("after_manifest", help="Path to post-fix manifest JSON")
    parser.add_argument("-o", "--output", required=True, help="Output markdown file path")
    args = parser.parse_args()

    out = write_diff_markdown(args.before_manifest, args.after_manifest, args.output)
    print(f"Diff report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
