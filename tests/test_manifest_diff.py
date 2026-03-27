import json
from pathlib import Path

from src.manifest_diff import write_diff_markdown


def test_write_diff_markdown_matches_by_stem(tmp_path):
    before = {
        "corpus": {
            "total_documents": 1,
            "avg_score": 65.0,
            "gate_decision_distribution": {"REMEDIATION_RECOMMENDED": 1},
        },
        "documents": [
            {
                "source_file": "lesson.docx",
                "overall_score": 65.0,
                "gate_decision": "REMEDIATION_RECOMMENDED",
                "issues_by_severity": {"critical": 0, "warning": 3, "info": 1},
                "criteria_scores": {
                    "self_containment": {"score": 75.0},
                    "paragraph_length": {"score": 0.0},
                },
            }
        ],
        "benchmarks": [
            {"retrieval_mode": "lexical", "recall_at_5": 0.6, "mrr": 0.4, "ndcg_at_5": 0.3, "query_count": 1}
        ],
    }
    after = {
        "corpus": {
            "total_documents": 1,
            "avg_score": 72.0,
            "gate_decision_distribution": {"PASS_WITH_NOTES": 1},
        },
        "documents": [
            {
                "source_file": "lesson.md",
                "overall_score": 72.0,
                "gate_decision": "PASS_WITH_NOTES",
                "issues_by_severity": {"critical": 0, "warning": 1, "info": 1},
                "criteria_scores": {
                    "self_containment": {"score": 95.0},
                    "paragraph_length": {"score": 30.0},
                },
            }
        ],
        "benchmarks": [
            {"retrieval_mode": "lexical", "recall_at_5": 0.7, "mrr": 0.5, "ndcg_at_5": 0.4, "query_count": 1}
        ],
    }

    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    out_path = tmp_path / "diff.md"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    written = write_diff_markdown(str(before_path), str(after_path), str(out_path))
    assert Path(written).exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Before vs After Diff" in content
    assert "| lesson | 65.0 | 72.0 | +7.0 | REMEDIATION_RECOMMENDED -> PASS_WITH_NOTES | 4 -> 2 |" in content
    assert "recall_at_5 (lexical)" in content
