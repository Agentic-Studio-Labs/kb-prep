import json
from pathlib import Path

from src.web_report import write_web_report


def test_write_web_report_creates_html(tmp_path):
    manifest = {
        "corpus": {
            "total_documents": 2,
            "total_chunks": 5,
            "avg_score": 82.5,
            "gate_decision_distribution": {"PASS": 1, "PASS_WITH_NOTES": 1},
        },
        "documents": [
            {
                "source_file": "a.md",
                "overall_score": 90,
                "readiness": "EXCELLENT",
                "gate_decision": "PASS",
                "domain": "education",
                "topics": ["budgeting"],
                "entity_count": 2,
                "relationship_count": 1,
                "chunk_count": 3,
                "issues_by_severity": {"critical": 0, "warning": 1, "info": 0},
                "criteria_scores": {
                    "self_containment": {
                        "label": "Self-Containment",
                        "score": 88.0,
                        "weight": 0.2,
                        "issue_count": 1,
                    }
                },
                "retrieval_quality_gate": {"retrieval_mode_hint": {"recommended_mode": "text_hybrid_default"}},
            },
            {
                "source_file": "b.pdf",
                "overall_score": 72,
                "readiness": "GOOD",
                "gate_decision": "PASS_WITH_NOTES",
                "domain": "education",
                "topics": ["planning"],
                "entity_count": 1,
                "relationship_count": 0,
                "chunk_count": 2,
                "issues_by_severity": {"critical": 0, "warning": 0, "info": 1},
                "criteria_scores": {
                    "structure": {
                        "label": "Structure",
                        "score": 70.0,
                        "weight": 0.1,
                        "issue_count": 1,
                    }
                },
                "retrieval_quality_gate": {"retrieval_mode_hint": {"recommended_mode": "hybrid_sparse_template"}},
            },
        ],
        "benchmarks": [],
    }
    out = tmp_path / "dash.html"
    path = write_web_report(str(out), manifest)
    assert Path(path).exists()
    text = Path(path).read_text(encoding="utf-8")
    assert "IngestGate Dashboard" in text
    assert json.dumps(manifest, ensure_ascii=False) in text
    assert "Gate Decision Distribution" in text
    assert "Document Drill-In" in text
    assert "data-clickable" in text
    assert "Issues by Severity" in text
    assert "Criteria Scores" in text
