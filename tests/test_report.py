import os

from click.testing import CliRunner

from src.models import Issue, ScoreCard, ScoringResult, Severity


def _make_card(filename="test.docx", score=85.0, issues=None):
    """Helper to build a ScoreCard for testing."""
    card = ScoreCard(file_path=f"/tmp/{filename}")
    card.results = [
        ScoringResult(
            category="structure",
            label="Document Structure",
            score=score,
            weight=0.15,
            issues=issues or [],
        )
    ]
    card.overall_score = score
    return card


def test_report_header():
    """_report_header returns markdown header with timestamp and file count."""
    from src.cli import _report_header

    lines = _report_header(command="score", file_count=5)
    text = "\n".join(lines)
    assert "# IngestGate" in text
    assert "score" in text.lower()
    assert "5" in text


def test_report_scores():
    """_report_scores returns markdown table rows."""
    from src.cli import _report_scores

    cards = [_make_card("a.docx", 90), _make_card("b.docx", 60)]
    lines = _report_scores(cards, detail=False)
    text = "\n".join(lines)
    assert "| a.docx" in text
    assert "| b.docx" in text
    assert "90" in text
    assert "Average score" in text


def test_report_scores_detail():
    """_report_scores with detail includes issue breakdown."""
    from src.cli import _report_scores

    issue = Issue(severity=Severity.WARNING, category="structure", message="Bad heading")
    cards = [_make_card("a.docx", 70, issues=[issue])]
    lines = _report_scores(cards, detail=True)
    text = "\n".join(lines)
    assert "Bad heading" in text


def test_report_scores_surfaces_parse_fidelity_warning():
    from src.cli import _report_scores

    issue = Issue(
        severity=Severity.WARNING,
        category="structure",
        message="Low parse fidelity: only 98 words extracted from 124 KB file",
    )
    cards = [_make_card("sparse.pdf", 62, issues=[issue])]
    lines = _report_scores(cards, detail=False)
    text = "\n".join(lines)
    assert "Parse fidelity warnings" in text
    assert "sparse.pdf" in text


def test_report_scores_surfaces_template_parse_fidelity_note():
    from src.cli import _report_scores

    issue = Issue(
        severity=Severity.INFO,
        category="structure",
        message="Low parse fidelity (template-like document): only 25 words extracted from 53 KB file",
    )
    cards = [_make_card("tracker.pdf", 88, issues=[issue])]
    lines = _report_scores(cards, detail=False)
    text = "\n".join(lines)
    assert "Parse fidelity notes (expected sparse templates)" in text
    assert "tracker.pdf" in text


def test_generate_report_path():
    """_generate_report_path returns timestamped filename."""
    from src.cli import _generate_report_path

    path = _generate_report_path("analyze")
    assert path.startswith("ingestgate-analyze-")
    assert path.endswith(".md")
    # Verify timestamp format YYYYMMDD-HHMMSS
    parts = path.replace("ingestgate-analyze-", "").replace(".md", "")
    assert len(parts) == 15  # YYYYMMDD-HHMMSS


def test_generate_web_report_path():
    from src.cli import _generate_web_report_path

    path = _generate_web_report_path("analyze")
    assert path.startswith("ingestgate-analyze-")
    assert path.endswith(".html")
    parts = path.replace("ingestgate-analyze-", "").replace(".html", "")
    assert len(parts) == 15  # YYYYMMDD-HHMMSS


def test_gate_decision_mapping_and_legacy_readiness():
    issue = Issue(severity=Severity.CRITICAL, category="structure", message="Major parse failure")

    passing = _make_card("pass.docx", 92)
    assert passing.readiness.value == "PASS"
    assert passing.legacy_readiness == "EXCELLENT"

    needs_notes = _make_card("notes.docx", 74)
    assert needs_notes.readiness.value == "PASS_WITH_NOTES"
    assert needs_notes.legacy_readiness == "GOOD"

    remediation = _make_card("remediate.docx", 55)
    assert remediation.readiness.value == "REMEDIATION_RECOMMENDED"
    assert remediation.legacy_readiness == "FAIR"

    critical = _make_card("critical.docx", 91, issues=[issue])
    assert critical.readiness.value == "HOLD_FOR_REVIEW"
    assert critical.legacy_readiness == "EXCELLENT"


def test_gate_decision_respects_custom_thresholds():
    custom = ScoreCard(
        file_path="/tmp/custom.docx",
        overall_score=82,
        gate_pass_threshold=90,
        gate_pass_with_notes_threshold=80,
        gate_remediation_threshold=60,
    )
    assert custom.readiness.value == "PASS_WITH_NOTES"
    assert custom.legacy_readiness == "GOOD"


def test_score_generates_report(tmp_path):
    """score command auto-generates a markdown report file."""
    # Create a test DOCX
    from src.cli import cli
    from tests.test_scoring import _create_test_docx

    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)

    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        result = runner.invoke(cli, ["score", test_file])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        # Find the generated report
        reports = [f for f in os.listdir(td) if f.startswith("ingestgate-score-")]
        assert len(reports) == 1, f"Expected 1 report, found: {reports}"
        content = open(reports[0]).read()
        assert "# IngestGate score Report" in content
        assert "test.docx" in content


def test_score_no_report_flag(tmp_path):
    """--no-report suppresses report generation."""
    from src.cli import cli
    from tests.test_scoring import _create_test_docx

    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)

    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        result = runner.invoke(cli, ["score", "--no-report", test_file])
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        reports = [f for f in os.listdir(td) if f.startswith("ingestgate-")]
        assert len(reports) == 0
