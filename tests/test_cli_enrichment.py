import json

from click.testing import CliRunner

from src.cli import _build_benchmark_query, cli
from src.corpus_analyzer import build_corpus_analysis
from src.models import DocumentMetadata, Paragraph, ParsedDocument
from tests.test_scoring import _create_test_docx


class _FakeAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    async def analyze_and_build_graph(self, docs):
        from src.models import ContentAnalysis

        return [ContentAnalysis(summary="ok") for _ in docs], None


class _FakeRecommender:
    def __init__(self, *args, **kwargs):
        pass

    async def recommend(self, docs, analyses):
        from src.models import FolderNode, FolderRecommendation

        return FolderRecommendation(root=FolderNode(name="Knowledge Base", description=""), file_assignments={})

    def validate_assignments(self, assignments, similarity_matrix, doc_labels):
        return 0.0, []


def test_analyze_skip_enrichment_hides_folder_section(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)
    monkeypatch.setattr("src.analyzer.ContentAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("src.recommender.FolderRecommender", _FakeRecommender)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--llm-key", "test", "--skip-enrichment"])
    assert result.exit_code == 0
    assert "Recommended Folder Structure" not in result.output


def test_analyze_chunk_export_flag_writes_chunks(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)
    monkeypatch.setattr("src.analyzer.ContentAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("src.recommender.FolderRecommender", _FakeRecommender)

    result = CliRunner().invoke(cli, ["analyze", str(tmp_path), "--llm-key", "test", "--export-chunks"])
    assert result.exit_code == 0
    assert list((tmp_path / ".ragprep").glob("*.chunks.json"))


def test_benchmark_query_prefers_headings_and_tfidf_over_filename_stem():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/4-5.FL.10 Handout B.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Budget Planning", level=1, style="Heading 1", index=0),
            Paragraph(text="Students create a weekly budget and savings plan.", level=0, style="Normal", index=1),
        ],
    )
    corpus = build_corpus_analysis([doc])

    query = _build_benchmark_query(doc, corpus)

    assert "budget" in query
    assert "handout" not in query


def test_analyze_run_benchmark_json_includes_query_source_note(tmp_path, monkeypatch):
    test_file = str(tmp_path / "test.docx")
    _create_test_docx(test_file)
    monkeypatch.setattr("src.analyzer.ContentAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("src.recommender.FolderRecommender", _FakeRecommender)
    monkeypatch.setattr("src.cli._print_score_table", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.cli._print_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.cli._print_graph_summary", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        cli,
        [str("analyze"), str(tmp_path), "--llm-key", "test", "--run-benchmark", "--json-output", "--skip-enrichment"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["benchmarks"]
    assert "query_source: heading+tfidf deterministic" in payload["benchmarks"][0]["notes"]
