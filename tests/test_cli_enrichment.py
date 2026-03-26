from click.testing import CliRunner

from src.cli import cli
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
