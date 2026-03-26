"""Tests for async ContentAnalyzer.

Verifies that the analyzer uses AsyncAnthropic, async methods,
semaphore-based concurrency, handles failures gracefully, and
filters low-confidence analyses from the knowledge graph.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.analyzer import ContentAnalyzer
from src.config import Config
from src.models import DocumentMetadata, Paragraph, ParsedDocument


def _make_doc(filename="test.docx", text="This is test content about math."):
    return ParsedDocument(
        metadata=DocumentMetadata(file_path=f"/tmp/{filename}", file_type="docx"),
        paragraphs=[Paragraph(text=text, level=0, style="Normal", index=0)],
    )


def _mock_llm_response(json_str: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=json_str)]
    return msg


def test_analyze_is_async():
    """ContentAnalyzer.analyze is a coroutine."""
    import inspect

    from src.analyzer import ContentAnalyzer

    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze)


def test_analyze_and_build_graph_is_async():
    """analyze_and_build_graph is a coroutine."""
    import inspect

    from src.analyzer import ContentAnalyzer

    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze_and_build_graph)


def test_concurrent_analysis():
    """analyze_and_build_graph processes multiple docs concurrently."""
    config = Config(anthropic_api_key="test-key", concurrency=2)

    json_response = '{"domain":"education","topics":["math"],"audience":"students","content_type":"lesson","key_concepts":["fractions"],"suggested_tags":["math"],"summary":"A lesson.","entities":[],"relationships":[]}'

    with patch("src.analyzer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.messages.create = AsyncMock(return_value=_mock_llm_response(json_response))

        analyzer = ContentAnalyzer(config)

        docs = [_make_doc(f"doc{i}.docx") for i in range(4)]
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 4
        assert all(a.domain == "education" for a in analyses)
        assert mock_instance.messages.create.call_count == 4


def test_analysis_failure_handled():
    """Failed analysis returns error ContentAnalysis, doesn't crash."""
    config = Config(anthropic_api_key="test-key", concurrency=2)

    with patch("src.analyzer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.messages.create = AsyncMock(side_effect=Exception("API error"))

        analyzer = ContentAnalyzer(config)
        docs = [_make_doc("fail.docx")]
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 1
        assert "failed" in analyses[0].summary.lower()


# ---------------------------------------------------------------------------
# Confidence gate tests
# ---------------------------------------------------------------------------


def test_low_confidence_failed_analysis():
    """Failed analyses are low confidence."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis

    doc = _make_doc("test.docx", "Some content about financial planning and budgeting strategies.")
    analysis = ContentAnalysis(summary="Analysis failed: API error")
    assert _analysis_is_low_confidence(doc, analysis) is True


def test_low_confidence_no_entities():
    """Analyses with zero entities are low confidence."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis

    doc = _make_doc("test.docx", "Some content about financial planning and budgeting strategies.")
    analysis = ContentAnalysis(summary="Good analysis", entities=[])
    assert _analysis_is_low_confidence(doc, analysis) is True


def test_low_confidence_sparse_entities():
    """Long document with only 1 entity is low confidence."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis, Entity

    doc = _make_doc("test.docx", " ".join(["word"] * 500))  # 500-word doc
    analysis = ContentAnalysis(
        summary="Sparse",
        entities=[Entity(name="Thing", entity_type="concept", source_file="test.docx")],
    )
    assert _analysis_is_low_confidence(doc, analysis) is True


def test_low_confidence_no_relationships():
    """3+ entities with zero relationships is low confidence."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis, Entity

    doc = _make_doc("test.docx", "Content here.")
    analysis = ContentAnalysis(
        summary="OK",
        entities=[
            Entity(name="A", entity_type="concept", source_file="test.docx"),
            Entity(name="B", entity_type="skill", source_file="test.docx"),
            Entity(name="C", entity_type="process", source_file="test.docx"),
        ],
        relationships=[],
    )
    assert _analysis_is_low_confidence(doc, analysis) is True


def test_low_confidence_uniform_types():
    """All entities same type suggests LLM defaulting."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis, Entity, Relationship

    doc = _make_doc("test.docx", "Content here.")
    analysis = ContentAnalysis(
        summary="OK",
        entities=[
            Entity(name="A", entity_type="concept", source_file="test.docx"),
            Entity(name="B", entity_type="concept", source_file="test.docx"),
            Entity(name="C", entity_type="concept", source_file="test.docx"),
        ],
        relationships=[
            Relationship(source="A", target="B", rel_type="related_to", source_file="test.docx"),
        ],
    )
    assert _analysis_is_low_confidence(doc, analysis) is True


def test_high_confidence_good_analysis():
    """A well-formed analysis with diverse entities and relationships passes."""
    from src.analyzer import _analysis_is_low_confidence
    from src.models import ContentAnalysis, Entity, Relationship

    doc = _make_doc("test.docx", "Content about budgeting and saving strategies for families.")
    analysis = ContentAnalysis(
        summary="Financial literacy content",
        entities=[
            Entity(name="Budgeting", entity_type="concept", source_file="test.docx"),
            Entity(name="Saving", entity_type="skill", source_file="test.docx"),
            Entity(name="Family Finance", entity_type="process", source_file="test.docx"),
        ],
        relationships=[
            Relationship(source="Budgeting", target="Saving", rel_type="related_to", source_file="test.docx"),
        ],
    )
    assert _analysis_is_low_confidence(doc, analysis) is False
