"""Tests for async ContentAnalyzer.

Verifies that the analyzer uses AsyncAnthropic, async methods,
semaphore-based concurrency, and handles failures gracefully.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer import ContentAnalyzer
from config import Config
from models import ParsedDocument, DocumentMetadata, Paragraph


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
    from analyzer import ContentAnalyzer
    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze)


def test_analyze_and_build_graph_is_async():
    """analyze_and_build_graph is a coroutine."""
    import inspect
    from analyzer import ContentAnalyzer
    assert inspect.iscoroutinefunction(ContentAnalyzer.analyze_and_build_graph)


def test_concurrent_analysis():
    """analyze_and_build_graph processes multiple docs concurrently."""
    config = Config(anthropic_api_key="test-key", concurrency=2)

    json_response = '{"domain":"education","topics":["math"],"audience":"students","content_type":"lesson","key_concepts":["fractions"],"suggested_tags":["math"],"summary":"A lesson.","entities":[],"relationships":[]}'

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.messages.create = AsyncMock(
            return_value=_mock_llm_response(json_response)
        )

        analyzer = ContentAnalyzer(config)

        docs = [_make_doc(f"doc{i}.docx") for i in range(4)]
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 4
        assert all(a.domain == "education" for a in analyses)
        assert mock_instance.messages.create.call_count == 4


def test_analysis_failure_handled():
    """Failed analysis returns error ContentAnalysis, doesn't crash."""
    config = Config(anthropic_api_key="test-key", concurrency=2)

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.messages.create = AsyncMock(
            side_effect=Exception("API error")
        )

        analyzer = ContentAnalyzer(config)
        docs = [_make_doc("fail.docx")]
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 1
        assert "failed" in analyses[0].summary.lower()
