import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import ParsedDocument, DocumentMetadata, Paragraph


MOCK_ANALYSIS_JSON = '{"domain":"education","topics":["math"],"audience":"students","content_type":"lesson","key_concepts":["fractions"],"suggested_tags":["math"],"summary":"A math lesson.","entities":[{"name":"Fractions","type":"concept","description":"Number parts"}],"relationships":[]}'


def _mock_response(text):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_docs(n=3):
    return [
        ParsedDocument(
            metadata=DocumentMetadata(file_path=f"/tmp/doc{i}.docx", file_type="docx"),
            paragraphs=[Paragraph(text=f"Content about math topic {i}.", level=0, style="Normal", index=0)],
        )
        for i in range(n)
    ]


def test_full_analyze_pipeline_concurrent():
    """analyze_and_build_graph runs concurrently and builds correct graph."""
    from analyzer import ContentAnalyzer

    config = Config(anthropic_api_key="test-key", concurrency=2)

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock = MockClient.return_value
        mock.messages.create = AsyncMock(return_value=_mock_response(MOCK_ANALYSIS_JSON))

        analyzer = ContentAnalyzer(config)
        docs = _make_docs(5)
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 5
        assert not graph.is_empty
        assert mock.messages.create.call_count == 5
        # All analyses should have domain "education"
        assert all(a.domain == "education" for a in analyses)
        # Graph should have entities — only explicit entities from analysis
        # (each doc produces the same entity "Fractions", deduplicated to 1)
        summary = graph.summarize()
        assert summary.total_entities >= 1


def test_concurrent_analysis_with_failures():
    """Pipeline handles mixed success/failure gracefully."""
    from analyzer import ContentAnalyzer

    config = Config(anthropic_api_key="test-key", concurrency=3)

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:  # Third call fails
            raise Exception("Simulated API error")
        return _mock_response(MOCK_ANALYSIS_JSON)

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock = MockClient.return_value
        mock.messages.create = AsyncMock(side_effect=mock_create)

        analyzer = ContentAnalyzer(config)
        docs = _make_docs(5)
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 5
        # One analysis should be failed
        failed = [a for a in analyses if "failed" in a.summary.lower()]
        succeeded = [a for a in analyses if a.domain == "education"]
        assert len(failed) == 1
        assert len(succeeded) == 4


def test_semaphore_limits_concurrency():
    """Semaphore actually limits concurrent LLM calls."""
    from analyzer import ContentAnalyzer

    config = Config(anthropic_api_key="test-key", concurrency=2)
    max_concurrent = 0
    current_concurrent = 0

    original_mock = _mock_response(MOCK_ANALYSIS_JSON)

    async def mock_create(**kwargs):
        nonlocal max_concurrent, current_concurrent
        current_concurrent += 1
        max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.01)  # Simulate API latency
        current_concurrent -= 1
        return original_mock

    with patch("analyzer.AsyncAnthropic") as MockClient:
        mock = MockClient.return_value
        mock.messages.create = AsyncMock(side_effect=mock_create)

        analyzer = ContentAnalyzer(config)
        docs = _make_docs(6)
        analyses, graph = asyncio.run(analyzer.analyze_and_build_graph(docs))

        assert len(analyses) == 6
        # Semaphore should limit to at most 2 concurrent calls
        assert max_concurrent <= 2, f"Max concurrent was {max_concurrent}, expected <= 2"
