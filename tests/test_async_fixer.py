"""Tests for async DocumentFixer.

Verifies that the fixer uses AsyncAnthropic, async methods,
semaphore-based concurrency, and preserves fix logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def test_fix_is_async():
    """DocumentFixer.fix is a coroutine."""
    import inspect

    from src.fixer import DocumentFixer

    assert inspect.iscoroutinefunction(DocumentFixer.fix)


def test_call_llm_is_async():
    """DocumentFixer._call_llm is a coroutine."""
    import inspect

    from src.fixer import DocumentFixer

    assert inspect.iscoroutinefunction(DocumentFixer._call_llm)


def test_fix_methods_are_async():
    """All _fix_* methods are coroutines."""
    import inspect

    from src.fixer import DocumentFixer

    for name in [
        "_fix_dangling_reference",
        "_fix_generic_heading",
        "_fix_long_paragraph",
        "_fix_acronym",
        "_generate_filename",
    ]:
        method = getattr(DocumentFixer, name)
        assert inspect.iscoroutinefunction(method), f"{name} should be async"


def test_call_llm_uses_semaphore():
    """_call_llm uses a semaphore for concurrency control."""
    from src.config import Config
    from src.fixer import DocumentFixer

    config = Config(anthropic_api_key="test-key", concurrency=3)

    with patch("src.fixer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        msg = MagicMock()
        msg.content = [MagicMock(text="fixed text")]
        mock_instance.messages.create = AsyncMock(return_value=msg)

        fixer = DocumentFixer(config)
        assert hasattr(fixer, "_semaphore")
        assert fixer._semaphore._value == 3

        result = asyncio.run(fixer._call_llm("test prompt"))
        assert result == "fixed text"
        mock_instance.messages.create.assert_called_once()


def test_fix_applies_dangling_reference():
    """fix() awaits _fix_dangling_reference for self_containment issues."""
    from src.config import Config
    from src.fixer import DocumentFixer
    from src.models import (
        DocumentMetadata,
        Issue,
        Paragraph,
        ParsedDocument,
        ScoreCard,
        ScoringResult,
        Severity,
    )

    config = Config(anthropic_api_key="test-key", output_dir="/tmp/test-fix-out")
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/test.docx", file_type="docx"),
        paragraphs=[
            Paragraph(text="See Unit 2 for details.", level=0, style="Normal", index=0),
        ],
    )
    scorecard = ScoreCard(
        file_path="/tmp/test.docx",
        results=[
            ScoringResult(
                category="self_containment",
                label="Self-Containment",
                score=50.0,
                weight=0.2,
                issues=[
                    Issue(severity=Severity.WARNING, category="self_containment", message="Dangling ref", location=0)
                ],
            ),
        ],
    )

    with patch("src.fixer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        msg = MagicMock()
        msg.content = [MagicMock(text="The details from Unit 2 are as follows.")]
        mock_instance.messages.create = AsyncMock(return_value=msg)

        fixer = DocumentFixer(config)
        report = asyncio.run(fixer.fix(doc, scorecard))

        assert len(report.actions) == 1
        assert report.actions[0].category == "self_containment"


def test_fix_generates_filename():
    """fix() awaits _generate_filename for filename_quality issues."""
    from src.config import Config
    from src.fixer import DocumentFixer
    from src.models import (
        DocumentMetadata,
        Issue,
        Paragraph,
        ParsedDocument,
        ScoreCard,
        ScoringResult,
        Severity,
    )

    config = Config(anthropic_api_key="test-key", output_dir="/tmp/test-fix-out")
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/doc1.docx", file_type="docx"),
        paragraphs=[
            Paragraph(text="Introduction to Fractions", level=1, style="Heading 1", index=0),
            Paragraph(text="Fractions are parts of a whole.", level=0, style="Normal", index=1),
        ],
    )
    scorecard = ScoreCard(
        file_path="/tmp/doc1.docx",
        results=[
            ScoringResult(
                category="filename_quality",
                label="Filename Quality",
                score=30.0,
                weight=0.1,
                issues=[Issue(severity=Severity.WARNING, category="filename_quality", message="Bad filename")],
            ),
        ],
    )

    with patch("src.fixer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        msg = MagicMock()
        msg.content = [MagicMock(text="introduction-to-fractions")]
        mock_instance.messages.create = AsyncMock(return_value=msg)

        fixer = DocumentFixer(config)
        report = asyncio.run(fixer.fix(doc, scorecard))

        assert report.new_filename == "introduction-to-fractions"


def test_sync_helpers_unchanged():
    """_find_paragraph and _get_graph_context_for_paragraph stay synchronous."""
    import inspect

    from src.fixer import DocumentFixer

    assert not inspect.iscoroutinefunction(DocumentFixer._find_paragraph)
    assert not inspect.iscoroutinefunction(DocumentFixer._get_graph_context_for_paragraph)
    assert not inspect.iscoroutinefunction(DocumentFixer._write_fixed)


def test_fix_acronym_skips_when_already_defined():
    from src.config import Config
    from src.fixer import DocumentFixer
    from src.models import Paragraph

    config = Config(anthropic_api_key="test-key", output_dir="/tmp/test-fix-out")
    with patch("src.fixer.AsyncAnthropic") as MockClient:
        mock_instance = MockClient.return_value
        msg = MagicMock()
        msg.content = [MagicMock(text="Specific, Measurable, Achievable, Relevant, and Time-bound")]
        mock_instance.messages.create = AsyncMock(return_value=msg)
        fixer = DocumentFixer(config)

        paragraphs = [
            Paragraph(
                text="SMART (Specific, Measurable, Achievable, Relevant, Time-bound) goals help.",
                level=0,
                style="Normal",
                index=0,
            )
        ]
        action = asyncio.run(fixer._fix_acronym(paragraphs, "SMART", paragraphs[0].text))
        assert action is None
        assert paragraphs[0].text.count("SMART") == 1
