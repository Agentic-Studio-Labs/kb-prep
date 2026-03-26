"""Tests for folder recommendation engine."""

from pathlib import Path

from src.models import DocumentMetadata, Paragraph, ParsedDocument
from src.recommender import _detect_format_duplicates


def _make_doc(filename: str) -> ParsedDocument:
    ext = Path(filename).suffix.lstrip(".") or "md"
    return ParsedDocument(
        metadata=DocumentMetadata(file_path=f"/tmp/{filename}", file_type=ext, file_size_bytes=100),
        paragraphs=[Paragraph(text="Content", level=0, style="Normal", index=0)],
    )


def test_detect_format_duplicates_pdf_docx():
    """PDF/DOCX pairs with same stem are detected as duplicates."""
    docs = [_make_doc("lesson.pdf"), _make_doc("lesson.docx"), _make_doc("other.md")]
    dupes = _detect_format_duplicates(docs)
    assert len(dupes) == 1
    # One of the pair is primary, the other is duplicate
    assert "lesson.docx" in dupes or "lesson.pdf" in dupes


def test_detect_format_duplicates_none():
    """No duplicates when all stems are unique."""
    docs = [_make_doc("lesson.pdf"), _make_doc("budget.docx"), _make_doc("guide.md")]
    dupes = _detect_format_duplicates(docs)
    assert len(dupes) == 0


def test_reassign_misplaced():
    """Misplaced docs should be reassigned to nearest folder."""
    import numpy as np

    from src.recommender import FolderRecommender

    recommender = FolderRecommender()
    assignments = {"a.md": "Insurance", "b.md": "Insurance", "c.md": "Budgeting", "d.md": "Insurance"}
    misplaced = [("d.md", -0.15)]  # d.md has negative silhouette in Insurance

    # Similarity: d.md is more similar to c.md (Budgeting) than a.md/b.md (Insurance)
    sim = np.array(
        [
            [1.0, 0.8, 0.1, 0.2],  # a.md
            [0.8, 1.0, 0.1, 0.2],  # b.md
            [0.1, 0.1, 1.0, 0.9],  # c.md
            [0.2, 0.2, 0.9, 1.0],  # d.md
        ]
    )
    doc_labels = ["a.md", "b.md", "c.md", "d.md"]

    new_assignments = recommender.reassign_misplaced(assignments, misplaced, sim, doc_labels)
    assert new_assignments["d.md"] == "Budgeting", f"d.md should move to Budgeting, got {new_assignments['d.md']}"
