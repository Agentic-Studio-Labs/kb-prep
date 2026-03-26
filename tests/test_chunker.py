from src.chunker import DocumentChunker
from src.models import DocumentMetadata, Paragraph, ParsedDocument


def test_chunker_preserves_heading_path():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/example.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Lesson 1", level=1, style="Heading 1", index=0),
            Paragraph(text="Budgeting starts with income.", level=0, style="Normal", index=1),
            Paragraph(text="Track expenses carefully.", level=0, style="Normal", index=2),
        ],
    )

    chunker = DocumentChunker(target_words=80, overlap_words=20)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) == 1
    assert chunk_set.chunks[0].heading_path == ["Lesson 1"]


def test_chunker_splits_large_sections_without_crossing_headings():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/example.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Heading A", level=1, style="Heading 1", index=0),
            Paragraph(text="One " * 120, level=0, style="Normal", index=1),
            Paragraph(text="Heading B", level=1, style="Heading 1", index=2),
            Paragraph(text="Two " * 40, level=0, style="Normal", index=3),
        ],
    )

    chunker = DocumentChunker(target_words=60, overlap_words=10)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) >= 2
    assert all("Heading B" not in c.text for c in chunk_set.chunks if c.heading_path == ["Heading A"])


def test_chunker_headless_document_still_chunks():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/no-headings.md", file_type="md"),
        paragraphs=[
            Paragraph(text="This paragraph has no heading.", level=0, style="Normal", index=0),
            Paragraph(text="Neither does this one.", level=0, style="Normal", index=1),
        ],
    )

    chunker = DocumentChunker(target_words=20, overlap_words=5)
    chunk_set = chunker.chunk_document(doc)

    assert len(chunk_set.chunks) >= 1
    assert chunk_set.chunks[0].heading_path == []
