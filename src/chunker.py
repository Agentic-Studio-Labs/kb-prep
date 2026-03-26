"""Structure-aware chunking for parsed documents."""

from dataclasses import dataclass

from .models import Chunk, ChunkSet, ParsedDocument


@dataclass
class DocumentChunker:
    """Create heading-aware chunks from parsed documents."""

    target_words: int = 220
    overlap_words: int = 40

    def chunk_document(self, doc: ParsedDocument) -> ChunkSet:
        chunks: list[Chunk] = []
        current_heading_path: list[str] = []
        section_paragraphs: list[tuple[int, str]] = []

        def flush_section() -> None:
            if not section_paragraphs:
                return
            chunks.extend(self._chunk_section(doc, current_heading_path, section_paragraphs, len(chunks)))

        for para in doc.paragraphs:
            if para.is_heading:
                flush_section()
                current_heading_path = self._update_heading_path(current_heading_path, para.level, para.text)
                section_paragraphs = []
                continue
            if para.text.strip():
                section_paragraphs.append((para.index, para.text))

        flush_section()
        return ChunkSet(document_id=doc.metadata.stem, source_file=doc.metadata.filename, chunks=chunks)

    def _update_heading_path(self, current: list[str], level: int, text: str) -> list[str]:
        # Heading jumps (e.g. H1 -> H3) preserve observed levels only.
        next_path = current[: max(level - 1, 0)]
        next_path.append(text.strip())
        return next_path

    def _chunk_section(
        self,
        doc: ParsedDocument,
        heading_path: list[str],
        section_paragraphs: list[tuple[int, str]],
        chunk_offset: int,
    ) -> list[Chunk]:
        joined_words: list[tuple[int, str]] = []
        for para_index, text in section_paragraphs:
            joined_words.extend((para_index, word) for word in text.split())

        output: list[Chunk] = []
        start = 0
        while start < len(joined_words):
            end = min(start + self.target_words, len(joined_words))
            window = joined_words[start:end]
            chunk_text = " ".join(word for _, word in window)
            output.append(
                Chunk(
                    chunk_id=f"{doc.metadata.stem}::{chunk_offset + len(output):03d}",
                    document_id=doc.metadata.stem,
                    source_file=doc.metadata.filename,
                    text=chunk_text,
                    heading_path=list(heading_path),
                    start_paragraph_index=window[0][0],
                    end_paragraph_index=window[-1][0],
                    token_estimate=max(1, len(chunk_text) // 4),
                    chunk_type="section",
                    quality_flags=[],
                )
            )
            if end == len(joined_words):
                break
            start = max(end - self.overlap_words, start + 1)

        return output
