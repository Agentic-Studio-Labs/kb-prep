import json
import tempfile
from pathlib import Path

import numpy as np

from src.export import build_manifest_data, write_chunk_sidecar
from src.models import (
    Chunk,
    ChunkSet,
    ContentAnalysis,
    CorpusAnalysis,
    DocumentMetadata,
    FolderNode,
    FolderRecommendation,
    Paragraph,
    ParsedDocument,
    ScoreCard,
)


def test_manifest_includes_chunk_summary():
    doc = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/test.md", file_type="md"),
        paragraphs=[Paragraph(text="Budgeting matters.", level=0, style="Normal", index=0)],
    )
    analysis = ContentAnalysis(summary="Budgeting lesson")
    card = ScoreCard(file_path="/tmp/test.md", overall_score=80.0)
    ca = CorpusAnalysis(
        tfidf_matrix=np.eye(1),
        feature_names=["budget"],
        doc_labels=["test.md"],
        similarity_matrix=np.eye(1),
    )
    rec = FolderRecommendation(root=FolderNode(name="Root", description=""), file_assignments={"test.md": "Budget"})
    chunk_set = ChunkSet(
        document_id="test",
        source_file="test.md",
        chunks=[
            Chunk(
                chunk_id="test::000",
                document_id="test",
                source_file="test.md",
                text="Budgeting matters.",
                heading_path=[],
                start_paragraph_index=0,
                end_paragraph_index=0,
                token_estimate=2,
            )
        ],
    )

    data = build_manifest_data(
        docs=[doc],
        analyses=[analysis],
        cards=[card],
        corpus_analysis=ca,
        recommendation=rec,
        graph=None,
        chunk_sets=[chunk_set],
        benchmarks=[],
    )

    assert data["corpus"]["total_chunks"] == 1
    assert data["documents"][0]["chunk_count"] == 1
    assert data["schema_version"] == "2.0"


def test_write_chunk_sidecar_creates_chunks_json():
    chunk_set = ChunkSet(
        document_id="test",
        source_file="test.md",
        chunks=[
            Chunk(
                chunk_id="test::000",
                document_id="test",
                source_file="test.md",
                text="Budgeting matters.",
                heading_path=[],
                start_paragraph_index=0,
                end_paragraph_index=0,
                token_estimate=2,
            )
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        out = write_chunk_sidecar(td, chunk_set)
        assert out.endswith(".chunks.json")
        content = json.loads(Path(out).read_text(encoding="utf-8"))
        assert content["document_id"] == "test"
        assert len(content["chunks"]) == 1
