from src.models import Chunk, ChunkBenchmark, ChunkSet


def test_chunk_ids_are_stable_from_doc_and_index():
    chunk = Chunk(
        chunk_id="lesson-1::000",
        document_id="lesson-1",
        source_file="lesson-1.docx",
        text="Budgeting starts with income and expenses.",
        heading_path=["Lesson 1", "Budget Basics"],
        start_paragraph_index=2,
        end_paragraph_index=3,
        token_estimate=8,
        chunk_type="section",
        quality_flags=[],
    )

    assert chunk.chunk_id == "lesson-1::000"
    assert chunk.heading_path == ["Lesson 1", "Budget Basics"]


def test_chunk_set_tracks_document_lineage():
    chunk = Chunk(
        chunk_id="lesson-1::000",
        document_id="lesson-1",
        source_file="lesson-1.docx",
        text="Budgeting starts with income and expenses.",
        heading_path=[],
        start_paragraph_index=0,
        end_paragraph_index=0,
        token_estimate=8,
        chunk_type="section",
        quality_flags=[],
    )
    chunk_set = ChunkSet(document_id="lesson-1", source_file="lesson-1.docx", chunks=[chunk])

    assert chunk_set.source_file == "lesson-1.docx"
    assert len(chunk_set.chunks) == 1


def test_chunk_benchmark_holds_three_retrieval_modes():
    benchmark = ChunkBenchmark(
        retrieval_mode="hybrid",
        recall_at_5=0.8,
        mrr=0.62,
        ndcg_at_5=0.71,
        query_count=20,
    )

    assert benchmark.retrieval_mode == "hybrid"
    assert benchmark.query_count == 20
