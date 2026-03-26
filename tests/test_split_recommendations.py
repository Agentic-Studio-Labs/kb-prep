import numpy as np

from src.export import build_manifest_data
from src.models import (
    ContentAnalysis,
    CorpusAnalysis,
    DocumentMetadata,
    FolderNode,
    FolderRecommendation,
    Paragraph,
    ParsedDocument,
    ScoreCard,
    SplitRecommendation,
)
from src.scorer import QualityScorer, generate_split_recommendations


def test_broad_documents_emit_split_recommendation():
    broad_document = ParsedDocument(
        metadata=DocumentMetadata(file_path="/tmp/broad.md", file_type="md"),
        paragraphs=[
            Paragraph(text="Saving", level=1, style="Heading 1", index=0),
            Paragraph(text="Save money each week for goals.", level=0, style="Normal", index=1),
            Paragraph(text="Credit", level=1, style="Heading 1", index=2),
            Paragraph(text="Credit cards and loans require repayment with interest.", level=0, style="Normal", index=3),
            Paragraph(text="Careers", level=1, style="Heading 1", index=4),
            Paragraph(text="Career planning connects education to future income.", level=0, style="Normal", index=5),
        ],
    )
    scorer = QualityScorer()
    card = scorer.score(broad_document)

    recommendations = generate_split_recommendations([broad_document], [card])

    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.source_file == "broad.md"
    assert "topic" in rec.reason.lower() or "broad" in rec.reason.lower() or "entropy" in rec.reason.lower()


def test_manifest_exports_split_recommendations():
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
    recommendation = FolderRecommendation(
        root=FolderNode(name="Root", description=""), file_assignments={"test.md": "Budget"}
    )
    split_recommendations = [
        SplitRecommendation(
            source_file="test.md",
            reason="High topic entropy (0.76) - document covers many disparate topics",
            proposed_boundaries=[2, 5],
            suggested_titles=["Budgeting Basics", "Credit and Debt"],
        )
    ]

    data = build_manifest_data(
        docs=[doc],
        analyses=[analysis],
        cards=[card],
        corpus_analysis=ca,
        recommendation=recommendation,
        split_recommendations=split_recommendations,
    )

    assert "split_recommendations" in data
    assert data["split_recommendations"][0]["source_file"] == "test.md"
    assert data["split_recommendations"][0]["proposed_boundaries"] == [2, 5]
