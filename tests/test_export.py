"""Tests for metadata export (sidecar JSON + corpus manifest)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.export import write_manifest, write_sidecar
from src.models import (
    ContentAnalysis,
    CorpusAnalysis,
    DocMetrics,
    DocumentMetadata,
    Entity,
    Issue,
    Paragraph,
    ParsedDocument,
    Relationship,
    ScoreCard,
    ScoringResult,
    Severity,
)


def _make_doc(filename="test.docx"):
    ext = Path(filename).suffix.lstrip(".") or "md"
    return ParsedDocument(
        metadata=DocumentMetadata(file_path=f"/tmp/{filename}", file_type=ext, file_size_bytes=100),
        paragraphs=[Paragraph(text="Test content.", level=0, style="Normal", index=0)],
    )


def _make_analysis():
    return ContentAnalysis(
        domain="education",
        topics=["math", "fractions"],
        audience="students",
        content_type="lesson",
        key_concepts=["denominators"],
        suggested_tags=["math"],
        summary="A lesson about fractions.",
        entities=[Entity(name="Fractions", entity_type="concept", source_file="test.docx", description="Math concept")],
        relationships=[
            Relationship(
                source="Fractions", target="Denominators", rel_type="related_to", source_file="test.docx", context="..."
            )
        ],
    )


def _make_card(filename="test.docx", score=75.0):
    card = ScoreCard(file_path=f"/tmp/{filename}")
    card.results = [
        ScoringResult(category="self_containment", label="Self-Containment", score=score, weight=0.20, issues=[]),
    ]
    card.overall_score = score
    return card


def _make_metrics():
    return DocMetrics(
        entropy=0.42,
        coherence=0.71,
        readability_grade=6.2,
        self_retrieval_score=0.65,
        info_density=[0.3, 0.5],
        topic_boundaries=[4],
    )


def test_write_sidecar_creates_file():
    with tempfile.TemporaryDirectory() as td:
        path = write_sidecar(td, "test-doc", _make_doc(), _make_analysis(), _make_card(), _make_metrics())
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["ragprep_version"] == "0.1.0"


def test_sidecar_schema_keys():
    with tempfile.TemporaryDirectory() as td:
        path = write_sidecar(td, "test-doc", _make_doc(), _make_analysis(), _make_card(), _make_metrics())
        data = json.loads(Path(path).read_text())
        required = {
            "ragprep_version",
            "source_file",
            "output_file",
            "analysis",
            "scores",
            "metrics",
            "entities",
            "relationships",
            "retrieval_quality_gate",
        }
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"


def test_sidecar_handles_missing_metrics():
    with tempfile.TemporaryDirectory() as td:
        path = write_sidecar(td, "test-doc", _make_doc(), _make_analysis(), _make_card(), None)
        data = json.loads(Path(path).read_text())
        assert data["metrics"] is None


def test_sidecar_unicode_preserved():
    with tempfile.TemporaryDirectory() as td:
        analysis = _make_analysis()
        analysis.topics = ["matemáticas", "frações"]
        path = write_sidecar(td, "test-doc", _make_doc(), analysis, _make_card(), None)
        content = Path(path).read_text(encoding="utf-8")
        assert "matemáticas" in content
        assert "frações" in content


def test_write_manifest_creates_file():
    with tempfile.TemporaryDirectory() as td:
        from scipy.sparse import csr_matrix

        ca = CorpusAnalysis(
            tfidf_matrix=csr_matrix(np.array([[1, 0], [0, 1]])),
            feature_names=["a", "b"],
            doc_labels=["test.docx"],
            similarity_matrix=np.eye(1),
            doc_metrics={"test.docx": _make_metrics()},
        )
        path = write_manifest(td, [_make_doc()], [_make_analysis()], [_make_card()], ca, None)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["corpus"]["total_documents"] == 1


def test_manifest_corpus_stats():
    with tempfile.TemporaryDirectory() as td:
        from scipy.sparse import csr_matrix

        docs = [_make_doc("a.docx"), _make_doc("b.docx")]
        analyses = [_make_analysis(), _make_analysis()]
        cards = [_make_card("a.docx", 80.0), _make_card("b.docx", 60.0)]
        ca = CorpusAnalysis(
            tfidf_matrix=csr_matrix(np.eye(2)),
            feature_names=["x"],
            doc_labels=["a.docx", "b.docx"],
            similarity_matrix=np.eye(2),
        )
        path = write_manifest(td, docs, analyses, cards, ca, None)
        data = json.loads(Path(path).read_text())
        assert data["corpus"]["total_documents"] == 2
        assert data["corpus"]["avg_score"] == 70.0
        assert "retrieval_mode_distribution" in data["corpus"]


def test_sidecar_retrieval_quality_gate_for_template_note():
    with tempfile.TemporaryDirectory() as td:
        doc = _make_doc("goal-tracker.pdf")
        doc.metadata.file_type = "pdf"
        card = _make_card("goal-tracker.pdf")
        card.results = [
            ScoringResult(
                category="structure",
                label="Structure",
                score=60.0,
                weight=0.1,
                issues=[
                    Issue(
                        severity=Severity.INFO,
                        category="structure",
                        message="Low parse fidelity (template-like document): only 25 words extracted from 53 KB file",
                    )
                ],
            )
        ]
        path = write_sidecar(td, "tracker", doc, _make_analysis(), card, _make_metrics())
        data = json.loads(Path(path).read_text())
        rqg = data["retrieval_quality_gate"]
        assert rqg["retrieval_mode_hint"]["recommended_mode"] == "hybrid_sparse_template"
        assert rqg["modality_readiness"]["template_like_document"] is True


def test_manifest_similarity_matrix_skipped_for_large_corpus():
    with tempfile.TemporaryDirectory() as td:
        from scipy.sparse import csr_matrix

        n = 101
        docs = [_make_doc(f"doc{i}.md") for i in range(n)]
        analyses = [_make_analysis() for _ in range(n)]
        cards = [_make_card(f"doc{i}.md") for i in range(n)]
        ca = CorpusAnalysis(
            tfidf_matrix=csr_matrix(np.eye(n)),
            feature_names=["x"],
            doc_labels=[f"doc{i}.md" for i in range(n)],
            similarity_matrix=np.eye(n),
        )
        path = write_manifest(td, docs, analyses, cards, ca, None)
        data = json.loads(Path(path).read_text())
        assert data["similarity_matrix"] is None


def test_manifest_knowledge_graph_section():
    with tempfile.TemporaryDirectory() as td:
        from scipy.sparse import csr_matrix

        from src.graph_builder import KnowledgeGraph

        graph = KnowledgeGraph()
        graph._add_entity(Entity(name="Budget", entity_type="concept", source_file="a.md"), "a.md")
        graph._add_entity(Entity(name="Saving", entity_type="skill", source_file="a.md"), "a.md")
        graph._add_relationship(
            Relationship(source="Budget", target="Saving", rel_type="related_to", source_file="a.md")
        )

        ca = CorpusAnalysis(
            tfidf_matrix=csr_matrix(np.eye(1)),
            feature_names=["x"],
            doc_labels=["a.md"],
            similarity_matrix=np.eye(1),
        )
        path = write_manifest(td, [_make_doc("a.md")], [_make_analysis()], [_make_card("a.md")], ca, graph)
        data = json.loads(Path(path).read_text())
        assert data["knowledge_graph"] is not None
        assert len(data["knowledge_graph"]["entities"]) == 2
        assert len(data["knowledge_graph"]["relationships"]) == 1
