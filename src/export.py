"""Export analysis results as machine-readable JSON.

Produces two formats:
- Sidecar files: one .meta.json per document, co-located with the fixed Markdown
- Corpus manifest: a single manifest.json with all documents, scores, and graph
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    ChunkBenchmark,
    ChunkSet,
    ContentAnalysis,
    CorpusAnalysis,
    DocMetrics,
    ParsedDocument,
    ScoreCard,
    SplitRecommendation,
)


def write_sidecar(
    output_dir: str,
    filename_stem: str,
    doc: ParsedDocument,
    analysis: ContentAnalysis,
    card: ScoreCard,
    metrics: Optional[DocMetrics],
) -> str:
    """Write a .meta.json sidecar file alongside the fixed Markdown."""
    retrieval_quality_gate = _build_retrieval_quality_gate(doc, card)
    data = {
        "ingestgate_version": "0.1.0",
        "source_file": doc.metadata.filename,
        "output_file": f"{filename_stem}.md",
        "analysis": {
            "domain": analysis.domain,
            "topics": analysis.topics,
            "audience": analysis.audience,
            "content_type": analysis.content_type,
            "key_concepts": analysis.key_concepts,
            "suggested_tags": analysis.suggested_tags,
            "summary": analysis.summary,
        },
        "scores": {
            "overall": card.overall_score,
            "readiness": card.readiness.value,
            "criteria": {
                r.category: {
                    "score": r.score,
                    "weight": r.weight,
                    "issues": len(r.issues),
                }
                for r in card.results
            },
        },
        "metrics": _serialize_metrics(metrics),
        "entities": [{"name": e.name, "type": e.entity_type, "description": e.description} for e in analysis.entities],
        "relationships": [
            {"source": r.source, "target": r.target, "type": r.rel_type, "context": r.context}
            for r in analysis.relationships
        ],
        "retrieval_quality_gate": retrieval_quality_gate,
    }

    out_path = Path(output_dir) / f"{filename_stem}.meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


def write_chunk_sidecar(output_dir: str, chunk_set: ChunkSet) -> str:
    """Write a per-document .chunks.json sidecar."""
    data = {
        "schema_version": "2.0",
        "document_id": chunk_set.document_id,
        "source_file": chunk_set.source_file,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "heading_path": c.heading_path,
                "start_paragraph_index": c.start_paragraph_index,
                "end_paragraph_index": c.end_paragraph_index,
                "token_estimate": c.token_estimate,
                "chunk_type": c.chunk_type,
                "quality_flags": c.quality_flags,
                "metadata": c.metadata,
            }
            for c in chunk_set.chunks
        ],
    }
    out_path = Path(output_dir) / f"{chunk_set.document_id}.chunks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


def build_manifest_data(
    docs: list[ParsedDocument],
    analyses: list[ContentAnalysis],
    cards: list[ScoreCard],
    corpus_analysis: CorpusAnalysis,
    graph=None,
    chunk_sets: Optional[list[ChunkSet]] = None,
    benchmarks: Optional[list[ChunkBenchmark]] = None,
    split_recommendations: Optional[list[SplitRecommendation]] = None,
) -> dict:
    """Build the corpus manifest as a JSON-serializable dict.

    Use this directly for --json-output (no file I/O needed).
    Use write_manifest() to also write it to disk.
    """
    chunk_sets = chunk_sets or []
    benchmarks = benchmarks or []
    split_recommendations = split_recommendations or []

    readiness_dist: dict[str, int] = {}
    for card in cards:
        r = card.readiness.value
        readiness_dist[r] = readiness_dist.get(r, 0) + 1

    total_entities = 0
    total_relationships = 0
    cross_doc_edges = 0
    if graph and not graph.is_empty:
        summary = graph.summarize()
        total_entities = summary.total_entities
        total_relationships = summary.total_relationships
        cross_doc_edges = summary.cross_document_edges

    avg_score = sum(c.overall_score for c in cards) / len(cards) if cards else 0.0

    chunk_count_by_file = {cs.source_file: len(cs.chunks) for cs in chunk_sets}
    retrieval_mode_distribution: dict[str, int] = {}

    doc_entries = []
    for doc, analysis, card in zip(docs, analyses, cards):
        rqg = _build_retrieval_quality_gate(doc, card)
        mode = rqg["retrieval_mode_hint"]["recommended_mode"]
        retrieval_mode_distribution[mode] = retrieval_mode_distribution.get(mode, 0) + 1
        doc_entries.append(
            {
                "source_file": doc.metadata.filename,
                "overall_score": round(card.overall_score, 1),
                "readiness": card.readiness.value,
                "domain": analysis.domain,
                "topics": analysis.topics,
                "entity_count": len(analysis.entities),
                "relationship_count": len(analysis.relationships),
                "chunk_count": chunk_count_by_file.get(doc.metadata.filename, 0),
                "retrieval_quality_gate": rqg,
            }
        )

    kg_data = None
    if graph and not graph.is_empty:
        kg_entities = [
            {"name": e.name, "type": e.entity_type, "source_file": e.source_file, "description": e.description}
            for e in graph._entities.values()
        ]
        kg_relationships = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.rel_type,
                "source_file": r.source_file,
                "context": r.context,
            }
            for r in graph._relationships
        ]
        kg_clusters = []
        for i, cluster in enumerate(graph.find_clusters()):
            kg_clusters.append({"label": f"Cluster {i + 1}", "entities": cluster})
        kg_data = {
            "entities": kg_entities,
            "relationships": kg_relationships,
            "clusters": kg_clusters,
        }

    sim_data = None
    if hasattr(corpus_analysis, "similarity_matrix") and corpus_analysis.similarity_matrix.size > 0:
        n = corpus_analysis.similarity_matrix.shape[0]
        if n <= 100:
            sim_data = {
                "labels": corpus_analysis.doc_labels,
                "matrix": corpus_analysis.similarity_matrix.tolist(),
            }

    return {
        "schema_version": "2.0",
        "ingestgate_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "total_documents": len(docs),
            "total_chunks": sum(len(cs.chunks) for cs in chunk_sets),
            "avg_score": round(avg_score, 1),
            "readiness_distribution": readiness_dist,
            "retrieval_mode_distribution": retrieval_mode_distribution,
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "cross_document_edges": cross_doc_edges,
        },
        "documents": doc_entries,
        "knowledge_graph": kg_data,
        "similarity_matrix": sim_data,
        "benchmarks": [
            {
                "retrieval_mode": b.retrieval_mode,
                "recall_at_5": b.recall_at_5,
                "mrr": b.mrr,
                "ndcg_at_5": b.ndcg_at_5,
                "query_count": b.query_count,
                "notes": b.notes,
            }
            for b in benchmarks
        ],
        "split_recommendations": [
            {
                "source_file": rec.source_file,
                "reason": rec.reason,
                "proposed_boundaries": rec.proposed_boundaries,
                "suggested_titles": rec.suggested_titles,
            }
            for rec in split_recommendations
        ],
    }


def write_manifest(
    output_dir: str,
    docs: list[ParsedDocument],
    analyses: list[ContentAnalysis],
    cards: list[ScoreCard],
    corpus_analysis: CorpusAnalysis,
    graph=None,
    chunk_sets: Optional[list[ChunkSet]] = None,
    benchmarks: Optional[list[ChunkBenchmark]] = None,
    split_recommendations: Optional[list[SplitRecommendation]] = None,
) -> str:
    """Write a corpus-level manifest.json to the output directory root."""
    data = build_manifest_data(
        docs,
        analyses,
        cards,
        corpus_analysis,
        graph,
        chunk_sets=chunk_sets,
        benchmarks=benchmarks,
        split_recommendations=split_recommendations,
    )
    out_path = Path(output_dir) / "manifest.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


def _serialize_metrics(metrics: Optional[DocMetrics]) -> Optional[dict]:
    """Serialize DocMetrics to a JSON-safe dict."""
    if metrics is None:
        return None
    return {
        "entropy": metrics.entropy,
        "coherence": metrics.coherence,
        "readability_grade": metrics.readability_grade,
        "self_retrieval_score": metrics.self_retrieval_score,
        "info_density": metrics.info_density,
        "topic_boundaries": metrics.topic_boundaries,
    }


def _build_retrieval_quality_gate(doc: ParsedDocument, card: ScoreCard) -> dict:
    body_count = len(doc.body_paragraphs)
    heading_count = len(doc.headings)
    total_words = sum(p.word_count for p in doc.paragraphs)
    short_label_count = sum(1 for p in doc.body_paragraphs if _looks_like_short_label(p.text))
    short_label_ratio = (short_label_count / body_count) if body_count else 0.0
    heading_density = (body_count / heading_count) if heading_count else None

    parse_fidelity_warning = any(
        "Low parse fidelity:" in issue.message and issue.severity.value in {"warning", "critical"}
        for issue in card.all_issues
    )
    parse_fidelity_template_note = any(
        "Low parse fidelity (template-like document)" in issue.message for issue in card.all_issues
    )
    heading_warning = any(
        issue.category == "heading_quality" and "no headings" in issue.message.lower() for issue in card.all_issues
    )

    layout_heavy_pdf = doc.metadata.file_type == "pdf" and (
        short_label_ratio >= 0.30 or (heading_count >= 10 and total_words < 500)
    )
    template_like = (
        parse_fidelity_template_note
        or _filename_has_template_marker(doc.metadata.filename)
        or short_label_ratio >= 0.50
    )

    if parse_fidelity_warning or layout_heavy_pdf:
        recommended_mode = "multimodal_or_ocr_review"
        confidence = "high" if parse_fidelity_warning else "medium"
    elif template_like:
        recommended_mode = "hybrid_sparse_template"
        confidence = "medium"
    elif heading_warning and total_words > 400:
        recommended_mode = "hybrid_with_structure_rewrite"
        confidence = "medium"
    else:
        recommended_mode = "text_hybrid_default"
        confidence = "high"

    reasons: list[str] = []
    if parse_fidelity_warning:
        reasons.append("parse_fidelity_warning")
    if parse_fidelity_template_note:
        reasons.append("template_like_sparse_content")
    if layout_heavy_pdf:
        reasons.append("layout_heavy_pdf")
    if heading_warning:
        reasons.append("low_structural_headings")
    if not reasons:
        reasons.append("clean_text_for_standard_retrieval")

    return {
        "retrieval_mode_hint": {
            "recommended_mode": recommended_mode,
            "confidence": confidence,
            "reasons": reasons,
        },
        "modality_readiness": {
            "text_only_ready": recommended_mode == "text_hybrid_default",
            "layout_heavy_pdf": layout_heavy_pdf,
            "template_like_document": template_like,
            "parse_fidelity_warning": parse_fidelity_warning,
            "parse_fidelity_template_note": parse_fidelity_template_note,
        },
        "evidence": {
            "file_type": doc.metadata.file_type,
            "total_words": total_words,
            "heading_count": heading_count,
            "body_paragraph_count": body_count,
            "short_label_ratio": round(short_label_ratio, 3),
            "heading_density": round(heading_density, 2) if heading_density is not None else None,
        },
    }


def _looks_like_short_label(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    words = len(normalized.split())
    if words > 4:
        return False
    if normalized.endswith(":"):
        return True
    return normalized.isupper() and words <= 4


def _filename_has_template_marker(filename: str) -> bool:
    lowered = filename.lower()
    markers = ("tracker", "rubric", "template", "worksheet", "handout")
    return any(marker in lowered for marker in markers)
