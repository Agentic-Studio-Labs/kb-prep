"""Folder structure recommendation engine.

Uses LLM analysis results to propose an anam.ai knowledge base
folder hierarchy with document assignments.
"""

import json
from typing import Optional

from anthropic import AsyncAnthropic

from .analyzer import extract_json
from .config import Config
from .models import (
    ContentAnalysis,
    FolderNode,
    FolderRecommendation,
    ParsedDocument,
)
from .prompts import RECOMMEND_FOLDERS


def _detect_format_duplicates(docs: list[ParsedDocument]) -> dict[str, str]:
    """Map duplicate filenames to their primary.

    Two files are duplicates if their stems match after normalizing
    (e.g., "4-5.FL.10 Handout B.pdf" and "4-5.FL.10 Handout B.docx").
    """
    from pathlib import Path

    stem_to_files: dict[str, list[str]] = {}
    for doc in docs:
        stem = Path(doc.metadata.filename).stem.strip().lower()
        stem_to_files.setdefault(stem, []).append(doc.metadata.filename)

    duplicates: dict[str, str] = {}
    for stem, files in stem_to_files.items():
        if len(files) > 1:
            primary = files[0]
            for dup in files[1:]:
                duplicates[dup] = primary
    return duplicates


class FolderRecommender:
    """Recommend anam.ai KB folder structure based on content analysis.

    Uses a knowledge graph (when available) for graph-based community
    detection, falling back to LLM or heuristic approaches.
    """

    def __init__(self, config: Optional[Config] = None, graph=None):
        self.config = config
        self.graph = graph  # Optional KnowledgeGraph
        self._client: Optional[AsyncAnthropic] = None

        if config and config.anthropic_api_key:
            self._client = AsyncAnthropic(api_key=config.anthropic_api_key)

    async def recommend(
        self,
        docs: list[ParsedDocument],
        analyses: list[ContentAnalysis],
    ) -> FolderRecommendation:
        """Generate folder structure recommendation.

        Priority:
        1. Graph-informed + LLM (best: uses relationship structure AND LLM naming)
        2. LLM only (good: smart naming but no structural awareness)
        3. Graph-only heuristic (decent: structural clustering, generic names)
        4. Simple heuristic (fallback: domain-based grouping)
        """
        if self.graph and not self.graph.is_empty and self._client:
            return await self._graph_llm_recommend(docs, analyses)
        elif self._client and self.config:
            return await self._llm_recommend(docs, analyses)
        elif self.graph and not self.graph.is_empty:
            return await self._graph_heuristic_recommend(docs, analyses)
        else:
            return await self._heuristic_recommend(docs, analyses)

    # ------------------------------------------------------------------
    # Graph + LLM recommendation (best quality)
    # ------------------------------------------------------------------

    async def _call_llm_for_folders(
        self,
        doc_summaries: list[dict],
    ) -> Optional[FolderRecommendation]:
        """Shared LLM call + JSON parse + tree build for folder recommendations."""
        hints = self.config.folder_hints if self.config else ""
        hint_block = f"\nDomain-specific guidance:\n{hints}\n" if hints else ""
        prompt = RECOMMEND_FOLDERS.format(
            documents_json=json.dumps(doc_summaries, indent=2),
            domain_hints=hint_block,
        )

        response = await self._client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        data = extract_json(response_text)
        if data is None:
            return None

        root = FolderNode(name="Knowledge Base", description="Root knowledge base container")
        for folder_data in data.get("folders", []):
            node = self._json_to_folder_node(folder_data)
            root.children.append(node)

        assignments = data.get("assignments", {})
        return FolderRecommendation(root=root, file_assignments=assignments)

    async def _graph_llm_recommend(
        self,
        docs: list[ParsedDocument],
        analyses: list[ContentAnalysis],
    ) -> FolderRecommendation:
        """Use graph community detection to cluster, then LLM to name folders."""
        file_clusters = self.graph.get_file_clusters()
        duplicates = _detect_format_duplicates(docs)

        # Build enriched document summaries with graph context
        doc_summaries = []
        for doc, analysis in zip(docs, analyses):
            # Find which cluster this file belongs to
            cluster_label = "Unassigned"
            for label, files in file_clusters.items():
                if doc.metadata.filename in files:
                    cluster_label = label
                    break

            entry: dict = {
                "filename": doc.metadata.filename,
                "domain": analysis.domain,
                "topics": analysis.topics,
                "audience": analysis.audience,
                "content_type": analysis.content_type,
                "summary": analysis.summary,
                "graph_cluster": cluster_label,
                "entity_count": len(analysis.entities),
                "cross_doc_connections": len(self.graph.get_cross_document_references(doc.metadata.filename)),
            }
            if doc.metadata.filename in duplicates:
                entry["duplicate_of"] = duplicates[doc.metadata.filename]
            doc_summaries.append(entry)

        result = await self._call_llm_for_folders(doc_summaries)
        if result is None:
            return await self._graph_heuristic_recommend(docs, analyses)
        if result:
            for dup, primary in duplicates.items():
                if primary in result.file_assignments:
                    result.file_assignments[dup] = result.file_assignments[primary]
        return result

    # ------------------------------------------------------------------
    # Graph-only heuristic (no LLM)
    # ------------------------------------------------------------------

    async def _graph_heuristic_recommend(
        self,
        docs: list[ParsedDocument],
        analyses: list[ContentAnalysis],
    ) -> FolderRecommendation:
        """Use graph community detection for folder structure without LLM."""
        file_clusters = self.graph.get_file_clusters()

        root = FolderNode(name="Knowledge Base", description="Root knowledge base container")
        assignments: dict[str, str] = {}

        for label, files in sorted(file_clusters.items()):
            # Clean up the label for a folder name
            folder_name = label.split(": ", 1)[-1].title() if ": " in label else label.title()
            folder_type = label.split(": ", 1)[0] if ": " in label else "general"

            folder_node = FolderNode(
                name=folder_name,
                description=f"Documents related to {folder_name.lower()} ({folder_type})",
                document_files=files,
            )
            root.children.append(folder_node)

            for f in files:
                assignments[f] = folder_name

        # Handle unassigned files
        all_assigned = set()
        for files in file_clusters.values():
            all_assigned.update(files)

        unassigned = [doc.metadata.filename for doc in docs if doc.metadata.filename not in all_assigned]
        if unassigned:
            general_node = FolderNode(
                name="General",
                description="Uncategorized documents",
                document_files=unassigned,
            )
            root.children.append(general_node)
            for f in unassigned:
                assignments[f] = "General"

        return FolderRecommendation(root=root, file_assignments=assignments)

    # ------------------------------------------------------------------
    # LLM-powered recommendation
    # ------------------------------------------------------------------

    async def _llm_recommend(
        self,
        docs: list[ParsedDocument],
        analyses: list[ContentAnalysis],
    ) -> FolderRecommendation:
        """Use LLM to design optimal folder structure."""
        duplicates = _detect_format_duplicates(docs)
        doc_summaries = []
        for doc, analysis in zip(docs, analyses):
            entry: dict = {
                "filename": doc.metadata.filename,
                "domain": analysis.domain,
                "topics": analysis.topics,
                "audience": analysis.audience,
                "content_type": analysis.content_type,
                "summary": analysis.summary,
            }
            if doc.metadata.filename in duplicates:
                entry["duplicate_of"] = duplicates[doc.metadata.filename]
            doc_summaries.append(entry)

        result = await self._call_llm_for_folders(doc_summaries)
        if result is None:
            return await self._heuristic_recommend(docs, analyses)
        if result:
            for dup, primary in duplicates.items():
                if primary in result.file_assignments:
                    result.file_assignments[dup] = result.file_assignments[primary]
        return result

    def _json_to_folder_node(self, data: dict) -> FolderNode:
        """Recursively convert JSON folder dict to FolderNode."""
        node = FolderNode(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
        )
        for child_data in data.get("children", []):
            child = self._json_to_folder_node(child_data)
            node.children.append(child)
        return node

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------

    async def _heuristic_recommend(
        self,
        docs: list[ParsedDocument],
        analyses: list[ContentAnalysis],
    ) -> FolderRecommendation:
        """Simple domain-based grouping without LLM."""
        root = FolderNode(
            name="Knowledge Base",
            description="Root knowledge base container",
        )

        # Group by domain
        by_domain: dict[str, list[tuple[ParsedDocument, ContentAnalysis]]] = {}
        for doc, analysis in zip(docs, analyses):
            domain = analysis.domain or "General"
            by_domain.setdefault(domain, []).append((doc, analysis))

        assignments: dict[str, str] = {}

        for domain, items in sorted(by_domain.items()):
            domain_node = FolderNode(
                name=domain.title(),
                description=f"Documents related to {domain}",
            )
            root.children.append(domain_node)

            # Sub-group by first topic if enough documents
            if len(items) > 3:
                by_topic: dict[str, list[tuple[ParsedDocument, ContentAnalysis]]] = {}
                for doc, analysis in items:
                    topic = analysis.topics[0] if analysis.topics else "General"
                    by_topic.setdefault(topic, []).append((doc, analysis))

                for topic, topic_items in sorted(by_topic.items()):
                    topic_node = FolderNode(
                        name=topic.title(),
                        description=f"{topic} materials in {domain}",
                    )
                    domain_node.children.append(topic_node)
                    for doc, _ in topic_items:
                        assignments[doc.metadata.filename] = f"{domain.title()}/{topic.title()}"
                        topic_node.document_files.append(doc.metadata.filename)
            else:
                for doc, _ in items:
                    assignments[doc.metadata.filename] = domain.title()
                    domain_node.document_files.append(doc.metadata.filename)

        return FolderRecommendation(root=root, file_assignments=assignments)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_assignments(
        self,
        assignments: dict[str, str],
        similarity_matrix,
        doc_labels: list[str],
    ) -> tuple[float, list[tuple[str, float]]]:
        """Validate folder assignments using silhouette analysis."""
        import numpy as np
        from sklearn.metrics import silhouette_samples, silhouette_score

        if len(set(assignments.values())) < 2 or len(doc_labels) < 3:
            return 0.0, []
        labels = [assignments.get(doc, "unknown") for doc in doc_labels]
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return 0.0, []
        numeric_labels = [unique_labels.index(lbl) for lbl in labels]
        distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)
        score = float(silhouette_score(distance_matrix, numeric_labels, metric="precomputed"))
        per_doc = silhouette_samples(distance_matrix, numeric_labels, metric="precomputed")
        misplaced = [(doc_labels[i], float(per_doc[i])) for i in range(len(per_doc)) if per_doc[i] < 0]
        return score, misplaced

    def reassign_misplaced(
        self,
        assignments: dict[str, str],
        misplaced: list[tuple[str, float]],
        similarity_matrix,
        doc_labels: list[str],
    ) -> dict[str, str]:
        """Reassign documents with negative silhouette scores to their nearest folder."""
        import numpy as np

        misplaced_files = {f for f, _ in misplaced}
        folder_centroids: dict[str, list[int]] = {}

        for i, label in enumerate(doc_labels):
            folder = assignments.get(label)
            if folder and label not in misplaced_files:
                folder_centroids.setdefault(folder, []).append(i)

        new_assignments = dict(assignments)
        for filename, _ in misplaced:
            if filename not in doc_labels:
                continue
            doc_idx = doc_labels.index(filename)
            best_folder = None
            best_sim = -1.0
            for folder, indices in folder_centroids.items():
                avg_sim = float(np.mean([similarity_matrix[doc_idx, j] for j in indices]))
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_folder = folder
            if best_folder:
                new_assignments[filename] = best_folder

        return new_assignments


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def format_folder_tree(node: FolderNode, indent: int = 0) -> str:
    """Format a folder tree as a string for display."""
    lines: list[str] = []
    prefix = "  " * indent
    icon = "📁" if node.children else "📄"

    line = f"{prefix}{icon} {node.name}"
    if node.description:
        line += f"  — {node.description}"
    lines.append(line)

    for child in node.children:
        lines.append(format_folder_tree(child, indent + 1))

    if node.document_files:
        for f in node.document_files:
            lines.append(f"{prefix}  📄 {f}")

    return "\n".join(lines)
