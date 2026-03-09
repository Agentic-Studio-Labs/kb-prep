"""Tests for knowledge graph builder improvements.

Tests three changes:
1. No implicit entities from topics/key_concepts
2. Fuzzy matching threshold raised to ≥8 chars
3. Louvain community detection instead of connected components
"""

import pytest

from graph_builder import KnowledgeGraph
from models import ContentAnalysis, Entity, ParsedDocument, Relationship


def _make_doc(filename: str, text: str = "Some content") -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    from pathlib import Path

    from models import DocumentMetadata, Paragraph

    ext = Path(filename).suffix.lstrip(".") or "md"
    return ParsedDocument(
        metadata=DocumentMetadata(
            file_path=f"/tmp/{filename}",
            file_type=ext,
            file_size_bytes=len(text),
        ),
        paragraphs=[Paragraph(text=text, level=0, style="Normal", index=0)],
    )


# ------------------------------------------------------------------
# 1. No implicit entities from topics/key_concepts
# ------------------------------------------------------------------


class TestNoImplicitEntities:
    """Topics and key_concepts should NOT be added as graph entities."""

    def test_topics_not_added_as_entities(self):
        graph = KnowledgeGraph()
        doc = _make_doc("lesson.md")
        analysis = ContentAnalysis(
            topics=["budgeting", "saving", "credit"],
            entities=[Entity(name="SMART Goals", entity_type="concept", source_file="lesson.md")],
        )
        graph.add_analysis(doc, analysis)

        entity_names = {e.name for e in graph._entities.values()}
        assert "SMART Goals" in entity_names, "Explicit entity should be present"
        assert "budgeting" not in entity_names, "Topic should NOT be added as entity"
        assert "saving" not in entity_names, "Topic should NOT be added as entity"
        assert "credit" not in entity_names, "Topic should NOT be added as entity"

    def test_key_concepts_not_added_as_entities(self):
        graph = KnowledgeGraph()
        doc = _make_doc("lesson.md")
        analysis = ContentAnalysis(
            key_concepts=["interest rates", "compound interest"],
            entities=[Entity(name="Banking", entity_type="topic", source_file="lesson.md")],
        )
        graph.add_analysis(doc, analysis)

        entity_names = {e.name for e in graph._entities.values()}
        assert "Banking" in entity_names
        assert "interest rates" not in entity_names, "Key concept should NOT be added as entity"
        assert "compound interest" not in entity_names, "Key concept should NOT be added as entity"

    def test_entity_count_matches_explicit_only(self):
        graph = KnowledgeGraph()
        doc = _make_doc("lesson.md")
        analysis = ContentAnalysis(
            topics=["topic1", "topic2", "topic3"],
            key_concepts=["concept1", "concept2"],
            entities=[
                Entity(name="Entity A", entity_type="concept", source_file="lesson.md"),
                Entity(name="Entity B", entity_type="skill", source_file="lesson.md"),
            ],
        )
        graph.add_analysis(doc, analysis)
        assert len(graph._entities) == 2, "Only explicit entities should be in the graph"


# ------------------------------------------------------------------
# 2. Fuzzy matching threshold raised to ≥8 chars
# ------------------------------------------------------------------


class TestFuzzyMatchingThreshold:
    """Substring fallback should only activate for names ≥8 chars."""

    def test_short_name_no_fuzzy_match(self):
        """'Goal' (4 chars) should NOT fuzzy-match 'Goal Setting'."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Goal Setting", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        # "Goal" is only 4 chars — should NOT match "Goal Setting"
        result = graph._find_entity_key("Goal")
        assert result is None, "Short name 'Goal' should not fuzzy-match 'Goal Setting'"

    def test_medium_name_no_fuzzy_match(self):
        """'Credit' (6 chars) should NOT fuzzy-match 'Credit Score'."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Credit Score", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("Credit")
        assert result is None, "6-char name should not fuzzy-match"

    def test_long_name_does_fuzzy_match(self):
        """'Financial Planning' (18 chars) SHOULD fuzzy-match 'Financial Planning Basics'."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Financial Planning Basics", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("Financial Planning")
        assert result is not None, "Long name should still fuzzy-match"

    def test_exact_match_always_works(self):
        """Exact match should work regardless of length."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Goal", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("Goal")
        assert result is not None, "Exact match should always work"

    def test_exact_match_case_insensitive(self):
        """Exact match should be case-insensitive."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="SMART Goals", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("smart goals")
        assert result is not None, "Case-insensitive exact match should work"


# ------------------------------------------------------------------
# 3. Louvain community detection
# ------------------------------------------------------------------


class TestLouvainClustering:
    """Clustering should use Louvain, producing finer-grained communities."""

    def _build_two_cluster_graph(self) -> KnowledgeGraph:
        """Build a graph with two distinct clusters connected by a weak link."""
        graph = KnowledgeGraph()

        # Cluster A: tightly connected
        for name in ["Budgeting Basics", "Saving Strategies", "Financial Goals"]:
            graph._add_entity(Entity(name=name, entity_type="concept", source_file="a.md"), "a.md")
        graph._add_relationship(Relationship(source="Budgeting Basics", target="Saving Strategies", rel_type="related_to", source_file="a.md"))
        graph._add_relationship(Relationship(source="Saving Strategies", target="Financial Goals", rel_type="related_to", source_file="a.md"))
        graph._add_relationship(Relationship(source="Financial Goals", target="Budgeting Basics", rel_type="related_to", source_file="a.md"))

        # Cluster B: tightly connected
        for name in ["Career Exploration", "College Preparation", "Dream Job Research"]:
            graph._add_entity(Entity(name=name, entity_type="concept", source_file="b.md"), "b.md")
        graph._add_relationship(Relationship(source="Career Exploration", target="College Preparation", rel_type="related_to", source_file="b.md"))
        graph._add_relationship(Relationship(source="College Preparation", target="Dream Job Research", rel_type="related_to", source_file="b.md"))
        graph._add_relationship(Relationship(source="Dream Job Research", target="Career Exploration", rel_type="related_to", source_file="b.md"))

        # Weak bridge between clusters
        graph._add_relationship(Relationship(source="Financial Goals", target="Career Exploration", rel_type="influences", source_file="a.md"))

        return graph

    def test_finds_multiple_clusters(self):
        """Louvain should find 2 communities, not 1 connected component."""
        graph = self._build_two_cluster_graph()
        clusters = graph.find_clusters()
        assert len(clusters) >= 2, (
            f"Expected ≥2 clusters from two distinct communities, got {len(clusters)}"
        )

    def test_file_clusters_separates_files(self):
        """Files from different communities should be in different clusters."""
        graph = self._build_two_cluster_graph()
        file_clusters = graph.get_file_clusters()
        assert len(file_clusters) >= 2, (
            f"Expected ≥2 file clusters, got {len(file_clusters)}: {file_clusters}"
        )
