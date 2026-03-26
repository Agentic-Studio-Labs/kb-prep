"""Tests for knowledge graph builder improvements.

Tests three changes:
1. No implicit entities from topics/key_concepts
2. Fuzzy matching threshold raised to ≥8 chars
3. Louvain community detection instead of connected components
"""

from src.graph_builder import KnowledgeGraph
from src.models import ContentAnalysis, Entity, ParsedDocument, Relationship


def _make_doc(filename: str, text: str = "Some content") -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    from pathlib import Path

    from src.models import DocumentMetadata, Paragraph

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
    """Cosine similarity on char n-grams should resolve entity names."""

    def test_short_name_cosine_match(self):
        """'Goal' should cosine-match 'Goal Setting' (high char n-gram overlap)."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Goal Setting", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("Goal")
        assert result is not None, "Short name 'Goal' should match 'Goal Setting' via cosine"

    def test_medium_name_cosine_match(self):
        """'Credit' should cosine-match 'Credit Score' (high char n-gram overlap)."""
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Credit Score", entity_type="concept", source_file="a.md"),
            "a.md",
        )

        result = graph._find_entity_key("Credit")
        assert result is not None, "6-char name should match via cosine"

    def test_long_name_does_fuzzy_match(self):
        """'Financial Planning' (18 chars) SHOULD cosine-match 'Financial Planning Basics'."""
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
# 2b. Cosine entity resolution (char n-gram TF-IDF)
# ------------------------------------------------------------------


class TestCosineEntityResolution:
    def test_morphological_match(self):
        graph = KnowledgeGraph()
        graph._add_entity(Entity(name="Budgeting Basics", entity_type="concept", source_file="a.md"), "a.md")
        result = graph._find_entity_key("Budget")
        assert result is not None, "'Budget' should fuzzy-match 'Budgeting Basics'"

    def test_no_false_positive(self):
        graph = KnowledgeGraph()
        graph._add_entity(Entity(name="Digital Safety", entity_type="concept", source_file="a.md"), "a.md")
        result = graph._find_entity_key("Financial Literacy")
        assert result is None, "Unrelated entities should not match"

    def test_deterministic_best_match(self):
        graph = KnowledgeGraph()
        graph._add_entity(
            Entity(name="Financial Literacy Standards", entity_type="concept", source_file="a.md"), "a.md"
        )
        graph._add_entity(Entity(name="Digital Financial Safety", entity_type="concept", source_file="b.md"), "b.md")
        result = graph._find_entity_key("Financial Literacy")
        entity = graph._entities[result]
        assert "Financial Literacy Standards" in entity.name


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
        graph._add_relationship(
            Relationship(
                source="Budgeting Basics", target="Saving Strategies", rel_type="related_to", source_file="a.md"
            )
        )
        graph._add_relationship(
            Relationship(
                source="Saving Strategies", target="Financial Goals", rel_type="related_to", source_file="a.md"
            )
        )
        graph._add_relationship(
            Relationship(source="Financial Goals", target="Budgeting Basics", rel_type="related_to", source_file="a.md")
        )

        # Cluster B: tightly connected
        for name in ["Career Exploration", "College Preparation", "Dream Job Research"]:
            graph._add_entity(Entity(name=name, entity_type="concept", source_file="b.md"), "b.md")
        graph._add_relationship(
            Relationship(
                source="Career Exploration", target="College Preparation", rel_type="related_to", source_file="b.md"
            )
        )
        graph._add_relationship(
            Relationship(
                source="College Preparation", target="Dream Job Research", rel_type="related_to", source_file="b.md"
            )
        )
        graph._add_relationship(
            Relationship(
                source="Dream Job Research", target="Career Exploration", rel_type="related_to", source_file="b.md"
            )
        )

        # Weak bridge between clusters
        graph._add_relationship(
            Relationship(
                source="Financial Goals", target="Career Exploration", rel_type="influences", source_file="a.md"
            )
        )

        return graph

    def test_finds_multiple_clusters(self):
        """Louvain should find 2 communities, not 1 connected component."""
        graph = self._build_two_cluster_graph()
        clusters = graph.find_clusters()
        assert len(clusters) >= 2, f"Expected ≥2 clusters from two distinct communities, got {len(clusters)}"

    def test_file_clusters_separates_files(self):
        """Files from different communities should be in different clusters."""
        graph = self._build_two_cluster_graph()
        file_clusters = graph.get_file_clusters()
        assert len(file_clusters) >= 2, f"Expected ≥2 file clusters, got {len(file_clusters)}: {file_clusters}"


# ------------------------------------------------------------------
# 4. Spectral clustering
# ------------------------------------------------------------------


class TestSpectralClustering:
    def test_deterministic_clustering(self):
        import numpy as np

        from src.graph_builder import spectral_cluster

        sim = np.array(
            [
                [1.0, 0.8, 0.1, 0.1],
                [0.8, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.9],
                [0.1, 0.1, 0.9, 1.0],
            ]
        )
        clusters_1 = spectral_cluster(sim)
        clusters_2 = spectral_cluster(sim)
        assert clusters_1 == clusters_2, "Should be deterministic"
        assert len(clusters_1) == 2, f"Should find 2 clusters, got {len(clusters_1)}"


# ------------------------------------------------------------------
# 5. PageRank + bridge entities
# ------------------------------------------------------------------


def test_bipartite_similarity():
    graph = KnowledgeGraph()
    graph._add_entity(Entity(name="Budgeting", entity_type="concept", source_file="a.md"), "a.md")
    graph._add_entity(Entity(name="Budgeting", entity_type="concept", source_file="b.md"), "b.md")
    graph._add_entity(Entity(name="Insurance", entity_type="concept", source_file="c.md"), "c.md")
    sim = graph.get_bipartite_doc_similarity()
    assert sim is not None
    files = sorted(graph._file_entities.keys())
    a_idx = files.index("a.md")
    b_idx = files.index("b.md")
    c_idx = files.index("c.md")
    assert sim[a_idx, b_idx] > sim[a_idx, c_idx]


def test_blend_similarity_matrices():
    import numpy as np

    from src.graph_builder import blend_similarity

    tfidf_sim = np.array([[1.0, 0.5], [0.5, 1.0]])
    entity_sim = np.array([[1.0, 0.8], [0.8, 1.0]])
    blended = blend_similarity(tfidf_sim, entity_sim, alpha=0.7)
    expected_01 = 0.7 * 0.5 + 0.3 * 0.8  # 0.59
    assert abs(blended[0, 1] - expected_01) < 0.01


def test_pagerank_returns_rankings():
    graph = KnowledgeGraph()
    leaf_names = ["Budgeting", "Saving", "Credit", "Insurance", "Investing"]
    for name in leaf_names:
        graph._add_entity(Entity(name=name, entity_type="concept", source_file="a.md"), "a.md")
    graph._add_entity(Entity(name="Financial Literacy", entity_type="concept", source_file="a.md"), "a.md")
    for name in leaf_names:
        graph._add_relationship(
            Relationship(source="Financial Literacy", target=name, rel_type="covers", source_file="a.md")
        )
    rankings = graph.get_pagerank()
    assert len(rankings) > 0
    # PageRank rewards nodes with many inbound edges; the 5 leaf concepts each
    # receive a link from "Financial Literacy", so they rank above it.
    top_entity = max(rankings, key=rankings.get)
    top_entity_name = graph._entities[top_entity].name if top_entity in graph._entities else top_entity
    leaf_keys = {e.key for e in graph._entities.values() if e.name in leaf_names}
    assert top_entity in leaf_keys, f"Top PageRank entity should be a leaf concept, got {top_entity_name!r}"


def test_file_clusters_use_pagerank_labels():
    """Cluster labels should use the highest-PageRank entity, not the first."""
    graph = KnowledgeGraph()
    # Create a cluster where "Insurance" has highest PageRank (most inbound links)
    for name in ["Risk Assessment", "Coverage Types", "Premium Calculation"]:
        graph._add_entity(Entity(name=name, entity_type="concept", source_file="a.md"), "a.md")
    graph._add_entity(Entity(name="Insurance", entity_type="concept", source_file="a.md"), "a.md")
    # All point to Insurance (giving it highest PageRank)
    for name in ["Risk Assessment", "Coverage Types", "Premium Calculation"]:
        graph._add_relationship(Relationship(source=name, target="Insurance", rel_type="part_of", source_file="a.md"))

    clusters = graph.get_file_clusters()
    # The label should contain "Insurance" (highest PR), not "Coverage Types" or "Premium Calculation"
    labels = list(clusters.keys())
    assert any("Insurance" in label for label in labels), f"Expected 'Insurance' in labels, got {labels}"
