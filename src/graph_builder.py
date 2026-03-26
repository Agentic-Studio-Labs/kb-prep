"""Knowledge graph builder and query engine.

Builds an in-memory networkx graph from entity/relationship data
extracted during LLM analysis. Used by the fixer (cross-document
reference resolution) and scorer (orphan/completeness detection).
"""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering as SklearnSpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .models import (
    ContentAnalysis,
    Entity,
    GraphSummary,
    ParsedDocument,
    Relationship,
)


class KnowledgeGraph:
    """In-memory knowledge graph built from document analysis."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}  # key → Entity
        self._name_index: dict[str, str] = {}  # normalized_name → entity_key (for O(1) exact lookup)
        self._relationships: list[Relationship] = []
        self._file_entities: dict[str, set[str]] = defaultdict(set)  # file → {entity keys}
        self._cached_components: Optional[list[set]] = None  # cached connected components
        self._entity_vectorizer = None
        self._entity_matrix = None
        self._entity_keys_list: list[str] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_analysis(self, doc: ParsedDocument, analysis: ContentAnalysis):
        """Merge entities and relationships from one document's analysis."""
        source_file = doc.metadata.filename

        for entity in analysis.entities:
            entity.source_file = entity.source_file or source_file
            self._add_entity(entity, source_file)

        for rel in analysis.relationships:
            rel.source_file = rel.source_file or source_file
            self._add_relationship(rel)

    def _add_entity(self, entity: Entity, source_file: str):
        """Add or merge an entity into the graph."""
        key = entity.key
        if key not in self._entities:
            self._entities[key] = entity
            self._name_index[entity.name.lower().strip()] = key
            self._cached_components = None  # invalidate
            self._entity_matrix = None  # invalidate cosine index
            self.graph.add_node(
                key,
                name=entity.name,
                entity_type=entity.entity_type,
                source_file=entity.source_file,
                description=entity.description,
            )
        else:
            # Merge: keep the longer description
            existing = self._entities[key]
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
                self.graph.nodes[key]["description"] = entity.description

        self._file_entities[source_file].add(key)

    def _add_relationship(self, rel: Relationship):
        """Add a relationship edge to the graph."""
        # Normalize entity references to keys
        source_key = self._find_entity_key(rel.source)
        target_key = self._find_entity_key(rel.target)

        if not source_key:
            # Create a placeholder entity for unresolved references
            placeholder = Entity(
                name=rel.source,
                entity_type="unresolved",
                source_file=rel.source_file,
            )
            self._add_entity(placeholder, rel.source_file)
            source_key = placeholder.key

        if not target_key:
            placeholder = Entity(
                name=rel.target,
                entity_type="unresolved",
                source_file=rel.source_file,
            )
            self._add_entity(placeholder, rel.source_file)
            target_key = placeholder.key

        self._cached_components = None  # invalidate
        self.graph.add_edge(
            source_key,
            target_key,
            rel_type=rel.rel_type,
            source_file=rel.source_file,
            context=rel.context,
        )
        self._relationships.append(rel)

    def _find_entity_key(self, name: str, threshold: float = 0.4) -> Optional[str]:
        """Find an entity key by name using TF-IDF cosine similarity."""
        normalized = name.lower().strip()
        # O(1) exact match via index
        if normalized in self._name_index:
            return self._name_index[normalized]
        # Cosine similarity on character n-grams
        if not self._entities:
            return None
        self._rebuild_entity_index()
        query_vec = self._entity_vectorizer.transform([name])
        sims = sklearn_cosine_similarity(query_vec, self._entity_matrix)[0]
        best_idx = sims.argmax()
        if sims[best_idx] >= threshold:
            return self._entity_keys_list[best_idx]
        return None

    def _rebuild_entity_index(self):
        """Rebuild the TF-IDF entity index when entities change."""
        if self._entity_matrix is not None:
            return  # already up to date
        entity_texts = [f"{e.name} {e.description}" for e in self._entities.values()]
        self._entity_keys_list = list(self._entities.keys())
        self._entity_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._entity_matrix = self._entity_vectorizer.fit_transform(entity_texts)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_related_content(self, entity_name: str, max_hops: int = 2) -> list[dict]:
        """Get content related to an entity within N hops.

        Returns a list of dicts with entity info and relationship path.
        Used by the fixer for cross-document reference resolution.
        """
        key = self._find_entity_key(entity_name)
        if not key:
            return []

        results = []
        visited = set()

        def traverse(current_key: str, depth: int, path: list[str]):
            if depth > max_hops or current_key in visited:
                return
            visited.add(current_key)

            node = self.graph.nodes.get(current_key, {})
            results.append(
                {
                    "entity": node.get("name", current_key),
                    "type": node.get("entity_type", "unknown"),
                    "source_file": node.get("source_file", ""),
                    "description": node.get("description", ""),
                    "path": list(path),
                    "depth": depth,
                }
            )

            # Traverse outgoing edges
            for _, neighbor, data in self.graph.out_edges(current_key, data=True):
                traverse(
                    neighbor,
                    depth + 1,
                    path + [f"--{data.get('rel_type', '?')}-->"],
                )

            # Traverse incoming edges
            for neighbor, _, data in self.graph.in_edges(current_key, data=True):
                traverse(
                    neighbor,
                    depth + 1,
                    path + [f"<--{data.get('rel_type', '?')}--"],
                )

        traverse(key, 0, [])
        return results

    def get_entities_for_file(self, filename: str) -> list[Entity]:
        """Get all entities associated with a specific file."""
        keys = self._file_entities.get(filename, [])
        return [self._entities[k] for k in keys if k in self._entities]

    def get_cross_document_references(self, filename: str) -> list[dict]:
        """Find entities in this file that are also referenced in other files.

        Used by the fixer to resolve "see Unit 2" type references.
        """
        file_keys = set(self._file_entities.get(filename, []))
        cross_refs = []

        for key in file_keys:
            # Check if this entity appears in other files
            for other_file, other_keys in self._file_entities.items():
                if other_file == filename:
                    continue
                if key in other_keys:
                    entity = self._entities[key]
                    cross_refs.append(
                        {
                            "entity": entity.name,
                            "type": entity.entity_type,
                            "also_in": other_file,
                        }
                    )

        return cross_refs

    def find_orphan_references(self) -> list[str]:
        """Find entities that are referenced in relationships but never defined
        in any document (type='unresolved').

        Used by the scorer for knowledge completeness checks.
        """
        orphans = []
        for key, entity in self._entities.items():
            if entity.entity_type == "unresolved":
                orphans.append(entity.name)
        return sorted(set(orphans))

    def _get_components(self) -> list[set]:
        """Return communities via Louvain detection (cached, invalidated on graph mutation)."""
        if self._cached_components is None:
            if self.graph.number_of_nodes() == 0:
                self._cached_components = []
            else:
                undirected = self.graph.to_undirected()
                self._cached_components = list(nx.community.louvain_communities(undirected, seed=42))
        return self._cached_components

    def find_clusters(self) -> list[list[str]]:
        """Detect topic communities using connected components on undirected projection.

        Returns groups of entity names that form natural clusters.
        """
        components = self._get_components()
        if not components:
            return []

        clusters = []
        for component in components:
            if len(component) < 2:
                continue
            names = [self.graph.nodes[key].get("name", key) for key in component]
            clusters.append(sorted(names))

        # Sort by size (largest first)
        clusters.sort(key=len, reverse=True)
        return clusters

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summarize(self) -> GraphSummary:
        """Generate summary statistics for the graph."""
        entity_types: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            entity_types[entity.entity_type] += 1

        rel_types: dict[str, int] = defaultdict(int)
        cross_doc = 0
        for u, v, data in self.graph.edges(data=True):
            rel_types[data.get("rel_type", "unknown")] += 1
            u_file = self.graph.nodes[u].get("source_file", "")
            v_file = self.graph.nodes[v].get("source_file", "")
            if u_file and v_file and u_file != v_file:
                cross_doc += 1

        return GraphSummary(
            total_entities=len(self._entities),
            total_relationships=self.graph.number_of_edges(),
            entity_types=dict(entity_types),
            relationship_types=dict(rel_types),
            orphan_references=self.find_orphan_references(),
            clusters=self.find_clusters(),
            cross_document_edges=cross_doc,
        )

    def get_bipartite_doc_similarity(self) -> Optional[np.ndarray]:
        """Project document-entity bipartite graph to document-document similarity."""
        files = sorted(self._file_entities.keys())
        if len(files) < 2:
            return None
        entity_keys = list(self._entities.keys())
        if not entity_keys:
            return None
        n_files = len(files)
        n_entities = len(entity_keys)
        adj = np.zeros((n_files, n_entities))
        for i, f in enumerate(files):
            for j, ek in enumerate(entity_keys):
                if ek in self._file_entities.get(f, set()):
                    adj[i, j] = 1.0
        doc_counts = adj.sum(axis=0)
        idf = np.log((n_files + 1) / (doc_counts + 1))
        weighted_adj = adj * idf
        sim = weighted_adj @ weighted_adj.T
        diag = np.sqrt(np.diag(sim))
        diag[diag == 0] = 1.0
        sim = sim / np.outer(diag, diag)
        return sim

    @property
    def is_empty(self) -> bool:
        return self.graph.number_of_nodes() == 0

    # ------------------------------------------------------------------
    # Centrality / ranking
    # ------------------------------------------------------------------

    def get_pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Rank entities by PageRank centrality."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, alpha=alpha)

    def get_bridge_entities(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Find bridge entities (high betweenness centrality)."""
        if self.graph.number_of_nodes() < 3:
            return []
        bc = nx.betweenness_centrality(self.graph)
        ranked = sorted(bc.items(), key=lambda x: -x[1])[:top_n]
        return [(self._entities[k].name if k in self._entities else k, v) for k, v in ranked if v > 0]


# ------------------------------------------------------------------
# Module-level spectral clustering utility
# ------------------------------------------------------------------


def spectral_cluster(similarity_matrix: np.ndarray, min_clusters: int = 2) -> list[list[int]]:
    """Cluster documents using spectral analysis of similarity matrix."""
    n = similarity_matrix.shape[0]
    if n < 2:
        return [list(range(n))]
    W = similarity_matrix.copy()
    np.fill_diagonal(W, 0)
    W[W < 0.05] = 0
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues = np.linalg.eigvalsh(L)
    max_k = min(10, n // 2, n - 1)
    if max_k < 2:
        max_k = 2
    gaps = np.diff(eigenvalues[1 : max_k + 1])
    n_clusters = int(np.argmax(gaps) + 2) if len(gaps) > 0 else 2
    n_clusters = max(min_clusters, min(n_clusters, n // 2))
    sc = SklearnSpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=42, assign_labels="kmeans"
    )
    sim = similarity_matrix.copy()
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, 0, 1)
    labels = sc.fit_predict(sim)
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return sorted(clusters.values(), key=len, reverse=True)


def blend_similarity(tfidf_sim: np.ndarray, entity_sim: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Blend TF-IDF and entity-based similarity matrices."""
    return alpha * tfidf_sim + (1 - alpha) * entity_sim
