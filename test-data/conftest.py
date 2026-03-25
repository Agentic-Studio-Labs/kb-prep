"""Shared pytest fixtures for the eval suite.

Loads corpora from test-data/corpora/ (created by setup.py) and provides
a session-scoped TF-IDF matrix and engine adapter.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CORPORA_DIR = Path(__file__).parent / "corpora"


# ---------------------------------------------------------------------------
# pytest CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--corpus", default="newsgroups", help="Default corpus for parametric tests")


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        pytest.skip(f"Dataset not found: {path}. Run setup.py first.")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def newsgroups_data():
    """20 Newsgroups — text, labels, categories."""
    records = _load_jsonl(CORPORA_DIR / "newsgroups" / "data.jsonl")
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    categories = [r["category"] for r in records]
    return {"texts": texts, "labels": labels, "categories": categories, "records": records}


@pytest.fixture(scope="session")
def squad_data():
    """SQuAD 1.1 — paragraphs with questions and answers."""
    return _load_jsonl(CORPORA_DIR / "squad" / "data.jsonl")


@pytest.fixture(scope="session")
def cuad_data():
    """CUAD — contract texts."""
    return _load_jsonl(CORPORA_DIR / "cuad" / "data.jsonl")


@pytest.fixture(scope="session")
def hotpotqa_data():
    """HotpotQA — multi-hop QA with supporting facts."""
    return _load_jsonl(CORPORA_DIR / "hotpotqa" / "data.jsonl")


@pytest.fixture(scope="session")
def arxiv_data():
    """arXiv sample — titles and abstracts."""
    return _load_jsonl(CORPORA_DIR / "arxiv_sample" / "data.jsonl")


@pytest.fixture(scope="session")
def choi_data():
    """Choi segmentation — documents with known boundaries."""
    return _load_jsonl(CORPORA_DIR / "choi" / "data.jsonl")


@pytest.fixture(scope="session")
def beir_scifact():
    """BEIR SciFact — corpus and queries."""
    corpus = _load_jsonl(CORPORA_DIR / "beir" / "scifact" / "corpus.jsonl")
    queries_path = CORPORA_DIR / "beir" / "scifact" / "queries.jsonl"
    queries = _load_jsonl(queries_path) if queries_path.exists() else []
    return {"corpus": corpus, "queries": queries}


@pytest.fixture(scope="session")
def beir_nfcorpus():
    """BEIR NFCorpus."""
    corpus = _load_jsonl(CORPORA_DIR / "beir" / "nfcorpus" / "corpus.jsonl")
    queries_path = CORPORA_DIR / "beir" / "nfcorpus" / "queries.jsonl"
    queries = _load_jsonl(queries_path) if queries_path.exists() else []
    return {"corpus": corpus, "queries": queries}


@pytest.fixture(scope="session")
def beir_trec_covid():
    """BEIR TREC-COVID."""
    corpus = _load_jsonl(CORPORA_DIR / "beir" / "trec-covid" / "corpus.jsonl")
    queries_path = CORPORA_DIR / "beir" / "trec-covid" / "queries.jsonl"
    queries = _load_jsonl(queries_path) if queries_path.exists() else []
    return {"corpus": corpus, "queries": queries}


@pytest.fixture(scope="session")
def fb15k237_data():
    """FB15k-237 knowledge graph triples."""
    data_dir = CORPORA_DIR / "fb15k237"
    if not data_dir.exists():
        pytest.skip("FB15k-237 not found. Run setup.py first.")
    triples = []
    for filename in ["train.txt", "valid.txt", "test.txt"]:
        fpath = data_dir / filename
        if fpath.exists():
            for line in fpath.read_text().strip().split("\n"):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples


@pytest.fixture(scope="session")
def sts_data():
    """STS Benchmark — sentence pairs with similarity scores."""
    return _load_jsonl(CORPORA_DIR / "sts" / "data.jsonl")


@pytest.fixture(scope="session")
def leipzig_er_data():
    """Leipzig ER benchmarks."""
    data_dir = CORPORA_DIR / "leipzig_er"
    if not data_dir.exists():
        pytest.skip("Leipzig ER not found. Run setup.py first.")
    datasets = {}
    for ds_name in ["Abt-Buy", "Amazon-Google", "DBLP-ACM", "DBLP-Scholar"]:
        ds_dir = data_dir / ds_name
        if not ds_dir.exists():
            continue
        csv_files = list(ds_dir.rglob("*.csv"))
        if csv_files:
            datasets[ds_name] = ds_dir
    return datasets


# ---------------------------------------------------------------------------
# TF-IDF fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def newsgroups_tfidf(newsgroups_data):
    """TF-IDF matrix for 20 Newsgroups."""
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(newsgroups_data["texts"])
    return {
        "matrix": matrix,
        "vectorizer": vectorizer,
        "texts": newsgroups_data["texts"],
        "labels": newsgroups_data["labels"],
    }


# ---------------------------------------------------------------------------
# Engine adapter — wraps src.* functions into the API the tests expect
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine():
    """Adapter that wraps kb-prep's src modules into the eval test API."""
    try:
        from src.corpus_analyzer import (
            _bm25_score,
            _compute_coherence,
            _compute_entropy,
            _compute_readability,
            _compute_self_retrieval_score,
            _compute_topic_boundaries,
            build_corpus_analysis,
            rocchio_expand_query,
            select_overlap_sentences,
        )
        from src.graph_builder import (
            KnowledgeGraph,
            blend_similarity,
            spectral_cluster,
        )
        from src.models import (
            DocumentMetadata,
            Entity,
            Paragraph,
            ParsedDocument,
            Relationship,
        )
    except ImportError as e:
        pytest.fail(f"Engine not found. Install kb-prep first: {e}")

    class EngineAdapter:
        """Wraps kb-prep functions into the API expected by eval tests."""

        # --- Layer 1: Information-Theoretic ---

        def shannon_entropy(self, text: str) -> float:
            """Compute Shannon entropy of a single document's TF-IDF vector."""
            vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            try:
                matrix = vectorizer.fit_transform([text])
            except ValueError:
                return 0.0
            vec = matrix[0].toarray().flatten()
            return _compute_entropy(vec)

        def shannon_entropy_from_vector(self, tfidf_vec) -> float:
            """Compute entropy from a pre-computed TF-IDF vector."""
            if hasattr(tfidf_vec, "toarray"):
                vec = tfidf_vec.toarray().flatten()
            else:
                vec = np.asarray(tfidf_vec).flatten()
            return _compute_entropy(vec)

        def heading_coherence(self, heading: str, body: str) -> float:
            """Compute coherence between a heading and body text."""
            return _compute_coherence([(heading, body)])

        def readability(self, text: str) -> float:
            """Compute Flesch-Kincaid grade level."""
            return _compute_readability(text)

        # --- Layer 2: Spectral Graph ---

        def entity_cosine_similarity(self, pairs: list[tuple[str, str]]) -> list[float]:
            """Compute cosine similarity between entity description pairs."""
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            all_texts = [t for pair in pairs for t in pair]
            try:
                matrix = vectorizer.fit_transform(all_texts)
            except ValueError:
                return [0.0] * len(pairs)
            sims = []
            for i in range(0, len(all_texts), 2):
                sim = cosine_similarity(matrix[i], matrix[i + 1])[0][0]
                sims.append(float(sim))
            return sims

        def spectral_cluster(self, similarity_matrix: np.ndarray, k: int = None) -> np.ndarray:
            """Run spectral clustering on a similarity matrix."""
            from sklearn.cluster import SpectralClustering

            n = similarity_matrix.shape[0]
            if k is None:
                k = min(20, n // 2)
            sim = similarity_matrix.copy()
            np.fill_diagonal(sim, 1.0)
            sim = np.clip(sim, 0, 1)
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )
            return sc.fit_predict(sim)

        def pagerank(self, graph, alpha: float = 0.85) -> dict:
            """Run PageRank on a networkx graph."""
            import networkx as nx

            return nx.pagerank(graph, alpha=alpha)

        # --- Layer 3: Semantic Chunking ---

        def texttile(self, paragraphs: list[str], block_size: int = 3) -> list[int]:
            """Find topic boundaries using TextTiling."""
            return _compute_topic_boundaries(paragraphs, block_size=block_size)

        def select_overlap(self, sentences: list[str], budget: int = 100) -> list[str]:
            """Select information-dense overlap sentences."""
            return select_overlap_sentences(sentences, budget=budget)

        # --- Layer 4: Retrieval ---

        def bm25_search(self, query: str, documents: list[str], top_k: int = 10) -> list[int]:
            """BM25 search returning ranked document indices."""
            scores = _bm25_score(query, documents)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            return ranked[:top_k]

        def bm25_scores(self, query: str, documents: list[str]) -> list[float]:
            """BM25 scores for all documents."""
            return _bm25_score(query, documents)

        def rocchio_expand(self, query: str, corpus: list[str], **kwargs) -> str:
            """Expand query using Rocchio pseudo-relevance feedback."""
            return rocchio_expand_query(query, corpus, **kwargs)

        def silhouette_score(self, distance_matrix: np.ndarray, labels) -> float:
            """Compute silhouette score for a clustering."""
            from sklearn.metrics import silhouette_score as sk_silhouette

            return float(sk_silhouette(distance_matrix, labels, metric="precomputed"))

        # --- Utilities ---

        def compute_tfidf(self, texts: list[str]) -> tuple:
            """Compute TF-IDF matrix for a corpus."""
            vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
            matrix = vectorizer.fit_transform(texts)
            return matrix, vectorizer

        def compute_similarity(self, tfidf_matrix) -> np.ndarray:
            """Compute cosine similarity matrix."""
            return cosine_similarity(tfidf_matrix)

    return EngineAdapter()


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    for marker in ["layer1", "layer2", "layer3", "layer4", "layer5", "cross_layer", "slow"]:
        config.addinivalue_line("markers", f"{marker}: {marker} tests")
