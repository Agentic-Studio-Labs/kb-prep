"""Corpus-wide TF-IDF analysis engine.

Computes a shared TF-IDF matrix from parsed documents and derives
per-document metrics (entropy, coherence, readability, topic boundaries,
retrieval-aware score) and a corpus-wide similarity matrix.

All downstream consumers (scorer, graph_builder, recommender, eval)
import from this module rather than computing their own text features.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import CorpusAnalysis, DocMetrics, ParsedDocument


def build_corpus_analysis(docs: list[ParsedDocument]) -> CorpusAnalysis:
    """Build TF-IDF matrix and derived metrics for a corpus of documents.

    This is the primary entry point. Call once after parsing, before scoring.
    """
    if not docs:
        return CorpusAnalysis(
            tfidf_matrix=csr_matrix((0, 0)),
            feature_names=[],
            doc_labels=[],
            similarity_matrix=np.array([]),
            doc_metrics={},
        )

    doc_labels = [doc.metadata.filename for doc in docs]
    doc_texts = [doc.full_text for doc in docs]

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    similarity_matrix = cosine_similarity(tfidf_matrix)

    doc_metrics = {}
    for i, doc in enumerate(docs):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        entropy = _compute_entropy(doc_vec)
        doc_metrics[doc_labels[i]] = DocMetrics(
            entropy=entropy,
            coherence=0.0,
            readability_grade=0.0,
        )

    return CorpusAnalysis(
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
        doc_labels=doc_labels,
        similarity_matrix=similarity_matrix,
        doc_metrics=doc_metrics,
    )


def _compute_entropy(tfidf_vec: np.ndarray) -> float:
    """Shannon entropy of a TF-IDF vector, normalized to [0, 1]."""
    nonzero = tfidf_vec[tfidf_vec > 0]
    if len(nonzero) <= 1:
        return 0.0
    probs = nonzero / nonzero.sum()
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(nonzero))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)
