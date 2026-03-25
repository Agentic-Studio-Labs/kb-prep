"""Corpus-wide TF-IDF analysis engine.

Computes a shared TF-IDF matrix from parsed documents and derives
per-document metrics (entropy, coherence, readability, topic boundaries,
retrieval-aware score) and a corpus-wide similarity matrix.

All downstream consumers (scorer, graph_builder, recommender, eval)
import from this module rather than computing their own text features.
"""

import itertools
import math
import re
from collections import Counter

import numpy as np
from scipy.signal import argrelmin
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import CorpusAnalysis, DocMetrics, ParsedDocument


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

    max_df = 0.95 if len(docs) > 1 else 1.0
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=max_df)
    try:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
    except ValueError:
        # All terms pruned (e.g., identical documents) — retry with max_df=1.0
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=1.0)
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    similarity_matrix = cosine_similarity(tfidf_matrix)

    doc_metrics = {}
    for i, doc in enumerate(docs):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        entropy = _compute_entropy(doc_vec)

        pairs = _extract_heading_content_pairs(doc)
        coherence = _compute_coherence(pairs)

        readability_grade = _compute_readability(doc_texts[i])

        para_texts = [p.text for p in doc.paragraphs if p.text.strip()]
        topic_boundaries = _compute_topic_boundaries(para_texts)

        self_retrieval = _compute_self_retrieval_score(
            doc_idx=i,
            doc_label=doc_labels[i],
            tfidf_matrix=tfidf_matrix,
            feature_names=feature_names,
            all_doc_texts=doc_texts,
            all_doc_labels=doc_labels,
        )

        info_density = _compute_info_density(doc)

        doc_metrics[doc_labels[i]] = DocMetrics(
            entropy=entropy,
            coherence=coherence,
            readability_grade=readability_grade,
            info_density=info_density,
            topic_boundaries=topic_boundaries,
            self_retrieval_score=self_retrieval,
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


# ---------------------------------------------------------------------------
# Task 4: Readability Scoring (Flesch-Kincaid)
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Estimate syllable count using vowel group heuristic."""
    word = word.lower().strip()
    if not word:
        return 1
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    count = len(re.findall(r"[aeiouy]+", word))
    return max(1, count)


def _compute_readability(text: str) -> float:
    """Flesch-Kincaid Grade Level."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    n_sentences = len(sentences)
    n_words = len(words)
    n_syllables = sum(_count_syllables(w) for w in words)
    grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    return max(0.0, grade)


# ---------------------------------------------------------------------------
# Task 5: TF-IDF Coherence (Heading vs Content)
# ---------------------------------------------------------------------------


def _compute_coherence(heading_content_pairs: list[tuple[str, str]]) -> float:
    """Average cosine similarity between heading text and content below it."""
    if not heading_content_pairs:
        return 0.0
    pairs_with_content = [(h, c) for h, c in heading_content_pairs if h.strip() and c.strip()]
    if not pairs_with_content:
        return 0.0
    all_texts = []
    for h, c in pairs_with_content:
        all_texts.extend([h, c])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        matrix = vec.fit_transform(all_texts)
    except ValueError:
        return 0.0
    similarities = []
    for i in range(0, len(all_texts), 2):
        sim = cosine_similarity(matrix[i], matrix[i + 1])[0][0]
        similarities.append(sim)
    return float(np.mean(similarities))


def _extract_heading_content_pairs(doc: ParsedDocument) -> list[tuple[str, str]]:
    """Extract (heading_text, content_below) pairs from a document."""
    pairs = []
    paragraphs = doc.paragraphs
    for i, para in enumerate(paragraphs):
        if not para.is_heading:
            continue
        content_parts = []
        for j in range(i + 1, len(paragraphs)):
            if paragraphs[j].is_heading:
                break
            content_parts.append(paragraphs[j].text)
        if content_parts:
            pairs.append((para.text, " ".join(content_parts)))
    return pairs


# ---------------------------------------------------------------------------
# Task 6: TextTiling Topic Boundaries
# ---------------------------------------------------------------------------


def _compute_topic_boundaries(paragraphs: list[str], block_size: int = 3) -> list[int]:
    """Find topic boundaries using TextTiling (Hearst 1997, adapted)."""
    if len(paragraphs) < 3:
        return []
    blocks = []
    for i in range(0, len(paragraphs), max(1, block_size)):
        block_text = " ".join(paragraphs[i : i + block_size])
        blocks.append(block_text)
    if len(blocks) < 2:
        return []
    vec = TfidfVectorizer(stop_words="english")
    try:
        block_matrix = vec.fit_transform(blocks)
    except ValueError:
        return []
    similarities = []
    for i in range(len(blocks) - 1):
        sim = cosine_similarity(block_matrix[i], block_matrix[i + 1])[0][0]
        similarities.append(float(sim))
    if len(similarities) < 2:
        return []
    sims = np.array(similarities)
    if len(sims) >= 7:
        try:
            from scipy.signal import savgol_filter

            window = min(5, len(sims) - (1 - len(sims) % 2)) | 1
            sims = savgol_filter(sims, window, min(2, window - 1))
        except Exception:
            pass
    valleys = argrelmin(sims, order=1)[0]
    if len(sims) > 1:
        threshold = float(np.mean(sims) - np.std(sims))
        valleys = [v for v in valleys if sims[v] < threshold]
    boundaries = [int(v * max(1, block_size) + block_size) for v in valleys]
    boundaries = [b for b in boundaries if b < len(paragraphs)]
    return boundaries


# ---------------------------------------------------------------------------
# Task 7: Retrieval-Aware Scoring
# ---------------------------------------------------------------------------


def rocchio_expand_query(
    query: str,
    corpus_texts: list[str],
    top_k: int = 3,
    n_expand: int = 5,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> str:
    """Expand a query using Rocchio pseudo-relevance feedback."""
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = corpus_texts + [query]
    try:
        matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return query
    query_vec = matrix[-1]
    doc_matrix = matrix[:-1]
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    relevant = doc_matrix[top_indices].mean(axis=0)
    corpus_mean = doc_matrix.mean(axis=0)
    expanded_vec = alpha * query_vec.toarray() + beta * np.array(relevant) - gamma * np.array(corpus_mean)
    feature_names = vectorizer.get_feature_names_out()
    query_terms = set(re.findall(r"\w+", query.lower()))
    expanded_flat = np.array(expanded_vec).flatten()
    term_scores = [
        (feature_names[i], expanded_flat[i])
        for i in range(len(feature_names))
        if feature_names[i] not in query_terms and expanded_flat[i] > 0
    ]
    term_scores.sort(key=lambda x: -x[1])
    expansion_terms = [t for t, _ in term_scores[:n_expand]]
    return query + " " + " ".join(expansion_terms)


def _bm25_score(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Compute BM25 scores for a query against all docs."""
    query_terms = re.findall(r"\w+", query.lower())
    doc_tokens = [re.findall(r"\w+", d.lower()) for d in docs]
    avg_dl = sum(len(d) for d in doc_tokens) / len(doc_tokens) if doc_tokens else 1
    n_docs = len(doc_tokens)
    df: Counter = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    scores = []
    for tokens in doc_tokens:
        dl = len(tokens)
        tf_map = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            term_df = df.get(term, 0)
            if term_df == 0:
                continue
            idf = math.log((n_docs - term_df + 0.5) / (term_df + 0.5) + 1)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores


def select_overlap_sentences(sentences: list[str], budget: int = 100) -> list[str]:
    """Pick the most informative sentences for chunk overlap.
    Scores sentences by TF-IDF vector L2 norm (information density).
    Returns highest-scoring sentences within word budget, in original order.
    """
    if not sentences:
        return []
    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform(sentences)
    except ValueError:
        return sentences[:1]
    scores = np.array(matrix.power(2).sum(axis=1)).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    selected_indices = []
    words_used = 0
    for idx, score in ranked:
        wc = len(sentences[idx].split())
        if words_used + wc > budget and selected_indices:
            break
        selected_indices.append(idx)
        words_used += wc
    return [sentences[i] for i in sorted(selected_indices)]


def _compute_info_density(doc: ParsedDocument) -> list[float]:
    """Compute information density (TF-IDF magnitude) per section."""
    sections = []
    current = []
    for para in doc.paragraphs:
        if para.is_heading and current:
            sections.append(" ".join(current))
            current = []
        elif not para.is_heading:
            current.append(para.text)
    if current:
        sections.append(" ".join(current))
    if not sections:
        return []
    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform(sections)
    except ValueError:
        return [0.0] * len(sections)
    densities = []
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().flatten()
        nonzero = row[row > 0]
        density = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0
        densities.append(density)
    return densities


def _compute_self_retrieval_score(
    doc_idx: int,
    doc_label: str,
    tfidf_matrix,
    feature_names: list[str],
    all_doc_texts: list[str],
    all_doc_labels: list[str],
    top_n_terms: int = 8,
    combo_size: int = 3,
    top_k: int = 5,
    max_queries: int = 20,
) -> float:
    """Score how well a document can be retrieved by queries about its own content."""
    doc_vec = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = doc_vec.argsort()[-top_n_terms:][::-1]
    top_terms = [feature_names[i] for i in top_indices if doc_vec[i] > 0]
    if len(top_terms) < combo_size:
        return 0.0
    combos = list(itertools.combinations(top_terms, combo_size))
    if len(combos) > max_queries:
        step = len(combos) // max_queries
        combos = combos[::step][:max_queries]
    queries = [" ".join(combo) for combo in combos]
    if not queries:
        return 0.0
    hits = 0
    for query in queries:
        scores = _bm25_score(query, all_doc_texts)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        if doc_idx in ranked:
            hits += 1
    return hits / len(queries)
