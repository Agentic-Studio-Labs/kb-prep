"""Naive baseline for RAG comparison — fixed-size chunks + standard BM25."""

import math
import re
from collections import Counter


class NaiveChunker:
    """Fixed-size chunker with fixed-offset overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into fixed-size token chunks."""
        tokens = text.split()
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = " ".join(tokens[i : i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
        return chunks if chunks else [text]


class NaiveBM25:
    """Standard BM25 without any improvements."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [re.findall(r"\w+", d.lower()) for d in documents]
        self.raw_docs = documents
        self.k1 = k1
        self.b = b
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs) if self.docs else 1
        self.n_docs = len(self.docs)
        self.df: Counter = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1

    def search(self, query: str, top_k: int = 10) -> list[int]:
        query_terms = re.findall(r"\w+", query.lower())
        scores = []
        for doc in self.docs:
            dl = len(doc)
            tf_map = Counter(doc)
            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                df = self.df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return ranked[:top_k]
