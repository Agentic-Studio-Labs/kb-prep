#!/usr/bin/env python3
"""RAG evaluation script for anam-prep output.

Tests document quality by simulating retrieval + generation against
a set of test questions with known ground truth answers.

Metrics:
  - Retrieval Hit Rate: Did the correct source doc appear in top-k?
  - Context Precision: How relevant were the retrieved chunks?
  - Faithfulness: Does the generated answer stick to retrieved context?
  - Answer Correctness: Does the answer match ground truth?

Usage:
    python eval/run_eval.py <rag-folder> --llm-key $ANTHROPIC_API_KEY
    python eval/run_eval.py rag-files-20260302-094034/ --llm-key $KEY --top-k 5
    python eval/run_eval.py rag-files-20260302-094034/ --llm-key $KEY --questions eval/finlit-test-questions.json
"""

import asyncio
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click
from anthropic import AsyncAnthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Allow imports from project root (parent of eval/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer import extract_json

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A retrievable text chunk from a document."""
    text: str
    source_file: str
    chunk_index: int

@dataclass
class RetrievalResult:
    """Chunks retrieved for a single query."""
    chunks: list[Chunk]
    scores: list[float]

@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    question_id: str
    question: str
    expected_source: str
    topic: str
    audience: str
    ground_truth: str
    retrieved_sources: list[str]
    retrieved_text: str
    generated_answer: str
    # Scores (0.0 - 1.0)
    retrieval_hit: float = 0.0       # Was expected source in top-k?
    context_precision: float = 0.0   # How relevant was retrieved context?
    faithfulness: float = 0.0        # Does answer stick to context?
    answer_correctness: float = 0.0  # Does answer match ground truth?


# ---------------------------------------------------------------------------
# Chunker: split markdown files into retrievable pieces
# ---------------------------------------------------------------------------

def load_and_chunk(folder: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Load all .md files from folder (recursively) and split into chunks."""
    chunks = []
    folder_path = Path(folder)

    for md_file in sorted(folder_path.rglob("*.md")):
        # Skip report files
        if md_file.name.startswith("anam-prep-"):
            continue

        text = md_file.read_text(encoding="utf-8", errors="replace")
        rel_path = str(md_file.relative_to(folder_path))

        # Split by paragraphs first, then merge into chunk_size windows
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        current_chunk = []
        current_len = 0

        for para in paragraphs:
            words = len(para.split())
            if current_len + words > chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=rel_path,
                    chunk_index=len(chunks),
                ))
                # Keep overlap
                overlap_words = 0
                while current_chunk and overlap_words < overlap:
                    overlap_words += len(current_chunk[-1].split())
                    current_chunk = current_chunk[-1:]
                current_len = overlap_words

            current_chunk.append(para)
            current_len += words

        # Flush remaining
        if current_chunk:
            chunks.append(Chunk(
                text="\n\n".join(current_chunk),
                source_file=rel_path,
                chunk_index=len(chunks),
            ))

    return chunks


# ---------------------------------------------------------------------------
# BM25 retriever (no external deps)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Simple BM25 retriever over text chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Tokenize
        self.docs = [self._tokenize(c.text) for c in chunks]
        self.avg_dl = sum(len(d) for d in self.docs) / len(self.docs) if self.docs else 1

        # IDF
        self.df: Counter = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(self.docs)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_terms = self._tokenize(query)
        scores = []

        for doc in self.docs:
            dl = len(doc)
            tf_map = Counter(doc)
            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * numerator / denominator
            scores.append(score)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        result_chunks = [self.chunks[i] for i, _ in ranked]
        result_scores = [s for _, s in ranked]

        return RetrievalResult(chunks=result_chunks, scores=result_scores)


class AnamRetriever:
    """Retriever that searches anam.ai KB folders via vector similarity API."""

    def __init__(self, anam_key: str, anam_base_url: str = "https://api.anam.ai"):
        from anam_client import AnamClient
        from config import Config

        config = Config(anam_api_key=anam_key, anam_base_url=anam_base_url)
        self.client = AnamClient(config)
        folders = self.client.list_folders()
        self.folder_ids = [f["id"] for f in folders]
        self.folder_names = {f["id"]: f.get("name", f["id"]) for f in folders}
        print(f"  AnamRetriever: found {len(self.folder_ids)} folders: "
              f"{', '.join(self.folder_names.values())}")
        self._logged_schema = False

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        all_hits: list[tuple[float, Chunk]] = []
        for folder_id in self.folder_ids:
            try:
                raw = self.client.search_folder(folder_id, query, limit=top_k)
            except Exception as e:
                logger.warning("Search failed for folder %s: %s", folder_id, e)
                continue
            if not self._logged_schema and raw:
                logger.info("Anam search response schema: %s",
                            json.dumps(raw[0], indent=2, default=str))
                self._logged_schema = True
            folder_name = self.folder_names.get(folder_id, folder_id)
            for i, hit in enumerate(raw):
                score = self._extract_score(hit)
                chunk = self._hit_to_chunk(hit, folder_name, i)
                all_hits.append((score, chunk))
        all_hits.sort(key=lambda x: -x[0])
        top = all_hits[:top_k]
        return RetrievalResult(
            chunks=[c for _, c in top],
            scores=[s for s, _ in top],
        )

    @staticmethod
    def _extract_score(hit: dict) -> float:
        for key in ("score", "similarity", "relevance", "distance", "_score"):
            if key in hit:
                val = float(hit[key])
                return (1.0 / (1.0 + val)) if key == "distance" else val
        return 0.0

    @staticmethod
    def _hit_to_chunk(hit: dict, folder_name: str, index: int) -> Chunk:
        text = (hit.get("text") or hit.get("content") or hit.get("chunk_text")
                or hit.get("passage") or str(hit))
        source = (hit.get("source_file") or hit.get("document_name")
                  or hit.get("fileName") or hit.get("file_name")
                  or hit.get("source") or hit.get("title")
                  or hit.get("documentName") or folder_name)
        if isinstance(hit.get("metadata"), dict):
            meta = hit["metadata"]
            source = (meta.get("source_file") or meta.get("document_name")
                      or meta.get("fileName") or meta.get("file_name")
                      or meta.get("source") or meta.get("title") or source)
        return Chunk(text=text, source_file=source, chunk_index=hit.get("chunk_index", index))


# ---------------------------------------------------------------------------
# LLM judge: evaluate with Claude
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system. Score the following on a scale of 0.0 to 1.0.

## Question
{question}

## Retrieved Context
{context}

## Generated Answer
{answer}

## Ground Truth Answer
{ground_truth}

Score these three dimensions:

1. **context_precision**: How relevant is the retrieved context to answering the question? (0.0 = completely irrelevant, 1.0 = perfectly relevant)
2. **faithfulness**: Does the generated answer ONLY use information from the retrieved context? (0.0 = makes up facts not in context, 1.0 = fully grounded in context)
3. **answer_correctness**: Does the generated answer convey the same information as the ground truth? (0.0 = completely wrong, 1.0 = captures all key points)

Return ONLY valid JSON:
{{"context_precision": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0}}
"""

ANSWER_PROMPT = """\
Answer the following question using ONLY the provided context. If the context doesn't contain enough information, say so.

## Question
{question}

## Context
{context}

Provide a clear, concise answer (2-4 sentences).
"""


async def generate_answer(client: AsyncAnthropic, question: str, context: str, model: str) -> str:
    """Generate an answer from retrieved context."""
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(
            question=question, context=context
        )}],
    )
    return resp.content[0].text.strip()


async def judge_answer(
    client: AsyncAnthropic,
    question: str,
    context: str,
    answer: str,
    ground_truth: str,
    model: str,
) -> dict[str, float]:
    """Use LLM-as-judge to score the answer."""
    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
        )}],
    )
    text = resp.content[0].text.strip()
    data = extract_json(text)
    if data is not None:
        return data
    return {"context_precision": 0.0, "faithfulness": 0.0, "answer_correctness": 0.0}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(
    folder: str,
    questions_file: str,
    llm_key: str,
    model: str,
    top_k: int,
    concurrency: int,
    anam_key: str | None = None,
    anam_url: str = "https://api.anam.ai",
) -> list[EvalResult]:
    """Run full evaluation pipeline."""
    # Load questions
    with open(questions_file) as f:
        questions = json.load(f)

    # Build retriever
    if anam_key:
        print(f"Using anam.ai vector search (key=...{anam_key[-4:]})")
        retriever = AnamRetriever(anam_key, anam_url)
    else:
        print(f"Loading and chunking documents from {folder}...")
        chunks = load_and_chunk(folder)
        print(f"  {len(chunks)} chunks from {len(set(c.source_file for c in chunks))} files")
        retriever = BM25Retriever(chunks)

    # Setup LLM client
    client = AsyncAnthropic(api_key=llm_key)
    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(q: dict) -> EvalResult:
        async with semaphore:
            # Retrieve
            result = retriever.search(q["question"], top_k=top_k)
            retrieved_sources = list(dict.fromkeys(c.source_file for c in result.chunks))
            context = "\n\n---\n\n".join(
                f"[Source: {c.source_file}]\n{c.text}" for c in result.chunks
            )

            # Check retrieval hit
            expected = q["expected_source"]
            hit = 1.0 if any(expected in src for src in retrieved_sources) else 0.0

            # Generate answer
            answer = await generate_answer(client, q["question"], context, model)

            # Judge
            scores = await judge_answer(
                client, q["question"], context, answer, q["ground_truth"], model
            )

            return EvalResult(
                question_id=q["id"],
                question=q["question"],
                expected_source=expected,
                topic=q.get("topic", ""),
                audience=q.get("audience", ""),
                ground_truth=q["ground_truth"],
                retrieved_sources=retrieved_sources,
                retrieved_text=context[:500],
                generated_answer=answer,
                retrieval_hit=hit,
                context_precision=scores.get("context_precision", 0.0),
                faithfulness=scores.get("faithfulness", 0.0),
                answer_correctness=scores.get("answer_correctness", 0.0),
            )

    # Run all evaluations concurrently
    print(f"Evaluating {len(questions)} questions (concurrency={concurrency})...")
    tasks = [eval_one(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: list[EvalResult], folder: str) -> str:
    """Print and return a markdown evaluation report."""
    lines = []
    lines.append(f"# RAG Evaluation Report")
    lines.append(f"")
    lines.append(f"**Source:** {folder}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Questions:** {len(results)}")
    lines.append(f"")

    # Aggregate scores
    avg = lambda vals: sum(vals) / len(vals) if vals else 0
    metrics = {
        "Retrieval Hit Rate": avg([r.retrieval_hit for r in results]),
        "Context Precision": avg([r.context_precision for r in results]),
        "Faithfulness": avg([r.faithfulness for r in results]),
        "Answer Correctness": avg([r.answer_correctness for r in results]),
    }

    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|------:|")
    for name, score in metrics.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"| {name} | {score:.1%} {bar} |")
    lines.append("")

    composite = avg(list(metrics.values()))
    lines.append(f"**Composite Score: {composite:.1%}**")
    lines.append("")

    # By topic
    topics = sorted(set(r.topic for r in results))
    if topics:
        lines.append("## By Topic")
        lines.append("")
        lines.append("| Topic | Hit Rate | Precision | Faithful | Correct | N |")
        lines.append("|-------|---------|-----------|----------|---------|---|")
        for topic in topics:
            tr = [r for r in results if r.topic == topic]
            lines.append(
                f"| {topic} "
                f"| {avg([r.retrieval_hit for r in tr]):.0%} "
                f"| {avg([r.context_precision for r in tr]):.0%} "
                f"| {avg([r.faithfulness for r in tr]):.0%} "
                f"| {avg([r.answer_correctness for r in tr]):.0%} "
                f"| {len(tr)} |"
            )
        lines.append("")

    # By audience
    audiences = sorted(set(r.audience for r in results))
    if audiences:
        lines.append("## By Audience")
        lines.append("")
        lines.append("| Audience | Hit Rate | Precision | Faithful | Correct | N |")
        lines.append("|----------|---------|-----------|----------|---------|---|")
        for aud in audiences:
            tr = [r for r in results if r.audience == aud]
            lines.append(
                f"| {aud} "
                f"| {avg([r.retrieval_hit for r in tr]):.0%} "
                f"| {avg([r.context_precision for r in tr]):.0%} "
                f"| {avg([r.faithfulness for r in tr]):.0%} "
                f"| {avg([r.answer_correctness for r in tr]):.0%} "
                f"| {len(tr)} |"
            )
        lines.append("")

    # Per-question detail
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| ID | Question | Hit | Prec | Faith | Correct | Top Source |")
    lines.append("|----|----------|-----|------|-------|---------|------------|")
    for r in results:
        top_src = r.retrieved_sources[0] if r.retrieved_sources else "none"
        # Truncate for table
        q_short = r.question[:50] + "..." if len(r.question) > 50 else r.question
        src_short = Path(top_src).stem[:30] if top_src else "none"
        lines.append(
            f"| {r.question_id} | {q_short} "
            f"| {r.retrieval_hit:.0%} | {r.context_precision:.0%} "
            f"| {r.faithfulness:.0%} | {r.answer_correctness:.0%} "
            f"| {src_short} |"
        )
    lines.append("")

    # Failures
    misses = [r for r in results if r.retrieval_hit < 1.0]
    if misses:
        lines.append("## Retrieval Misses")
        lines.append("")
        for r in misses:
            lines.append(f"- **{r.question_id}**: Expected `{r.expected_source}`, got: {', '.join(r.retrieved_sources[:3])}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option("--llm-key", envvar="ANTHROPIC_API_KEY", required=True, help="Anthropic API key")
@click.option("--model", default="claude-sonnet-4-20250514", help="Model for generation and judging")
@click.option("--questions", default="eval/finlit-test-questions.json", help="Test questions JSON file")
@click.option("--top-k", default=5, help="Number of chunks to retrieve per question")
@click.option("--concurrency", default=5, help="Max parallel LLM calls")
@click.option("--output", "-o", default=None, help="Save report to file")
@click.option("--anam-key", envvar="ANAM_API_KEY", default=None,
              help="Anam API key - uses anam.ai vector search instead of local BM25")
@click.option("--anam-url", envvar="ANAM_BASE_URL", default="https://api.anam.ai",
              help="Anam API base URL")
def main(folder, llm_key, model, questions, top_k, concurrency, output, anam_key, anam_url):
    """Evaluate RAG document quality against test questions."""
    results = asyncio.run(run_evaluation(
        folder, questions, llm_key, model, top_k, concurrency,
        anam_key=anam_key, anam_url=anam_url,
    ))
    report = print_report(results, folder)

    if output:
        out_path = output
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = f"eval/eval-report-{ts}.md"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
