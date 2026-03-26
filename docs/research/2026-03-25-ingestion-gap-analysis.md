# Ingestion Gap Analysis: What RAG Platforms Actually Handle

Research date: 2026-03-25
Sources: Official documentation, API references, and blog posts from Pinecone, Weaviate, and pgvector ecosystem.

---

## Executive Summary

The research confirms that ragprep fills a real gap — but the gap is narrower than it first appears, and the README needs to be precise about what "document problems" means versus what platforms already handle.

**The honest positioning:** ragprep is a pre-ingestion quality layer. It does things no vector database does (scoring, self-containment fixes, retrieval benchmarking). But it should not imply that platforms leave users completely helpless — Pinecone Assistant does full parsing+chunking, Weaviate auto-vectorizes, and pgai is adding parsing+chunking to PostgreSQL.

---

## Capability Matrix (Verified Against Primary Sources)

### Legend
- ✅ = Built-in, works out of the box
- ⚠️ = Partial / requires configuration or separate product
- ❌ = Not provided, user responsibility

| Capability | Pinecone (DB) | Pinecone Assistant | Weaviate | pgvector | pgai (Timescale) | ragprep |
|---|---|---|---|---|---|---|
| **Document parsing (PDF/DOCX)** | ❌ | ✅ (PDF, DOCX, MD, TXT, JSON) | ❌ | ❌ | ⚠️ (PDF, HTML, MD) | ✅ |
| **Text chunking** | ❌ | ✅ (automatic) | ❌ | ❌ | ⚠️ (recursive char splitter) | ✅ (heading-aware, respects hierarchy) |
| **Embedding generation** | ✅ (Inference API) | ✅ (automatic) | ✅ (vectorizer modules) | ❌ | ✅ (multiple providers) | ❌ |
| **Vector storage + search** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Hybrid search (BM25+vector)** | ✅ (sparse-dense) | ✅ | ✅ (BM25F + vector fusion) | ⚠️ (needs ParadeDB or tsvector) | ⚠️ | ❌ (BM25+ used internally for scoring, not search) |
| **Metadata storage/filtering** | ✅ (40KB/record) | ✅ | ✅ (schema-based) | ✅ (SQL columns) | ✅ | ✅ (exports JSON) |
| **Quality scoring** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (10 criteria) |
| **Self-containment checks** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Dangling reference rewrites** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (LLM-powered) |
| **Header/footer cleanup** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (deterministic — drops page numbers, repeated headers) |
| **Retrieval benchmarking** | ⚠️ (Assistant Eval API, 2025) | ⚠️ (response quality only) | ❌ | ❌ | ❌ | ✅ (BM25+ self-retrieval) |
| **Knowledge graph** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Document grouping/taxonomy** | ❌ (namespaces are manual) | ❌ | ❌ | ❌ | ❌ | ✅ (Louvain + spectral) |
| **Reranking** | ✅ (pinecone-rerank-v0) | ✅ | ❌ (external) | ❌ | ❌ | ❌ |
| **Auto content generation (RAG)** | ❌ | ✅ | ✅ (generative modules) | ❌ | ❌ | ❌ |

---

## Pressure-Testing Each README Claim

### Claim 1: "Most RAG failures aren't embedding problems or chunk size problems. They're document problems."

**Verdict: DEFENSIBLE but needs nuance.**

This is a strong opening and directionally correct. The research confirms that none of the three platforms offer quality scoring, self-containment checking, or content-level fixes. However:

- The claim is hard to verify empirically. "Most RAG failures" is a strong quantifier. There's no published study that breaks down RAG failure modes by category.
- Embedding model choice and chunk size DO cause real failures — dismissing them risks alienating users who've experienced those problems.
- **Recommendation:** Soften slightly. Instead of "most RAG failures aren't embedding problems," consider "RAG failures often start before embeddings — with the documents themselves." This is equally compelling but less falsifiable.

### Claim 2: "Works with any vector database (Pinecone, Weaviate, Qdrant, Chroma, etc.)"

**Verdict: TRUE and safe.**

ragprep outputs Markdown files and JSON metadata. These are format-agnostic. The tool never calls any vector database API. This claim is accurate.

### Claim 3: "Scores documents across 10 criteria including a retrieval-aware metric that simulates search queries"

**Verdict: TRUE and unique.**

Confirmed: no vector database offers document-level quality scoring. Pinecone's 2025 Evaluation API scores *response quality* (answer correctness), not *document quality* (retrieval readiness). Weaviate documents retrieval metrics (recall@K, MRR, nDCG) but provides no built-in evaluation — users must build their own. pgvector has zero evaluation capability.

The retrieval-aware scorer using BM25+ self-retrieval is genuinely novel in this space. This is ragprep's strongest differentiator and the README should lean into it.

### Claim 4: "Fixes issues automatically — rewrites dangling references, splits long paragraphs, replaces generic headings, defines acronyms"

**Verdict: TRUE and unique.**

No platform offers content-level fixes. Pinecone Assistant parses and chunks documents but does not rewrite them. Weaviate's Transformation Agent (2025) can augment/enrich schema properties but does not modify source text for retrieval quality. pgai vectorizes text but doesn't clean it.

### Claim 5: "Recommends document groupings using Louvain community detection and TF-IDF similarity"

**Verdict: TRUE.**

No platform auto-generates taxonomy. Pinecone has namespaces (manual), Weaviate has collections and multi-tenancy (manual), pgvector uses SQL tables (manual). None cluster documents by topic automatically.

Note: the current README bullet uses algorithm names (Louvain, TF-IDF) which may not resonate with non-technical users. Consider rephrasing for the intro, keeping algorithm names in the technical section.

### Claim 6: "Exports machine-readable metadata — per-document .meta.json sidecars and a corpus-level manifest.json"

**Verdict: TRUE.**

This is a pipeline integration feature. Platforms handle metadata storage but not metadata *generation* — they expect users to provide metadata at ingestion time. ragprep generating structured metadata (entities, relationships, scores, topics) is genuinely useful for users who want to populate Pinecone metadata fields or Weaviate schema properties.

### Claim 7 (implicit): "downstream RAG pipelines (LlamaIndex, LangChain, Pinecone, Weaviate, etc.) can consume the analysis results programmatically"

**Verdict: ASPIRATIONAL.**

The JSON files are well-structured, but there's no actual integration code — no LlamaIndex reader, no LangChain loader, no Pinecone upload script. The metadata *could* be consumed by these tools, but saying they "can consume" it implies plug-and-play compatibility.

**Recommendation:** Either add a simple example showing how to load manifest.json into LlamaIndex/LangChain, or soften the language to "designed for integration with" rather than "can consume."

---

## What the README Should NOT Claim

Based on the research, avoid these traps:

1. **Don't imply platforms can't parse documents at all.** Pinecone Assistant handles PDF/DOCX parsing automatically. pgai (Timescale) is adding PDF/HTML/MD parsing. The gap is in *quality-aware* parsing, not parsing itself.

2. **Don't imply platforms have no chunking.** Pinecone Assistant chunks automatically. pgai has configurable chunking. The gap is in *structure-aware* chunking that respects heading hierarchy and generates quality metadata per chunk — which is what the chunk-first plan adds.

3. **Don't position against LlamaIndex/LangChain as competitors.** They're complementary. LlamaIndex has 100+ document loaders and chunking strategies. ragprep's value is the quality layer that runs *between* parsing and ingestion, not replacing the parsing frameworks.

4. **Don't claim "60-80% of the work is on you."** This was a rough estimate from our earlier conversation. It's directionally correct but not defensible with citations. The actual split depends heavily on which product and which use case.

---

## What the README SHOULD Emphasize

These are ragprep's genuinely unique, defensible capabilities:

1. **Retrieval-aware scoring** — No platform or framework does this. Self-retrieval rate using BM25+ against the corpus is novel.

2. **Content-level fixes** — No platform rewrites dangling references, expands acronyms, or replaces generic headings. This is uniquely ragprep.

3. **Pre-ingestion quality gate** — The concept of scoring documents *before* they enter the vector database and flagging problems is not something any platform offers. Pinecone's Eval API scores *after* retrieval.

4. **Knowledge graph across corpus** — Entity extraction + graph analysis (Louvain, PageRank, spectral clustering) across a document corpus is not available in any of these platforms. Weaviate has cross-references but they're manual.

5. **Machine-readable quality metadata** — Generating structured JSON with scores, entities, relationships, and folder assignments that can be ingested as metadata alongside vectors. No platform generates this.

6. **Document grouping with validation** — Automatic taxonomy suggestion with silhouette validation. Manual in all platforms.

---

## Nuances Worth Noting

### Pinecone Assistant changes the picture
Pinecone Assistant (GA 2025) is a managed RAG service that handles parsing, chunking, embedding, and answer generation end-to-end. For users on Pinecone Assistant, ragprep's parsing/chunking is redundant — but the quality scoring, fixes, and knowledge graph still add value. The README should acknowledge this tier exists.

### Weaviate's new agents
Weaviate's Transformation Agent (2025) can automatically augment schema properties — e.g., adding a "summary" field to every object using an LLM. This is post-ingestion enrichment, not pre-ingestion quality. But it's worth knowing about because it addresses some of the "metadata gap" from a different angle.

### pgai is the closest competitor to ragprep's parsing/chunking
Timescale's pgai Vectorizer (Early Access, 2025) parses PDFs, chunks text, and generates embeddings — all inside PostgreSQL. It's the most complete ingestion pipeline in the pgvector ecosystem. It doesn't do quality scoring or fixes, but it does automated parsing+chunking+embedding, which overlaps with ragprep's parse step. However, pgai's chunking is a recursive character splitter — it doesn't respect document heading hierarchy. ragprep's chunker never crosses heading boundaries and preserves the full heading path per chunk, and its retrieval benchmarking (Recall@5, MRR, nDCG@5) measures whether the chunks actually work for search. pgai has no equivalent evaluation step.

### The real gap is quality, not capability
The honest story: parsing and chunking are increasingly handled by platforms and frameworks. What nobody handles is *measuring whether the result is good*. That's ragprep's lane — quality scoring, content fixes, retrieval benchmarking, and structured metadata export.
