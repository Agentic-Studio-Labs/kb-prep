# Ingesting ragprep Output into Weaviate

This guide walks through loading ragprep's fixed Markdown, chunk sidecars, and quality metadata into a Weaviate instance for hybrid retrieval.

## Prerequisites

- ragprep output from `fix` or `analyze` (a directory with `.md`, `.meta.json`, `.chunks.json`, and `manifest.json`)
- Weaviate instance running (local Docker or Weaviate Cloud)
- Python 3.10+ with `weaviate-client` v4 installed

```bash
pip install weaviate-client
```

## What ragprep produces

After running `fix`:

```
rag-files-20260326/
├── api-design-guide.md
├── api-design-guide.meta.json
├── api-design-guide.chunks.json
├── onboarding-checklist.md
├── onboarding-checklist.meta.json
├── onboarding-checklist.chunks.json
├── incident-response-runbook.md
├── incident-response-runbook.meta.json
├── incident-response-runbook.chunks.json
└── manifest.json
```

Each `.chunks.json` contains heading-aware chunks with metadata. Each `.meta.json` contains scores, entities, relationships, and a `retrieval_quality_gate` block. The `manifest.json` has corpus-level stats and per-document summaries.

## Step 1: Review manifest signals before ingesting

Before uploading anything, check the retrieval quality gate signals:

```bash
# Which docs need special handling?
jq -r '.documents[]
  | select(.retrieval_quality_gate.retrieval_mode_hint.recommended_mode != "text_hybrid_default")
  | [.source_file, .retrieval_quality_gate.retrieval_mode_hint.recommended_mode]
  | @tsv' rag-files-20260326/manifest.json
```

| Recommended mode | Action |
|---|---|
| `text_hybrid_default` | Ingest normally |
| `hybrid_sparse_template` | Ingest with metadata filters, lower embedding weight |
| `hybrid_with_structure_rewrite` | Consider running `fix` first, then re-analyze |
| `multimodal_or_ocr_review` | Manual review — text extraction may be incomplete |

Skip or quarantine `multimodal_or_ocr_review` documents until you have a layout-aware pipeline.

## Step 2: Define Weaviate collection schema

Create a collection that stores chunks with ragprep metadata as filterable properties:

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()  # or connect_to_weaviate_cloud(...)

client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),  # or any vectorizer
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="source_file", data_type=DataType.TEXT),
        Property(name="chunk_id", data_type=DataType.TEXT),
        Property(name="heading_path", data_type=DataType.TEXT_ARRAY),
        Property(name="chunk_type", data_type=DataType.TEXT),
        Property(name="domain", data_type=DataType.TEXT),
        Property(name="topics", data_type=DataType.TEXT_ARRAY),
        Property(name="overall_score", data_type=DataType.NUMBER),
        Property(name="readiness", data_type=DataType.TEXT),
        Property(name="retrieval_mode", data_type=DataType.TEXT),
        Property(name="token_estimate", data_type=DataType.INT),
    ],
)
```

**Why these properties?**
- `heading_path` enables scoped retrieval ("search only within the API Authentication section")
- `domain` and `topics` enable metadata-filtered hybrid search
- `overall_score` lets you filter out low-quality documents at query time
- `retrieval_mode` lets your query router treat template docs differently from prose docs

## Step 3: Load and upload chunks

```python
import json
from pathlib import Path


def load_ragprep_output(output_dir: str) -> list[dict]:
    """Load all chunks with their document-level metadata."""
    output_path = Path(output_dir)
    manifest = json.loads((output_path / "manifest.json").read_text())

    # Build lookup from manifest for document-level fields
    doc_lookup = {}
    for doc_entry in manifest["documents"]:
        doc_lookup[doc_entry["source_file"]] = doc_entry

    chunks_to_upload = []

    for chunks_file in output_path.glob("*.chunks.json"):
        chunk_data = json.loads(chunks_file.read_text())
        source_file = chunk_data["source_file"]
        doc_meta = doc_lookup.get(source_file, {})
        rqg = doc_meta.get("retrieval_quality_gate", {})
        mode = rqg.get("retrieval_mode_hint", {}).get("recommended_mode", "text_hybrid_default")

        # Skip documents flagged for manual review
        if mode == "multimodal_or_ocr_review":
            print(f"  Skipping {source_file} (needs OCR/multimodal review)")
            continue

        for chunk in chunk_data["chunks"]:
            chunks_to_upload.append({
                "text": chunk["text"],
                "source_file": source_file,
                "chunk_id": chunk["chunk_id"],
                "heading_path": chunk["heading_path"],
                "chunk_type": chunk["chunk_type"],
                "token_estimate": chunk["token_estimate"],
                "domain": doc_meta.get("domain", ""),
                "topics": doc_meta.get("topics", []),
                "overall_score": doc_meta.get("overall_score", 0.0),
                "readiness": doc_meta.get("readiness", ""),
                "retrieval_mode": mode,
            })

    return chunks_to_upload


def upload_to_weaviate(client, chunks: list[dict]):
    """Batch-upload chunks to Weaviate."""
    collection = client.collections.get("Document")

    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            batch.add_object(properties=chunk)

    print(f"Uploaded {len(chunks)} chunks to Weaviate")
```

Usage:

```python
chunks = load_ragprep_output("rag-files-20260326/")
upload_to_weaviate(client, chunks)
```

## Step 4: Query with metadata filters

### Basic hybrid search

```python
collection = client.collections.get("Document")

response = collection.query.hybrid(
    query="how do we handle API authentication?",
    limit=5,
)

for obj in response.objects:
    print(f"[{obj.properties['source_file']}] {obj.properties['heading_path']}")
    print(f"  {obj.properties['text'][:200]}...")
    print()
```

### Filtered by domain

```python
from weaviate.classes.query import Filter

response = collection.query.hybrid(
    query="incident response escalation",
    filters=Filter.by_property("domain").equal("engineering"),
    limit=5,
)
```

### Filtered by quality score

```python
# Only return chunks from well-scored documents
response = collection.query.hybrid(
    query="onboarding steps for new engineers",
    filters=Filter.by_property("overall_score").greater_or_equal(70.0),
    limit=5,
)
```

### Scoped to a heading path

```python
# Search within a specific section
response = collection.query.hybrid(
    query="timeout configuration",
    filters=Filter.by_property("heading_path").contains_any(["API Configuration"]),
    limit=5,
)
```

## Step 5: Verify retrieval

After uploading, run a few spot checks:

```python
# Count chunks by retrieval mode
for mode in ["text_hybrid_default", "hybrid_sparse_template", "hybrid_with_structure_rewrite"]:
    count = collection.aggregate.over_all(
        filters=Filter.by_property("retrieval_mode").equal(mode),
        total_count=True,
    )
    print(f"  {mode}: {count.total_count} chunks")

# Verify a specific document's chunks are present
response = collection.query.fetch_objects(
    filters=Filter.by_property("source_file").equal("api-design-guide.md"),
    limit=100,
)
print(f"api-design-guide.md: {len(response.objects)} chunks")
```

## Adapting for other vector databases

The `load_ragprep_output()` function is database-agnostic — it returns a list of dicts. To adapt for a different target:

| Vector DB | What changes |
|---|---|
| **Pinecone** | Use `pinecone.Index.upsert()` with `chunk_id` as the vector ID and other fields as metadata |
| **Qdrant** | Use `qdrant_client.upsert()` with a `PointStruct` per chunk |
| **Chroma** | Use `collection.add()` with documents, metadatas, and ids |
| **pgvector** | INSERT into a table with a vector column + metadata columns |

The ragprep sidecar format is designed so that `text` is always the content to embed, and everything else is structured metadata for filtering.

## Entity-powered retrieval (advanced)

Each `.meta.json` sidecar includes extracted entities and relationships. You can use these to build a secondary retrieval path:

```python
# Load entities from a sidecar
meta = json.loads(Path("rag-files-20260326/api-design-guide.meta.json").read_text())

for entity in meta["entities"]:
    print(f"  {entity['type']}: {entity['name']} — {entity['description']}")
```

Use cases:
- **Entity-augmented queries** — expand "OAuth" to include related entities found in the graph
- **Cross-document linking** — entities shared across documents create implicit connections
- **Knowledge graph overlay** — load `manifest.json` → `knowledge_graph` into a graph DB alongside your vector index

## Workflow summary

```
ragprep fix ./docs/ --llm-key $KEY
    ↓
review manifest.json signals
    ↓
skip multimodal_or_ocr_review docs
    ↓
load_ragprep_output() → list of chunk dicts
    ↓
upload_to_weaviate() → Weaviate collection
    ↓
hybrid query with metadata filters
```
