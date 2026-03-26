# Manifest Deep Dive

This guide explains how to interpret `.ragprep/manifest.json` and convert signals into concrete next actions.

## Quick Triage Order

1. Start with `corpus.retrieval_mode_distribution` to see how many docs need special handling.
2. Check `benchmarks` (if `--run-benchmark` was used) for objective retrieval performance.
3. Drill into `documents[*].retrieval_quality_gate` for per-document mode hints and evidence.
4. Use `documents[*].overall_score` and issue counts to prioritize fixes.

## Corpus-Level Fields

- `corpus.total_documents`, `corpus.total_chunks`, `corpus.avg_score`: basic coverage and quality snapshot.
- `corpus.readiness_distribution`: EXCELLENT/GOOD/FAIR/POOR spread.
- `corpus.retrieval_mode_distribution`: count of docs by recommended retrieval strategy.
  - `text_hybrid_default`: normal lexical+vector retrieval should be fine.
  - `hybrid_sparse_template`: use hybrid retrieval with metadata filters for form-like docs.
  - `hybrid_with_structure_rewrite`: retrieval may improve after document restructuring.
  - `multimodal_or_ocr_review`: text extraction likely insufficient; consider OCR/layout-aware pipeline.

## Benchmarks (when `--run-benchmark`)

Located under `benchmarks`.

- `recall_at_5`: "Can we find relevant chunks in top 5?"
- `mrr`: "How early is the first relevant chunk returned?"
- `ndcg_at_5`: ranking quality across top 5.
- `notes`: includes query provenance (for example deterministic heading+TF-IDF query source).

Suggested guardrails:

- `recall_at_5 < 0.60`: likely retrieval risk.
- `mrr < 0.50`: relevant chunks exist but ranking is weak.
- `ndcg_at_5 < 0.60`: top results are not well ordered.

## Per-Document Retrieval Quality Gate

Located at `documents[*].retrieval_quality_gate`.

### `retrieval_mode_hint`

- `recommended_mode`: operational recommendation for this document.
- `confidence`: how strongly the heuristic supports the recommendation (`high`/`medium`).
- `reasons`: why that mode was selected.

### `modality_readiness`

- `text_only_ready`: safe for default text pipeline.
- `layout_heavy_pdf`: likely layout-dependent PDF (tables/forms/visual structure).
- `template_like_document`: sparse/form-like structure likely.
- `parse_fidelity_warning`: suspiciously sparse extraction (warning severity).
- `parse_fidelity_template_note`: sparse extraction expected for template-like docs.

### `evidence`

Useful diagnostics for audits and threshold tuning:

- `file_type`, `total_words`
- `heading_count`, `body_paragraph_count`
- `short_label_ratio` (form-like signal)
- `heading_density`

## Action Playbook by Recommended Mode

- `text_hybrid_default`
  - Use standard hybrid retrieval.
  - No immediate parsing action needed.

- `hybrid_sparse_template`
  - Keep in corpus when semantically useful.
  - Add metadata filters (doc type, grade, unit, handout).
  - Avoid over-weighting embeddings for short label-heavy docs.

- `hybrid_with_structure_rewrite`
  - Run `fix`, then re-run `analyze --run-benchmark`.
  - Prioritize heading quality and self-containment fixes.

- `multimodal_or_ocr_review`
  - Consider OCR/layout-aware extraction and/or multimodal retrieval.
  - Validate whether tables/forms are preserved before ingestion.

## Useful jq Snippets

Assume:

```bash
MANIFEST=.ragprep/manifest.json
```

Show retrieval mode distribution:

```bash
jq '.corpus.retrieval_mode_distribution' "$MANIFEST"
```

List docs not in default mode:

```bash
jq -r '.documents[]
  | select(.retrieval_quality_gate.retrieval_mode_hint.recommended_mode != "text_hybrid_default")
  | [.source_file, .retrieval_quality_gate.retrieval_mode_hint.recommended_mode, .retrieval_quality_gate.retrieval_mode_hint.confidence]
  | @tsv' "$MANIFEST"
```

List docs with parse fidelity risk/note:

```bash
jq -r '.documents[]
  | select(
      .retrieval_quality_gate.modality_readiness.parse_fidelity_warning
      or .retrieval_quality_gate.modality_readiness.parse_fidelity_template_note
    )
  | [.source_file, .retrieval_quality_gate.modality_readiness]
  | @json' "$MANIFEST"
```

Show benchmark metrics by retrieval mode:

```bash
jq -r '.benchmarks[] | [.retrieval_mode, .recall_at_5, .mrr, .ndcg_at_5, .query_count] | @tsv' "$MANIFEST"
```

## Recommended Workflow

1. `score` for quick baseline.
2. `analyze --run-benchmark` to generate manifest signals and retrieval metrics.
3. Triage with this guide.
4. `fix` only when signals/benchmarks indicate risk.
5. Re-run `analyze --run-benchmark` and compare.
