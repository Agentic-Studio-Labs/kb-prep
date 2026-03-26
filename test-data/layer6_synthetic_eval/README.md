## Layer 6 Synthetic Eval

This suite is a small mixed corpus designed to make meaningful before/after deltas easy to see when running IngestGate `score`, `analyze --run-benchmark`, and `fix`.

It is intentionally adversarial rather than realistic at corpus scale. Each document is shaped to stress a different part of the pipeline.

### Corpus files

- `01_clean_control.md`
  - Clean control document. Should remain stable and score well.
- `02_fix_heavy_policy.md`
  - Dense with dangling references, generic headings, undefined acronyms, and an overlong paragraph.
- `03_table_layout_lesson.docx`
  - Table-heavy DOCX with split-run cell text to exercise DOCX table parsing.
- `04_sparse_tracker_template.pdf`
  - Label-heavy template PDF with split heading lines to exercise layout-aware parsing and retrieval-mode hints.
- `05_topic_overlap_packet.md`
  - Broad, multi-topic packet with overlapping vocabulary intended to trigger weaker retrieval signals and a split recommendation.

### What this suite is for

- Manual CLI runs where the user wants obvious fix actions and retrieval-quality-gate signals.
- Lightweight automated checks that ensure the parser/scorer/export pipeline sees the expected patterns.

### Suggested manual workflow

```bash
ingestgate score test-data/layer6_synthetic_eval/corpus/
ingestgate analyze test-data/layer6_synthetic_eval/corpus/ --llm-key $ANTHROPIC_API_KEY --run-benchmark
ingestgate fix test-data/layer6_synthetic_eval/corpus/ --llm-key $ANTHROPIC_API_KEY
ingestgate analyze test-data/layer6_synthetic_eval/corpus/ --llm-key $ANTHROPIC_API_KEY --run-benchmark
```

`analyze` writes metadata to `test-data/layer6_synthetic_eval/corpus/.ingestgate/`.

### Expected directional outcomes

- `01_clean_control.md` should score high and need few or no changes.
- `02_fix_heavy_policy.md` should show the clearest fix actions.
- `03_table_layout_lesson.docx` should preserve table content and split-run phrases.
- `04_sparse_tracker_template.pdf` should lean toward non-default retrieval mode hints.
- `05_topic_overlap_packet.md` should be the best candidate for `split_recommendations`.
