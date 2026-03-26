---
name: prep-and-eval
description: Use when running the document preparation pipeline on a folder of files and evaluating retrieval quality afterward. Triggers on "run the pipeline", "fix and evaluate", "prep these docs", or "how good is retrieval".
---

# Prep and Eval

Run the ragprep fix pipeline on a document folder, then evaluate retrieval quality with the RAG eval script.

## Quick Reference

| Step | Command | Requires |
|------|---------|----------|
| Score only | `python cli.py score <folder>` | Nothing |
| Fix + organize | `python cli.py fix <folder> --llm-key $ANTHROPIC_API_KEY` | Anthropic key |
| Fix with hints | `python cli.py fix <folder> --llm-key $ANTHROPIC_API_KEY --folder-hints hints/education-k5.txt` | Anthropic key |
| Eval (BM25) | `python eval/run_eval.py <output-folder> --llm-key $ANTHROPIC_API_KEY` | Anthropic key |
| Eval (anam vector) | `python eval/run_eval.py <output-folder> --llm-key $ANTHROPIC_API_KEY --anam-key $ANAM_API_KEY` | Both keys |

All commands must use the venv Python: `.venv/bin/python`

## Workflow

1. **Score first** (free, no LLM) to see baseline quality
2. **Fix** to auto-correct issues and organize into folders. Output goes to `rag-files-{timestamp}/`
3. **Evaluate** the output folder against the 25 finlit test questions
4. **Compare** composite scores across runs

## Common Flags

- `--concurrency N` — parallel LLM calls (default: 5)
- `--exclude PATTERN` — skip files matching substring (repeatable)
- `--folder-hints FILE` — domain-specific folder guidance
- `--no-report` — suppress markdown report generation
- `--detail` — show per-issue breakdown in score output

## Eval Metrics

| Metric | What it measures |
|--------|-----------------|
| Retrieval Hit Rate | Was the expected source doc in top-k results? |
| Context Precision | How relevant were the retrieved chunks? |
| Faithfulness | Does the answer stick to retrieved context? |
| Answer Correctness | Does the answer match ground truth? |
| **Composite** | Average of all four |

## Example Full Run

```bash
# 1. Score
.venv/bin/python cli.py score ~/path/to/docs/

# 2. Fix with education folder hints
.venv/bin/python cli.py fix ~/path/to/docs/ \
  --llm-key $ANTHROPIC_API_KEY \
  --folder-hints hints/education-k5.txt

# 3. Evaluate the output (uses latest rag-files-* dir)
.venv/bin/python eval/run_eval.py rag-files-YYYYMMDD-HHMMSS/ \
  --llm-key $ANTHROPIC_API_KEY

# 4. Check report
cat eval/eval-report-*.md
```

## Previous Results

| Run | Hit Rate | Precision | Faithful | Correct | Composite |
|-----|----------|-----------|----------|---------|-----------|
| With hints (Mar 2) | 76% | 65% | 99% | 49% | 72.3% |
| No hints (Mar 2) | 80% | 68% | 100% | 57% | 76.2% |

Weakness: Assessment topic (0% hit rate). Strength: Digital Financial Safety (100%/85%).
