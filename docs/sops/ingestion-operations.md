# IngestGate Ingestion Operations SOP

This SOP is the primary operational playbook for running IngestGate and deciding what gets ingested.

Use `docs/manifest-deep-dive.md` as the field-level reference when you need to inspect exact manifest keys.

## Default Profile

Use the balanced profile unless a run explicitly calls for stricter gating:

- `--pass-threshold 85`
- `--pass-with-notes-threshold 70`
- `--remediation-threshold 50`

## Decision Policy

Apply gate decisions in this order:

1. `PASS` -> ingest automatically
2. `PASS_WITH_NOTES` -> ingest, open follow-up cleanup item
3. `REMEDIATION_RECOMMENDED` -> block ingestion, run `fix`, then re-analyze
4. `HOLD_FOR_REVIEW` -> block ingestion, manual owner review required

Treat `gate_decision` as the primary policy control. Treat `retrieval_mode_hint` as handling guidance.

## Standard Runbook

1. Baseline score
2. Analyze with benchmark
3. Review manifest triage signals
4. Apply decision policy
5. Fix blocked docs
6. Re-analyze and compare
7. Ingest approved docs

### Commands

```bash
# 1) Baseline
ingestgate score ./my-docs/ \
  --pass-threshold 85 --pass-with-notes-threshold 70 --remediation-threshold 50

# 2) Analyze + benchmark
ingestgate analyze ./my-docs/ --llm-key $ANTHROPIC_API_KEY --run-benchmark \
  --pass-threshold 85 --pass-with-notes-threshold 70 --remediation-threshold 50

# 3) If needed, remediate
ingestgate fix ./my-docs/ --llm-key $ANTHROPIC_API_KEY \
  --pass-threshold 85 --pass-with-notes-threshold 70 --remediation-threshold 50

# 4) Re-check after fixes
ingestgate analyze ./my-docs/ --llm-key $ANTHROPIC_API_KEY --run-benchmark \
  --pass-threshold 85 --pass-with-notes-threshold 70 --remediation-threshold 50

# 5) Generate before/after diff markdown
python -m src.manifest_diff \
  ./run-folder/01-before-manifest.json \
  ./run-folder/03-after-clean-manifest.json \
  -o ./run-folder/04-before-after-diff.md
```

## Manifest Triage Checklist

Review in this order:

1. `corpus.gate_decision_distribution`
2. `corpus.retrieval_mode_distribution`
3. `benchmarks` (`recall_at_5`, `mrr`, `ndcg_at_5`)
4. `documents[*].gate_decision`
5. `documents[*].retrieval_quality_gate`

## Fast Triage Snippets

```bash
MANIFEST=.ingestgate/manifest.json

# Gate decision distribution
jq '.corpus.gate_decision_distribution' "$MANIFEST"

# Blocked docs
jq -r '.documents[]
  | select(.gate_decision == "REMEDIATION_RECOMMENDED" or .gate_decision == "HOLD_FOR_REVIEW")
  | [.source_file, .gate_decision, .retrieval_quality_gate.retrieval_mode_hint.recommended_mode]
  | @tsv' "$MANIFEST"

# Non-default retrieval modes
jq -r '.documents[]
  | select(.retrieval_quality_gate.retrieval_mode_hint.recommended_mode != "text_hybrid_default")
  | [.source_file, .retrieval_quality_gate.retrieval_mode_hint.recommended_mode]
  | @tsv' "$MANIFEST"
```

## Escalation Rules

- Escalate immediately when `HOLD_FOR_REVIEW` count is non-zero.
- Escalate when benchmark guardrails degrade (`recall_at_5 < 0.60` or `mrr < 0.50`).
- Escalate when blocked docs exceed agreed operational threshold for the run.

## Handoff Artifacts

At the end of each run, capture:

- Manifest path
- Gate decision distribution
- Blocked doc list with reasons
- Benchmark summary
- Fix/re-run delta (if remediation was performed)
- Before/after diff markdown in the run folder (for example: `04-before-after-diff.md` comparing baseline and clean post-fix manifests)
