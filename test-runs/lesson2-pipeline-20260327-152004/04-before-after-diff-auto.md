# Before vs After Diff

Run folder: `test-runs/lesson2-pipeline-20260327-152004`

Compared artifacts:
- `01-before-manifest.json`
- `03-after-clean-manifest.json`

## Corpus Summary

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Total documents | 6 | 6 | +0 |
| Average score | 84.5 | 86.1 | +1.6 |
| Gate distribution | PASS: 5, REMEDIATION_RECOMMENDED: 1 | PASS: 5, REMEDIATION_RECOMMENDED: 1 | n/a |
| recall_at_5 (lexical) | 0.667 | 0.500 | -0.167 |
| mrr (lexical) | 0.389 | 0.399 | +0.010 |
| ndcg_at_5 (lexical) | 0.309 | 0.299 | -0.009 |
| query_count (lexical) | 6 | 6 | +0 |

## Document-Level Delta

| Document stem | Before | After | Delta | Gate before -> after | Issues before -> after |
|---|---:|---:|---:|---|---:|
| 4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams | 88.5 | 91.2 | +2.7 | PASS -> PASS | 9 -> 7 |
| 4-5.FL.2 Handout A. SMART Goals at Home Letter | 91.4 | 91.7 | +0.3 | PASS -> PASS | 8 -> 7 |
| 4-5.FL.2 Handout B. SMART Goal Brainstorm and Planning | 89.0 | 89.4 | +0.4 | PASS -> PASS | 10 -> 9 |
| 4-5.FL.2 Handout C. SMART Goal Progress Tracker | 87.7 | 88.1 | +0.4 | PASS -> PASS | 8 -> 6 |
| 4-5.FL.2 Lesson - How Can SMART Goals Help You Achieve Your Dreams_ | 64.0 | 69.7 | +5.7 | REMEDIATION_RECOMMENDED -> REMEDIATION_RECOMMENDED | 14 -> 10 |
| 4-5.FL.2 Rubric for Handout B. SMART Goal Brainstorm and Planning | 86.3 | 86.3 | +0.0 | PASS -> PASS | 8 -> 8 |

## Why A Doc Is Still Blocked

Remaining `REMEDIATION_RECOMMENDED` document:
- `4-5.FL.2 Lesson - How Can SMART Goals Help You Achieve Your Dreams_.md`

Notable criterion changes:
- `acronym_definitions`: 84.0 -> 92.0 (+8.0)
- `knowledge_completeness`: 85.0 -> 100.0 (+15.0)
- `self_containment`: 76.0 -> 100.0 (+24.0)

Criteria still limiting gate decision:
- `file_focus`: 40.0
- `heading_quality`: 30.0
- `paragraph_length`: 0.0
- `structure`: 50.0

## Notes

- This comparison matches documents by filename stem so extension changes don't break diffing.
- Recommended comparison dashboard: `03-after-clean-web-report.html`.
