# Before vs After Diff

Run folder: `test-runs/lesson2-pipeline-20260327-152004`

Compared artifacts:

- `01-before-manifest.json`
- `03-after-clean-manifest.json`

## Corpus Summary


| Metric             | Before                              | After                               | Delta     |
| ------------------ | ----------------------------------- | ----------------------------------- | --------- |
| Total documents    | 6                                   | 6                                   | 0         |
| Average score      | 84.5                                | 86.1                                | +1.6      |
| Gate distribution  | PASS: 5, REMEDIATION_RECOMMENDED: 1 | PASS: 5, REMEDIATION_RECOMMENDED: 1 | no change |
| Recall@5 (lexical) | 0.667                               | 0.500                               | -0.167    |
| MRR (lexical)      | 0.389                               | 0.399                               | +0.010    |
| nDCG@5 (lexical)   | 0.309                               | 0.299                               | -0.009    |


## Document-Level Delta


| Document stem                                                       | Before | After | Delta | Gate before -> after                               | Issues before -> after |
| ------------------------------------------------------------------- | ------ | ----- | ----- | -------------------------------------------------- | ---------------------- |
| 4-5.FL.2 Anchor Chart - SMART Goals and Achieving Your Dreams       | 88.5   | 91.2  | +2.7  | PASS -> PASS                                       | 9 -> 7                 |
| 4-5.FL.2 Handout A. SMART Goals at Home Letter                      | 91.4   | 91.7  | +0.3  | PASS -> PASS                                       | 8 -> 7                 |
| 4-5.FL.2 Handout B. SMART Goal Brainstorm and Planning              | 89.0   | 89.4  | +0.4  | PASS -> PASS                                       | 10 -> 9                |
| 4-5.FL.2 Handout C. SMART Goal Progress Tracker                     | 87.7   | 88.1  | +0.4  | PASS -> PASS                                       | 8 -> 6                 |
| 4-5.FL.2 Lesson - How Can SMART Goals Help You Achieve Your Dreams_ | 64.0   | 69.7  | +5.7  | REMEDIATION_RECOMMENDED -> REMEDIATION_RECOMMENDED | 14 -> 10               |
| 4-5.FL.2 Rubric for Handout B. SMART Goal Brainstorm and Planning   | 86.3   | 86.3  | +0.0  | PASS -> PASS                                       | 8 -> 8                 |


## Why One Doc Is Still Blocked

Remaining `REMEDIATION_RECOMMENDED` document:

- `4-5.FL.2 Lesson - How Can SMART Goals Help You Achieve Your Dreams_.md`

Biggest criterion improvements:

- `self_containment`: 76 -> 100 (+24)
- `knowledge_completeness`: 85 -> 100 (+15)
- `acronym_definitions`: 84 -> 92 (+8)

Criteria still limiting gate decision:

- `paragraph_length`: 0
- `heading_quality`: 30
- `structure`: 50
- `file_focus`: 40

## Notes

- This is the clean comparison (post-fix analyze excluded `ingestgate-fix-*` report files).
- Recommended primary web view for this run: `03-after-clean-web-report.html`.

