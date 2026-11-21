# Historical ANLS and page-disjoint split correction — 21 November 2026

## Purpose and provenance

This addendum records the scorer audit of the archived 18 November 2026 DocVQA resolution/energy experiment and the page-disjoint split frozen on 21 November 2026. The historical request CSV and original energy telemetry are retained; this correction does not rewrite the archived source data.

Audit implementation:
- `scripts/audit_anls.py`
- `src/evaluation/metrics.py`
- Reference implementation: [herobd/layoutlmv2 `eval_docvqa.py`](https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py)

Inputs:
- Manifest: `data/docvqa/manifest.jsonl`
- Historical requests: `publication/energy_measurement_2026-11-18/energy_requests.csv`

SHA-256 of the manifest: `1ef38c7abf0de4c74963ec24d06296efbf19bec48c17b718ff5b385e66c6c5ac`

SHA-256 of the historical request CSV: `cb517e17c5102fe6869248878e51aab93b2606c464a9d034448cb45569d59619`

## Corrected quality results

There are 50 unique questions and 150 measured requests per resolution (three repeated executions per question). Repetitions are not counted as independent questions. Scores below are averaged at the question level.

| Pixel budget | Unique questions | Corrected ANLS | Corrected exact match |
|---:|---:|---:|---:|
| 200,704 | 50 | 0.725571 | 56% (28/50) |
| 401,408 | 50 | 0.769369 | 66% (33/50) |
| 802,816 | 50 | 0.770841 | 72% (36/50) |

### Correction against the originally reported values

| Pixel budget | Previously reported ANLS-style | Recomputed ANLS | Difference | Previously reported exact match | Recomputed exact match |
|---:|---:|---:|---:|---:|---:|
| 200,704 | 0.738000 | 0.725571 | -0.012429 | 58% (29/50) | 56% (28/50) |
| 401,408 | 0.780200 | 0.769369 | -0.010831 | 66% (33/50) | 66% (33/50) |
| 802,816 | 0.792750 | 0.770841 | -0.021909 | 74% (37/50) | 72% (36/50) |

The corrected reference-aligned ANLS is lower at all three settings. The resolution ranking remains ordered from lowest to highest ANLS as low, medium, high, but the high-resolution result is only slightly above the medium-resolution result (0.770841 versus 0.769369). This difference is descriptive on this 50-question cohort; it is not evidence of a statistically significant advantage.

The archived values are not overwritten. The corrected values in this addendum are the values to use in new analysis and tables.

## Corrected energy-per-correct-generation derivation

The measured incremental GPU board energy, latency, and visual-token measurements in the 18 November energy report are not recalculated by this scorer audit. The derived energy per correct generation does change because its denominator depends on the corrected exact-match count.

Each of the three executions for a question was consistent within the corresponding resolution condition. Thus the corrected number of exact-match generations is the number of correct questions multiplied by three:

| Pixel budget | Measured incremental energy | Correct questions | Correct generations (3 repetitions) | Corrected energy per correct generation |
|---:|---:|---:|---:|---:|
| 200,704 | 0.6589 Wh | 28/50 | 84/150 | 7.844 mWh |
| 401,408 | 0.7278 Wh | 33/50 | 99/150 | 7.352 mWh |
| 802,816 | 0.8483 Wh | 36/50 | 108/150 | 7.855 mWh |

Calculation: incremental GPU energy in Wh × 1,000 / correct measured generations. These remain approximate GPU-board-energy estimates because they inherit the original one-second NVML sampling, idle-baseline subtraction, and interpolation limitations. They are not wall-plug or whole-system energy.

## What the mismatch means

The archived `anls_style` and `exact_match` values are not guaranteed to match the scorer now used in the project. The audit recomputes each archived prediction against its archived reference answer(s), compares both metric columns independently, and reports the discrepancies; it does not modify the request CSV.

The reference-aligned ANLS implementation lowercases strings, applies character-level Levenshtein distance normalized by the longer string, uses a strict normalized-distance cutoff of 0.5, and takes the maximum over accepted references. It does **not** strip whitespace or normalize punctuation, internal spacing, abbreviations, numeric formats, currencies, or dates. For example, `M.D.` against `M. D.` receives ANLS 0.8 rather than 1.0.

The project's exact-match metric is a separate measure: it is case-insensitive and strips leading/trailing whitespace, but it does not normalize punctuation or internal formatting. The audit found at least one archived exact-match disagreement (request `30185` at the lowest resolution); use the current recomputed metric rather than carrying forward the legacy flag. The full reference/prediction pair should be inspected before assigning a specific cause to each individual mismatch.

The scorer audit corrects metric comparability; it does not prove semantic answer correctness. Exact match and ANLS should both be reported, and raw answers/references should remain available for inspection.

## Frozen 200-question page-disjoint pilot split

The split was created by grouping samples on SHA-256 of the image bytes. Every manifest question whose image hash matches one of the 50 historically evaluated development questions was assigned to development. This keeps all questions from a historically used page out of the holdout, even where the question IDs themselves were not in the original 50.

| Partition | Questions | Unique image hashes |
|---|---:|---:|
| Development | 54 | 48 |
| Holdout | 146 | 138 |
| Shared image hashes | 0 | 0 |

The 50 historical development IDs account for 48 image hashes. Four additional questions in the 200-question manifest shared those development images and were moved to development. The partition therefore contains 54 development questions and 146 holdout questions.

Frozen artifacts:
- `publication/rl_data_split_2026-11-21/development.jsonl`
- `publication/rl_data_split_2026-11-21/holdout.jsonl`
- `publication/rl_data_split_2026-11-21/split_summary.json`

The split summary records the manifest and historical CSV hashes listed above, the partition sample IDs and counts, and `shared_image_hashes: 0`. The split-generation code refuses to overwrite existing outputs unless explicitly passed `--overwrite`.

### Scope limitation

This is a page-disjoint partition of the existing 200-question validation sample. It is a useful pilot split and an improvement over a question-ID-only split, but it is **not** a sufficiently large RL policy-training corpus and should not be presented as the final benchmark. The next data stage must assemble the full training partition and a properly isolated validation/test design, grouping by source document/image and keeping the historically used validation pages out of any locked final test.

## Reproduction

Run the audit from the repository root:

```bash
python scripts/audit_anls.py \
  --input publication/energy_measurement_2026-11-18/energy_requests.csv
```

To regenerate the split in a new empty output directory (do not overwrite the frozen artifacts):

```bash
python scripts/freeze_validation_holdout.py \
  --manifest data/docvqa/manifest.jsonl \
  --historical-results publication/energy_measurement_2026-11-18/energy_requests.csv \
  --output-dir publication/rl_data_split_2026-11-21-reproduction
```

The split script reads the inputs and image files, computes content hashes, verifies zero image-hash overlap, and writes new artifacts. It does not modify the manifest or historical results CSV.
