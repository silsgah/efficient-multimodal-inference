# DocVQA answer-scoring audit

## Purpose

The historical request CSVs contain `exact_match` and `anls_style` fields, but the repository did not contain the implementation that produced those fields. This audit adds an independently testable scorer and a read-only comparison script. Historical files are never rewritten.

The ANLS implementation is aligned with the public DocVQA evaluation code at [herobd/layoutlmv2 eval_docvqa.py](https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py). The project's exact-match metric is reported separately and retains its previously implemented normalization.

## ANLS metric definition

For prediction (p) and ground-truth answer (g), lowercase both strings **without stripping leading or trailing whitespace**, then compute character-level Levenshtein distance (D(p,g)). Let:

\[
d_n(p,g)=\frac{D(p,g)}{\max(|p|,|g|)}
\]

The single-reference normalized similarity is:

\[
s(p,g)=
\begin{cases}
1-d_n(p,g), & d_n(p,g)<0.5\\
0, & d_n(p,g)\geq0.5
\end{cases}
\]

For a question with multiple accepted answers, use the maximum score over references. Dataset ANLS is the arithmetic mean across unique questions.

The upstream evaluator's formula is undefined when both prediction and reference are empty because its denominator is zero. This implementation raises a `ValueError` for that malformed case rather than assigning an unsupported score. Empty prediction against a non-empty reference (and the reverse) receives zero under the formula.

## Separate exact-match behavior

The repository's `exact_match_score` remains case-insensitive and strips outer whitespace. This is deliberately separate from ANLS because the upstream ANLS evaluator does not strip whitespace. The historical `exact_match` and `anls_style` columns are independently compared against their respective recomputed values.

## Deliberate normalization limits

ANLS lowercases only. It does not normalize:
- leading or trailing whitespace,
- punctuation or internal whitespace,
- abbreviations such as `M.D.` versus `M. D.`,
- numeric formatting such as `500` versus `500.00`,
- currencies or units,
- dates,
- answer prefixes or explanatory sentences.

For example, the reference ANLS for `M.D.` against `M. D.` is 0.8, not 1.0. This is a character edit, not a formatting normalization.

Any additional normalization must be separately versioned, justified, regression-tested, and reported. It must not be silently folded into the reference scorer.

The official DocVQA 2026 competition evaluator is a distinct task-specific pipeline with output-format requirements and additional answer-format rules. It is not used as the scorer for our original DocVQA validation experiment.

## Historical audit

Run:

```bash
python scripts/audit_anls.py \
  --input publication/energy_measurement_2026-11-18/energy_requests.csv
```

The output compares recomputed scores against archived columns and reports per-resolution means. It is read-only. Mismatches must be investigated before declaring the historical baseline reproduced. In particular, archived `anls_style` scores may differ if the historical scorer stripped or otherwise normalized whitespace; the audit must report those differences rather than silently modifying the historical results.

## RL implications

The reference-aligned ANLS score is a terminal, verifiable task-quality reward candidate. It should not be treated as a calibrated measure of semantic correctness, especially for numeric, date, and unit answers. Report exact match alongside ANLS, and preserve raw predictions and references for audit. Reward shaping or answer normalization requires a separate experiment and must be fixed before policy training.
