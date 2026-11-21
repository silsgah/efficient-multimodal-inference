# DocVQA answer-scoring audit

## Purpose

The historical request CSVs contain `exact_match` and `anls_style` fields, but the repository did not contain the implementation that produced those fields. This audit adds an independently testable, reference-aligned scorer and a read-only comparison script. Historical files are never rewritten.

## Metric definition

For prediction (p) and ground-truth answer (g), normalize case and strip outer whitespace, then compute character-level Levenshtein distance (D(p,g)). Let:

[
d_n(p,g)=\frac{D(p,g)}{\max(|p|,|g|)}
]

The single-reference normalized similarity is:

[
s(p,g)=
\begin{cases}
1-d_n(p,g), & d_n(p,g)<0.5\\
0, & d_n(p,g)\geq0.5
\end{cases}
]

For a question with multiple accepted answers, use the maximum score over references. Dataset ANLS is the arithmetic mean across unique questions.

This follows the public DocVQA evaluation implementation at
[herobd/layoutlmv2 eval_docvqa.py](https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py)
and the documented threshold used in DocVQA ANLS implementations. The public Pix2Struct metric implementation also documents this definition and points back to the same evaluator.

## Deliberate normalization limits

This implementation performs only lowercasing and outer whitespace stripping. It does not normalize:
- punctuation or internal whitespace,
- abbreviations such as `M.D.` versus `M. D.`,
- numeric formatting such as `500` versus `500.00`,
- currencies or units,
- dates,
- answer prefixes or explanatory sentences.

These cases can materially affect ANLS. Any additional normalization must be separately versioned, justified, regression-tested, and reported. It must not be silently folded into the reference scorer.

The official DocVQA 2026 competition evaluator is a distinct task-specific pipeline with output-format requirements and strict number/unit/date checks. It is not used as the scorer for our original DocVQA validation experiment.

## Historical audit

Run:

```bash
python scripts/audit_anls.py \
  --input publication/energy_measurement_2026-11-18/energy_requests.csv
```

The output compares recomputed scores against archived columns and reports per-resolution means. It is read-only. Mismatches must be investigated before declaring the historical baseline reproduced.

## RL implications

The reference-aligned ANLS score is a terminal, verifiable task-quality reward candidate. It should not be treated as a calibrated measure of semantic correctness, especially for numeric, date, and unit answers. Report exact match alongside ANLS, and preserve raw predictions and references for audit. Reward shaping or answer normalization requires a separate experiment and must be fixed before policy training.
