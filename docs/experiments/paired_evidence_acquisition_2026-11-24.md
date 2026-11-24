# Paired visual evidence acquisition experiment — 24 November 2026

## Scope

This experiment studies per-question, per-page evidence acquisition for
Qwen2.5-VL-3B-Instruct on the DocVQA single-page questions dataset. The frozen
source snapshot is `pixparse/docvqa-single-page-questions` at revision
`13619c00b32b50729b6e50b4bb0aa4b355698f3f`.

The selected 128-question exploratory sample is drawn from `policy_train`,
with one question per document group and page. It is not a final
generalization result. `policy_validation` and `locked_evaluation` remain
untouched for model selection and final evaluation, respectively. The official
DocVQA test split was not loaded.

## Action conditions

All conditions use the same frozen VLM, greedy decoding (`temperature=0`,
`top_p=1`), and the same page/question pairing.

- **STOP:** initial answer at 200,704 max pixels.
- **LOW-RES REVISION CONTROL:** supply the initial answer back to the model
  and ask it to reconsider at 200,704 max pixels.
- **FULL_UPSCALE:** same revision prompt, but reprocess the original full page
  at 802,816 max pixels.
- **ROI_CROP:** same revision prompt with a deterministic OCR/BM25 proposal
  contact sheet of up to two spatially related regions. The current single-image
  backend receives the composite contact sheet as one image.

ROI produced candidates for 104 of 128 questions; 24 had no lexical OCR match
and were explicitly recorded as skipped. ROI-only metrics are therefore
reported on the 104 eligible questions. A separate availability-aware result
uses STOP when ROI is unavailable.

## Metric definition

The scorer computes case-insensitive exact match after trimming outer
whitespace. ANLS lowercases and trims the strings, computes character-level
Levenshtein distance divided by the longer string length, and returns
(1 - mathrm{NLD}) only when NLD is strictly below 0.5. For multiple accepted
reference answers, the maximum score is used. Punctuation and internal
whitespace are preserved.

This is documented as the metric convention used by this experiment; for
publication, the exact evaluator/version should be cross-checked against the
official DocVQA evaluation implementation.

## Measured results

### Main paired action run (128 questions)

| Condition | N | EM | ANLS |
|---|---:|---:|---:|
| STOP | 128 | 0.4922 | 0.6822 |
| FULL_UPSCALE revision | 128 | 0.8281 | 0.9407 |
| ROI_CROP, eligible only | 104 | 0.7596 | 0.8416 |
| ROI when available, else STOP | 128 | 0.7422 | 0.8403 |

Paired comparisons:

| Comparison | N | Delta ANLS | 95% paired bootstrap CI | Delta EM |
|---|---:|---:|---:|---:|
| FULL_UPSCALE revision vs STOP | 128 | +0.2586 | [+0.1929, +0.3285] | +0.3359 |
| ROI vs STOP, ROI-eligible | 104 | +0.1946 | [+0.0995, +0.2896] | +0.3077 |
| ROI vs FULL_UPSCALE, ROI-eligible | 104 | -0.1002 | [-0.1729, -0.0339] | -0.0673 |
| ROI-if-available else STOP vs STOP | 128 | +0.1581 | [+0.0796, +0.2381] | +0.2500 |

### Low-resolution revision control

| Condition | EM | ANLS | Answers changed vs STOP |
|---|---:|---:|---:|
| STOP | 0.4922 | 0.6822 | 0 |
| LOW-RES REVISION CONTROL | 0.5000 | 0.6974 | 4 |
| FULL_UPSCALE revision | 0.8281 | 0.9407 | 56 |

The more informative visual-resolution contrast is FULL_UPSCALE revision vs
LOW-RES REVISION CONTROL:

- N = 128
- Delta ANLS = +0.2433
- 95% paired bootstrap CI = [+0.1795, +0.3113]
- Delta EM = +0.3281
- Per-question outcomes: 52 improve, 76 tie, 0 regress in ANLS.

This control reduces the alternative explanation that the improvement comes
only from asking the model to reconsider its earlier answer. The contrast is
still subject to request-order effects and the same model/prompt setup.

## Exploratory predictors

Univariate Spearman correlations with the visual-only ANLS delta
(FULL_UPSCALE revision minus LOW-RES revision control) include:

| Feature | Spearman rho |
|---|---:|
| Initial answer is numeric | +0.307 |
| Initial answer has a digit | +0.290 |
| Page height | +0.254 |
| Page line count | +0.253 |
| ROI context line count | +0.244 |
| Page pixel area | +0.238 |
| Page word count | +0.235 |

The feature audit is exploratory. It uses only pre-action observables as
candidate features and uses gold-derived quality deltas only as retrospective
outcomes. Orientation is encoded as sine/cosine because degrees are circular.
Correlations are univariate, are not causal evidence, and are not validated
predictors.

## Energy measurement caveat

Aggregate GPU board energy was estimated with NVML power samples and
pre/post idle subtraction. Reported aggregate estimates were:

| Phase | Requests | Estimated incremental board energy |
|---|---:|---:|
| Initial pass | 128 | 666.30 mWh |
| ROI revision | 104 | 1,016.03 mWh |
| Full-upscale revision | 128 | 1,382.37 mWh |
| Low-resolution revision control | 128 | 698.71 mWh |

These are aggregate phase estimates, not per-request ground-truth energy.
They should not be treated as precise marginal energy costs without a more
carefully controlled repeated measurement protocol.

## Interpretation and next steps

1. Full-page resolution increase provides a substantial paired quality gain
   beyond low-resolution answer revision on this exploratory sample.
2. ROI acquisition helps relative to STOP on eligible questions, but is less
   accurate than full-page escalation here and does not cover all questions.
3. The 52 positive visual-only deltas and 76 ties motivate selective escalation:
   the next question is whether pre-action signals can identify likely gains.
4. Do not train PPO/GRPO from these results. The implemented decision has one
   acquisition decision followed by termination; a contextual-bandit framing
   is currently more appropriate. A multi-step/sequential RL formulation
   requires a genuinely sequential environment with meaningful state
   transitions and future value.
5. Increase the policy-training sample, fit a regularized predictor using only
   decision-time information, tune on `policy_validation`, and report once on
   the untouched `locked_evaluation`.

## Reproduction scripts

- `scripts/run_paired_action_benchmark.py`
- `scripts/score_paired_action_benchmark.py`
- `scripts/run_lowres_revision_control.py`
- `scripts/score_lowres_revision_control.py`
- `scripts/analyze_action_predictors.py`
- `scripts/publish_docvqa_corpus.py`

The inference runners save append-only JSONL records and keep gold labels out
of the inference process. The corpus is not committed to GitHub; use the
Hugging Face publisher to archive it in a private dataset repository. The
publisher creates a sibling staging tree, shards content-addressed image files
under two-character hash-prefix folders to respect the Hub's per-folder file
limit, hard-links images rather than making a second local copy, rewrites
staged JSON/JSONL image paths, and records per-file integrity hashes. The
original frozen corpus directory remains unchanged. Uploads use the Hub's
resumable folder upload API and are verified by remote file count and bytes.

## Data provenance

- Source dataset card: https://huggingface.co/datasets/pixparse/docvqa-single-page-questions
- Official DocVQA dataset page and download terms notice: https://site.docvqa.org/datasets
- DocVQA paper: https://arxiv.org/abs/2007.00398

The source Hugging Face card declares MIT. The official DocVQA page separately
instructs users to read challenge download terms and conditions; users remain
responsible for complying with upstream rights and terms.
