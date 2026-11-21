# Method and Experimental Protocol
## Energy-Aware Adaptive Visual Evidence Acquisition for Document-Centric VLM Inference

**Status:** Initial protocol draft — design freeze pending implementation review  
**Project:** Efficient Multimodal Inference  
**Date:** 2026-11-20

## 1. What happens to the initial experiments?

The completed fixed-resolution, concurrency, visual-token, ROI-pilot, and GPU-energy experiments remain part of this project. They are not being replaced, deleted, or relabelled as adaptive-method results.

They provide:
- Fixed-resolution reference points for quality, visual-token count, latency, and GPU board-energy estimates.
- A measured vLLM serving baseline and a separate concurrency experiment.
- A verified Qwen2.5-VL visual-token accounting procedure.
- Qualitative examples of OCR, small-glyph, table, and document-layout failure modes.
- A small manual/oracle ROI pilot that motivates adaptive acquisition but is not a generalizable automated-ROI result.

All historical raw files and reports are preserved in their dated result folders and public archives. New adaptive runs must be written to new, separately named result directories. Never overwrite historical CSV, JSON, or reports.

### 1.1 Historical baseline results

| Pixel budget | Mean visual tokens/image | Exact match | ANLS-style | Mean latency | Incremental GPU energy/request |
|---:|---:|---:|---:|---:|---:|
| 200,704 | 245.08 | 56% (28/50) | 0.725571 | 282.0 ms | 4.393 mWh |
| 401,408 | 484.64 | 66% (33/50) | 0.769369 | 301.0 ms | 4.852 mWh |
| 802,816 | 998.32 | 72% (36/50) | 0.770841 | 318.3 ms | 5.655 mWh |

These values are from the 18 November 2026 energy experiment on one NVIDIA A40. Each setting used 50 unique questions with three repetitions (150 requests). GPU energy is estimated from approximately one-second NVML board-power samples, subtracting a model-loaded idle baseline measured before and after each workload. It is not whole-system or wall-plug energy.

The archived energy-run scorer originally marked 37/50 correct at the highest resolution; the reference-aligned recomputation gives 36/50 (72%). The earlier independent high-resolution run reported 38/50 (76%) and has not been re-audited against the same request-level CSV. The runs are retained separately; the one-question answer variation is part of the historical record. The complete scorer audit, including 48 mismatching stored rows among 450 requests and zero repeat inconsistencies, is recorded in the [21 November correction addendum](../reports/ANLS_AND_SPLIT_CORRECTION_2026-11-21.md).

### 1.2 Important separation of historical experiments

The concurrency experiment used a different prompt and generation cap from the resolution/energy quality experiments. Its throughput and latency results are useful as a serving baseline, but its accuracy must not be combined with the resolution/energy quality results.

The initial ROI experiments used manually selected crops, including crops informed by observed errors. They are oracle-selected pilot cases and must not be used as an unbiased test of an automatic crop-selection policy.

## 2. Research question

Can a document-centric VLM serving system select the most cost-effective visual evidence acquisition action for each question, using expected answer-quality improvement relative to measured resource cost, rather than relying only on fixed resolution, crop area, or visual-token count?

The initial research objective is an inference-time adaptive controller. A learned policy is a later stage, conditional on the transparent controller producing useful and reproducible trajectories.

## 3. Scope and non-claims

The initial implementation will:
- Keep Qwen2.5-VL-3B-Instruct frozen.
- Use the existing vLLM backend and image-processing pipeline where feasible.
- Evaluate a small, explicit action set.
- Make decisions outside the VLM model weights.
- Log every decision, image transform, visual-token count, latency measurement, and available power telemetry.

The initial implementation will not:
- Claim to reproduce AdaptVision's DTPO or reinforcement-learning training.
- Claim to reproduce SieveServe's custom vLLM 0.16 integration.
- Modify the model's internal vision or language layers for token pruning.
- Treat manual ROI crops as an automatic ROI policy.
- Claim per-request energy from the current one-second NVML sampler.
- Combine historical experiments with different prompts/configurations into a single accuracy comparison.

## 4. Adaptive action space

For each image-question request, the first action is the economical low-resolution representation.

After observing the first response and evidence signals, the controller can select:

A0 — STOP: accept the current response.

A1 — FULL_UPSCALE: acquire a higher-resolution full-image representation and run a follow-up inference.

A2 — ROI_CROP: request or select a region of interest, process the crop at a defined resolution, and run a follow-up inference.

A3 — DENSE_PRESERVE: preserve a high-information representation (initially, do not apply lossy token pruning; use the configured resolution cap only when validated).

The initial transparent policy may initially support only A0 and A1 until the crop-selection mechanism is validated. A2 must not be implemented as manual/oracle ROI at evaluation time.

## 5. Controller signals and cost model

The controller must not rely on a single uncalibrated confidence score. It should log the following candidate signals separately:

- Answer-form/evidence sufficiency signals from the first model response.
- Query-side properties, including query length and question category.
- Image-side properties, including aspect ratio, effective resized dimensions, patch grid and merged visual-token count.
- OCR/document density proxies that can be computed without a second expensive VLM call.
- Model-generated uncertainty signals, only if they can be extracted reliably from the serving interface.
- Available serving context such as queue/load state, when the experiment supports controlled measurement.

For each action a, record estimates and then measurements of:
- Additional visual tokens.
- Additional wall latency and backend latency.
- Incremental GPU board energy at workload/batch level.
- Answer-quality change.

The estimated cost model is a predictor for decision-making, not a replacement for measured energy. It must be trained/calibrated only on development data and validated on disjoint data.

## 6. Candidate decision objective

A first formulation is:

a* = argmax_a [ E(Delta Q | x,a) - lambda_E E(Delta Energy | x,a) - lambda_L E(Delta Latency | x,a) ]

subject to an explicit minimum acceptable answer-quality constraint.

Definitions:
- x: observed request and first-pass state.
- a: candidate acquisition action.
- Delta Q: expected improvement in answer quality.
- Delta Energy: additional GPU board energy attributable to the action, measured at a controlled workload level.
- Delta Latency: end-to-end latency increase.
- lambda_E and lambda_L: declared operating preferences or policy hyperparameters.

This is a design hypothesis. It is not yet a trained model or validated optimization objective. We will first compare deterministic, interpretable policies and inspect the quality/resource Pareto frontier.

## 7. Baselines and experimental conditions

Required baselines:

1. FIXED_LOW: always use 200,704 maximum pixels.
2. FIXED_MEDIUM: always use 401,408 maximum pixels.
3. FIXED_HIGH: always use 802,816 maximum pixels.
4. ALWAYS_ESCALATE: low-resolution initial inference followed by high-resolution inference for every request. Its total two-pass cost must be counted.
5. SIMPLE_EVIDENCE_TRIGGER: low-resolution first pass and escalation when a frozen, transparent evidence-insufficiency rule fires.
6. ENERGY_AWARE_CONTROLLER: chooses among validated actions using expected marginal quality improvement and resource cost.

Additional comparisons:
- AdaptVision-inspired coarse-to-fine acquisition baseline, implemented without claiming to reproduce its RL-trained policy.
- SieveServe-inspired query-risk and dense-image-risk routing ablations, implemented without claiming to reproduce its full vLLM serving integration.

All methods compared for answer quality must use the same model checkpoint, prompt, decoding settings, input cohort, answer normalizer, and reference handling wherever technically possible. Any unavoidable differences must be reported rather than hidden.

## 8. Data protocol

The existing 50-question cohort has already been used for analysis, error inspection, and manual ROI design. It is development data only for the adaptive method.

Before final evaluation:
- Identify and freeze a disjoint evaluation cohort from the available DocVQA validation data.
- Keep question-level identities disjoint between development and final evaluation.
- If the remaining available manifest is insufficient, obtain a larger permitted evaluation subset.
- Verify the dataset license/terms and redistribution restrictions before publishing examples or source images.
- Do not use manually selected ROI crops or labels derived from final evaluation errors to tune the controller.

All train/development/test manifests should be versioned and hashed. The final evaluation manifest must be frozen before evaluating the final controller.

## 9. Metrics

Primary:
- Exact match using a documented, frozen normalizer and any-reference matching.
- ANLS-style score using character Levenshtein distance, maximum over references, and the stated 0.5 threshold. Clearly label this local implementation; do not call it the official DocVQA evaluator unless validated against the official evaluator.
- Incremental GPU board energy per request/workload, with the sampling/integration method disclosed.
- Energy per correct response at the cohort level.
- End-to-end latency and P50/P95/P99.
- Visual tokens per request and total visual tokens.

Secondary:
- Escalation rate.
- Action frequencies.
- Additional tokens, latency, and energy by action.
- Failure/abstention rate.
- Answer consistency across repetitions.
- Quality–energy and quality–latency Pareto frontiers.

Scoring regression tests must verify that ANLS lowercases without stripping whitespace or normalizing dotted initials (e.g. M.D. versus M. D.) and that multiple reference answers are handled by maximum-over-references scoring. Exact match remains separate and strips outer whitespace. The scorer was audited against the archived historical request-level scores on 21 November 2026; use the recomputed metrics and preserve the historical CSV unchanged.

## 10. Energy measurement protocol

The current NVML sampling approach is suitable for controlled aggregate workload comparisons, not precise sub-second per-request energy attribution.

For action-cost calibration:
- Keep the model loaded and warmed.
- Use repeated, sufficiently long action-specific workloads.
- Measure pre- and post-workload loaded-idle power.
- Sample board power at approximately one-second intervals.
- Integrate the time series and subtract the corresponding loaded-idle estimate.
- Repeat conditions and report variation/sensitivity to baseline drift.
- Keep GPU board energy distinct from server wall-plug energy.
- Randomize or counterbalance condition order when practical to reduce drift and thermal/order effects.

If action-level marginal energy cannot be measured reliably at the individual request level, fit action costs from aggregate randomized batches and report the granularity honestly.

## 11. Statistical and reporting requirements

- Unit of quality analysis: unique question, not repeated generation.
- Repetitions estimate execution variability; they are not independent questions.
- Report paired question-level differences for methods evaluated on the same items.
- Use bootstrap confidence intervals over questions for accuracy/ANLS and, where appropriate, energy metrics.
- Report the number of unique questions and repeated executions separately.
- Report energy-meter limitations and baseline drift.
- Do not claim an energy-efficiency winner from overlapping point estimates without uncertainty analysis.
- Do not infer that fewer visual tokens necessarily mean proportionally lower GPU energy.

## 12. Implementation milestones

M0 — Protocol and data freeze:
- Finalize this protocol.
- Build disjoint development/evaluation manifests.
- Freeze the scorer and verify historical score reproduction.

M1 — Controller skeleton:
- Implement typed action/state/result structures.
- Implement image acquisition operations.
- Implement orchestration and immutable structured logging.
- No GPU run required for initial unit tests.

M2 — GPU smoke test:
- Load the existing Qwen2.5-VL-3B-Instruct checkpoint on the existing A40.
- Run one low-resolution request and one controlled escalation.
- Verify image transforms, visual-token accounting, response capture and logs.

M3 — Transparent adaptive baselines:
- Implement always-escalate and simple evidence-triggered policies.
- Validate action costs using aggregate workload measurement.

M4 — Energy-aware controller:
- Add cost model and policy selection.
- Evaluate on disjoint questions.
- Produce quality–energy Pareto results.

M5 — Lightweight RL/bandit policy:
- Only after valid trajectories and a reproducible heuristic baseline exist.
- Keep the VLM frozen initially.
- Compare against the heuristic and fixed baselines.
- Do not fine-tune the 3B VLM unless results establish a need.

## 13. RunPod resource policy

Continue using the existing NVIDIA A40 48 GB pod for inference and initial policy experiments. Do not rent H100s or multi-GPU infrastructure for the initial controller. A separate multi-GPU RL setup is not justified until the lightweight-policy experiments demonstrate a limitation that requires it.

Stop the pod when it is not running experiments. Preserve the persistent Network Volume and all historical result directories.

## 14. Required reproducibility artifacts

For each new experiment, preserve:
- Protocol/version identifier.
- Code commit.
- Model and processor identifiers/revisions.
- Runtime package versions.
- Data manifest hash.
- Prompt and generation configuration.
- Per-request action and acquisition trace.
- Per-request visual-token counts.
- Request-level answers and scoring fields.
- GPU power telemetry and baseline measurements.
- Aggregate metrics and analysis scripts.

New adaptive results must use new dated directories. Historical results are immutable.

## 15. Open design decisions to resolve before M0 freeze

1. Exact disjoint DocVQA evaluation cohort and permissible dataset terms.
2. Initial evidence-insufficiency signals that are observable without another expensive model call.
3. Whether ROI selection is initially geometry/OCR-based or generated by a separate proposal model.
4. How to estimate marginal energy per action from aggregate controlled measurements.
5. ~~Exact scorer regression tests and reconciliation with historical outputs.~~ Completed 21 November 2026; see the ANLS and split correction addendum.
6. Whether the first policy is a contextual bandit or a small supervised cost/benefit predictor before any RL update.

---
This document is a working protocol, not a claim that the proposed method has already been implemented or validated.
## 16. Data and scorer status update — 21 November 2026

The historical 18 November energy request CSV was audited using the reference-aligned DocVQA scorer. Corrected quality scores are shown in Section 1.1; raw GPU energy, latency and visual-token telemetry remain unchanged. This scorer does not trim whitespace or normalize punctuation. The historical request CSV remains immutable.

The existing 200-question validation pilot has a frozen page-content split (54 development questions / 146 holdout questions; zero shared page hashes). It is a pilot partition, not the full policy-learning corpus or final locked test.

The full-corpus builder is scripts/build_docvqa_corpus.py. Its source schema and split rules are documented in [FULL_CORPUS_BUILD_PROTOCOL.md](FULL_CORPUS_BUILD_PROTOCOL.md). The source is pinned to dataset revision 13619c00b32b50729b6e50b4bb0aa4b355698f3f. Full corpus materialization and all resulting split counts remain pending; no adaptive policy training has been performed.

