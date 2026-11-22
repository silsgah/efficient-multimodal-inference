# Efficient Multimodal Inference

**Visual Token Economics in Vision-Language Model Serving**

This project evaluates how image resolution and concurrent request load affect the quality, latency and throughput of a vision-language model served through vLLM.

## Research question

How do visual input settings and concurrent multimodal workloads affect inference efficiency and task quality?

## Current experimental setup

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- GPU: NVIDIA A40
- Serving engine: vLLM OpenAI-compatible API
- Evaluation cohort: 50 unique DocVQA questions
- Image pixel budgets: 200,704; 401,408; 802,816

## Completed experiments

### Earlier independent resolution run (preserved separately)

Three pixel budgets were evaluated with 150 measured executions per setting (50 questions repeated three times).

| Pixel budget | Exact match | ANLS-style score | Mean wall latency |
|---:|---:|---:|---:|
| 200,704 | 58% | 0.7380 | 266 ms |
| 401,408 | 66% | 0.7802 | 298 ms |
| 802,816 | 76% | 0.8128 | 309 ms |

The table above is the earlier independent fixed-resolution run. It is distinct from the 18 November energy-measurement run and was not part of the 21 November request-CSV scorer audit. Preserve it as a separate historical experiment; do not combine its metrics with the energy run. Accuracy is calculated over 50 unique questions, not 150 repeated executions.

### Concurrency sweep

At the 802,816-pixel setting, the same 50-question cohort was sent through the vLLM OpenAI-compatible API at concurrency 1, 2, 4 and 8 (150 measured requests per level).

| Concurrency | Throughput (req/s) | Mean latency (ms) | P95 (ms) | Failures |
|---:|---:|---:|---:|---:|
| 1 | 2.94 | 340 | 558 | 0 |
| 2 | 5.25 | 380 | 575 | 0 |
| 4 | 7.32 | 534 | 802 | 0 |
| 8 | 8.63 | 885 | 1,227 | 0 |

Across 600 measured requests, no requests failed. Throughput increased with concurrency, while mean and tail latency also increased. This is a preliminary result for one model, one GPU and this workload; it does not establish a general production capacity limit.

Detailed concurrency methodology and interpretation: [Concurrency benchmark report](reports/CONCURRENCY_RESULTS_2026-11-17.md).

## Adaptive visual evidence acquisition — research extension

The original resolution, concurrency, visual-token, ROI-pilot, and GPU-energy experiments are preserved as historical baselines. They are not replaced by adaptive-method results. The next research phase develops an energy-aware controller for document-centric visual evidence acquisition.

The working experimental protocol is [METHOD_AND_EXPERIMENT_PROTOCOL.md](docs/METHOD_AND_EXPERIMENT_PROTOCOL.md). It defines the action space, comparison baselines, data split, metrics, energy-measurement constraints, and implementation milestones. The protocol is a draft pending data and evaluation freeze; no adaptive method results are claimed yet.

### GPU energy and visual-token sweep

On 18 November 2026, GPU board power was sampled with NVML while the same 50-question cohort was evaluated at the three image pixel budgets (150 measured requests per condition).

| Pixel budget | Mean visual tokens/image | Exact match | ANLS-style | Mean latency | Incremental energy/request | Incremental energy/correct generation |
|---:|---:|---:|---:|---:|---:|---:|
| 200,704 | 245.08 | 56% (28/50) | 0.725571 | 282 ms | 4.393 mWh | 7.844 mWh |
| 401,408 | 484.64 | 66% (33/50) | 0.769369 | 301 ms | 4.852 mWh | **7.352 mWh** |
| 802,816 | 998.32 | 72% (36/50) | 0.770841 | 318 ms | 5.655 mWh | 7.855 mWh |

Energy is an estimate of incremental GPU board energy, not whole-system energy. It is calculated by integrating approximately one-second NVML power samples and subtracting a model-loaded idle baseline measured before and after each workload. Pre/post idle drift and the short 50-question cohort limit the strength of the conclusions. The medium setting has the lowest energy-per-correct point estimate, but the differences are small and do not establish a statistically significant winner.

The archived energy-run scorer originally marked 37/50 correct at 802,816 pixels; the reference-aligned re-evaluation gives 36/50 (72%). The earlier independent resolution run reported 38/50 (76%) and has not been re-audited against the same request-level CSV. One substantive answer varied between the two runs (Bahawalpur versus Muzaffarabad on the Army peas-location question). The runs and raw results remain separate.

**Scoring correction (21 November 2026):** The energy-run ANLS and exact-match columns above have been recomputed using the reference-aligned scorer; the corrected energy-per-correct-generation values use the corrected exact-match counts. The audit processed 450 request rows across 50 unique questions and found 48 stored-score mismatches, with zero within-question/repeat inconsistencies. The historical CSV and GPU telemetry were not rewritten. The reference-aligned ANLS lowercases only; it does not strip whitespace or normalize punctuation/internal spacing. See the [ANLS and split correction addendum](reports/ANLS_AND_SPLIT_CORRECTION_2026-11-21.md) and [full corpus build protocol](docs/FULL_CORPUS_BUILD_PROTOCOL.md).


Detailed methodology, measurement caveats and interpretation: [GPU energy benchmark report](reports/ENERGY_RESULTS_2026-11-18.md). Machine-readable summary: [energy resolution summary](results/energy_measurement_2026-11-18/summary.json).

## Known measurement limitations

- The concurrency run reports client-observed end-to-end latency; TTFT and TPOT were not separately instrumented.
- GPU utilization and peak GPU memory were not sampled during the concurrency run.
- Actual visual token counts were not exposed by the benchmark.
- Accuracy was not evaluated for the concise prompt used in the concurrency sweep.
- The concurrency run's prompt and output cap differ from the resolution sweep; their accuracy figures must not be conflated.
- The cohort contains 50 unique questions repeated three times.
- Original DocVQA images and manifest are not included in the public artifact archive.

## Reproducibility and artifacts

Experiment outputs, configuration files, comparison tables and the concurrency benchmark package are archived in the [Hugging Face dataset repository](https://huggingface.co/datasets/gahsilas/efficient-multimodal-inference-benchmarks).

Repository folders:
- `configs/`: experiment and serving configurations.
- `src/`: model backends, data handling, evaluation and benchmark logic.
- `scripts/`: experiment runners.
- `tests/`: automated tests.
- `reports/`: experiment reports.

