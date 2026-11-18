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

### Resolution sweep

Three pixel budgets were evaluated with 150 measured executions per setting (50 questions repeated three times).

| Pixel budget | Exact match | ANLS-style score | Mean wall latency |
|---:|---:|---:|---:|
| 200,704 | 58% | 0.7380 | 266 ms |
| 401,408 | 66% | 0.7802 | 298 ms |
| 802,816 | 76% | 0.8128 | 309 ms |

The locally implemented ANLS-style score is a project metric, not a claim of an official DocVQA evaluation. Accuracy is calculated over 50 unique questions, not 150 repeated executions.

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

### GPU energy and visual-token sweep

On 18 November 2026, GPU board power was sampled with NVML while the same 50-question cohort was evaluated at the three image pixel budgets (150 measured requests per condition).

| Pixel budget | Mean visual tokens/image | Exact match | ANLS-style | Mean latency | Incremental energy/request | Incremental energy/correct generation |
|---:|---:|---:|---:|---:|---:|---:|
| 200,704 | 245.08 | 58% | 0.7380 | 282 ms | 4.393 mWh | 7.574 mWh |
| 401,408 | 484.64 | 66% | 0.7802 | 301 ms | 4.852 mWh | **7.351 mWh** |
| 802,816 | 998.32 | 74% | 0.79275 | 318 ms | 5.655 mWh | 7.642 mWh |

Energy is an estimate of incremental GPU board energy, not whole-system energy. It is calculated by integrating approximately one-second NVML power samples and subtracting a model-loaded idle baseline measured before and after each workload. Pre/post idle drift and the short 50-question cohort limit the strength of the conclusions. The medium setting has the lowest energy-per-correct point estimate, but the differences are small and do not establish a statistically significant winner.

The energy run at 802,816 pixels produced 37/50 correct answers, compared with 38/50 in the earlier independent resolution run. One substantive answer changed (Bahawalpur to Muzaffarabad on the Army peas-location question). The runs are preserved separately; the original result has not been overwritten.

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

