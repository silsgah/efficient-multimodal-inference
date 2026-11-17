# Multimodal Inference Concurrency Benchmark — 17 November 2026

## Experiment setup

- Model: Qwen/Qwen2.5-VL-3B-Instruct
- GPU: NVIDIA A40
- Image pixel budget: 802,816
- Cohort: 50 unique DocVQA questions, repeated three times
- Measured requests: 150 per concurrency level (600 total)
- Warmups: 5 per concurrency level
- Serving interface: vLLM OpenAI-compatible chat completions API
- Client: asynchronous Python client
- Prompt: concise direct-answer instruction
- Temperature: 0; top_p: 1; max output tokens: 32

## Results

| Concurrency | Successful | Failed | Throughput (req/s) | Mean latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Output tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150 | 0 | 2.94 | 340.2 | 328.0 | 557.6 | 627.5 | 19.93 |
| 2 | 150 | 0 | 5.25 | 380.0 | 363.7 | 575.1 | 820.2 | 35.61 |
| 4 | 150 | 0 | 7.32 | 533.6 | 556.1 | 802.2 | 931.4 | 49.64 |
| 8 | 150 | 0 | 8.63 | 885.1 | 878.2 | 1226.7 | 1341.4 | 58.48 |

## Interpretation

Throughput increased from 2.94 requests/s at concurrency 1 to 8.63 requests/s at concurrency 8 (approximately 2.94x). Mean latency increased from 340 ms to 885 ms, and P95 latency increased from 558 ms to 1,227 ms. All 600 measured requests succeeded.

This indicates that concurrency improved aggregate throughput on this single A40 setup, with diminishing throughput gains and increasing per-request latency. It does not establish a general production capacity limit or prove GPU saturation.

## Limitations

- Client-observed end-to-end latency only; TTFT and TPOT were not instrumented.
- GPU utilization and peak GPU memory were not sampled during this run.
- Actual visual token counts were not exposed by this benchmark.
- Accuracy was not evaluated in this concurrency experiment.
- The cohort contains 50 unique questions repeated three times, not 150 independent questions.
- The prompt and 32-token response cap differ from the earlier resolution sweep; do not combine their accuracy scores as if prompts were identical.
- This is a preliminary systems benchmark on one model, GPU, dataset cohort and serving configuration.

## Artifacts

The full request-level CSV, summary JSON, executed benchmark script and project-local result directory are archived in the Hugging Face dataset repository:
https://huggingface.co/datasets/gahsilas/efficient-multimodal-inference-benchmarks

Local project path:
results/concurrency_20261117_124612/
