# GPU Energy and Visual-Token Resolution Sweep — 18 November 2026

> **Scoring correction (21 November 2026):** The accuracy figures and derived energy-per-correct-generation values in the original 18 November report below are the values as first calculated and are now superseded. A reference-aligned ANLS audit recomputed the historical request CSV without modifying it. See [the scoring and split correction addendum](ANLS_AND_SPLIT_CORRECTION_2026-11-21.md) for corrected scores, corrected derived efficiency values, input hashes, and the frozen image-disjoint pilot split. The GPU telemetry, measured energy, latency, and visual-token measurements below were not recomputed by this scoring correction.

## Summary

This experiment measured GPU board power while running the same 50-question DocVQA cohort at three image pixel budgets. Each condition used 5 warm-up requests followed by 150 measured requests (50 unique questions × 3 repetitions). The model was Qwen/Qwen2.5-VL-3B-Instruct served through the project's vLLM backend on one NVIDIA A40.

The experiment estimates incremental GPU board energy by subtracting a model-loaded idle baseline measured immediately before and after each workload. It does not measure whole-system or wall-plug energy.

## Results

| Pixel budget | Mean visual tokens/image | Exact match | ANLS-style | Mean latency | Mean GPU power | Loaded idle baseline | Incremental GPU energy | Incremental energy/request | Incremental energy/correct generation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200,704 | 245.08 | 58% (29/50) | 0.73800 | 282.0 ms | 176.50 W | 122.02 W | 0.6589 Wh | 4.393 mWh | 7.574 mWh |
| 401,408 | 484.64 | 66% (33/50) | 0.78020 | 301.0 ms | 181.41 W | 125.08 W | 0.7278 Wh | 4.852 mWh | **7.351 mWh** |
| 802,816 | 998.32 | 74% (37/50) | 0.79275 | 318.3 ms | 191.60 W | 128.52 W | 0.8483 Wh | 5.655 mWh | 7.642 mWh |

Each condition contains 150 measured generations. The number of correct generations is 87, 99 and 111 respectively. Accuracy and ANLS-style scores are calculated over 50 unique questions, not 150 repeated requests. All 50 questions produced consistent answers within each condition's three repetitions.

## Measurement method

- GPU: NVIDIA A40 (reported power limit 300 W).
- Power telemetry: NVML via pynvml, sampled at approximately one-second intervals during each workload.
- For each pixel budget, the model backend was loaded and warmed before sampling the model-loaded idle baseline.
- Idle power was sampled for approximately 10 seconds before and after the measured workload. The arithmetic mean of the pre- and post-workload idle power estimates was used as the idle baseline.
- Workload energy was integrated from the sampled power trace using trapezoidal integration.
- Incremental energy = gross workload GPU energy − idle-baseline energy over the same workload duration.
- Five warm-ups were excluded; each setting then ran 50 questions × 3 repetitions.
- Generation settings: temperature 0, top_p 1, max_new_tokens 128. Resolution budgets used min_pixels 200,704.

The idle pre/post measurements were 123.56/120.49 W, 127.64/122.52 W and 131.10/125.93 W for the three budgets. This drift is retained as a measurement limitation. The NVML readings represent GPU board power, not total server energy; the one-second telemetry cadence and interpolation make the integrated estimates approximate.

## Interpretation

Across the lowest and highest budgets, mean merged visual-token count increased by approximately 4.07×, while incremental GPU energy per request increased by approximately 28.7%. This is not a linear token-to-energy relationship: merged visual sequence positions are not the same as all pre-merger vision-encoder patch computation, and the measured workload includes other GPU inference work.

The 401,408 setting had the lowest point estimate for incremental energy per correct generation (7.351 mWh), compared with 7.574 and 7.642 mWh. These point estimates are close, and the experiment does not establish a statistically significant efficiency winner. Higher resolution improved measured exact match in this cohort, but with higher token consumption and energy per request.

The high-resolution energy run scored 37/50 (74%), whereas the earlier independent high-resolution resolution run scored 38/50 (76%). The two runs differed on one substantive answer: the earlier run answered “Bahawalpur” to the Army peas-location question, while the energy run answered “Muzaffarabad”. A fine-fragrances answer changed wording but failed both exact-match and ANLS-style thresholds in both runs. The subsequent reference-aligned scoring audit found that internal dotted-initial spacing is **not** normalized: “M.D.” and “M. D.” remain distinct character sequences. See the 21 November scoring correction addendum. The archived historical results are retained rather than silently rewritten.

## Limitations

- Only one NVIDIA A40, one model, one serving backend and one 50-question cohort were tested.
- The cohort has 50 unique questions repeated three times; repeated generations are not independent questions.
- NVML board-power measurements are sampled at approximately one-second intervals and are not equivalent to wall-plug or data-centre energy.
- The model-loaded idle baseline drifted between pre- and post-workload measurements.
- The result is a preliminary comparison, not a formal statistical demonstration that one pixel budget is more energy-efficient.
- Visual-token count is the merged visual sequence length; it does not capture all pre-merger encoder computation.
- Earlier concurrency results used a different prompt/output cap and must not be combined with these quality scores.
- Manual ROI experiments elsewhere in the project are small oracle-selected pilots, not evidence of a generalized automated ROI policy.

## Archived raw artifacts

The matching request-level outputs and telemetry are intended to be archived under:

`energy_measurement/2026-11-18/`

in the Hugging Face dataset repository:

https://huggingface.co/datasets/gahsilas/efficient-multimodal-inference-benchmarks

Expected files include `energy_requests.csv`, `gpu_power_samples.csv`, `energy_resolution_summary.json`, `idle_power_samples.csv` and `idle_power_baseline.json` (where present in the run directory).
