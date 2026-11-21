# Adaptive visual evidence acquisition implementation

## What is implemented

- Typed request, evidence, acquisition-action, cost-estimate, and policy-decision records.
- A deterministic controller that selects between STOP, FULL_UPSCALE, and ROI_CROP.
- A dense-information guard that vetoes ROI cropping when the risk signal crosses its configured threshold.
- Full-image resolution escalation through the image processor's max_pixels setting.
- ROI crop materialization with bounding-box validation; the original image is not modified.
- One-follow-up orchestration with JSON decision traces.
- VLLMAdaptiveAdapter, which maps acquired inputs to the existing model-backend interface.
- CPU unit tests and a synthetic CPU smoke test.

## What is deliberately not claimed

The current implementation does not include a calibrated evidence analyzer or action-cost predictor. Those components are explicit interfaces:

- EvidenceAnalyzer must produce evidence-sufficiency, dense-information-risk, query-complexity, and optional ROI signals.
- ActionCostEstimator must estimate marginal quality gain, incremental GPU energy, incremental latency, and visual-token change for each candidate action.

The default thresholds and penalty weights in configs/adaptive.yaml are engineering defaults only. They are not learned or experimentally validated. The synthetic CPU smoke test validates control flow only; it is not model inference and does not produce research measurements.

## First GPU validation

With the persistent RunPod Network Volume mounted at /workspace, update the repository and run:

    cd /workspace/projects/efficient-multimodal-inference
    source /opt/efficient-multimodal-inference-venv/bin/activate
    python -m pip install -r requirements/dev.txt
    python -m pytest -q
    python scripts/smoke_test_adaptive_backend.py --index 0 --max-pixels 200704

The smoke script loads the vLLM backend and performs one real model generation through VLLMAdaptiveAdapter. It does not run the adaptive policy and does not measure GPU energy.

## Research next step

Before comparing adaptive policies, freeze a disjoint evaluation cohort and implement/validate the evidence analyzer and action-cost estimator using development-only trajectories. Measure energy with randomized aggregate action batches and the loaded-model idle protocol; do not use the existing one-second NVML sampler to claim per-request energy. Keep all 50 previously used questions as development material, not as the final held-out test cohort.


## Sequential episode runner

The additive `SequentialEpisodeRunner` preserves the original one-follow-up
`AdaptiveOrchestrator` and runs a bounded sequence:

1. Initial low-resolution inference and evidence analysis.
2. Policy decision over currently feasible acquisition actions.
3. Acquisition, a fresh VLM inference, and fresh evidence analysis.
4. A new policy decision using the updated answer, evidence, action history,
   cumulative wall latency, measured energy (when available), and remaining
   budgets.
5. Termination at STOP, the acquisition limit, or a budget guard.

The initial action limit is two acquisitions. `FULL_UPSCALE` and `ROI_CROP`
are each available at most once per episode. A full-page acquisition always
uses the original page image. ROI coordinates are interpreted in the original
page coordinate frame.

`ImageAcquirer` accepts independent `roi_min_pixels` and `roi_max_pixels`.
The initial controlled comparison fixes both to 200704 pixels for ROI
acquisition while full-page escalation can use 802816 pixels. These are
processor pixel constraints, not a guarantee of the exact visual-token count.
Actual token counts must be validated against the installed vLLM/model output.

## Resource accounting and limitations

`SequentialEpisodeRunner` separates:

- measured NVML gross device energy for the initial pass and each subsequent
  acquisition/inference/analyzer operation;
- model-reported inference latency;
- end-to-end wall latency for each step and the complete episode;
- predicted marginal action energy/latency used for pre-action admission.

`NvmlEnergyMeter` integrates sampled GPU board power with the trapezoidal rule.
It reports gross device energy and does not attribute power to the inference
process. Shared-GPU activity and idle power therefore contaminate attribution;
controlled measurements should run with exclusive GPU access. Sampling
interval, device identity, sample count, mean power and peak power are retained
in the trace.

Budget guards filter actions whose predicted incremental costs exceed the
remaining measured/elapsed budget. In-flight VLM generations cannot be safely
preempted. If measured energy or wall latency overshoots after an admitted
action, the runner records `budget_exceeded` and terminates the episode.
Therefore these are admission guards with overrun detection, not hard real-time
preemption guarantees.

An energy budget cannot be configured without an `EnergyMeter`. Install the
optional NVML dependency with:

    python -m pip install -r requirements/energy.txt

The CPU tests use a deterministic fake meter; they do not constitute GPU
measurement validation.

## First sequential CPU validation

    cd /tmp/emi-anls-audit-20261121
    python -m pytest -q tests/test_adaptive_acquisition.py tests/test_adaptive_controller.py tests/test_adaptive_orchestrator.py tests/test_vllm_adaptive_adapter.py tests/test_sequential_episode.py tests/test_adaptive_energy.py
    python scripts/validate_adaptive_cpu.py

These tests verify control flow, action masking, updated observations, ROI/full
page sequencing, budget admission and energy integration math using mocks.
They do not validate Qwen image processing, actual visual-token counts, NVML
hardware sampling or learned-policy performance.
