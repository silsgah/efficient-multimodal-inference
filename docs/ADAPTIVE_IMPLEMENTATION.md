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
