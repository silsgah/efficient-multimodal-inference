"""CPU-only smoke test for adaptive decision and orchestration plumbing.

Uses synthetic model outputs and explicitly synthetic action estimates. It does
not load a VLM, measure energy, or produce research results.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from src.adaptive.acquisition import ImageAcquirer
from src.adaptive.controller import AdaptiveController
from src.adaptive.orchestrator import (
    AdaptiveOrchestrator,
    AdaptiveRequest,
    ModelResult,
)
from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    BoundingBox,
    EvidenceSignals,
    PolicyConfig,
)


class SyntheticBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return ModelResult(
            answer="uncertain" if self.calls == 1 else "synthetic follow-up answer",
            latency_seconds=0.01,
            visual_tokens=250 if self.calls == 1 else 180,
            input_tokens=400,
            output_tokens=3,
        )


class SyntheticEvidenceAnalyzer:
    def analyze(self, request, acquired_input, result):
        return EvidenceSignals(
            evidence_sufficiency=0.25,
            dense_information_risk=0.2,
            query_complexity=0.8,
            roi_bbox=BoundingBox(40, 30, 220, 150),
            source="synthetic_cpu_smoke_test",
        )


class SyntheticEstimator:
    estimator_id = "synthetic-only"

    def estimate(self, context, action):
        if action is AcquisitionAction.ROI_CROP:
            return ActionEstimate(action, 0.30, 0.5, 100.0, 180, self.estimator_id)
        if action is AcquisitionAction.FULL_UPSCALE:
            return ActionEstimate(action, 0.20, 2.0, 300.0, 500, self.estimator_id)
        return None


def main():
    with tempfile.TemporaryDirectory(prefix="adaptive-cpu-") as temp:
        root = Path(temp)
        image_path = root / "sample.png"
        Image.new("RGB", (300, 200), color="white").save(image_path)

        orchestrator = AdaptiveOrchestrator(
            backend=SyntheticBackend(),
            evidence_analyzer=SyntheticEvidenceAnalyzer(),
            cost_estimator=SyntheticEstimator(),
            controller=AdaptiveController(PolicyConfig()),
            acquirer=ImageAcquirer(root / "crops"),
            trace_dir=root / "traces",
        )
        result = orchestrator.run(
            AdaptiveRequest(
                request_id="cpu-smoke-001",
                question="What date is printed?",
                image_path=image_path,
            )
        )

        report = {
            "status": "passed",
            "mode": "synthetic_cpu_smoke_test",
            "calls": 2,
            "chosen_action": result.decision.action.value,
            "initial_answer": result.initial_result.answer,
            "final_answer": result.final_result.answer,
            "trace_written": result.trace_path is not None and result.trace_path.exists(),
            "research_measurements": False,
            "energy_measurement": None,
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
