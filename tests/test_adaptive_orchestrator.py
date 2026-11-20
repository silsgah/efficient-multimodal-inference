from pathlib import Path

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


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        answer = "uncertain" if len(self.calls) == 1 else "2026-11-20"
        return ModelResult(
            answer=answer,
            latency_seconds=0.2,
            visual_tokens=250 if len(self.calls) == 1 else 700,
            input_tokens=400,
            output_tokens=4,
        )


class FakeAnalyzer:
    def analyze(self, request, acquired_input, result):
        return EvidenceSignals(
            evidence_sufficiency=0.2,
            dense_information_risk=0.1,
            roi_bbox=BoundingBox(10, 10, 100, 100),
            source="unit-test",
        )


class FakeEstimator:
    estimator_id = "unit-test-table"

    def estimate(self, context, action):
        if action is AcquisitionAction.ROI_CROP:
            return ActionEstimate(
                action=action,
                expected_quality_gain=0.4,
                incremental_energy_mwh=0.5,
                incremental_latency_ms=100,
                additional_visual_tokens=200,
                estimator_id=self.estimator_id,
            )
        if action is AcquisitionAction.FULL_UPSCALE:
            return ActionEstimate(
                action=action,
                expected_quality_gain=0.2,
                incremental_energy_mwh=2.0,
                incremental_latency_ms=300,
                additional_visual_tokens=500,
                estimator_id=self.estimator_id,
            )
        return None


def test_orchestrator_runs_first_pass_then_roi_followup_and_writes_trace(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 200), color="white").save(image_path)

    backend = FakeBackend()
    orchestrator = AdaptiveOrchestrator(
        backend=backend,
        evidence_analyzer=FakeAnalyzer(),
        cost_estimator=FakeEstimator(),
        controller=AdaptiveController(PolicyConfig()),
        acquirer=ImageAcquirer(tmp_path / "crops"),
        trace_dir=tmp_path / "traces",
    )
    request = AdaptiveRequest(
        request_id="sample-01",
        question="What is the date?",
        image_path=image_path,
    )

    result = orchestrator.run(request)

    assert len(backend.calls) == 2
    assert result.followup_action is AcquisitionAction.ROI_CROP
    assert result.initial_result.answer == "uncertain"
    assert result.final_result.answer == "2026-11-20"
    assert result.total_model_latency_seconds == 0.4
    assert result.trace_path is not None and result.trace_path.exists()
