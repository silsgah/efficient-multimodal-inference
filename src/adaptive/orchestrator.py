import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.adaptive.acquisition import AcquiredInput, ImageAcquirer
from src.adaptive.controller import AdaptiveController
from src.adaptive.cost_model import ActionCostEstimator
from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    DecisionContext,
    EvidenceSignals,
    PolicyDecision,
)


@dataclass(frozen=True)
class AdaptiveRequest:
    request_id: str
    question: str
    image_path: Path
    initial_max_pixels: int = 200704
    maximum_max_pixels: int = 802816
    minimum_pixels: int = 200704

    def __post_init__(self) -> None:
        if not self.request_id or not self.question.strip():
            raise ValueError("request_id and question are required.")
        if self.initial_max_pixels <= 0 or self.initial_max_pixels % 784:
            raise ValueError("initial_max_pixels must be a positive multiple of 784.")
        if self.maximum_max_pixels < self.initial_max_pixels:
            raise ValueError("maximum_max_pixels cannot be below initial_max_pixels.")
        if self.minimum_pixels <= 0 or self.minimum_pixels > self.initial_max_pixels:
            raise ValueError("minimum_pixels must be positive and no greater than initial_max_pixels.")


@dataclass(frozen=True)
class ModelResult:
    answer: str
    latency_seconds: float
    visual_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class AdaptiveInferenceBackend(Protocol):
    def generate(self, request: AcquiredInput) -> ModelResult:
        """Generate an answer for an acquired image representation."""


class EvidenceAnalyzer(Protocol):
    def analyze(
        self,
        request: AdaptiveRequest,
        acquired_input: AcquiredInput,
        result: ModelResult,
    ) -> EvidenceSignals:
        """Return request- and image-level signals after first-pass inference."""


@dataclass(frozen=True)
class AdaptiveRunResult:
    request_id: str
    initial_result: ModelResult
    final_result: ModelResult
    decision: PolicyDecision
    followup_action: AcquisitionAction | None
    total_model_latency_seconds: float
    trace_path: Path | None = None


class AdaptiveOrchestrator:
    """One-escalation adaptive inference loop; no hidden model calls."""

    def __init__(
        self,
        backend: AdaptiveInferenceBackend,
        evidence_analyzer: EvidenceAnalyzer,
        cost_estimator: ActionCostEstimator,
        controller: AdaptiveController,
        acquirer: ImageAcquirer,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.evidence_analyzer = evidence_analyzer
        self.cost_estimator = cost_estimator
        self.controller = controller
        self.acquirer = acquirer
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None

    def run(self, request: AdaptiveRequest) -> AdaptiveRunResult:
        initial_input = AcquiredInput(
            request_id=request.request_id,
            question=request.question,
            image_path=request.image_path,
            max_pixels=request.initial_max_pixels,
            min_pixels=request.minimum_pixels,
            action=AcquisitionAction.FIRST_PASS,
        )
        first_result = self.backend.generate(initial_input)
        signals = self.evidence_analyzer.analyze(
            request,
            initial_input,
            first_result,
        )

        context = DecisionContext(
            request_id=request.request_id,
            question=request.question,
            image_path=request.image_path,
            current_max_pixels=request.initial_max_pixels,
            maximum_max_pixels=request.maximum_max_pixels,
            evidence=signals,
            current_visual_tokens=first_result.visual_tokens,
        )

        estimates: list[ActionEstimate] = []
        for action in (AcquisitionAction.FULL_UPSCALE, AcquisitionAction.ROI_CROP):
            estimate = self.cost_estimator.estimate(context, action)
            if estimate is not None:
                estimates.append(estimate)

        decision = self.controller.decide(context, estimates)
        final_result = first_result
        followup_action: AcquisitionAction | None = None

        if decision.action is not AcquisitionAction.STOP:
            acquired = self.acquirer.acquire(
                context,
                decision.action,
                roi_bbox=signals.roi_bbox,
            )
            final_result = self.backend.generate(acquired)
            followup_action = decision.action

        total_latency = (
            first_result.latency_seconds
            + (final_result.latency_seconds if followup_action is not None else 0.0)
        )

        trace_path = self._write_trace(
            request=request,
            initial_input=initial_input,
            initial_result=first_result,
            signals=signals,
            estimates=estimates,
            decision=decision,
            followup_action=followup_action,
            final_result=final_result,
            total_latency=total_latency,
        )

        return AdaptiveRunResult(
            request_id=request.request_id,
            initial_result=first_result,
            final_result=final_result,
            decision=decision,
            followup_action=followup_action,
            total_model_latency_seconds=total_latency,
            trace_path=trace_path,
        )

    def _write_trace(
        self,
        request: AdaptiveRequest,
        initial_input: AcquiredInput,
        initial_result: ModelResult,
        signals: EvidenceSignals,
        estimates: list[ActionEstimate],
        decision: PolicyDecision,
        followup_action: AcquisitionAction | None,
        final_result: ModelResult,
        total_latency: float,
    ) -> Path | None:
        if self.trace_dir is None:
            return None

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request.request_id,
            "question": request.question,
            "initial": {
                "image_path": str(initial_input.image_path),
                "max_pixels": initial_input.max_pixels,
                "answer": initial_result.answer,
                "latency_seconds": initial_result.latency_seconds,
                "visual_tokens": initial_result.visual_tokens,
            },
            "evidence_signals": asdict(signals),
            "action_estimates": [asdict(item) for item in estimates],
            "decision": {
                "action": decision.action.value,
                "reason": decision.reason,
                "utility": decision.utility,
                "rejected": list(decision.rejected),
            },
            "followup_action": (
                followup_action.value if followup_action is not None else None
            ),
            "final": {
                "answer": final_result.answer,
                "latency_seconds": final_result.latency_seconds,
                "visual_tokens": final_result.visual_tokens,
            },
            "total_model_latency_seconds": total_latency,
            "energy_measurement": None,
            "note": (
                "No per-request GPU energy is claimed. Energy must be measured "
                "using controlled aggregate workloads and the documented NVML protocol."
            ),
        }

        # Enum values nested in dataclasses need explicit JSON-safe conversion.
        for item in payload["action_estimates"]:
            item["action"] = item["action"].value
        if payload["evidence_signals"]["roi_bbox"] is not None:
            payload["evidence_signals"]["roi_bbox"] = asdict(signals.roi_bbox)

        trace_path = self.trace_dir / f"{request.request_id}.json"
        with trace_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return trace_path
