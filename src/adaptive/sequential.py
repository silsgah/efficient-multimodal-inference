"""Bounded sequential visual-acquisition episodes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence

from src.adaptive.acquisition import AcquiredInput, ImageAcquirer
from src.adaptive.cost_model import ActionCostEstimator
from src.adaptive.energy import EnergyMeter, GpuEnergyMeasurement
from src.adaptive.orchestrator import (
    AdaptiveInferenceBackend,
    AdaptiveRequest,
    EvidenceAnalyzer,
    ModelResult,
)
from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    DecisionContext,
    PolicyDecision,
)


@dataclass(frozen=True)
class SequentialEpisodeConfig:
    """Episode limits. An energy budget requires an active NVML meter."""

    max_acquisition_actions: int = 2
    energy_budget_mwh: float | None = None
    latency_budget_ms: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.max_acquisition_actions <= 2:
            raise ValueError(
                "max_acquisition_actions must be between 0 and 2 because "
                "FULL_UPSCALE and ROI_CROP are each available at most once."
            )
        if self.energy_budget_mwh is not None and self.energy_budget_mwh <= 0:
            raise ValueError("energy_budget_mwh must be positive when supplied.")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive when supplied.")


@dataclass(frozen=True)
class EpisodeObservation:
    """Policy-visible state; contains no ground-truth answer or source metadata."""

    context: DecisionContext
    current_input: AcquiredInput
    current_result: ModelResult
    timestep: int
    acquisition_actions_taken: tuple[AcquisitionAction, ...]
    cumulative_wall_latency_ms: float
    cumulative_model_latency_seconds: float
    cumulative_energy_mwh: float | None
    estimated_acquisition_energy_mwh: float
    estimated_acquisition_latency_ms: float
    remaining_energy_budget_mwh: float | None
    remaining_latency_budget_ms: float | None


class SequentialPolicy(Protocol):
    def decide(
        self,
        observation: EpisodeObservation,
        estimates: Sequence[ActionEstimate],
    ) -> PolicyDecision:
        """Choose the next action from the currently feasible candidates."""


class HeuristicSequentialPolicy:
    """Adapter that preserves AdaptiveController as a sequential baseline."""

    def __init__(self, controller) -> None:
        self.controller = controller

    def decide(
        self,
        observation: EpisodeObservation,
        estimates: Sequence[ActionEstimate],
    ) -> PolicyDecision:
        return self.controller.decide(observation.context, estimates)


@dataclass(frozen=True)
class SequentialEpisodeResult:
    request_id: str
    initial_result: ModelResult
    final_result: ModelResult
    acquisition_actions: tuple[AcquisitionAction, ...]
    total_model_latency_seconds: float
    total_wall_latency_ms: float
    measured_gpu_energy_mwh: float | None
    estimated_acquisition_energy_mwh: float
    estimated_acquisition_latency_ms: float
    termination_reason: str
    energy_budget_guard_enabled: bool
    latency_budget_guard_enabled: bool
    budget_exceeded: bool
    steps: tuple[dict[str, Any], ...]
    trace_path: Path | None = None


class SequentialEpisodeRunner:
    """Run a first pass followed by zero to two policy-controlled acquisitions.

    Each acquisition is followed by a fresh VLM inference and evidence analysis.
    The policy therefore sees the updated answer and evidence at its next
    decision. Action estimates filter candidate actions before execution.
    Measured energy and wall latency are recorded separately from estimates.

    Budget guards prevent scheduling actions whose predicted incremental costs
    exceed the remaining budget. Since an inference call cannot be preempted
    safely, actual measurements can still reveal an overrun; such overruns are
    explicitly recorded and terminate the episode.
    """

    def __init__(
        self,
        backend: AdaptiveInferenceBackend,
        evidence_analyzer: EvidenceAnalyzer,
        cost_estimator: ActionCostEstimator,
        policy: SequentialPolicy,
        acquirer: ImageAcquirer,
        config: SequentialEpisodeConfig | None = None,
        energy_meter: EnergyMeter | None = None,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.evidence_analyzer = evidence_analyzer
        self.cost_estimator = cost_estimator
        self.policy = policy
        self.acquirer = acquirer
        self.config = config or SequentialEpisodeConfig()
        self.energy_meter = energy_meter
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None

        if self.config.energy_budget_mwh is not None and self.energy_meter is None:
            raise ValueError(
                "An energy budget requires a measured EnergyMeter. "
                "Estimated action costs alone are not accepted as measured energy."
            )

    def _measure(
        self,
        operation,
        label: str,
    ) -> tuple[Any, GpuEnergyMeasurement | None, float]:
        started = perf_counter()
        if self.energy_meter is None:
            value = operation()
            measurement = None
        else:
            value, measurement = self.energy_meter.measure(
                operation,
                label=label,
            )
        wall_latency_ms = (perf_counter() - started) * 1000.0
        return value, measurement, wall_latency_ms

    @staticmethod
    def _evidence_to_dict(evidence) -> dict[str, Any]:
        bbox = evidence.roi_bbox
        return {
            "evidence_sufficiency": evidence.evidence_sufficiency,
            "dense_information_risk": evidence.dense_information_risk,
            "query_complexity": evidence.query_complexity,
            "roi_bbox": asdict(bbox) if bbox is not None else None,
            "source": evidence.source,
        }

    @staticmethod
    def _estimate_to_dict(estimate: ActionEstimate | None) -> dict[str, Any] | None:
        if estimate is None:
            return None
        return {
            "action": estimate.action.value,
            "expected_quality_gain": estimate.expected_quality_gain,
            "incremental_energy_mwh": estimate.incremental_energy_mwh,
            "incremental_latency_ms": estimate.incremental_latency_ms,
            "additional_visual_tokens": estimate.additional_visual_tokens,
            "estimator_id": estimate.estimator_id,
        }

    @classmethod
    def _decision_to_dict(cls, decision: PolicyDecision) -> dict[str, Any]:
        return {
            "action": decision.action.value,
            "reason": decision.reason,
            "utility": decision.utility,
            "selected_estimate": cls._estimate_to_dict(
                decision.selected_estimate
            ),
            "rejected": list(decision.rejected),
        }

    @staticmethod
    def _measurement_to_dict(
        measurement: GpuEnergyMeasurement | None,
    ) -> dict[str, Any] | None:
        return asdict(measurement) if measurement is not None else None

    @classmethod
    def _step_record(
        cls,
        step_index: int,
        acquired: AcquiredInput,
        result: ModelResult,
        evidence,
        wall_latency_ms: float,
        energy: GpuEnergyMeasurement | None,
    ) -> dict[str, Any]:
        return {
            "step": step_index,
            "action": acquired.action.value,
            "image_path": str(acquired.image_path),
            "min_pixels": acquired.min_pixels,
            "max_pixels": acquired.max_pixels,
            "answer": result.answer,
            "model_latency_seconds": result.latency_seconds,
            "wall_latency_ms": wall_latency_ms,
            "visual_tokens": result.visual_tokens,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "evidence_signals": cls._evidence_to_dict(evidence),
            "energy_measurement": cls._measurement_to_dict(energy),
            "decision_after_step": None,
        }

    def run(self, request: AdaptiveRequest) -> SequentialEpisodeResult:
        source_image = Path(request.image_path)
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        episode_started = perf_counter()
        initial_input = AcquiredInput(
            request_id=request.request_id,
            question=request.question,
            image_path=source_image,
            max_pixels=request.initial_max_pixels,
            min_pixels=request.minimum_pixels,
            action=AcquisitionAction.FIRST_PASS,
        )

        def initial_operation():
            result = self.backend.generate(initial_input)
            evidence = self.evidence_analyzer.analyze(
                request,
                initial_input,
                result,
            )
            return result, evidence

        (initial_result, initial_evidence), initial_energy, initial_wall_ms = (
            self._measure(initial_operation, "first_pass")
        )

        current_input = initial_input
        current_result = initial_result
        current_evidence = initial_evidence
        current_max_pixels = request.initial_max_pixels
        total_model_latency_seconds = initial_result.latency_seconds
        measured_energy_mwh = (
            initial_energy.energy_mwh if initial_energy is not None else None
        )
        estimated_acquisition_energy_mwh = 0.0
        estimated_acquisition_latency_ms = 0.0
        acquisition_actions: list[AcquisitionAction] = []
        steps: list[dict[str, Any]] = [
            self._step_record(
                0,
                initial_input,
                initial_result,
                initial_evidence,
                initial_wall_ms,
                initial_energy,
            )
        ]
        termination_reason: str | None = None

        def elapsed_wall_ms() -> float:
            return (perf_counter() - episode_started) * 1000.0

        def check_budget() -> str | None:
            elapsed = elapsed_wall_ms()
            if (
                self.config.latency_budget_ms is not None
                and elapsed >= self.config.latency_budget_ms
            ):
                return "latency_budget_exhausted"
            if (
                self.config.energy_budget_mwh is not None
                and measured_energy_mwh is not None
                and measured_energy_mwh >= self.config.energy_budget_mwh
            ):
                return "energy_budget_exhausted"
            return None

        termination_reason = check_budget()

        while (
            termination_reason is None
            and len(acquisition_actions) < self.config.max_acquisition_actions
        ):
            context = DecisionContext(
                request_id=request.request_id,
                question=request.question,
                # Always retain the original full-page coordinate frame.
                image_path=source_image,
                current_max_pixels=current_max_pixels,
                maximum_max_pixels=request.maximum_max_pixels,
                evidence=current_evidence,
                current_visual_tokens=current_result.visual_tokens,
            )

            elapsed = elapsed_wall_ms()
            remaining_latency = (
                self.config.latency_budget_ms - elapsed
                if self.config.latency_budget_ms is not None
                else None
            )
            remaining_energy = (
                self.config.energy_budget_mwh - measured_energy_mwh
                if (
                    self.config.energy_budget_mwh is not None
                    and measured_energy_mwh is not None
                )
                else None
            )

            feasible_estimates: list[ActionEstimate] = []
            used_actions = set(acquisition_actions)

            for action in (
                AcquisitionAction.FULL_UPSCALE,
                AcquisitionAction.ROI_CROP,
            ):
                if action in used_actions:
                    continue
                if (
                    action is AcquisitionAction.FULL_UPSCALE
                    and current_max_pixels
                    >= min(
                        request.maximum_max_pixels,
                        self.acquirer.upscale_max_pixels,
                    )
                ):
                    continue
                if (
                    action is AcquisitionAction.ROI_CROP
                    and current_evidence.roi_bbox is None
                ):
                    continue

                estimate = self.cost_estimator.estimate(context, action)
                if estimate is None:
                    continue
                if estimate.action is not action:
                    raise ValueError(
                        "Cost estimator returned an estimate for the wrong action."
                    )
                if (
                    remaining_latency is not None
                    and estimate.incremental_latency_ms > remaining_latency
                ):
                    continue
                if (
                    remaining_energy is not None
                    and estimate.incremental_energy_mwh > remaining_energy
                ):
                    continue
                feasible_estimates.append(estimate)

            observation = EpisodeObservation(
                context=context,
                current_input=current_input,
                current_result=current_result,
                timestep=len(steps) - 1,
                acquisition_actions_taken=tuple(acquisition_actions),
                cumulative_wall_latency_ms=elapsed,
                cumulative_model_latency_seconds=total_model_latency_seconds,
                cumulative_energy_mwh=measured_energy_mwh,
                estimated_acquisition_energy_mwh=estimated_acquisition_energy_mwh,
                estimated_acquisition_latency_ms=estimated_acquisition_latency_ms,
                remaining_energy_budget_mwh=remaining_energy,
                remaining_latency_budget_ms=remaining_latency,
            )
            decision = self.policy.decide(observation, feasible_estimates)
            steps[-1]["decision_after_step"] = self._decision_to_dict(decision)

            if decision.action is AcquisitionAction.STOP:
                termination_reason = "policy_stop"
                break
            if decision.action not in (
                AcquisitionAction.FULL_UPSCALE,
                AcquisitionAction.ROI_CROP,
            ):
                raise ValueError(
                    f"Policy returned unsupported episode action: {decision.action}"
                )

            selected_estimate = next(
                (
                    item for item in feasible_estimates
                    if item.action is decision.action
                ),
                None,
            )
            if selected_estimate is None:
                raise ValueError(
                    "Policy selected an action outside the feasible candidate set. "
                    "Learned policies must respect the action mask."
                )

            def acquisition_operation():
                acquired = self.acquirer.acquire(
                    context,
                    decision.action,
                    roi_bbox=current_evidence.roi_bbox,
                )
                result = self.backend.generate(acquired)
                evidence = self.evidence_analyzer.analyze(
                    request,
                    acquired,
                    result,
                )
                return acquired, result, evidence

            (acquired, result, evidence), energy, wall_ms = self._measure(
                acquisition_operation,
                decision.action.value,
            )

            acquisition_actions.append(decision.action)
            estimated_acquisition_energy_mwh += (
                selected_estimate.incremental_energy_mwh
            )
            estimated_acquisition_latency_ms += (
                selected_estimate.incremental_latency_ms
            )
            total_model_latency_seconds += result.latency_seconds
            if energy is not None:
                measured_energy_mwh += energy.energy_mwh

            steps.append(
                self._step_record(
                    len(steps),
                    acquired,
                    result,
                    evidence,
                    wall_ms,
                    energy,
                )
            )
            current_input = acquired
            current_result = result
            current_evidence = evidence

            # An ROI does not change the source-page resolution state. A
            # subsequent full-page action therefore uses the original image.
            if decision.action is AcquisitionAction.FULL_UPSCALE:
                current_max_pixels = acquired.max_pixels

            termination_reason = check_budget()
            if termination_reason is not None:
                break

        if termination_reason is None:
            if len(acquisition_actions) >= self.config.max_acquisition_actions:
                termination_reason = "maximum_acquisition_actions_reached"
            else:
                termination_reason = "policy_stop"

        total_wall_latency_ms = elapsed_wall_ms()
        energy_budget_guard_enabled = self.config.energy_budget_mwh is not None
        latency_budget_guard_enabled = self.config.latency_budget_ms is not None
        budget_exceeded = (
            (
                energy_budget_guard_enabled
                and measured_energy_mwh is not None
                and measured_energy_mwh > self.config.energy_budget_mwh
            )
            or (
                latency_budget_guard_enabled
                and total_wall_latency_ms > self.config.latency_budget_ms
            )
        )

        result = SequentialEpisodeResult(
            request_id=request.request_id,
            initial_result=initial_result,
            final_result=current_result,
            acquisition_actions=tuple(acquisition_actions),
            total_model_latency_seconds=total_model_latency_seconds,
            total_wall_latency_ms=total_wall_latency_ms,
            measured_gpu_energy_mwh=measured_energy_mwh,
            estimated_acquisition_energy_mwh=estimated_acquisition_energy_mwh,
            estimated_acquisition_latency_ms=estimated_acquisition_latency_ms,
            termination_reason=termination_reason,
            energy_budget_guard_enabled=energy_budget_guard_enabled,
            latency_budget_guard_enabled=latency_budget_guard_enabled,
            budget_exceeded=budget_exceeded,
            steps=tuple(steps),
        )

        trace_path = self._write_trace(request, result)
        return replace(result, trace_path=trace_path)

    def _write_trace(
        self,
        request: AdaptiveRequest,
        result: SequentialEpisodeResult,
    ) -> Path | None:
        if self.trace_dir is None:
            return None

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        safe_id = hashlib.sha256(
            request.request_id.encode("utf-8")
        ).hexdigest()[:20]
        trace_path = self.trace_dir / f"{safe_id}.json"

        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request.request_id,
            "question": request.question,
            "budgets": {
                "max_acquisition_actions": self.config.max_acquisition_actions,
                "energy_budget_mwh": self.config.energy_budget_mwh,
                "latency_budget_ms": self.config.latency_budget_ms,
                "energy_budget_guard_enabled": result.energy_budget_guard_enabled,
                "latency_budget_guard_enabled": result.latency_budget_guard_enabled,
            },
            "steps": list(result.steps),
            "acquisition_actions": [
                action.value for action in result.acquisition_actions
            ],
            "final_answer": result.final_result.answer,
            "total_model_latency_seconds": result.total_model_latency_seconds,
            "total_wall_latency_ms": result.total_wall_latency_ms,
            "measured_gpu_energy_mwh": result.measured_gpu_energy_mwh,
            "estimated_acquisition_energy_mwh": (
                result.estimated_acquisition_energy_mwh
            ),
            "estimated_acquisition_latency_ms": (
                result.estimated_acquisition_latency_ms
            ),
            "termination_reason": result.termination_reason,
            "budget_exceeded": result.budget_exceeded,
            "energy_measurement_note": (
                "Energy was recorded by the configured EnergyMeter. Interpret "
                "its attribution and integration method using each step's "
                "energy_measurement metadata."
                if self.energy_meter is not None
                else "No GPU energy meter was active; energy is unmeasured."
            ),
        }
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return trace_path
