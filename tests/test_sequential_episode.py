from pathlib import Path
import json

import pytest
from PIL import Image

from src.adaptive.acquisition import ImageAcquirer
from src.adaptive.energy import GpuEnergyMeasurement
from src.adaptive.orchestrator import AdaptiveRequest, ModelResult
from src.adaptive.sequential import (
    SequentialEpisodeConfig,
    SequentialEpisodeRunner,
)
from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    BoundingBox,
    EvidenceSignals,
    PolicyDecision,
)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, acquired):
        self.calls.append(acquired)
        call_id = len(self.calls)
        return ModelResult(
            answer=f"answer-{call_id}",
            latency_seconds=0.01,
            visual_tokens=250 + call_id,
            input_tokens=20,
            output_tokens=2,
        )


class FakeAnalyzer:
    def analyze(self, request, acquired, result):
        return EvidenceSignals(
            evidence_sufficiency=0.1 * int(result.answer.split("-")[-1]),
            dense_information_risk=0.1,
            roi_bbox=BoundingBox(10, 10, 100, 100),
            source=f"fake-analyzer:{result.answer}",
        )


class FakeEstimator:
    estimator_id = "test-estimator"

    def __init__(self, energy_mwh=0.1, latency_ms=10.0):
        self.energy_mwh = energy_mwh
        self.latency_ms = latency_ms

    def estimate(self, context, action):
        return ActionEstimate(
            action=action,
            expected_quality_gain=0.4,
            incremental_energy_mwh=self.energy_mwh,
            incremental_latency_ms=self.latency_ms,
            additional_visual_tokens=200,
            estimator_id=self.estimator_id,
        )


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.observations = []
        self.candidate_actions = []

    def decide(self, observation, estimates):
        self.observations.append(observation)
        self.candidate_actions.append(tuple(item.action for item in estimates))
        action = self.actions[len(self.observations) - 1]
        if action is AcquisitionAction.STOP:
            return PolicyDecision(action=action, reason="scripted stop")
        selected = next(
            (item for item in estimates if item.action is action),
            None,
        )
        return PolicyDecision(
            action=action,
            reason="scripted test action",
            selected_estimate=selected,
        )


class FirstFeasiblePolicy:
    def decide(self, observation, estimates):
        if not estimates:
            return PolicyDecision(
                action=AcquisitionAction.STOP,
                reason="no feasible actions",
            )
        selected = estimates[0]
        return PolicyDecision(
            action=selected.action,
            reason="first feasible action",
            selected_estimate=selected,
        )


class SequencedEnergyMeter:
    def __init__(self, energies):
        self.energies = list(energies)

    def measure(self, operation, *, label):
        value = operation()
        energy = self.energies.pop(0)
        mean_power_w = energy * 3.6 / 0.01
        measurement = GpuEnergyMeasurement(
            label=label,
            energy_mwh=energy,
            duration_seconds=0.01,
            mean_power_w=mean_power_w,
            peak_power_w=mean_power_w,
            sample_count=2,
            device_index=0,
            device_name="fake-gpu",
            sampling_interval_seconds=0.05,
        )
        return value, measurement


class FixedEnergyMeter:
    def __init__(self, energy_mwh):
        self.energy_mwh = energy_mwh
        self.labels = []

    def measure(self, operation, *, label):
        value = operation()
        self.labels.append(label)
        mean_power_w = self.energy_mwh * 3.6 / 0.01
        measurement = GpuEnergyMeasurement(
            label=label,
            energy_mwh=self.energy_mwh,
            duration_seconds=0.01,
            mean_power_w=mean_power_w,
            peak_power_w=mean_power_w,
            sample_count=2,
            device_index=0,
            device_name="fake-gpu",
            sampling_interval_seconds=0.05,
        )
        return value, measurement


def make_runner(tmp_path, policy, *, config=None, meter=None, estimator=None):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 200), color="white").save(image_path)

    backend = FakeBackend()
    acquirer = ImageAcquirer(
        tmp_path / "crops",
        roi_min_pixels=200704,
        roi_max_pixels=200704,
    )
    runner = SequentialEpisodeRunner(
        backend=backend,
        evidence_analyzer=FakeAnalyzer(),
        cost_estimator=estimator or FakeEstimator(),
        policy=policy,
        acquirer=acquirer,
        config=config or SequentialEpisodeConfig(),
        energy_meter=meter,
        trace_dir=tmp_path / "traces",
    )
    request = AdaptiveRequest(
        request_id="sequential-test-01",
        question="What is printed on the page?",
        image_path=image_path,
    )
    return runner, request, backend, image_path


def test_sequential_runner_updates_observation_and_runs_two_acquisitions(tmp_path):
    policy = ScriptedPolicy([
        AcquisitionAction.ROI_CROP,
        AcquisitionAction.FULL_UPSCALE,
        AcquisitionAction.STOP,
    ])
    runner, request, backend, original = make_runner(tmp_path, policy)

    result = runner.run(request)

    assert len(backend.calls) == 3
    assert result.acquisition_actions == (
        AcquisitionAction.ROI_CROP,
        AcquisitionAction.FULL_UPSCALE,
    )
    assert result.initial_result.answer == "answer-1"
    assert result.final_result.answer == "answer-3"
    assert [obs.current_result.answer for obs in policy.observations] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert policy.observations[1].acquisition_actions_taken == (
        AcquisitionAction.ROI_CROP,
    )
    assert policy.observations[2].acquisition_actions_taken == (
        AcquisitionAction.ROI_CROP,
        AcquisitionAction.FULL_UPSCALE,
    )
    assert backend.calls[1].image_path != original
    assert backend.calls[1].max_pixels == 200704
    assert backend.calls[1].min_pixels == 200704
    assert backend.calls[2].image_path == original
    assert result.termination_reason == "policy_stop"
    assert len(result.steps) == 3
    assert result.trace_path is not None and result.trace_path.is_file()
    trace = json.loads(result.trace_path.read_text())
    assert trace["steps"][0]["decision_after_step"]["feasible_action_estimates"]


def test_sequential_runner_never_offers_an_action_twice(tmp_path):
    policy = ScriptedPolicy([
        AcquisitionAction.ROI_CROP,
        AcquisitionAction.ROI_CROP,
    ])
    runner, request, _, _ = make_runner(tmp_path, policy)

    with pytest.raises(ValueError, match="outside the feasible"):
        runner.run(request)

    assert AcquisitionAction.ROI_CROP not in policy.candidate_actions[1]


def test_energy_budget_requires_a_measured_energy_meter(tmp_path):
    with pytest.raises(ValueError, match="requires a measured EnergyMeter"):
        make_runner(
            tmp_path,
            FirstFeasiblePolicy(),
            config=SequentialEpisodeConfig(energy_budget_mwh=1.0),
        )


def test_measured_energy_budget_blocks_action_predicted_to_exceed_remaining(tmp_path):
    policy = FirstFeasiblePolicy()
    meter = FixedEnergyMeter(energy_mwh=0.6)
    runner, request, backend, _ = make_runner(
        tmp_path,
        policy,
        config=SequentialEpisodeConfig(energy_budget_mwh=1.0),
        meter=meter,
        estimator=FakeEstimator(energy_mwh=0.5, latency_ms=10),
    )

    result = runner.run(request)

    assert len(backend.calls) == 1
    assert result.measured_gpu_energy_mwh == pytest.approx(0.6)
    assert result.energy_budget_guard_enabled
    assert not result.budget_exceeded
    assert result.termination_reason == "policy_stop"


def test_latency_budget_filters_actions_using_remaining_estimated_cost(tmp_path):
    policy = FirstFeasiblePolicy()
    runner, request, backend, _ = make_runner(
        tmp_path,
        policy,
        config=SequentialEpisodeConfig(latency_budget_ms=1000),
        estimator=FakeEstimator(energy_mwh=0.1, latency_ms=2000),
    )

    result = runner.run(request)

    assert len(backend.calls) == 1
    assert result.latency_budget_guard_enabled
    assert not result.budget_exceeded
    assert result.termination_reason == "policy_stop"


def test_max_acquisition_actions_cannot_exceed_available_unique_actions():
    with pytest.raises(ValueError, match="between 0 and 2"):
        SequentialEpisodeConfig(max_acquisition_actions=3)


def test_trace_separates_unmeasured_energy_from_estimates(tmp_path):
    runner, request, _, _ = make_runner(tmp_path, FirstFeasiblePolicy())

    result = runner.run(request)
    payload = json.loads(result.trace_path.read_text())

    assert payload["measured_gpu_energy_mwh"] is None
    assert payload["estimated_acquisition_energy_mwh"] > 0
    assert "unmeasured" in payload["energy_measurement_note"]



def test_measured_energy_overrun_is_recorded_and_terminates_episode(tmp_path):
    meter = SequencedEnergyMeter([0.2, 0.5])
    runner, request, backend, _ = make_runner(
        tmp_path,
        FirstFeasiblePolicy(),
        config=SequentialEpisodeConfig(energy_budget_mwh=0.6),
        meter=meter,
        estimator=FakeEstimator(energy_mwh=0.2, latency_ms=10),
    )

    result = runner.run(request)

    assert len(backend.calls) == 2
    assert result.measured_gpu_energy_mwh == pytest.approx(0.7)
    assert result.termination_reason == "energy_budget_exhausted"
    assert result.budget_exceeded
