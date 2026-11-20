from pathlib import Path

import pytest

from src.adaptive.controller import AdaptiveController
from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    BoundingBox,
    DecisionContext,
    EvidenceSignals,
    PolicyConfig,
)


def context(
    sufficiency=0.2,
    dense_risk=0.1,
    roi=True,
    current_pixels=200704,
):
    return DecisionContext(
        request_id="q-001",
        question="What is the invoice total?",
        image_path=Path("invoice.png"),
        current_max_pixels=current_pixels,
        maximum_max_pixels=802816,
        evidence=EvidenceSignals(
            evidence_sufficiency=sufficiency,
            dense_information_risk=dense_risk,
            roi_bbox=BoundingBox(10, 10, 100, 100) if roi else None,
            source="test-fixture",
        ),
    )


def estimate(action, gain, energy, latency, tokens=100, estimator="test-estimator"):
    return ActionEstimate(
        action=action,
        expected_quality_gain=gain,
        incremental_energy_mwh=energy,
        incremental_latency_ms=latency,
        additional_visual_tokens=tokens,
        estimator_id=estimator,
    )


def test_stop_when_evidence_is_sufficient():
    controller = AdaptiveController()
    decision = controller.decide(
        context(sufficiency=0.95),
        [estimate(AcquisitionAction.FULL_UPSCALE, 0.5, 1, 100)],
    )
    assert decision.action is AcquisitionAction.STOP


def test_dense_information_vetoes_roi_but_allows_full_upscale():
    controller = AdaptiveController()
    decision = controller.decide(
        context(dense_risk=0.9),
        [
            estimate(AcquisitionAction.ROI_CROP, 0.4, 0.1, 50),
            estimate(AcquisitionAction.FULL_UPSCALE, 0.2, 1.0, 200),
        ],
    )
    assert decision.action is AcquisitionAction.FULL_UPSCALE
    assert any("dense-information guard" in reason for reason in decision.rejected)


def test_roi_selected_when_highest_net_utility():
    controller = AdaptiveController()
    decision = controller.decide(
        context(),
        [
            estimate(AcquisitionAction.ROI_CROP, 0.35, 0.4, 100, 120),
            estimate(AcquisitionAction.FULL_UPSCALE, 0.25, 2.0, 350, 500),
        ],
    )
    assert decision.action is AcquisitionAction.ROI_CROP
    assert decision.selected_estimate is not None


def test_roi_requires_bbox():
    controller = AdaptiveController()
    decision = controller.decide(
        context(roi=False),
        [estimate(AcquisitionAction.ROI_CROP, 0.5, 0.1, 50)],
    )
    assert decision.action is AcquisitionAction.STOP
    assert any("no valid ROI" in reason for reason in decision.rejected)


def test_no_escalation_when_already_at_max_resolution():
    controller = AdaptiveController()
    decision = controller.decide(
        context(current_pixels=802816),
        [estimate(AcquisitionAction.FULL_UPSCALE, 0.5, 0.1, 10)],
    )
    assert decision.action is AcquisitionAction.STOP


def test_action_over_budget_is_rejected():
    config = PolicyConfig(maximum_incremental_energy_mwh=2.0)
    controller = AdaptiveController(config)
    decision = controller.decide(
        context(),
        [estimate(AcquisitionAction.FULL_UPSCALE, 0.8, 3.0, 50)],
    )
    assert decision.action is AcquisitionAction.STOP
    assert any("energy budget" in reason for reason in decision.rejected)


def test_non_positive_utility_stops():
    config = PolicyConfig(
        energy_penalty_weight=1.0,
        latency_penalty_weight=1.0,
    )
    controller = AdaptiveController(config)
    decision = controller.decide(
        context(),
        [estimate(AcquisitionAction.FULL_UPSCALE, 0.05, 10, 1000)],
    )
    assert decision.action is AcquisitionAction.STOP


def test_invalid_evidence_score_rejected():
    with pytest.raises(ValueError):
        EvidenceSignals(
            evidence_sufficiency=1.1,
            dense_information_risk=0.0,
        )
