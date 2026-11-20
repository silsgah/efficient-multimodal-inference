from collections.abc import Iterable

from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    DecisionContext,
    PolicyConfig,
    PolicyDecision,
)


class AdaptiveController:
    """Transparent cost-aware selector over externally estimated action values.

    The controller deliberately does not invent quality-gain estimates. A
    calibrated estimator must supply ActionEstimate values. This keeps the
    decision rule testable and prevents unvalidated heuristic scores from
    being mistaken for empirical model confidence.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def _utility(self, estimate: ActionEstimate) -> float:
        energy_fraction = (
            estimate.incremental_energy_mwh
            / self.config.maximum_incremental_energy_mwh
        )
        latency_fraction = (
            estimate.incremental_latency_ms
            / self.config.maximum_incremental_latency_ms
        )
        return (
            estimate.expected_quality_gain
            - self.config.energy_penalty_weight * energy_fraction
            - self.config.latency_penalty_weight * latency_fraction
        )

    def decide(
        self,
        context: DecisionContext,
        estimates: Iterable[ActionEstimate],
    ) -> PolicyDecision:
        if (
            context.evidence.evidence_sufficiency
            >= self.config.stop_sufficiency_threshold
        ):
            return PolicyDecision(
                action=AcquisitionAction.STOP,
                reason="Initial evidence sufficiency meets the stop threshold.",
            )

        valid: list[tuple[float, ActionEstimate]] = []
        rejected: list[str] = []

        for estimate in estimates:
            action = estimate.action

            if action is AcquisitionAction.FULL_UPSCALE:
                if context.current_max_pixels >= min(
                    context.maximum_max_pixels,
                    self.config.upscale_max_pixels,
                ):
                    rejected.append("FULL_UPSCALE: already at the maximum resolution.")
                    continue

            elif action is AcquisitionAction.ROI_CROP:
                if context.evidence.roi_bbox is None:
                    rejected.append("ROI_CROP: no valid ROI proposal was supplied.")
                    continue
                if (
                    context.evidence.dense_information_risk
                    >= self.config.dense_information_threshold
                ):
                    rejected.append(
                        "ROI_CROP: dense-information guard vetoed lossy crop acquisition."
                    )
                    continue
            else:
                rejected.append(f"{action.value}: unsupported acquisition action.")
                continue

            if (
                estimate.expected_quality_gain
                < self.config.minimum_expected_quality_gain
            ):
                rejected.append(f"{action.value}: expected quality gain is too small.")
                continue

            if (
                estimate.incremental_energy_mwh
                > self.config.maximum_incremental_energy_mwh
            ):
                rejected.append(f"{action.value}: exceeds the energy budget.")
                continue

            if (
                estimate.incremental_latency_ms
                > self.config.maximum_incremental_latency_ms
            ):
                rejected.append(f"{action.value}: exceeds the latency budget.")
                continue

            utility = self._utility(estimate)
            if utility <= self.config.minimum_utility:
                rejected.append(f"{action.value}: non-positive net utility ({utility:.6f}).")
                continue

            valid.append((utility, estimate))

        if not valid:
            return PolicyDecision(
                action=AcquisitionAction.STOP,
                reason="No acquisition action satisfied the quality and resource constraints.",
                rejected=tuple(rejected),
            )

        # Deterministic tie breaking: utility, lower energy, lower latency.
        utility, selected = max(
            valid,
            key=lambda item: (
                item[0],
                -item[1].incremental_energy_mwh,
                -item[1].incremental_latency_ms,
                item[1].action.value,
            ),
        )
        return PolicyDecision(
            action=selected.action,
            reason="Selected the feasible action with the highest net expected utility.",
            utility=utility,
            selected_estimate=selected,
            rejected=tuple(rejected),
        )
