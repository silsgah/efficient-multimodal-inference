from typing import Protocol

from src.adaptive.types import (
    AcquisitionAction,
    ActionEstimate,
    DecisionContext,
)


class ActionCostEstimator(Protocol):
    """Estimates expected marginal quality and resource cost for an action."""

    estimator_id: str

    def estimate(
        self,
        context: DecisionContext,
        action: AcquisitionAction,
    ) -> ActionEstimate | None:
        """Return None when the action has no valid estimate."""
