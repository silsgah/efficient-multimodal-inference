from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AcquisitionAction(str, Enum):
    FIRST_PASS = "first_pass"
    STOP = "stop"
    FULL_UPSCALE = "full_upscale"
    ROI_CROP = "roi_crop"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-coordinate box in x0, y0, x1, y1 order."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(not isinstance(value, int) for value in values):
            raise TypeError("Bounding-box coordinates must be integers.")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError("Bounding-box origin cannot be negative.")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Bounding box must have positive width and height.")

    def validate_image_bounds(self, width: int, height: int) -> None:
        if self.x1 > width or self.y1 > height:
            raise ValueError(
                f"Bounding box {self} exceeds image dimensions {width}x{height}."
            )


@dataclass(frozen=True)
class EvidenceSignals:
    """Signals exposed by an evidence analyzer after the first inference.

    The fields are scores in [0, 1]. They are not assumed to be calibrated
    probabilities; calibration must be established on development data.
    """

    evidence_sufficiency: float
    dense_information_risk: float
    query_complexity: float = 0.5
    roi_bbox: BoundingBox | None = None
    source: str = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "evidence_sufficiency",
            "dense_information_risk",
            "query_complexity",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not self.source:
            raise ValueError("Evidence signal source must be identified.")


@dataclass(frozen=True)
class DecisionContext:
    request_id: str
    question: str
    image_path: Path
    current_max_pixels: int
    maximum_max_pixels: int
    evidence: EvidenceSignals
    current_visual_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required.")
        if not self.question.strip():
            raise ValueError("question cannot be empty.")
        if self.current_max_pixels <= 0 or self.maximum_max_pixels <= 0:
            raise ValueError("Pixel budgets must be positive.")
        if self.current_max_pixels % 784 or self.maximum_max_pixels % 784:
            raise ValueError("Pixel budgets must be multiples of 28 * 28.")
        if self.current_max_pixels > self.maximum_max_pixels:
            raise ValueError("Current pixel budget exceeds the maximum budget.")
        if (
            self.current_visual_tokens is not None
            and self.current_visual_tokens < 0
        ):
            raise ValueError("current_visual_tokens cannot be negative.")


@dataclass(frozen=True)
class ActionEstimate:
    """Estimated marginal change relative to the current first-pass state."""

    action: AcquisitionAction
    expected_quality_gain: float
    incremental_energy_mwh: float
    incremental_latency_ms: float
    additional_visual_tokens: int
    estimator_id: str

    def __post_init__(self) -> None:
        if self.action is AcquisitionAction.STOP:
            raise ValueError("STOP is not an acquisition action estimate.")
        if not 0.0 <= self.expected_quality_gain <= 1.0:
            raise ValueError("expected_quality_gain must be in [0, 1].")
        if self.incremental_energy_mwh < 0:
            raise ValueError("incremental_energy_mwh cannot be negative.")
        if self.incremental_latency_ms < 0:
            raise ValueError("incremental_latency_ms cannot be negative.")
        if self.additional_visual_tokens < 0:
            raise ValueError("additional_visual_tokens cannot be negative.")
        if not self.estimator_id:
            raise ValueError("estimator_id is required.")


@dataclass(frozen=True)
class PolicyConfig:
    stop_sufficiency_threshold: float = 0.85
    dense_information_threshold: float = 0.55
    minimum_expected_quality_gain: float = 0.02
    maximum_incremental_energy_mwh: float = 20.0
    maximum_incremental_latency_ms: float = 1500.0
    energy_penalty_weight: float = 0.15
    latency_penalty_weight: float = 0.15
    minimum_utility: float = 0.0
    upscale_max_pixels: int = 802816

    def __post_init__(self) -> None:
        for name in (
            "stop_sufficiency_threshold",
            "dense_information_threshold",
            "minimum_expected_quality_gain",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.maximum_incremental_energy_mwh <= 0:
            raise ValueError("maximum_incremental_energy_mwh must be positive.")
        if self.maximum_incremental_latency_ms <= 0:
            raise ValueError("maximum_incremental_latency_ms must be positive.")
        if self.energy_penalty_weight < 0 or self.latency_penalty_weight < 0:
            raise ValueError("Penalty weights cannot be negative.")
        if self.upscale_max_pixels <= 0 or self.upscale_max_pixels % 784:
            raise ValueError("upscale_max_pixels must be a positive multiple of 784.")


@dataclass(frozen=True)
class PolicyDecision:
    action: AcquisitionAction
    reason: str
    utility: float = 0.0
    selected_estimate: ActionEstimate | None = None
    rejected: tuple[str, ...] = ()
