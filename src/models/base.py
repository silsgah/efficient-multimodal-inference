from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")


@dataclass(frozen=True)
class InferenceRequest:
    question: str
    image: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question cannot be empty.")
        if not self.image:
            raise ValueError("image is required.")


@dataclass(frozen=True)
class InferenceResponse:
    answer: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    visual_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative.")


class ModelBackend(ABC):
    backend_name = "abstract"

    @abstractmethod
    def load(self) -> None:
        """Load the model and its inference runtime."""

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
        generation_config: GenerationConfig,
    ) -> InferenceResponse:
        """Generate one answer for an image-question request."""

    @abstractmethod
    def unload(self) -> None:
        """Release model and runtime resources."""
