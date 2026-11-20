from src.adaptive.acquisition import AcquiredInput
from src.adaptive.orchestrator import AdaptiveInferenceBackend, ModelResult
from src.models.base import GenerationConfig, InferenceRequest, ModelBackend


class VLLMAdaptiveAdapter(AdaptiveInferenceBackend):
    """Adapt the project's ModelBackend contract to adaptive inference."""

    def __init__(
        self,
        backend: ModelBackend,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.backend = backend
        self.generation_config = generation_config or GenerationConfig()

    def load(self) -> None:
        self.backend.load()

    def unload(self) -> None:
        self.backend.unload()

    def generate(self, request: AcquiredInput) -> ModelResult:
        inference_request = InferenceRequest(
            question=request.question,
            image=str(request.image_path),
            request_id=request.request_id,
            metadata={
                "min_pixels": request.min_pixels,
                "max_pixels": request.max_pixels,
            },
        )
        response = self.backend.generate(
            inference_request,
            self.generation_config,
        )
        return ModelResult(
            answer=response.answer,
            latency_seconds=response.latency_seconds,
            visual_tokens=response.visual_tokens,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
