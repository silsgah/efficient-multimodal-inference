from src.adaptive.acquisition import AcquiredInput
from src.adaptive.types import AcquisitionAction
from src.adaptive.vllm_adapter import VLLMAdaptiveAdapter
from src.models.base import (
    GenerationConfig,
    InferenceResponse,
    ModelBackend,
)


class RecordingBackend(ModelBackend):
    backend_name = "recording"

    def __init__(self):
        self.loaded = False
        self.last_request = None
        self.last_config = None

    def load(self):
        self.loaded = True

    def generate(self, request, generation_config):
        assert self.loaded
        self.last_request = request
        self.last_config = generation_config
        return InferenceResponse(
            answer="Invoice total: 120.00",
            latency_seconds=0.25,
            request_id=request.request_id,
            input_tokens=50,
            output_tokens=6,
            visual_tokens=12,
        )

    def unload(self):
        self.loaded = False


def test_adapter_maps_acquired_input_to_model_backend():
    backend = RecordingBackend()
    adapter = VLLMAdaptiveAdapter(
        backend,
        GenerationConfig(max_new_tokens=64),
    )
    adapter.load()
    try:
        acquired = AcquiredInput(
            request_id="invoice-01",
            question="What is the total?",
            image_path="invoice.png",
            max_pixels=401408,
            min_pixels=200704,
            action=AcquisitionAction.FULL_UPSCALE,
        )
        result = adapter.generate(acquired)
    finally:
        adapter.unload()

    assert backend.last_request.request_id == "invoice-01"
    assert backend.last_request.metadata == {
        "min_pixels": 200704,
        "max_pixels": 401408,
    }
    assert backend.last_config.max_new_tokens == 64
    assert result.answer == "Invoice total: 120.00"
    assert result.visual_tokens == 12
    assert not backend.loaded
