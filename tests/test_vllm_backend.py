import pytest

from src.models.base import (
    GenerationConfig,
    InferenceRequest,
    ModelBackend,
)
from src.models.vllm_backend import VLLMBackend


def test_vllm_backend_implements_model_backend():
    backend = VLLMBackend()
    assert isinstance(backend, ModelBackend)
    assert backend.backend_name == "vllm"


def test_vllm_backend_default_configuration():
    backend = VLLMBackend()
    assert backend.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert backend.max_model_len == 8192
    assert backend.gpu_memory_utilization == 0.85
    assert backend.engine is None
    assert backend.processor is None


def test_vllm_backend_accepts_custom_configuration():
    backend = VLLMBackend(
        model_id="custom/model",
        max_model_len=4096,
        gpu_memory_utilization=0.75,
    )
    assert backend.model_id == "custom/model"
    assert backend.max_model_len == 4096
    assert backend.gpu_memory_utilization == 0.75


def test_generate_requires_loaded_backend():
    backend = VLLMBackend()
    request = InferenceRequest(
        question="Describe this image",
        image="sample.jpg",
    )
    with pytest.raises(RuntimeError, match="Backend is not loaded"):
        backend.generate(request, GenerationConfig())


def test_unload_resets_backend_state():
    backend = VLLMBackend()
    backend.engine = object()
    backend.processor = object()

    backend.unload()

    assert backend.engine is None
    assert backend.processor is None


def test_image_content_accepts_pixel_budget():
    backend = VLLMBackend()

    content = backend.build_image_content(
        "sample.jpg",
        {"max_pixels": 401408},
    )

    assert content["max_pixels"] == 401408
    assert content["image"] == "sample.jpg"


def test_image_content_accepts_minimum_and_maximum():
    backend = VLLMBackend()

    content = backend.build_image_content(
        "sample.jpg",
        {
            "min_pixels": 200704,
            "max_pixels": 401408,
        },
    )

    assert content["min_pixels"] == 200704
    assert content["max_pixels"] == 401408


def test_image_content_rejects_invalid_pixel_budget():
    backend = VLLMBackend()

    with pytest.raises(ValueError, match="multiple"):
        backend.build_image_content(
            "sample.jpg",
            {"max_pixels": 200000},
        )


def test_image_content_rejects_minimum_above_maximum():
    backend = VLLMBackend()

    with pytest.raises(ValueError, match="cannot exceed"):
        backend.build_image_content(
            "sample.jpg",
            {
                "min_pixels": 401408,
                "max_pixels": 200704,
            },
        )
