import pytest

from src.models.base import (
    ModelBackend,
    GenerationConfig,
    InferenceRequest,
    InferenceResponse,
)


def test_generation_config_defaults():
    config = GenerationConfig()

    assert config.max_new_tokens == 128
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.do_sample is False


def test_inference_request():
    request = InferenceRequest(
        question="What is in this image?",
        image="sample.jpg",
        request_id="test-001",
    )

    assert request.question == "What is in this image?"
    assert request.image == "sample.jpg"
    assert request.request_id == "test-001"


def test_inference_response():
    response = InferenceResponse(
        answer="A vehicle",
        latency_seconds=0.25,
        input_tokens=100,
        output_tokens=5,
    )

    assert response.answer == "A vehicle"
    assert response.latency_seconds == 0.25
    assert response.input_tokens == 100
    assert response.output_tokens == 5


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        ModelBackend()
