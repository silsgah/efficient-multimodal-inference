import pytest

from src.config import (
    load_project_config,
    validate_model_config,
    validate_serving_config,
    validate_experiment_config,
)


def test_all_project_configs_load_and_validate():
    configs = load_project_config()
    assert set(configs) == {"model", "serving", "experiments"}


def test_model_config_rejects_missing_model_id():
    config = {
        "model": {"backend": "vllm"},
        "generation": {"max_new_tokens": 128},
        "input": {"max_images_per_request": 1},
    }

    with pytest.raises(ValueError, match="Model ID"):
        validate_model_config(config)


def test_serving_config_rejects_invalid_gpu_memory_fraction():
    config = {
        "engine": {
            "name": "vllm",
            "gpu_memory_utilization": 1.5,
            "max_model_len": 8192,
        },
        "benchmark": {
            "warmup_requests": 5,
            "measured_requests": 50,
            "repetitions": 3,
        },
        "concurrency": {"levels": [1, 2]},
    }

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        validate_serving_config(config)


def test_experiment_config_rejects_invalid_pixel_budget():
    config = {
        "experiment": {"name": "test"},
        "factors": {
            "image_max_pixels": [200000],
            "concurrency": [1, 2],
        },
    }

    with pytest.raises(ValueError, match="multiples of 28"):
        validate_experiment_config(config)
