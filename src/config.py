from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return data


def validate_model_config(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    generation = config.get("generation", {})
    input_config = config.get("input", {})

    if not model.get("id"):
        raise ValueError("Model ID is required.")

    if model.get("backend") != "vllm":
        raise ValueError("This project currently supports the vLLM backend.")

    if generation.get("max_new_tokens", 0) <= 0:
        raise ValueError("max_new_tokens must be positive.")

    if input_config.get("max_images_per_request", 0) < 1:
        raise ValueError("At least one image must be allowed per request.")


def validate_serving_config(config: dict[str, Any]) -> None:
    engine = config.get("engine", {})
    benchmark = config.get("benchmark", {})
    concurrency = config.get("concurrency", {})

    if engine.get("name") != "vllm":
        raise ValueError("Serving engine must be vllm.")

    memory_fraction = engine.get("gpu_memory_utilization", 0)
    if not 0 < memory_fraction <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")

    if engine.get("max_model_len", 0) <= 0:
        raise ValueError("max_model_len must be positive.")

    if benchmark.get("warmup_requests", -1) < 0:
        raise ValueError("warmup_requests cannot be negative.")

    if benchmark.get("measured_requests", 0) <= 0:
        raise ValueError("measured_requests must be positive.")

    if benchmark.get("repetitions", 0) <= 0:
        raise ValueError("repetitions must be positive.")

    levels = concurrency.get("levels", [])
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("Concurrency levels must be positive.")

    if len(levels) != len(set(levels)):
        raise ValueError("Concurrency levels must be unique.")


def validate_experiment_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    factors = config.get("factors", {})

    if not experiment.get("name"):
        raise ValueError("Experiment name is required.")

    pixel_budgets = factors.get("image_max_pixels", [])
    if not pixel_budgets or any(value <= 0 for value in pixel_budgets):
        raise ValueError("Image pixel budgets must be positive.")

    patch_area = 28 * 28
    if any(value % patch_area != 0 for value in pixel_budgets):
        raise ValueError("Pixel budgets must be multiples of 28 * 28.")

    concurrency = factors.get("concurrency", [])
    if not concurrency or any(value <= 0 for value in concurrency):
        raise ValueError("Experiment concurrency levels must be positive.")


def load_project_config() -> dict[str, dict[str, Any]]:
    configs = {
        "model": load_yaml(CONFIG_DIR / "model.yaml"),
        "serving": load_yaml(CONFIG_DIR / "serving.yaml"),
        "experiments": load_yaml(CONFIG_DIR / "experiments.yaml"),
    }

    validate_model_config(configs["model"])
    validate_serving_config(configs["serving"])
    validate_experiment_config(configs["experiments"])

    return configs
