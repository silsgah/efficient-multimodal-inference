import argparse
from pathlib import Path

from src.benchmark.runner import BenchmarkRunner
from src.config import load_project_config
from src.data.dataset import BenchmarkDataset
from src.models.base import GenerationConfig
from src.models.vllm_backend import VLLMBackend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run visual token economics benchmark."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the JSONL benchmark manifest.",
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Image pixel budget. Defaults to first configured budget.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_project_config()

    model_config = configs["model"]
    serving_config = configs["serving"]
    experiment_config = configs["experiments"]

    model = model_config["model"]
    generation = model_config["generation"]
    engine = serving_config["engine"]
    benchmark = serving_config["benchmark"]

    pixel_budgets = experiment_config["factors"]["image_max_pixels"]

    max_pixels = (
        args.max_pixels
        if args.max_pixels is not None
        else pixel_budgets[0]
    )

    if max_pixels not in pixel_budgets:
        raise ValueError(
            f"{max_pixels} is not in the configured pixel budgets: "
            f"{pixel_budgets}"
        )

    dataset = BenchmarkDataset(args.manifest)

    backend = VLLMBackend(
        model_id=model["id"],
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
    )

    generation_config = GenerationConfig(
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        do_sample=generation["do_sample"],
    )

    runner = BenchmarkRunner(
        backend=backend,
        dataset=dataset,
        generation_config=generation_config,
        max_pixels=max_pixels,
        warmup_requests=benchmark["warmup_requests"],
        measured_requests=benchmark["measured_requests"],
        repetitions=benchmark["repetitions"],
        results_dir=args.results_dir,
    )

    summary = runner.run()

    print("\nBenchmark completed.")
    print(f"Backend: {summary['backend']}")
    print(f"Pixel budget: {summary['max_pixels']}")
    print(f"Measured requests: {summary['total_measured_requests']}")
    print(f"Duration: {summary['benchmark_duration_seconds']:.2f}s")
    print(f"Output tokens/sec: {summary['output_tokens_per_second']:.2f}")
    print(f"Results: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
