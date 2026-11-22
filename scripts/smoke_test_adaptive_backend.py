"""One-request GPU smoke test through the actual adaptive vLLM adapter.

This loads the configured model and generates exactly one answer. It does not
run the adaptive policy, estimate expected quality gain, or measure energy.
"""
import argparse
import json
from pathlib import Path

from src.adaptive.acquisition import AcquiredInput
from src.adaptive.types import AcquisitionAction
from src.adaptive.vllm_adapter import VLLMAdaptiveAdapter
from src.data.dataset import BenchmarkDataset
from src.models.base import GenerationConfig
from src.models.vllm_backend import VLLMBackend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/docvqa/manifest.jsonl"),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = BenchmarkDataset(args.manifest)
    sample = dataset[args.index]
    acquired = AcquiredInput(
        request_id=sample.sample_id,
        question=sample.question,
        image_path=sample.image,
        max_pixels=args.max_pixels,
        min_pixels=200704,
        action=AcquisitionAction.FIRST_PASS,
    )

    # Match the 0.80 A40 memory allocation used by the original VLM smoke test.
    backend = VLLMBackend(
        max_num_seqs=1,
        gpu_memory_utilization=0.80,
    )
    adapter = VLLMAdaptiveAdapter(
        backend,
        GenerationConfig(max_new_tokens=args.max_new_tokens),
    )

    adapter.load()
    try:
        result = adapter.generate(acquired)
    finally:
        adapter.unload()

    report = {
        "status": "passed",
        "mode": "single_real_vllm_inference",
        "request_id": sample.sample_id,
        "max_pixels": args.max_pixels,
        "answer": result.answer,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "visual_tokens": result.visual_tokens,
        "energy_measurement": None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
