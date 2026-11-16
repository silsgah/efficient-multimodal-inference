import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.dataset import BenchmarkDataset
from src.models.base import GenerationConfig, ModelBackend


class BenchmarkRunner:
    """Sequential benchmark runner with repeatable result recording."""

    def __init__(
        self,
        backend: ModelBackend,
        dataset: BenchmarkDataset,
        generation_config: GenerationConfig,
        max_pixels: int,
        warmup_requests: int = 5,
        measured_requests: int = 50,
        repetitions: int = 3,
        results_dir: str | Path = "results",
    ):
        if max_pixels <= 0 or max_pixels % (28 * 28) != 0:
            raise ValueError("max_pixels must be a positive multiple of 784.")

        if warmup_requests < 0:
            raise ValueError("warmup_requests cannot be negative.")

        if measured_requests <= 0 or repetitions <= 0:
            raise ValueError("measured_requests and repetitions must be positive.")

        self.backend = backend
        self.dataset = dataset
        self.generation_config = generation_config
        self.max_pixels = max_pixels
        self.warmup_requests = warmup_requests
        self.measured_requests = measured_requests
        self.repetitions = repetitions
        self.results_dir = Path(results_dir)

    def _request(self, index: int):
        return self.dataset.to_request(
            index % len(self.dataset),
            max_pixels=self.max_pixels,
        )

    def run(self) -> dict[str, Any]:
        if not self.dataset.samples:
            raise ValueError("Cannot benchmark an empty dataset.")

        self.backend.load()

        try:
            # Warmup requests are excluded from measured results.
            for index in range(self.warmup_requests):
                self.backend.generate(
                    self._request(index),
                    self.generation_config,
                )

            records = []
            benchmark_start = time.perf_counter()

            for repetition in range(1, self.repetitions + 1):
                for index in range(self.measured_requests):
                    request = self._request(index)
                    started = time.perf_counter()

                    response = self.backend.generate(
                        request,
                        self.generation_config,
                    )

                    wall_latency = time.perf_counter() - started

                    records.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "repetition": repetition,
                        "request_id": request.request_id,
                        "max_pixels": self.max_pixels,
                        "question": request.question,
                        "answer": response.answer,
                        "wall_latency_seconds": wall_latency,
                        "backend_latency_seconds": response.latency_seconds,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "visual_tokens": response.visual_tokens,
                        "ttft_seconds": None,
                        "tpot_seconds": None,
                        "itl_seconds": None,
                    })

            benchmark_duration = time.perf_counter() - benchmark_start

        finally:
            self.backend.unload()

        total_output_tokens = sum(
            record["output_tokens"] or 0 for record in records
        )

        summary = {
            "backend": self.backend.backend_name,
            "max_pixels": self.max_pixels,
            "warmup_requests": self.warmup_requests,
            "measured_requests_per_repetition": self.measured_requests,
            "repetitions": self.repetitions,
            "total_measured_requests": len(records),
            "benchmark_duration_seconds": benchmark_duration,
            "total_output_tokens": total_output_tokens,
            "output_tokens_per_second": (
                total_output_tokens / benchmark_duration
                if benchmark_duration > 0 else 0
            ),
            "ttft_available": False,
            "tpot_available": False,
            "visual_tokens_available": False,
            "records": records,
        }

        self._save(summary)
        return summary

    def _save(self, summary: dict[str, Any]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        stem = f"benchmark_{timestamp}"

        json_path = self.results_dir / f"{stem}.json"
        csv_path = self.results_dir / f"{stem}.csv"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        records = summary["records"]

        if records:
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(records[0].keys()),
                )
                writer.writeheader()
                writer.writerows(records)

        summary["json_path"] = str(json_path)
        summary["csv_path"] = str(csv_path)
