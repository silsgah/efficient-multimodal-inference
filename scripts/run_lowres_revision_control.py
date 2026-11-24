#!/usr/bin/env python3
"""Low-resolution answer-revision control for the paired DocVQA benchmark.

Uses the exact same Qwen runner and revision prompt as FULL_UPSCALE/ROI_CROP,
but retains the initial 200704-pixel image budget. Gold labels are not loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUNNER_PATH = Path("/root/run_paired_action_benchmark.py")
DEFAULT_BASE = Path("/workspace/projects/efficient-multimodal-inference/results/action_benchmark_2026-11-24")


def load_shared_runner():
    if not RUNNER_PATH.is_file():
        raise FileNotFoundError(
            f"Expected shared inference runner at {RUNNER_PATH}; copy the benchmark runner there first."
        )
    spec = importlib.util.spec_from_file_location("paired_action_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import paired benchmark runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_no_existing_engine():
    import subprocess
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    active = []
    for line in result.stdout.splitlines():
        if "VLLM::EngineCore" in line and "grep" not in line:
            active.append(line.strip())
    if active:
        raise RuntimeError(
            "Another vLLM EngineCore appears to be active. Stop this control run until it exits:\n"
            + "\n".join(active)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    base = args.benchmark_dir.resolve()
    inputs_path = base / "benchmark128_inputs.jsonl"
    predictions_path = base / "inference" / "predictions.jsonl"
    output_dir = base / "inference"
    output_path = output_dir / "revision_control.jsonl"
    energy_path = output_dir / "revision_control_energy.json"
    metadata_path = output_dir / "revision_control_metadata.json"

    if not inputs_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Input manifest or paired inference predictions are missing.")

    shared = load_shared_runner()
    inputs = shared.read_jsonl(inputs_path)[:args.limit]
    records = shared.read_jsonl(predictions_path)

    baseline = {}
    for row in records:
        if row.get("stage") == "first_pass" and row.get("status") == "completed":
            baseline[str(row["sample_id"])] = row

    missing_baseline = [str(row["sample_id"]) for row in inputs if str(row["sample_id"]) not in baseline]
    if missing_baseline:
        raise RuntimeError(f"Missing initial answers for {len(missing_baseline)} questions.")

    done = {}
    existing = shared.read_jsonl(output_path) if output_path.exists() else []
    for row in existing:
        done[str(row["sample_id"])] = row

    pending = [row for row in inputs if str(row["sample_id"]) not in done]

    metadata = {
        "experiment": "low_resolution_revision_control",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": shared.MODEL_ID,
        "resolution_max_pixels": shared.LOW_PIXELS,
        "prompt": "Same revision prompt as FULL_UPSCALE and ROI_CROP; previous first-pass answer supplied.",
        "gold_labels_loaded": False,
        "input_count": len(inputs),
        "pending_at_start": len(pending),
        "input_manifest_sha256": shared.sha256_file(inputs_path),
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("input_manifest_sha256") != metadata["input_manifest_sha256"]:
            raise RuntimeError("Existing revision-control output belongs to another input manifest.")
        if old.get("input_count") != len(inputs):
            raise RuntimeError("Existing revision-control output uses a different --limit.")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not pending:
        print(f"Revision-control already complete: {len(done)}/{len(inputs)}")
        return

    ensure_no_existing_engine()
    runner = shared.VLLMRunner()
    meter = shared.NVMLPowerMeter()

    try:
        runner.load()
        warm = pending[0]
        warm_sid = str(warm["sample_id"])
        print("[warmup] low-resolution revision-control prompt", flush=True)
        runner.generate(
            warm,
            Path(warm["image_path"]),
            shared.LOW_PIXELS,
            previous_answer=baseline[warm_sid]["answer"],
        )

        meter_ok = meter.start()
        if meter_ok:
            print("[energy] collecting 10s pre-workload idle baseline", flush=True)
            time.sleep(10)
        else:
            print(f"[energy] unavailable: {meter.error}", flush=True)

        start = time.monotonic()
        completed = 0
        try:
            for sample in pending:
                sid = str(sample["sample_id"])
                previous = baseline[sid]["answer"]
                response = runner.generate(
                    sample,
                    Path(sample["image_path"]),
                    shared.LOW_PIXELS,
                    previous_answer=previous,
                )
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "sample_id": sid,
                    "stage": "low_resolution_revision_control",
                    "action": "LOW_RESOLUTION_REVISION_CONTROL",
                    "status": "completed",
                    "question": sample["question"],
                    "question_category": sample.get("question_category"),
                    "image_sha256": sample["image_sha256"],
                    "image_used": sample["image_path"],
                    "previous_answer": previous,
                    "answer": response["answer"],
                    "max_pixels": shared.LOW_PIXELS,
                    "latency_seconds": response["wall_latency_seconds"],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "visual_tokens": response["visual_tokens"],
                }
                shared.append_jsonl_sync(output_path, record)
                completed += 1
                if completed == 1 or completed % 8 == 0 or completed == len(pending):
                    print(
                        f"[lowres_revision_control] completed {completed}/{len(pending)}; "
                        f"sample={sid}; latency={response['wall_latency_seconds']:.3f}s",
                        flush=True,
                    )
        finally:
            end = time.monotonic()
            if meter_ok:
                print("[energy] collecting 10s post-workload idle baseline", flush=True)
                time.sleep(10)
            power_samples = meter.stop() if meter_ok else []

        energy = (
            shared.estimate_phase_energy(power_samples, start, end)
            if meter_ok
            else {"available": False, "reason": meter.error or "NVML unavailable", "sample_count": 0}
        )
        energy_row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "low_resolution_revision_control",
            "requests_in_segment": completed,
            **energy,
        }
        shared.append_jsonl_sync(
            output_dir / "revision_control_energy_segments.jsonl",
            energy_row,
        )
        energy_path.write_text(json.dumps(energy_row, indent=2) + "\n", encoding="utf-8")
    finally:
        runner.unload()

    final = shared.read_jsonl(output_path)
    summary = {
        "status": "complete" if len({str(row["sample_id"]) for row in final}) >= len(inputs) else "incomplete",
        "completed": len({str(row["sample_id"]) for row in final}),
        "expected": len(inputs),
        "predictions": str(output_path),
        "energy": str(energy_path),
        "gold_labels_loaded": False,
    }
    (output_dir / "revision_control_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
