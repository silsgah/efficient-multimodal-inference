#!/usr/bin/env python3
"""Paired STOP / FULL_UPSCALE / OCR-ROI DocVQA benchmark.

Gold labels are never loaded by this inference runner. Each successful request
is appended and fsynced immediately. ROI crops are deterministic BM25 + local
spatial-context proposals saved as a one-image contact sheet for the current
single-image vLLM backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/projects/efficient-multimodal-inference")
CORPUS = Path("/workspace/worktrees/emi-docvqa-corpus/data/docvqa/full_corpus_2026-11-22")
DEFAULT_BENCH = ROOT / "results/action_benchmark_2026-11-24"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LOW_PIXELS = 200704
HIGH_PIXELS = 802816
MIN_PIXELS = 200704
SEED = 20261124
MAX_NEW_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
STOP_WORDS = set("""
a an the is are was were be been being of in on at to for from by with
and or but what when where who whom which how why does do did can could
would should this that these those it its as no
""".split())


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {number}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl_sync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token not in STOP_WORDS]


def bm25(query: str, lines: list[dict[str, Any]]) -> list[tuple[float, int]]:
    docs = [tokenize(line.get("text", "")) for line in lines]
    n = len(docs)
    if not n:
        return []
    df = Counter(token for doc in docs for token in set(doc))
    avgdl = sum(map(len, docs)) / max(n, 1)
    terms = set(tokenize(query))
    ranked = []
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for term in terms:
            frequency = tf[term]
            if not frequency:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * len(doc) / max(avgdl, 1.0)
            )
            score += idf * frequency * 2.5 / denominator
        if score > 0:
            ranked.append((score, index))
    return sorted(ranked, reverse=True)


def line_box(line: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    raw = line.get("bounding_box")
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError(f"Unexpected OCR bounding box: {raw}")
    xs = [float(value) for value in raw[0::2]]
    ys = [float(value) for value in raw[1::2]]
    x0, x1 = max(0, int(min(xs))), min(width, int(max(xs) + 0.999))
    y0, y1 = max(0, int(min(ys))), min(height, int(max(ys) + 0.999))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate OCR box: {raw}")
    return x0, y0, x1, y1


def spatially_related(a: tuple[int, int, int, int],
                      b: tuple[int, int, int, int],
                      width: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ah, bh = ay1 - ay0, by1 - by0
    aw, bw = ax1 - ax0, bx1 - bx0
    gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
    gap_y = max(0, max(ay0, by0) - min(ay1, by1))
    overlap_x = max(0, min(ax1, bx1) - max(ax0, bx0))
    column_overlap = overlap_x / max(1, min(aw, bw))
    same_row = (
        abs((ay0 + ay1) / 2 - (by0 + by1) / 2) <= max(1.8 * max(ah, bh), 20)
        and gap_x <= 0.30 * width
    )
    same_column = (
        column_overlap >= 0.25
        and gap_y <= max(4 * max(ah, bh), 45)
    )
    return same_row or same_column


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def propose_rois(question: str, ocr: dict[str, Any], max_regions: int = 2) -> list[dict[str, Any]]:
    lines = ocr.get("lines", [])
    width, height = int(ocr["width"]), int(ocr["height"])
    boxes = [line_box(line, width, height) for line in lines]
    heights = [box[3] - box[1] for box in boxes if box[3] > box[1]]
    pad_y = max(15, int(2 * statistics.median(heights))) if heights else 20
    ranked = bm25(question, lines)
    proposals: list[dict[str, Any]] = []

    for score, anchor_index in ranked[:8]:
        anchor = boxes[anchor_index]
        context = [anchor_index]
        for index, candidate in enumerate(boxes):
            if index != anchor_index and spatially_related(anchor, candidate, width):
                context.append(index)

        anchor_x = (anchor[0] + anchor[2]) / 2
        anchor_y = (anchor[1] + anchor[3]) / 2

        context = sorted(
            set(context),
            key=lambda index: (
                abs((boxes[index][1] + boxes[index][3]) / 2 - anchor_y) / max(1, height)
                + abs((boxes[index][0] + boxes[index][2]) / 2 - anchor_x) / max(1, width)
            ),
        )[:10]

        roi = (
            max(0, int(min(boxes[index][0] for index in context) - 0.02 * width)),
            max(0, int(min(boxes[index][1] for index in context) - pad_y)),
            min(width, int(max(boxes[index][2] for index in context) + 0.02 * width)),
            min(height, int(max(boxes[index][3] for index in context) + pad_y)),
        )

        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            continue
        if any(box_iou(roi, previous["bbox"]) > 0.50 for previous in proposals):
            continue

        fraction = ((roi[2] - roi[0]) * (roi[3] - roi[1])) / (width * height)
        proposals.append({
            "rank": len(proposals) + 1,
            "bm25_score": float(score),
            "anchor_line_index": anchor_index,
            "anchor_text": str(lines[anchor_index].get("text", "")),
            "context_lines": [str(lines[index].get("text", "")) for index in context if index != anchor_index],
            "bbox": roi,
            "page_area_fraction": fraction,
        })
        if len(proposals) >= max_regions:
            break

    return proposals


def materialize_roi_contact_sheet(
    sample: dict[str, Any],
    ocr: dict[str, Any],
    output_dir: Path,
) -> tuple[Path | None, list[dict[str, Any]]]:
    proposals = propose_rois(sample["question"], ocr, max_regions=2)
    if not proposals:
        return None, []

    from PIL import Image, ImageDraw, ImageFont

    source = Path(sample["image_path"])
    panels = []
    with Image.open(source) as opened:
        page = opened.convert("RGB")
        expected = (int(ocr["width"]), int(ocr["height"]))
        if page.size != expected:
            raise RuntimeError(
                f"Image/OCR dimensions differ for {sample['sample_id']}: "
                f"image={page.size}, OCR={expected}"
            )
        for proposal in proposals:
            crop = page.crop(tuple(proposal["bbox"]))
            panel = Image.new("RGB", (crop.width, crop.height + 24), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((6, 5), f"ROI {proposal['rank']}", fill="black", font=ImageFont.load_default())
            panel.paste(crop, (0, 24))
            panels.append(panel)

    gap = 12
    canvas_width = max(panel.width for panel in panels)
    canvas_height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    y = 0
    for panel in panels:
        x = (canvas_width - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + gap

    crop_dir = output_dir / "roi_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample["sample_id"]))
    crop_path = crop_dir / f"{safe_id}_roi_contact_sheet.png"
    canvas.save(crop_path, format="PNG")
    return crop_path, proposals


class VLLMRunner:
    def __init__(self) -> None:
        self.engine = None
        self.processor = None
        self.sampling_params = None

    def load(self) -> None:
        os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf-cache/hub")
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        log(f"[model] loading {MODEL_ID}")
        self.engine = LLM(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_seqs=1,
            trust_remote_code=False,
            limit_mm_per_prompt={"image": 1},
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
        )
        log("[model] loaded and ready")

    def generate(
        self,
        sample: dict[str, Any],
        image_path: Path,
        max_pixels: int,
        previous_answer: str | None = None,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Model is not loaded.")
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        if previous_answer is None:
            task_text = (
                "Answer the document question using only information visible "
                "in the supplied page image. Be concise. If the answer is not "
                "readable or supported, say that you cannot determine it."
            )
        else:
            task_text = (
                "You are receiving additional visual evidence for a question. "
                "Your previous answer from the initial low-resolution page was: "
                + json.dumps(previous_answer, ensure_ascii=False)
                + ". Reconsider that answer using the new image. The previous "
                "answer may be right or wrong. Retain it if supported; correct "
                "it if the new visual evidence supports a different or more "
                "precise answer. Return only the concise final answer."
            )

        prompt = (
            "<|im_start|>system\n"
            "You are a careful document question-answering assistant. "
            "Do not invent unreadable text or unsupported details.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            + task_text
            + "\nQuestion: "
            + sample["question"]
            + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        prompt_item = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "min_pixels": MIN_PIXELS,
                "max_pixels": max_pixels,
            },
        }

        started = time.perf_counter()
        outputs = self.engine.generate(
            [prompt_item],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        wall_seconds = time.perf_counter() - started

        if not outputs or not outputs[0].outputs:
            raise RuntimeError(f"vLLM returned no generation for {sample['sample_id']}")

        request_output = outputs[0]
        generation = request_output.outputs[0]
        prompt_ids = getattr(request_output, "prompt_token_ids", None) or []
        output_ids = getattr(generation, "token_ids", None) or []

        visual_tokens = None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if isinstance(image_pad_id, int):
                visual_tokens = sum(token == image_pad_id for token in prompt_ids)

        return {
            "answer": str(generation.text).strip(),
            "wall_latency_seconds": wall_seconds,
            "input_tokens": len(prompt_ids) if prompt_ids else None,
            "output_tokens": len(output_ids) if output_ids else None,
            "visual_tokens": visual_tokens,
        }

    def unload(self) -> None:
        self.engine = None
        self.processor = None
        self.sampling_params = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class NVMLPowerMeter:
    """One-second GPU board-power sampler for aggregate phase energy only."""
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.handle = None
        self.nvml = None

    def start(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self.error = f"NVML unavailable: {type(exc).__name__}: {exc}"
            return False

        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return True

    def _sample_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((time.monotonic(), float(watts)))
            except Exception as exc:
                self.error = f"NVML sampling failed: {type(exc).__name__}: {exc}"
                self.stop_event.set()
                break
            self.stop_event.wait(self.interval)

    def stop(self) -> list[tuple[float, float]]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return list(self.samples)


def interpolate_power(t: float, left: tuple[float, float], right: tuple[float, float]) -> float:
    t0, p0 = left
    t1, p1 = right
    if t1 <= t0:
        return p0
    ratio = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return p0 + ratio * (p1 - p0)


def estimate_phase_energy(
    samples: list[tuple[float, float]],
    start: float,
    end: float,
) -> dict[str, Any]:
    before = [power for timestamp, power in samples if timestamp < start]
    after = [power for timestamp, power in samples if timestamp > end]
    if len(before) < 2 or len(after) < 2 or len(samples) < 5:
        return {
            "available": False,
            "reason": "Insufficient idle baseline or power samples.",
            "sample_count": len(samples),
        }

    idle_before = statistics.mean(before)
    idle_after = statistics.mean(after)
    idle_watts = (idle_before + idle_after) / 2.0
    joules = 0.0

    for left, right in zip(samples, samples[1:]):
        t0, _ = left
        t1, _ = right
        lo, hi = max(t0, start), min(t1, end)
        if hi <= lo:
            continue
        p_lo = interpolate_power(lo, left, right)
        p_hi = interpolate_power(hi, left, right)
        net_watts = max(0.0, (p_lo + p_hi) / 2.0 - idle_watts)
        joules += net_watts * (hi - lo)

    return {
        "available": True,
        "method": "NVML one-second board-power integration with pre/post idle subtraction",
        "idle_before_watts": idle_before,
        "idle_after_watts": idle_after,
        "idle_reference_watts": idle_watts,
        "incremental_board_energy_joules": joules,
        "incremental_board_energy_mwh": joules / 3.6,
        "phase_duration_seconds": end - start,
        "sample_count": len(samples),
        "scope": "aggregate phase estimate; not per-request ground-truth energy",
    }


def run_phase(
    stage: str,
    jobs: list[dict[str, Any]],
    runner: VLLMRunner,
    records_path: Path,
    energy_path: Path,
    sample_by_id: dict[str, dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    roi_by_id: dict[str, tuple[Path | None, list[dict[str, Any]]]],
) -> None:
    if not jobs:
        log(f"[{stage}] no pending requests")
        return

    meter = NVMLPowerMeter()
    meter_ok = meter.start()
    if meter_ok:
        log(f"[{stage}] measuring aggregate GPU board power; 10s pre/post idle windows")
        time.sleep(10)
    else:
        log(f"[{stage}] energy measurement unavailable: {meter.error}")

    started = time.monotonic()
    completed = 0
    try:
        for sample in jobs:
            sid = str(sample["sample_id"])
            previous = None
            image_path = Path(sample["image_path"])
            pixel_budget = LOW_PIXELS

            if stage == "first_pass":
                pixel_budget = LOW_PIXELS
            elif stage == "full_upscale":
                pixel_budget = HIGH_PIXELS
                previous = baseline_by_id[sid]["answer"]
            elif stage == "roi_crop":
                pixel_budget = HIGH_PIXELS
                previous = baseline_by_id[sid]["answer"]
                image_path = roi_by_id[sid][0]
                if image_path is None:
                    raise RuntimeError(f"ROI job {sid} has no materialized crop.")

            response = runner.generate(
                sample=sample,
                image_path=image_path,
                max_pixels=pixel_budget,
                previous_answer=previous,
            )

            roi_path, roi_proposals = roi_by_id.get(sid, (None, []))
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sample_id": sid,
                "stage": stage,
                "action": {
                    "first_pass": "STOP",
                    "full_upscale": "FULL_UPSCALE",
                    "roi_crop": "ROI_CROP",
                }[stage],
                "status": "completed",
                "question": sample["question"],
                "question_category": sample.get("question_category"),
                "image_sha256": sample["image_sha256"],
                "image_used": str(image_path),
                "previous_answer": previous,
                "answer": response["answer"],
                "max_pixels": pixel_budget,
                "latency_seconds": response["wall_latency_seconds"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "visual_tokens": response["visual_tokens"],
                "roi_proposals": roi_proposals if stage == "roi_crop" else [],
                "roi_image_sha256": sha256_file(roi_path) if stage == "roi_crop" and roi_path else None,
                "roi_summed_page_area_fraction": sum(
                    item["page_area_fraction"] for item in roi_proposals
                ) if stage == "roi_crop" else None,
            }
            append_jsonl_sync(records_path, record)
            completed += 1
            if completed == 1 or completed % 8 == 0 or completed == len(jobs):
                log(f"[{stage}] completed {completed}/{len(jobs)}; last={sid}; latency={response['wall_latency_seconds']:.3f}s")

    finally:
        ended = time.monotonic()
        samples = meter.stop() if meter_ok else []
        if meter_ok:
            time.sleep(10)

    energy = estimate_phase_energy(samples, started, ended) if meter_ok else {
        "available": False,
        "reason": meter.error or "NVML sampler not started",
        "sample_count": 0,
    }
    energy_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "requests_in_segment": completed,
        **energy,
    }
    append_jsonl_sync(energy_path, energy_row)
    log(f"[{stage}] phase complete: {completed} requests; energy={energy.get('incremental_board_energy_mwh', 'unavailable')} mWh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    corpus_dir = args.corpus_dir.resolve()
    input_path = benchmark_dir / "benchmark128_inputs.jsonl"
    output_dir = benchmark_dir / "inference"
    records_path = output_dir / "predictions.jsonl"
    energy_path = output_dir / "energy_segments.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    input_rows = read_jsonl(input_path)
    if args.limit > len(input_rows):
        raise ValueError(f"--limit {args.limit} exceeds available sample size {len(input_rows)}.")
    input_rows = input_rows[:args.limit]

    manifest_hash = sha256_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_meta.get("input_manifest_sha256") != manifest_hash:
            raise RuntimeError("Existing inference output belongs to a different input manifest.")
        if old_meta.get("sample_count") != args.limit:
            raise RuntimeError("Existing inference output uses a different --limit.")
        phase_order = old_meta["escalation_phase_order"]
    else:
        phase_order = ["full_upscale", "roi_crop"]
        random.Random(args.seed).shuffle(phase_order)
        metadata = {
            "experiment": "paired_adaptive_evidence_acquisition_128",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_max_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "generation": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "initial_max_pixels": LOW_PIXELS,
            "escalation_max_pixels": HIGH_PIXELS,
            "minimum_pixels": MIN_PIXELS,
            "sample_count": args.limit,
            "seed": args.seed,
            "input_manifest_sha256": manifest_hash,
            "gold_labels_loaded": False,
            "escalation_phase_order": phase_order,
            "energy_method": "NVML aggregate stage power integration with idle subtraction; not per-request energy",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rows_by_id = {str(row["sample_id"]): row for row in input_rows}
    if len(rows_by_id) != len(input_rows):
        raise RuntimeError("Duplicate sample IDs in benchmark manifest.")

    # Load the page OCR only for the selected pages.
    wanted_hashes = {row["image_sha256"] for row in input_rows}
    ocr_by_hash: dict[str, dict[str, Any]] = {}
    pages_path = corpus_dir / "pages.jsonl"
    with pages_path.open(encoding="utf-8") as file:
        for line in file:
            page = json.loads(line)
            page_hash = page.get("image_sha256")
            if page_hash in wanted_hashes:
                ocr_by_hash[page_hash] = page["ocr_results"]
    if len(ocr_by_hash) != len(input_rows):
        raise RuntimeError(f"Found OCR for {len(ocr_by_hash)}/{len(input_rows)} selected pages.")

    log("[roi] preparing deterministic two-region contact sheets")
    roi_by_id: dict[str, tuple[Path | None, list[dict[str, Any]]]] = {}
    roi_available = 0
    for index, sample in enumerate(input_rows, 1):
        image_path = Path(sample["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        roi_path, proposals = materialize_roi_contact_sheet(
            sample, ocr_by_hash[sample["image_sha256"]], output_dir
        )
        roi_by_id[str(sample["sample_id"])] = (roi_path, proposals)
        roi_available += roi_path is not None
        if index % 32 == 0 or index == len(input_rows):
            log(f"[roi] prepared {index}/{len(input_rows)}; available={roi_available}")

    # Read prior records so an interrupted run resumes without repeating requests.
    existing = read_jsonl(records_path) if records_path.exists() else []
    done: dict[tuple[str, str], dict[str, Any]] = {}
    for record in existing:
        done[(str(record["sample_id"]), str(record["stage"]))] = record

    # Persist explicit unavailable-ROI records: no fabricated crop or VLM call.
    for sample in input_rows:
        sid = str(sample["sample_id"])
        if roi_by_id[sid][0] is None and (sid, "roi_crop") not in done:
            skip = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sample_id": sid,
                "stage": "roi_crop",
                "action": "ROI_CROP",
                "status": "skipped_no_ocr_match",
                "question": sample["question"],
                "question_category": sample.get("question_category"),
                "image_sha256": sample["image_sha256"],
                "answer": None,
                "roi_proposals": [],
            }
            append_jsonl_sync(records_path, skip)
            done[(sid, "roi_crop")] = skip

    runner = VLLMRunner()
    try:
        runner.load()

        # Warm the exact configured model with one low and one high-resolution request.
        warm_sample = input_rows[0]
        log("[warmup] low-resolution request")
        runner.generate(warm_sample, Path(warm_sample["image_path"]), LOW_PIXELS)
        log("[warmup] full-upscale request")
        runner.generate(
            warm_sample,
            Path(warm_sample["image_path"]),
            HIGH_PIXELS,
            previous_answer="warmup",
        )
        warm_roi, _ = roi_by_id[str(warm_sample["sample_id"])]
        if warm_roi is not None:
            log("[warmup] ROI contact-sheet request")
            runner.generate(warm_sample, warm_roi, HIGH_PIXELS, previous_answer="warmup")

        baseline_jobs = [
            row for row in input_rows
            if (str(row["sample_id"]), "first_pass") not in done
        ]
        run_phase(
            "first_pass", baseline_jobs, runner, records_path, energy_path,
            rows_by_id, {}, roi_by_id,
        )

        all_records = read_jsonl(records_path) if records_path.exists() else []
        latest = {}
        for record in all_records:
            latest[(str(record["sample_id"]), str(record["stage"]))] = record
        baseline_by_id = {
            sid: latest[(sid, "first_pass")]
            for sid in rows_by_id
            if (sid, "first_pass") in latest
        }
        missing_baseline = set(rows_by_id) - set(baseline_by_id)
        if missing_baseline:
            raise RuntimeError(f"Cannot escalate without first-pass answers for {len(missing_baseline)} questions.")

        for stage in phase_order:
            if stage == "full_upscale":
                jobs = [
                    row for row in input_rows
                    if (str(row["sample_id"]), stage) not in latest
                ]
            else:
                jobs = [
                    row for row in input_rows
                    if roi_by_id[str(row["sample_id"])][0] is not None
                    and (str(row["sample_id"]), stage) not in latest
                ]
            run_phase(
                stage, jobs, runner, records_path, energy_path,
                rows_by_id, baseline_by_id, roi_by_id,
            )
            all_records = read_jsonl(records_path) if records_path.exists() else []
            latest = {}
            for record in all_records:
                latest[(str(record["sample_id"]), str(record["stage"]))] = record

    finally:
        runner.unload()

    final_records = read_jsonl(records_path)
    summary = {
        "status": "inference_complete" if all(
            (str(row["sample_id"]), stage) in {
                (str(record["sample_id"]), str(record["stage"]))
                for record in final_records
            }
            for row in input_rows
            for stage in ("first_pass", "full_upscale", "roi_crop")
        ) else "inference_incomplete",
        "sample_count": len(input_rows),
        "completed_first_pass": sum(
            1 for row in input_rows
            if any(r["sample_id"] == str(row["sample_id"]) and r["stage"] == "first_pass" and r.get("status") == "completed" for r in final_records)
        ),
        "completed_full_upscale": sum(
            1 for row in input_rows
            if any(r["sample_id"] == str(row["sample_id"]) and r["stage"] == "full_upscale" and r.get("status") == "completed" for r in final_records)
        ),
        "completed_roi_crop": sum(
            1 for row in input_rows
            if any(r["sample_id"] == str(row["sample_id"]) and r["stage"] == "roi_crop" and r.get("status") == "completed" for r in final_records)
        ),
        "roi_skipped_no_ocr_match": sum(
            1 for r in final_records if r["stage"] == "roi_crop" and r.get("status") == "skipped_no_ocr_match"
        ),
        "predictions_jsonl": str(records_path),
        "energy_segments_jsonl": str(energy_path),
        "gold_labels_loaded": False,
        "input_manifest_sha256": manifest_hash,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
