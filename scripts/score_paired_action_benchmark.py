#!/usr/bin/env python3
"""Score the paired DocVQA acquisition benchmark using DocVQA-style ANLS/EM.

ANLS convention:
  * lowercase and trim outer whitespace (no punctuation/article normalization)
  * character-level Levenshtein distance / max string length
  * similarity = 1 - normalized distance only when normalized distance < 0.5
  * best score over the question's accepted references

ROI is reported on its actually eligible subset (104 in the 128-question run).
An additional availability-aware policy column uses ROI where a crop exists and
STOP otherwise. That is explicitly a composite policy, not an ROI-only result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def levenshtein(a: str, b: str) -> int:
    """Memory-efficient character-level Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def score_pair(prediction: str, reference: str, threshold: float = 0.5) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    distance = levenshtein(pred, gold)
    normalized_distance = distance / max(len(pred), len(gold))
    return 1.0 - normalized_distance if normalized_distance < threshold else 0.0


def score_answer(prediction: str, references: list[str]) -> tuple[float, int, str]:
    if not references:
        raise ValueError("A question has no reference answers.")
    pred = normalize_answer(prediction)
    normalized_refs = [normalize_answer(ref) for ref in references]
    em = int(any(pred == ref for ref in normalized_refs))
    best_reference = max(
        normalized_refs,
        key=lambda ref: score_pair(pred, ref),
    )
    anls = score_pair(pred, best_reference)
    return anls, em, best_reference


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_ci(
    deltas: list[float],
    seed: int = 20261124,
    repetitions: int = 10000,
) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    estimates = []
    for _ in range(repetitions):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistics.fmean(sample))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_comparison(
    name: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError(f"Paired comparison {name} has different sample sizes.")
    if [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise ValueError(f"Paired comparison {name} is not aligned by sample_id.")

    anls_delta = [float(a[left_key]["anls"]) - float(b[right_key]["anls"]) for a, b in zip(left, right)]
    em_delta = [int(a[left_key]["em"]) - int(b[right_key]["em"]) for a, b in zip(left, right)]
    ci_low, ci_high = bootstrap_ci(anls_delta, seed=20261124 + len(name))
    em_ci_low, em_ci_high = bootstrap_ci([float(x) for x in em_delta], seed=20261201 + len(name))

    anls_wins = sum(delta > 1e-12 for delta in anls_delta)
    anls_losses = sum(delta < -1e-12 for delta in anls_delta)
    em_wins = sum(delta > 0 for delta in em_delta)
    em_losses = sum(delta < 0 for delta in em_delta)

    return {
        "comparison": name,
        "n": len(anls_delta),
        "anls_left_mean": mean([float(row[left_key]["anls"]) for row in left]),
        "anls_right_mean": mean([float(row[right_key]["anls"]) for row in right]),
        "anls_delta_left_minus_right": mean(anls_delta),
        "anls_delta_bootstrap_95ci": [ci_low, ci_high],
        "anls_wins_ties_losses": [
            anls_wins,
            len(anls_delta) - anls_wins - anls_losses,
            anls_losses,
        ],
        "em_left": mean([float(row[left_key]["em"]) for row in left]),
        "em_right": mean([float(row[right_key]["em"]) for row in right]),
        "em_delta_left_minus_right": mean([float(x) for x in em_delta]),
        "em_delta_bootstrap_95ci": [em_ci_low, em_ci_high],
        "em_wins_ties_losses": [
            em_wins,
            len(em_delta) - em_wins - em_losses,
            em_losses,
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/workspace/projects/efficient-multimodal-inference/results/action_benchmark_2026-11-24"),
    )
    args = parser.parse_args()

    base = args.benchmark_dir.resolve()
    labels_path = base / "benchmark128_labels.jsonl"
    inputs_path = base / "benchmark128_inputs.jsonl"
    predictions_path = base / "inference" / "predictions.jsonl"
    energy_path = base / "inference" / "energy_segments.jsonl"
    output_dir = base / "inference" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (labels_path, inputs_path, predictions_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    labels_rows = read_jsonl(labels_path)
    input_rows = read_jsonl(inputs_path)
    prediction_rows = read_jsonl(predictions_path)

    labels = {str(row["sample_id"]): row["answers"] for row in labels_rows}
    inputs = {str(row["sample_id"]): row for row in input_rows}
    if len(labels) != len(labels_rows) or len(inputs) != len(input_rows):
        raise ValueError("Duplicate sample_id in labels or input manifest.")
    if set(labels) != set(inputs):
        raise ValueError("Gold-label IDs and benchmark input IDs do not match.")

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in prediction_rows:
        latest[(str(row["sample_id"]), str(row["stage"]))] = row

    scored: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for sid in sorted(inputs, key=lambda x: int(x) if x.isdigit() else x):
        sample = inputs[sid]
        references = labels[sid]
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list) or not references:
            raise ValueError(f"Invalid reference list for sample {sid}.")
        row: dict[str, Any] = {
            "sample_id": sid,
            "question": sample.get("question", ""),
            "question_category": sample.get("question_category", "uncategorized"),
            "answers": references,
        }
        for stage in ("first_pass", "full_upscale", "roi_crop"):
            record = latest.get((sid, stage))
            if record is None:
                missing.append((sid, stage))
                continue
            if record.get("status") == "completed":
                anls, em, matched_ref = score_answer(record.get("answer", ""), references)
                row[stage] = {
                    "status": "completed",
                    "answer": record.get("answer", ""),
                    "anls": anls,
                    "em": em,
                    "matched_reference": matched_ref,
                    "latency_seconds": record.get("latency_seconds"),
                }
            elif record.get("status") == "skipped_no_ocr_match":
                row[stage] = {"status": "skipped_no_ocr_match"}
            else:
                row[stage] = {"status": record.get("status", "unknown")}
        if "first_pass" not in row or "full_upscale" not in row:
            missing.append((sid, "required_first_or_full"))
        roi = row.get("roi_crop", {})
        if roi.get("status") == "completed":
            row["roi_or_stop"] = row["roi_crop"]
            row["roi_or_stop_policy_action"] = "ROI_CROP"
        else:
            row["roi_or_stop"] = row["first_pass"]
            row["roi_or_stop_policy_action"] = "STOP"
        scored.append(row)

    if missing:
        raise RuntimeError(f"Missing or incomplete prediction records: {missing[:10]} (total {len(missing)})")

    completed = {
        stage: [row for row in scored if row.get(stage, {}).get("status") == "completed"]
        for stage in ("first_pass", "full_upscale", "roi_crop")
    }
    all_scored = scored
    eligible_ids = {row["sample_id"] for row in completed["roi_crop"]}
    roi_eligible = [row for row in scored if row["sample_id"] in eligible_ids]

    def aggregate(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = [row[stage] for row in rows]
        return {
            "n": len(values),
            "coverage_over_128": len(values) / len(scored),
            "exact_match": mean([float(value["em"]) for value in values]),
            "anls": mean([float(value["anls"]) for value in values]),
            "mean_latency_seconds": mean([
                float(value["latency_seconds"]) for value in values
                if value.get("latency_seconds") is not None
            ]),
            "p50_latency_seconds": percentile([
                float(value["latency_seconds"]) for value in values
                if value.get("latency_seconds") is not None
            ], 0.50),
            "p95_latency_seconds": percentile([
                float(value["latency_seconds"]) for value in values
                if value.get("latency_seconds") is not None
            ], 0.95),
        }

    overall = {
        "STOP": aggregate("first_pass", all_scored),
        "FULL_UPSCALE": aggregate("full_upscale", all_scored),
        "ROI_CROP_eligible_only": aggregate("roi_crop", roi_eligible),
        "ROI_when_available_else_STOP": {
            "n": len(scored),
            "coverage_over_128": 1.0,
            "exact_match": mean([float(row["roi_or_stop"]["em"]) for row in scored]),
            "anls": mean([float(row["roi_or_stop"]["anls"]) for row in scored]),
            "policy_actions": dict(Counter(row["roi_or_stop_policy_action"] for row in scored)),
        },
    }

    # Pairwise comparisons use identical questions on both sides.
    comparisons = [
        paired_comparison(
            "FULL_UPSCALE vs STOP (all 128)",
            all_scored, all_scored, "full_upscale", "first_pass",
        ),
        paired_comparison(
            "ROI_CROP vs STOP (104 ROI-eligible questions)",
            roi_eligible, roi_eligible, "roi_crop", "first_pass",
        ),
        paired_comparison(
            "ROI_CROP vs FULL_UPSCALE (104 ROI-eligible questions)",
            roi_eligible, roi_eligible, "roi_crop", "full_upscale",
        ),
        paired_comparison(
            "ROI-when-available-else-STOP vs STOP (all 128)",
            all_scored, all_scored, "roi_or_stop", "first_pass",
        ),
    ]

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        categories[str(row.get("question_category") or "uncategorized")].append(row)
    category_rows = []
    for category in sorted(categories):
        rows = categories[category]
        eligible = [row for row in rows if row["roi_crop"]["status"] == "completed"]
        category_rows.append({
            "question_category": category,
            "n": len(rows),
            "roi_eligible_n": len(eligible),
            "stop_em": mean([float(row["first_pass"]["em"]) for row in rows]),
            "stop_anls": mean([float(row["first_pass"]["anls"]) for row in rows]),
            "full_em": mean([float(row["full_upscale"]["em"]) for row in rows]),
            "full_anls": mean([float(row["full_upscale"]["anls"]) for row in rows]),
            "roi_em_eligible": mean([float(row["roi_crop"]["em"]) for row in eligible]),
            "roi_anls_eligible": mean([float(row["roi_crop"]["anls"]) for row in eligible]),
        })

    energy_rows = read_jsonl(energy_path) if energy_path.is_file() else []
    energy_by_stage = {row["stage"]: row for row in energy_rows}
    energy_summary = {}
    for stage, prediction_stage in (
        ("first_pass", "first_pass"),
        ("full_upscale", "full_upscale"),
        ("roi_crop", "roi_crop"),
    ):
        measure = energy_by_stage.get(stage, {})
        n = len(completed[prediction_stage])
        mwh = measure.get("incremental_board_energy_mwh")
        energy_summary[stage] = {
            "incremental_board_energy_mwh": mwh,
            "completed_requests": n,
            "mean_mwh_per_completed_request": (float(mwh) / n) if mwh is not None and n else None,
            "measurement_available": bool(measure.get("available")),
            "measurement_scope": measure.get("scope"),
            "method": measure.get("method"),
        }

    # Per-question CSV keeps original answers and all paired metric deltas.
    detail_rows = []
    for row in scored:
        base_row = {
            "sample_id": row["sample_id"],
            "question_category": row["question_category"],
            "question": row["question"],
            "gold_answers": " || ".join(map(str, row["answers"])),
            "stop_answer": row["first_pass"]["answer"],
            "stop_em": row["first_pass"]["em"],
            "stop_anls": row["first_pass"]["anls"],
            "full_answer": row["full_upscale"]["answer"],
            "full_em": row["full_upscale"]["em"],
            "full_anls": row["full_upscale"]["anls"],
            "full_minus_stop_anls": row["full_upscale"]["anls"] - row["first_pass"]["anls"],
            "full_minus_stop_em": row["full_upscale"]["em"] - row["first_pass"]["em"],
            "roi_status": row["roi_crop"]["status"],
            "roi_answer": row["roi_crop"].get("answer", ""),
            "roi_em": row["roi_crop"].get("em", ""),
            "roi_anls": row["roi_crop"].get("anls", ""),
            "roi_minus_stop_anls": (
                row["roi_crop"]["anls"] - row["first_pass"]["anls"]
                if row["roi_crop"]["status"] == "completed" else ""
            ),
            "roi_minus_full_anls": (
                row["roi_crop"]["anls"] - row["full_upscale"]["anls"]
                if row["roi_crop"]["status"] == "completed" else ""
            ),
            "roi_or_stop_action": row["roi_or_stop_policy_action"],
            "roi_or_stop_em": row["roi_or_stop"]["em"],
            "roi_or_stop_anls": row["roi_or_stop"]["anls"],
        }
        detail_rows.append(base_row)

    summary = {
        "experiment": "paired_adaptive_evidence_acquisition_128",
        "n_questions": len(scored),
        "n_roi_eligible": len(roi_eligible),
        "n_roi_unavailable": len(scored) - len(roi_eligible),
        "metric_definition": {
            "em": "case-insensitive exact string match after trimming outer whitespace; max over references",
            "anls": "lowercase + trim outer whitespace; character Levenshtein / max string length; score 1-NLD only if NLD < 0.5; max over references",
            "punctuation_and_internal_whitespace": "preserved",
        },
        "overall": overall,
        "paired_comparisons": comparisons,
        "by_question_category": category_rows,
        "energy": energy_summary,
        "inference_artifacts": {
            "predictions": str(predictions_path),
            "labels": str(labels_path),
            "energy_segments": str(energy_path),
        },
    }

    (output_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "paired_question_scores.csv", detail_rows)
    write_csv(output_dir / "category_scores.csv", category_rows)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)

    lines = [
        "# Paired DocVQA acquisition benchmark — scoring",
        "",
        f"- Questions: {len(scored)}",
        f"- ROI eligible: {len(roi_eligible)}",
        f"- ROI unavailable: {len(scored) - len(roi_eligible)}",
        "- Metric: lowercase/trimmed character-level DocVQA ANLS, threshold 0.5; case-insensitive EM",
        "",
        "| Condition | N | EM | ANLS |",
        "|---|---:|---:|---:|",
    ]
    for name, key in (
        ("STOP", "STOP"),
        ("FULL_UPSCALE", "FULL_UPSCALE"),
        ("ROI_CROP (eligible only)", "ROI_CROP_eligible_only"),
        ("ROI when available, else STOP", "ROI_when_available_else_STOP"),
    ):
        item = overall[key]
        lines.append(
            f"| {name} | {item['n']} | {item['exact_match']:.4f} | {item['anls']:.4f} |"
        )
    lines += ["", "## Paired deltas", "", "| Comparison | N | Δ ANLS | 95% bootstrap CI | Δ EM | ANLS wins / ties / losses |", "|---|---:|---:|---:|---:|---:|"]
    for item in comparisons:
        ci = item["anls_delta_bootstrap_95ci"]
        lines.append(
            f"| {item['comparison']} | {item['n']} | "
            f"{item['anls_delta_left_minus_right']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{item['em_delta_left_minus_right']:+.4f} | "
            f"{item['anls_wins_ties_losses'][0]} / {item['anls_wins_ties_losses'][1]} / {item['anls_wins_ties_losses'][2]} |"
        )
    lines += ["", "## Energy estimates", "", "| Phase | Total measured energy (mWh) | Completed calls | Mean mWh/call |", "|---|---:|---:|---:|"]
    for stage, item in energy_summary.items():
        total = item["incremental_board_energy_mwh"]
        total_text = f"{total:.2f}" if total is not None else "Unavailable"
        per_call = item["mean_mwh_per_completed_request"]
        per_text = f"{per_call:.2f}" if per_call is not None else "Unavailable"
        lines.append(f"| {stage} | {total_text} | {item['completed_requests']} | {per_text} |")
    lines += [
        "",
        "Energy figures are aggregate NVML board-power estimates with pre/post idle subtraction, not per-request ground-truth energy.",
        "ROI eligible-only results are conditional on the deterministic OCR proposal returning at least one region. The availability-aware row uses STOP for the 24 ineligible questions and must not be described as ROI-only.",
        "",
    ]
    (output_dir / "score_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved:")
    print(output_dir / "score_summary.json")
    print(output_dir / "score_summary.md")
    print(output_dir / "paired_question_scores.csv")
    print(output_dir / "category_scores.csv")
    print(output_dir / "paired_comparisons.csv")


if __name__ == "__main__":
    main()
