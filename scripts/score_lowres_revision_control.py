#!/usr/bin/env python3
"""Score low-resolution revision control against STOP and FULL_UPSCALE."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def anls_em(prediction: Any, references: list[Any]) -> tuple[float, int]:
    p = norm(prediction)
    refs = [norm(ref) for ref in references]
    em = int(any(p == ref for ref in refs))
    scores = []
    for ref in refs:
        if not p and not ref:
            scores.append(1.0)
        elif not p or not ref:
            scores.append(0.0)
        else:
            nld = levenshtein(p, ref) / max(len(p), len(ref))
            scores.append(1.0 - nld if nld < 0.5 else 0.0)
    return max(scores, default=0.0), em


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] * (hi-index) + values[hi] * (index-lo)


def bootstrap_ci(values: list[float], seed: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(10000)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def compare(name: str, left: list[dict[str, Any]], right: list[dict[str, Any]], lk: str, rk: str):
    if [r["sample_id"] for r in left] != [r["sample_id"] for r in right]:
        raise ValueError(f"Unpaired rows in {name}")
    da = [r[lk]["anls"] - r[rk]["anls"] for r in left]
    de = [r[lk]["em"] - r[rk]["em"] for r in left]
    return {
        "comparison": name,
        "n": len(left),
        "delta_anls": mean(da),
        "delta_anls_ci95": bootstrap_ci(da, 20261124 + len(name)),
        "delta_em": mean([float(x) for x in de]),
        "delta_em_ci95": bootstrap_ci([float(x) for x in de], 20261224 + len(name)),
        "anls_wins_ties_losses": [
            sum(x > 1e-12 for x in da),
            sum(abs(x) <= 1e-12 for x in da),
            sum(x < -1e-12 for x in da),
        ],
        "em_wins_ties_losses": [
            sum(x > 0 for x in de),
            sum(x == 0 for x in de),
            sum(x < 0 for x in de),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/workspace/projects/efficient-multimodal-inference/results/action_benchmark_2026-11-24"),
    )
    args = parser.parse_args()
    base = args.benchmark_dir.resolve()
    inf = base / "inference"
    labels_rows = read_jsonl(base / "benchmark128_labels.jsonl")
    inputs_rows = read_jsonl(base / "benchmark128_inputs.jsonl")
    predictions = read_jsonl(inf / "predictions.jsonl")
    revision_path = inf / "revision_control.jsonl"
    if not revision_path.is_file():
        raise FileNotFoundError(revision_path)
    revisions = read_jsonl(revision_path)

    labels = {str(row["sample_id"]): row["answers"] for row in labels_rows}
    inputs = {str(row["sample_id"]): row for row in inputs_rows}
    if set(labels) != set(inputs):
        raise ValueError("Gold labels and inputs are not one-to-one.")

    base_records = {}
    for row in predictions:
        if row.get("status") == "completed" and row.get("stage") in ("first_pass", "full_upscale"):
            base_records[(str(row["sample_id"]), row["stage"])] = row

    revision_records = {}
    for row in revisions:
        if row.get("status") == "completed":
            revision_records[str(row["sample_id"])] = row

    expected = set(inputs)
    if set(revision_records) != expected:
        missing = sorted(expected - set(revision_records))
        extra = sorted(set(revision_records) - expected)
        raise RuntimeError(f"Revision control coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")

    rows = []
    for sid in sorted(expected, key=lambda x: int(x) if x.isdigit() else x):
        if (sid, "first_pass") not in base_records or (sid, "full_upscale") not in base_records:
            raise RuntimeError(f"Missing STOP or FULL_UPSCALE prediction for {sid}")
        refs = labels[sid]
        if isinstance(refs, str):
            refs = [refs]
        stop = base_records[(sid, "first_pass")]
        full = base_records[(sid, "full_upscale")]
        lowrev = revision_records[sid]

        stop_a, stop_e = anls_em(stop.get("answer", ""), refs)
        lowrev_a, lowrev_e = anls_em(lowrev.get("answer", ""), refs)
        full_a, full_e = anls_em(full.get("answer", ""), refs)

        rows.append({
            "sample_id": sid,
            "question": inputs[sid].get("question", ""),
            "question_category": inputs[sid].get("question_category", "uncategorized"),
            "answers": refs,
            "stop": {
                "answer": stop.get("answer", ""),
                "anls": stop_a,
                "em": stop_e,
                "latency_seconds": stop.get("latency_seconds"),
            },
            "lowres_revision": {
                "answer": lowrev.get("answer", ""),
                "anls": lowrev_a,
                "em": lowrev_e,
                "latency_seconds": lowrev.get("latency_seconds"),
            },
            "full_upscale": {
                "answer": full.get("answer", ""),
                "anls": full_a,
                "em": full_e,
                "latency_seconds": full.get("latency_seconds"),
            },
        })

    comparisons = [
        compare("LOW_RESOLUTION_REVISION vs STOP (all 128)", rows, rows, "lowres_revision", "stop"),
        compare("FULL_UPSCALE_REVISION vs LOW_RESOLUTION_REVISION (all 128)", rows, rows, "full_upscale", "lowres_revision"),
        compare("FULL_UPSCALE_REVISION vs STOP (all 128)", rows, rows, "full_upscale", "stop"),
    ]

    def aggregate(key: str) -> dict[str, float]:
        return {
            "em": mean([float(row[key]["em"]) for row in rows]),
            "anls": mean([float(row[key]["anls"]) for row in rows]),
            "mean_latency_seconds": mean([
                float(row[key]["latency_seconds"])
                for row in rows if row[key].get("latency_seconds") is not None
            ]),
            "answer_changed_vs_stop": sum(
                norm(row[key]["answer"]) != norm(row["stop"]["answer"]) for row in rows
            ),
        }

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[row["question_category"]].append(row)
    category_rows = []
    for category, items in sorted(by_cat.items()):
        category_rows.append({
            "category": category,
            "n": len(items),
            "stop_anls": mean([x["stop"]["anls"] for x in items]),
            "lowres_revision_anls": mean([x["lowres_revision"]["anls"] for x in items]),
            "full_upscale_anls": mean([x["full_upscale"]["anls"] for x in items]),
            "lowres_revision_delta": mean([
                x["lowres_revision"]["anls"] - x["stop"]["anls"] for x in items
            ]),
            "full_over_lowres_revision_delta": mean([
                x["full_upscale"]["anls"] - x["lowres_revision"]["anls"] for x in items
            ]),
        })

    energy_path = inf / "revision_control_energy.json"
    energy = json.loads(energy_path.read_text(encoding="utf-8")) if energy_path.exists() else {}
    energy_mwh = energy.get("incremental_board_energy_mwh")

    summary = {
        "experiment": "low_resolution_revision_control",
        "n": len(rows),
        "metric": "same case-insensitive trimmed character-level ANLS/EM definition as the paired benchmark scorer",
        "conditions": {
            "STOP": aggregate("stop"),
            "LOW_RESOLUTION_REVISION_CONTROL": aggregate("lowres_revision"),
            "FULL_UPSCALE_REVISION": aggregate("full_upscale"),
        },
        "paired_comparisons": comparisons,
        "by_question_category": category_rows,
        "energy": {
            "low_resolution_revision_incremental_board_energy_mwh": energy_mwh,
            "requests": energy.get("requests_in_segment"),
            "mean_mwh_per_revision_request": (
                float(energy_mwh) / int(energy["requests_in_segment"])
                if energy_mwh is not None and energy.get("requests_in_segment") else None
            ),
            "method": energy.get("method"),
            "scope": energy.get("scope"),
            "available": energy.get("available", False),
        },
        "interpretation": (
            "STOP vs low-resolution revision estimates the effect of asking the model to reconsider "
            "its previous answer without adding image pixels. Full-upscale revision vs low-resolution "
            "revision isolates the additional image-resolution effect under the shared revision prompt, "
            "subject to execution-order and aggregate-energy measurement limitations."
        ),
    }

    out = inf / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "revision_control_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    detail = []
    for row in rows:
        detail.append({
            "sample_id": row["sample_id"],
            "question_category": row["question_category"],
            "question": row["question"],
            "gold_answers": " || ".join(map(str, row["answers"])),
            "stop_answer": row["stop"]["answer"],
            "stop_em": row["stop"]["em"],
            "stop_anls": row["stop"]["anls"],
            "lowres_revision_answer": row["lowres_revision"]["answer"],
            "lowres_revision_em": row["lowres_revision"]["em"],
            "lowres_revision_anls": row["lowres_revision"]["anls"],
            "lowres_revision_minus_stop_anls": row["lowres_revision"]["anls"] - row["stop"]["anls"],
            "full_upscale_answer": row["full_upscale"]["answer"],
            "full_upscale_em": row["full_upscale"]["em"],
            "full_upscale_anls": row["full_upscale"]["anls"],
            "full_upscale_minus_lowres_revision_anls": row["full_upscale"]["anls"] - row["lowres_revision"]["anls"],
        })
    with (out / "revision_control_question_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    with (out / "revision_control_category_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    lines = [
        "# Low-resolution revision control",
        "",
        f"- Questions: {len(rows)}",
        "- All conditions use the same revision prompt for low-resolution revision and full-upscale revision.",
        "",
        "| Condition | EM | ANLS | Answers changed vs STOP |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("STOP", "STOP"),
        ("LOW_RESOLUTION_REVISION_CONTROL", "LOW_RESOLUTION_REVISION_CONTROL"),
        ("FULL_UPSCALE_REVISION", "FULL_UPSCALE_REVISION"),
    ):
        x = summary["conditions"][key]
        lines.append(
            f"| {label} | {x['em']:.4f} | {x['anls']:.4f} | {x['answer_changed_vs_stop']} |"
        )
    lines += [
        "",
        "| Paired comparison | N | Δ ANLS | 95% bootstrap CI | Δ EM | ANLS wins/ties/losses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        ci = item["delta_anls_ci95"]
        lines.append(
            f"| {item['comparison']} | {item['n']} | {item['delta_anls']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {item['delta_em']:+.4f} | "
            f"{item['anls_wins_ties_losses'][0]}/{item['anls_wins_ties_losses'][1]}/{item['anls_wins_ties_losses'][2]} |"
        )
    if energy_mwh is not None:
        lines += [
            "",
            f"Low-resolution revision aggregate incremental board energy: {energy_mwh:.2f} mWh "
            f"for {energy.get('requests_in_segment')} requests.",
        ]
    lines += [
        "",
        summary["interpretation"],
        "Energy is an aggregate NVML estimate with pre/post idle subtraction, not per-request ground-truth energy.",
        "",
    ]
    (out / "revision_control_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved:")
    print(out / "revision_control_summary.json")
    print(out / "revision_control_summary.md")
    print(out / "revision_control_question_scores.csv")
    print(out / "revision_control_category_scores.csv")


if __name__ == "__main__":
    main()
