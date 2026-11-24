#!/usr/bin/env python3
"""Build inference-time feature audit for the 128-question evidence-acquisition sample.

Uses only observable pre-action context as predictor features. Gold-derived
ANLS deltas are used strictly as outcomes for retrospective analysis, never as
input features. No policy model is fitted in this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
ABSTENTION_TERMS = ("cannot determine", "can't determine", "not readable", "not visible", "unable to read", "not provided")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample_id in {path}")
    return result


def tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def truthy_number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = math.sqrt(sum(x*x for x in da) * sum(y*y for y in db))
    if denom == 0:
        return None
    return sum(x*y for x, y in zip(da, db)) / denom


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(rank_values(a), rank_values(b))


def quantile(values: list[float], q: float) -> float:
    data = sorted(values)
    if not data:
        return float("nan")
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["visual_delta"]) for row in rows]
    return {
        "n": len(rows),
        "mean_visual_delta_anls": statistics.fmean(deltas) if deltas else None,
        "median_visual_delta_anls": statistics.median(deltas) if deltas else None,
        "improved_n": sum(x > 1e-9 for x in deltas),
        "tied_n": sum(abs(x) <= 1e-9 for x in deltas),
        "regressed_n": sum(x < -1e-9 for x in deltas),
        "improvement_rate": sum(x > 1e-9 for x in deltas) / len(deltas) if deltas else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/workspace/projects/efficient-multimodal-inference/results/action_benchmark_2026-11-24"),
    )
    args = parser.parse_args()

    base = args.benchmark_dir.resolve()
    inputs_path = base / "benchmark128_inputs.jsonl"
    predictions_path = base / "inference" / "predictions.jsonl"
    paired_path = base / "inference" / "evaluation" / "paired_question_scores.csv"
    revision_path = base / "inference" / "evaluation" / "revision_control_question_scores.csv"

    for path in (inputs_path, predictions_path, paired_path, revision_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = read_jsonl(inputs_path)
    predictions = read_jsonl(predictions_path)
    paired = read_csv(paired_path)
    revision = read_csv(revision_path)
    if len(inputs) != 128:
        print(f"NOTE: expected the planned 128-question sample; found {len(inputs)}.")

    pred_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions:
        pred_by_id[(str(row["sample_id"]), str(row["stage"]))] = row

    rows: list[dict[str, Any]] = []
    missing = []
    for sample in inputs:
        sid = str(sample["sample_id"])
        if sid not in paired or sid not in revision:
            missing.append(sid)
            continue
        first = pred_by_id.get((sid, "first_pass"))
        roi = pred_by_id.get((sid, "roi_crop"))
        if first is None or roi is None:
            missing.append(sid)
            continue

        question = str(sample.get("question", ""))
        q_tokens = tokens(question)
        q_unique = sorted(set(q_tokens))
        page = sample.get("page") or {}
        proposals = roi.get("roi_proposals") or []
        proposals = [p for p in proposals if isinstance(p, dict)]

        anchor_tokens = set()
        context_line_count = 0
        context_word_count = 0
        for proposal in proposals:
            anchor_tokens.update(tokens(proposal.get("anchor_text", "")))
            for text in proposal.get("context_lines", []) or []:
                context_line_count += 1
                context_word_count += len(tokens(text))

        best_proposal = max(proposals, key=lambda p: truthy_number(p.get("bm25_score")), default={})
        bm25_scores = sorted(
            [truthy_number(p.get("bm25_score")) for p in proposals],
            reverse=True,
        )
        roi_areas = [truthy_number(p.get("page_area_fraction")) for p in proposals]
        width = truthy_number(page.get("width"))
        height = truthy_number(page.get("height"))
        pixels = width * height
        word_count = truthy_number(page.get("word_count"))
        line_count = truthy_number(page.get("line_count"))
        initial_answer = str(first.get("answer") or "")
        answer_tokens = tokens(initial_answer)

        features = {
            "sample_id": sid,
            "question_category": str(sample.get("question_category") or "uncategorized"),
            "question_chars": len(question),
            "question_token_count": len(q_tokens),
            "question_unique_token_count": len(q_unique),
            "question_avg_token_length": statistics.fmean([len(t) for t in q_tokens]) if q_tokens else 0.0,
            "question_has_digit": int(bool(NUMBER_RE.search(question))),
            "question_numeric_pattern_count": len(NUMBER_RE.findall(question)),
            "question_has_date_cue": int(bool(re.search(r"\b(date|day|month|year|when|dated)\b", question, re.I))),
            "question_has_numeric_cue": int(bool(re.search(r"\b(amount|total|number|percentage|percent|how many|how much|rate|price|cost|value)\b", question, re.I))),
            "question_has_location_cue": int(bool(re.search(r"\b(where|location|address|city|country|place)\b", question, re.I))),
            "question_has_person_cue": int(bool(re.search(r"\b(who|name|person|director|manager|author|president)\b", question, re.I))),
            "page_width": width,
            "page_height": height,
            "page_pixels": pixels,
            "page_aspect_ratio": width / height if height else 0.0,
            "page_line_count": line_count,
            "page_word_count": word_count,
            "page_words_per_line": word_count / line_count if line_count else 0.0,
            "page_words_per_megapixel": word_count / (pixels / 1e6) if pixels else 0.0,
            "page_lines_per_megapixel": line_count / (pixels / 1e6) if pixels else 0.0,
            "page_orientation": truthy_number(page.get("orientation")),
            "roi_available": int(bool(proposals)),
            "roi_proposal_count": len(proposals),
            "roi_best_bm25_score": bm25_scores[0] if bm25_scores else 0.0,
            "roi_second_bm25_score": bm25_scores[1] if len(bm25_scores) > 1 else 0.0,
            "roi_score_gap": (bm25_scores[0] - bm25_scores[1]) if len(bm25_scores) > 1 else (bm25_scores[0] if bm25_scores else 0.0),
            "roi_area_sum_fraction": sum(roi_areas),
            "roi_area_max_fraction": max(roi_areas, default=0.0),
            "roi_anchor_query_token_overlap": len(anchor_tokens.intersection(q_unique)),
            "roi_anchor_query_overlap_fraction": (
                len(anchor_tokens.intersection(q_unique)) / len(q_unique) if q_unique else 0.0
            ),
            "roi_context_line_count": context_line_count,
            "roi_context_word_count": context_word_count,
            "initial_answer_chars": len(initial_answer.strip()),
            "initial_answer_token_count": len(answer_tokens),
            "initial_answer_has_digit": int(bool(NUMBER_RE.search(initial_answer))),
            "initial_answer_is_numeric": int(bool(re.fullmatch(r"[\s$£€¥+\-()\d.,%/]+", initial_answer))),
            "initial_answer_is_abstention": int(any(term in initial_answer.lower() for term in ABSTENTION_TERMS)),
        }

        paired_row = paired[sid]
        revision_row = revision[sid]
        visual_delta = truthy_number(revision_row.get("full_upscale_minus_lowres_revision_anls"))
        lowrev_delta = truthy_number(revision_row.get("lowres_revision_minus_stop_anls"))
        total_delta = truthy_number(paired_row.get("full_minus_stop_anls"))
        features.update({
            "visual_delta": visual_delta,
            "visual_improved": int(visual_delta > 1e-9),
            "visual_tied": int(abs(visual_delta) <= 1e-9),
            "visual_regressed": int(visual_delta < -1e-9),
            "lowres_revision_delta": lowrev_delta,
            "full_total_delta_vs_stop": total_delta,
            "roi_delta_vs_stop": (
                truthy_number(paired_row.get("roi_minus_stop_anls"))
                if paired_row.get("roi_minus_stop_anls") not in (None, "") else None
            ),
        })
        rows.append(features)

    if missing:
        raise RuntimeError(f"Missing paired inference/scoring records for: {missing[:12]} (total={len(missing)})")
    if len(rows) != len(inputs):
        raise RuntimeError(f"Feature rows {len(rows)} do not match input rows {len(inputs)}.")

    output = base / "inference" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "action_predictor_feature_matrix.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    numeric_features = [
        key for key, value in rows[0].items()
        if key not in {
            "sample_id", "question_category", "visual_delta", "visual_improved",
            "visual_tied", "visual_regressed", "lowres_revision_delta",
            "full_total_delta_vs_stop", "roi_delta_vs_stop",
        } and isinstance(value, (int, float))
    ]
    feature_correlations = []
    target = [float(row["visual_delta"]) for row in rows]
    binary_target = [float(row["visual_improved"]) for row in rows]
    for feature in numeric_features:
        values = [float(row[feature]) for row in rows]
        rho = spearman(values, target)
        rho_binary = spearman(values, binary_target)
        feature_correlations.append({
            "feature": feature,
            "spearman_rho_vs_visual_delta": rho,
            "spearman_rho_vs_improvement_binary": rho_binary,
            "unique_values": len(set(values)),
        })
    feature_correlations.sort(
        key=lambda row: abs(row["spearman_rho_vs_visual_delta"] or 0.0),
        reverse=True,
    )

    quartile_summaries = {}
    for feature in numeric_features:
        values = [float(row[feature]) for row in rows]
        if len(set(values)) < 4:
            continue
        cuts = [quantile(values, q) for q in (0.25, 0.50, 0.75)]
        bins = [[], [], [], []]
        for row in rows:
            value = float(row[feature])
            index = 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2 if value <= cuts[2] else 3
            bins[index].append(row)
        quartile_summaries[feature] = [
            {
                "bin": label,
                "n": len(group),
                **summarize_group(group),
                "range_min": min(float(r[feature]) for r in group) if group else None,
                "range_max": max(float(r[feature]) for r in group) if group else None,
            }
            for label, group in zip(("Q1", "Q2", "Q3", "Q4"), bins)
        ]

    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roi_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_groups[str(row["question_category"])].append(row)
        roi_groups["ROI_available" if row["roi_available"] else "ROI_unavailable"].append(row)

    summary = {
        "experiment": "action_predictor_feature_audit_128",
        "n": len(rows),
        "target": {
            "name": "visual_delta",
            "definition": "ANLS(FULL_UPSCALE_REVISION) - ANLS(LOW_RESOLUTION_REVISION_CONTROL)",
            "improved_threshold": "strictly greater than 1e-9",
            "improved_n": sum(binary_target),
            "tied_n": sum(1 for x in target if abs(x) <= 1e-9),
            "regressed_n": sum(1 for x in target if x < -1e-9),
            "mean_delta": statistics.fmean(target),
            "median_delta": statistics.median(target),
        },
        "feature_provenance": {
            "question_and_page_features": "benchmark128_inputs.jsonl",
            "roi_retrieval_features": "stored ROI proposal metadata generated before ROI VLM inference",
            "initial_answer_features": "first_pass answer text generated before acquisition",
            "gold_labels": "used only indirectly to calculate retrospective outcome deltas; never included as feature columns",
            "low_resolution_revision_answer": "excluded from feature columns because it requires an extra inference call and is not available in the single-pass state",
            "revision_or_full_answer_text": "excluded from feature columns",
        },
        "features": numeric_features + ["question_category"],
        "spearman_correlations": feature_correlations,
        "quartile_summaries": quartile_summaries,
        "by_question_category": {
            key: summarize_group(group) for key, group in sorted(category_groups.items())
        },
        "by_roi_availability": {
            key: summarize_group(group) for key, group in sorted(roi_groups.items())
        },
        "files": {
            "feature_matrix_csv": str(csv_path),
            "summary_json": str(output / "action_predictor_feature_audit.json"),
        },
        "caution": (
            "This is an exploratory 128-question policy_train sample. Correlations are univariate, "
            "not causal or validated predictors. Do not choose thresholds or report generalization "
            "from this audit; fit on policy_train and tune on the separate policy_validation partition."
        ),
    }
    summary_path = output / "action_predictor_feature_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("========== OUTCOME ==========")
    print(json.dumps(summary["target"], indent=2))
    print("\n========== TOP FEATURE CORRELATIONS ==========")
    for item in feature_correlations[:15]:
        rho = item["spearman_rho_vs_visual_delta"]
        rb = item["spearman_rho_vs_improvement_binary"]
        print(
            f"{item['feature']:42} "
            f"rho_delta={rho if rho is not None else float('nan'):+.3f} "
            f"rho_improved={rb if rb is not None else float('nan'):+.3f} "
            f"unique={item['unique_values']}"
        )
    print("\n========== ROI AVAILABILITY ==========")
    print(json.dumps(summary["by_roi_availability"], indent=2))
    print("\n========== CATEGORY ==========")
    print(json.dumps(summary["by_question_category"], indent=2))
    print("\n========== STRONGEST QUARTILE PATTERNS ==========")
    for item in feature_correlations[:8]:
        feature = item["feature"]
        print(f"\n{feature}")
        for bin_row in quartile_summaries.get(feature, []):
            print(
                f"  {bin_row['bin']} n={bin_row['n']:3} "
                f"range=[{bin_row['range_min']:.4g},{bin_row['range_max']:.4g}] "
                f"gain={bin_row['mean_visual_delta_anls']:+.4f} "
                f"improve={bin_row['improved_n']}/{bin_row['n']}"
            )
    print("\nSaved:", csv_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
