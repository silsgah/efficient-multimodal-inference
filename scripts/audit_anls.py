"""Compare archived DocVQA scores against the reference-aligned scorer.

Example:
  python scripts/audit_anls.py \
    --input publication/energy_measurement_2026-11-18/energy_requests.csv

The script is read-only and never modifies historical result files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.evaluation.metrics import score_answer


def parse_references(raw: str) -> list[str]:
    value = json.loads(raw)
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError("reference must encode a non-empty JSON list or string")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("all reference answers must be strings")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "publication/energy_measurement_2026-11-18/energy_requests.csv"
        ),
    )
    parser.add_argument("--max-mismatches", type=int, default=20)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    question_scores: dict[str, list[dict[str, object]]] = defaultdict(list)
    mismatch_rows: list[dict[str, object]] = []

    with args.input.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"request_id", "budget", "repetition", "answer", "reference"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        has_legacy_anls = "anls_style" in (reader.fieldnames or [])
        has_legacy_em = "exact_match" in (reader.fieldnames or [])

        row_count = 0
        for row in reader:
            row_count += 1
            references = parse_references(row["reference"])
            recomputed = score_answer(row["answer"], references)

            record = {
                "request_id": row["request_id"],
                "budget": row["budget"],
                "repetition": row["repetition"],
                **recomputed,
            }
            question_scores[row["request_id"]].append(record)

            differences = {}
            if has_legacy_anls:
                legacy = float(row["anls_style"])
                if not math.isclose(
                    legacy, recomputed["anls"], rel_tol=0.0, abs_tol=1e-9
                ):
                    differences["legacy_anls_style"] = legacy
                    differences["recomputed_anls"] = recomputed["anls"]
            if has_legacy_em:
                legacy_em = float(row["exact_match"])
                if not math.isclose(
                    legacy_em, recomputed["exact_match"],
                    rel_tol=0.0, abs_tol=1e-9
                ):
                    differences["legacy_exact_match"] = legacy_em
                    differences["recomputed_exact_match"] = recomputed["exact_match"]

            if differences:
                mismatch_rows.append({**record, **differences})

    unique_question_count = len(question_scores)
    anls_by_question = []
    em_by_question = []
    repeat_inconsistencies = []

    for request_id, records in question_scores.items():
        # Keep each question separate by resolution: each resolution has its
        # own answer and score; repetitions are executions, not new questions.
        by_budget: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            by_budget[str(record["budget"])].append(record)

        for budget, budget_records in by_budget.items():
            anls_values = {float(r["anls"]) for r in budget_records}
            em_values = {float(r["exact_match"]) for r in budget_records}
            answers_are_stable = len({
                (str(r["anls"]), str(r["exact_match"])) for r in budget_records
            }) == 1
            if not answers_are_stable:
                repeat_inconsistencies.append({
                    "request_id": request_id,
                    "budget": budget,
                    "records": budget_records,
                })
            anls_by_question.append((request_id, budget, sum(anls_values) / len(anls_values)))
            em_by_question.append((request_id, budget, sum(em_values) / len(em_values)))

    print("SCORER: case-insensitive, outer-strip-only DocVQA ANLS")
    print("ANLS cutoff: normalized edit distance strictly below 0.5")
    print("Input:", args.input)
    print("Rows:", row_count)
    print("Unique questions:", unique_question_count)
    print("Question-resolution groups:", len(anls_by_question))
    print("Stored-score mismatching rows:", len(mismatch_rows))
    print("Within-question/repeat score inconsistencies:", len(repeat_inconsistencies))

    print("\nRECOMPUTED MEANS BY RESOLUTION")
    for budget in sorted({budget for _, budget, _ in anls_by_question}, key=int):
        anls_values = [score for _, b, score in anls_by_question if b == budget]
        em_values = [
            score for _, b, score in em_by_question if b == budget
        ]
        print({
            "budget": budget,
            "questions": len(anls_values),
            "anls": round(sum(anls_values) / len(anls_values), 6),
            "exact_match": round(sum(em_values) / len(em_values), 6),
        })

    if mismatch_rows:
        print("\nFIRST SCORE MISMATCHES")
        for item in mismatch_rows[:args.max_mismatches]:
            print(item)

    if repeat_inconsistencies:
        print("\nFIRST REPEAT INCONSISTENCIES")
        for item in repeat_inconsistencies[:args.max_mismatches]:
            print(item)


if __name__ == "__main__":
    main()
