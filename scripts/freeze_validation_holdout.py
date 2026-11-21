"""Freeze a page-disjoint holdout from the existing 200-question manifest.

Historical development question IDs are read from the archived energy CSV.
Every manifest question whose image content matches a historical development
image is assigned to the development side, even if its question ID was not
benchmarked. Inputs are read-only; split artifacts are written to a new folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def resolve_image_path(image_value: str, manifest_path: Path) -> Path:
    image_path = Path(image_value)
    if image_path.is_absolute():
        if image_path.is_file():
            return image_path
        raise FileNotFoundError(image_path)

    search_roots = [Path.cwd(), manifest_path.parent]
    search_roots.extend(Path.cwd().parents)
    search_roots.extend(manifest_path.parent.parents)

    checked: set[Path] = set()
    for root in search_roots:
        candidate = (root / image_path).resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve image {image_value!r} referenced by {manifest_path}"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id on line {line_number}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r}")
            if not record.get("image"):
                raise ValueError(f"Missing image on line {line_number}")
            seen_ids.add(sample_id)
            record["sample_id"] = sample_id
            records.append(record)
    if not records:
        raise ValueError(f"No samples found in {path}")
    return records


def read_historical_development_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "request_id" not in (reader.fieldnames or []):
            raise ValueError("Historical results CSV must contain request_id")
        for row in reader:
            value = str(row.get("request_id", "")).strip()
            if value:
                ids.add(value)
    if not ids:
        raise ValueError(f"No request_id values found in {path}")
    return ids


def partition_by_document(
    samples: list[dict[str, Any]],
    historical_development_ids: set[str],
    image_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    sample_ids = {str(sample["sample_id"]) for sample in samples}
    missing = historical_development_ids - sample_ids
    if missing:
        raise ValueError(
            "Historical development IDs absent from manifest: "
            + ", ".join(sorted(missing)[:20])
        )

    development_image_hashes = {
        image_hashes[sample_id] for sample_id in historical_development_ids
    }
    development = [
        sample for sample in samples
        if str(sample["sample_id"]) in historical_development_ids
        or image_hashes[str(sample["sample_id"])] in development_image_hashes
    ]
    development_ids = {str(sample["sample_id"]) for sample in development}
    holdout = [
        sample for sample in samples
        if str(sample["sample_id"]) not in development_ids
    ]

    development_hashes = {
        image_hashes[str(sample["sample_id"])] for sample in development
    }
    holdout_hashes = {
        image_hashes[str(sample["sample_id"])] for sample in holdout
    }
    overlap = development_hashes & holdout_hashes
    if overlap:
        raise RuntimeError(
            f"Image leakage remains after partitioning: {len(overlap)} hashes"
        )
    if not holdout:
        raise ValueError("No holdout samples remain after removing image overlap")

    return development, holdout, development_image_hashes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/docvqa/manifest.jsonl")
    )
    parser.add_argument(
        "--historical-results",
        type=Path,
        default=Path(
            "publication/energy_measurement_2026-11-18/energy_requests.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("publication/rl_data_split_2026-11-21"),
    )
    args = parser.parse_args()

    samples = read_jsonl(args.manifest)
    historical_ids = read_historical_development_ids(args.historical_results)

    image_hashes: dict[str, str] = {}
    image_paths: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        path = resolve_image_path(str(sample["image"]), args.manifest)
        image_paths[sample_id] = str(path)
        image_hashes[sample_id] = sha256_file(path)

    development, holdout, historical_image_hashes = partition_by_document(
        samples, historical_ids, image_hashes
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_path = args.output_dir / "development.jsonl"
    holdout_path = args.output_dir / "holdout.jsonl"
    summary_path = args.output_dir / "split_summary.json"

    write_jsonl(development_path, development)
    write_jsonl(holdout_path, holdout)

    development_hashes = {
        image_hashes[str(sample["sample_id"])] for sample in development
    }
    holdout_hashes = {
        image_hashes[str(sample["sample_id"])] for sample in holdout
    }

    summary = {
        "schema_version": 1,
        "method": "group_by_sha256_image_content",
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_bytes(args.manifest.read_bytes()),
        "historical_results_path": str(args.historical_results),
        "historical_results_sha256": sha256_bytes(
            args.historical_results.read_bytes()
        ),
        "historical_development_question_ids": len(historical_ids),
        "historical_development_images": len(historical_image_hashes),
        "development_questions_in_output": len(development),
        "development_unique_images": len(development_hashes),
        "holdout_questions_in_output": len(holdout),
        "holdout_unique_images": len(holdout_hashes),
        "shared_image_hashes": len(development_hashes & holdout_hashes),
        "development_manifest": str(development_path),
        "holdout_manifest": str(holdout_path),
        "development_sample_ids": sorted(
            str(sample["sample_id"]) for sample in development
        ),
        "holdout_sample_ids": sorted(
            str(sample["sample_id"]) for sample in holdout
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"development_sample_ids", "holdout_sample_ids"}
    }, indent=2))


if __name__ == "__main__":
    main()
