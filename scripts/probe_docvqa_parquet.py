"""Probe one upstream Parquet shard per split with bounded memory.

Downloads the first train and validation Parquet shard from a pinned Hub
revision, inspects Parquet metadata/schema, and reads at most one row per
split with PyArrow. It does not extract images or write corpus artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "pixparse/docvqa-single-page-questions"
EXPECTED_PREFIXES = {"train": "data/train-", "validation": "data/validation-"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/hf-cache"))
    args = parser.parse_args()

    api = HfApi()
    info = api.dataset_info(args.dataset_id, revision=args.revision or "main")
    revision = args.revision or info.sha
    if not revision:
        raise RuntimeError("Could not resolve immutable Hub commit SHA")

    files = api.list_repo_files(
        repo_id=args.dataset_id, repo_type="dataset", revision=revision
    )
    print(json.dumps({
        "dataset_id": args.dataset_id,
        "resolved_revision": revision,
        "parquet_files": {
            split: len([name for name in files if name.startswith(prefix) and name.endswith(".parquet")])
            for split, prefix in EXPECTED_PREFIXES.items()
        },
    }, indent=2), flush=True)

    required_columns = [
        "question_id", "question", "answers", "data_split",
        "ocr_results", "other_metadata",
    ]
    for split, prefix in EXPECTED_PREFIXES.items():
        candidates = sorted(
            name for name in files
            if name.startswith(prefix) and name.endswith(".parquet")
        )
        if not candidates:
            raise FileNotFoundError(f"No {split} parquet shards found at {revision}")
        filename = candidates[0]
        local_path = hf_hub_download(
            repo_id=args.dataset_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(args.cache_dir),
        )
        parquet_file = pq.ParquetFile(local_path)
        available = parquet_file.schema_arrow.names
        missing = set(required_columns + ["image"]) - set(available)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

        print(f"===== {split.upper()} PARQUET =====", flush=True)
        arrow_schema = parquet_file.schema_arrow
        image_field = arrow_schema.field("image")
        print(json.dumps({
            "file": filename,
            "local_file": local_path,
            "file_bytes": Path(local_path).stat().st_size,
            "file_sha256": sha256_file(Path(local_path)),
            "row_count_in_shard": parquet_file.metadata.num_rows,
            "row_group_count": parquet_file.metadata.num_row_groups,
            "image_arrow_type": str(image_field.type),
            "image_field_nullable": image_field.nullable,
            "schema": str(arrow_schema),
        }, indent=2), flush=True)

        # Read one metadata/OCR row only. The image binary column is deliberately
        # omitted: the Arrow schema above records its representation without
        # pulling image bytes into Python memory.
        batch = next(parquet_file.iter_batches(
            batch_size=1, columns=required_columns, use_threads=False
        ))
        record = batch.to_pylist()[0]
        metadata = record["other_metadata"]
        ocr = record["ocr_results"]
        ocr_lines = ocr.get("lines", []) if isinstance(ocr, dict) else []
        first_line = ocr_lines[0] if ocr_lines else {}
        print(json.dumps({
            "question_id": str(record["question_id"]),
            "question": record["question"],
            "answers": record["answers"],
            "data_split": record["data_split"],
            "other_metadata": metadata,
            "ocr_keys": sorted(ocr.keys()) if isinstance(ocr, dict) else None,
            "ocr_line_count": len(ocr_lines),
            "first_ocr_line_keys": sorted(first_line.keys()) if isinstance(first_line, dict) else None,
            "first_ocr_line_text": first_line.get("text") if isinstance(first_line, dict) else None,
            "first_ocr_line_bbox": first_line.get("bounding_box") if isinstance(first_line, dict) else None,
        }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
