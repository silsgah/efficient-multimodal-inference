"""Inspect DocVQA schema through the lightweight Hugging Face Dataset Viewer API.

This intentionally avoids local Parquet streaming and image decoding. The
Dataset Viewer returns small JSON slices, so schema inspection does not
materialize large image shards or create decoder thread pools.

The viewer is used only for schema/record inspection. Full corpus materialization
will later use a separately pinned Hub commit and a memory-bounded Parquet reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

DATASET_ID = "pixparse/docvqa-single-page-questions"
API_ROOT = "https://datasets-server.huggingface.co"


def summarize_nested(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, list):
        first = value[0] if value else None
        return {
            "type": "array",
            "length": len(value),
            "first_item_type": type(first).__name__ if first is not None else None,
            "first_item_keys": sorted(first.keys()) if isinstance(first, dict) else None,
        }
    if isinstance(value, str):
        return {"type": "string", "length": len(value), "preview": value[:100]}
    return {"type": type(value).__name__, "value": value}


def request_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = API_ROOT + path
    response = requests.get(url, params=params, timeout=(10, 90))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Dataset Viewer endpoint {path} returned non-object JSON")
    if payload.get("error"):
        raise RuntimeError(f"Dataset Viewer endpoint {path}: {payload['error']}")
    return payload


def summarize_row(row: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    required = {
        "question_id", "question", "answers", "image",
        "data_split", "ocr_results", "other_metadata",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"{split} row {row_index} missing fields: {sorted(missing)}")

    answers = row["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"{split} row {row_index}: answers must be a non-empty list")

    metadata = row["other_metadata"]
    if not isinstance(metadata, dict):
        raise TypeError(f"{split} row {row_index}: other_metadata must be an object")
    document_id = metadata.get("ucsf_document_id")
    if document_id in (None, ""):
        raise ValueError(f"{split} row {row_index}: missing ucsf_document_id")
    if "ucsf_document_page_no" not in metadata:
        raise ValueError(f"{split} row {row_index}: missing ucsf_document_page_no")

    image = row["image"]
    ocr = row["ocr_results"]
    lines = ocr.get("lines", []) if isinstance(ocr, dict) else None

    return {
        "row_index": row_index,
        "question_id": str(row["question_id"]),
        "question": row["question"],
        "accepted_answers": [str(answer) for answer in answers],
        "data_split": row["data_split"],
        "image_cell": summarize_nested(image),
        "ocr_results": summarize_nested(ocr),
        "ocr_line_count": len(lines) if isinstance(lines, list) else None,
        "first_ocr_line": summarize_nested(lines[0]) if isinstance(lines, list) and lines else None,
        "other_metadata_keys": sorted(metadata.keys()),
        "ucsf_document_id": str(document_id),
        "ucsf_document_page_no": metadata.get("ucsf_document_page_no"),
        "doc_id": metadata.get("doc_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--rows-per-split", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.rows_per_split <= 10:
        raise ValueError("--rows-per-split must be between 1 and 10")

    splits_payload = request_json("/splits", {"dataset": args.dataset_id})
    split_entries = splits_payload.get("splits", [])
    pending = splits_payload.get("pending", [])
    failed = splits_payload.get("failed", [])
    configs = sorted({entry["config"] for entry in split_entries})
    if not split_entries:
        raise RuntimeError(
            "Dataset Viewer returned no ready splits. "
            f"pending={pending!r}; failed={failed!r}"
        )
    config = "default" if "default" in configs else configs[0]
    available_splits = sorted(
        entry["split"] for entry in split_entries if entry["config"] == config
    )

    size_payload = request_json("/size", {"dataset": args.dataset_id})
    size_splits = size_payload.get("size", {}).get("splits", [])
    sizes = {
        entry["split"]: {
            "num_rows": entry.get("num_rows"),
            "num_bytes_parquet_files": entry.get("num_bytes_parquet_files"),
        }
        for entry in size_splits if entry.get("config") == config
    }

    print(json.dumps({
        "dataset_id": args.dataset_id,
        "dataset_viewer_config": config,
        "available_splits": available_splits,
        "split_sizes": sizes,
        "inspection": "Dataset Viewer JSON API; no local Parquet/image decoding",
    }, indent=2), flush=True)

    for split in ("train", "validation"):
        if split not in available_splits:
            raise RuntimeError(f"Required split {split!r} is not available for config {config!r}")
        payload = request_json("/rows", {
            "dataset": args.dataset_id,
            "config": config,
            "split": split,
            "offset": 0,
            "length": args.rows_per_split,
        })
        rows = payload.get("rows", [])
        if len(rows) != args.rows_per_split:
            raise RuntimeError(
                f"Expected {args.rows_per_split} {split} rows; got {len(rows)}. "
                f"Dataset Viewer response: {payload.get('error', 'no error field')}"
            )
        print(f"===== {split.upper()} SAMPLE =====", flush=True)
        for i, item in enumerate(rows, start=1):
            wrapped = item.get("row", item)
            if not isinstance(wrapped, dict):
                raise TypeError(f"{split} row {i} is not a JSON object")
            print(json.dumps(
                summarize_row(wrapped, split, i),
                ensure_ascii=False,
            ), flush=True)


if __name__ == "__main__":
    main()
