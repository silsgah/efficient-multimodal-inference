"""Read-only schema and small-sample inspection for the upstream DocVQA dataset.

This script streams a few rows only. It does not write image files or manifests.
It records the resolved Hub revision so the full corpus build can pin the same
immutable dataset snapshot.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

DATASET_ID = "pixparse/docvqa-single-page-questions"


def summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return f"dict(keys={keys[:20]})"
    if isinstance(value, (list, tuple)):
        first_type = type(value[0]).__name__ if value else "empty"
        return f"{type(value).__name__}(len={len(value)}, first_type={first_type})"
    if isinstance(value, (bytes, bytearray)):
        return f"{type(value).__name__}(length={len(value)})"
    return f"{type(value).__name__}: {str(value)[:160]}"


def inspect_record(record: dict[str, Any], split: str, row_number: int) -> dict[str, Any]:
    required = {"question_id", "question", "answers", "image", "data_split", "ocr_results", "other_metadata"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{split} row {row_number} missing fields: {sorted(missing)}")
    metadata = record["other_metadata"]
    if not isinstance(metadata, dict):
        raise TypeError(f"{split} row {row_number}: other_metadata must be a dict")
    if not metadata.get("ucsf_document_id"):
        raise ValueError(f"{split} row {row_number}: missing ucsf_document_id")
    if "ucsf_document_page_no" not in metadata:
        raise ValueError(f"{split} row {row_number}: missing ucsf_document_page_no")
    answers = record["answers"]
    if not isinstance(answers, (list, tuple)) or not answers:
        raise ValueError(f"{split} row {row_number}: answers must be a non-empty list")
    image = record["image"]
    if isinstance(image, dict):
        if not (image.get("bytes") or image.get("path")):
            raise TypeError(f"{split} row {row_number}: image dict has no bytes/path")
        image_summary = summarize_value(image)
        image_bytes = len(image["bytes"]) if image.get("bytes") else None
    else:
        # Hugging Face Image features may decode directly to PIL.Image.Image.
        try:
            from PIL import Image as PILImage
            is_pil_image = isinstance(image, PILImage.Image)
        except ImportError:
            is_pil_image = False
        if not is_pil_image:
            raise TypeError(
                f"{split} row {row_number}: unexpected image representation "
                f"{type(image).__name__}"
            )
        image_summary = f"PIL.Image.Image(size={image.size}, mode={image.mode})"
        image_bytes = None
    ocr = record["ocr_results"]
    if not isinstance(ocr, dict):
        raise TypeError(f"{split} row {row_number}: ocr_results must be a dict")
    lines = ocr.get("lines", [])
    return {
        "question_id": str(record["question_id"]),
        "source_split": split,
        "question_chars": len(str(record["question"])),
        "accepted_answers": [str(answer) for answer in answers],
        "image_representation": image_summary,
        "image_bytes": image_bytes,
        "document_id": str(metadata["ucsf_document_id"]),
        "document_page_no": str(metadata["ucsf_document_page_no"]),
        "doc_id": metadata.get("doc_id"),
        "metadata_keys": sorted(metadata.keys()),
        "ocr_keys": sorted(ocr.keys()),
        "ocr_line_count": len(lines) if isinstance(lines, (list, tuple)) else None,
        "first_ocr_line": lines[0] if isinstance(lines, (list, tuple)) and lines else None,
        "data_split": record["data_split"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/hf-cache/datasets"))
    parser.add_argument("--rows-per-split", type=int, default=3)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    if args.rows_per_split < 1:
        raise ValueError("--rows-per-split must be at least 1")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(args.dataset_id, revision=args.revision or "main")
    revision = args.revision or info.sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable dataset revision")
    # HfApi.dataset_info() returns Hub metadata (DatasetInfo), not the
    # datasets-library DatasetInfo object that carries split statistics.
    # Split counts are not needed for this inspection; report the immutable
    # commit and let the streaming reader validate the actual records.
    print(json.dumps({
        "dataset_id": args.dataset_id,
        "resolved_revision": revision,
        "inspection_mode": "streaming; no rows or images written",
    }, indent=2), flush=True)

    for split in ("train", "validation"):
        stream = load_dataset(
            args.dataset_id,
            split=split,
            revision=revision,
            cache_dir=str(args.cache_dir),
            streaming=True,
        )
        print(f"===== {split.upper()} SAMPLE =====", flush=True)
        for row_number, record in enumerate(itertools.islice(iter(stream), args.rows_per_split), start=1):
            print(json.dumps(inspect_record(record, split, row_number), ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
