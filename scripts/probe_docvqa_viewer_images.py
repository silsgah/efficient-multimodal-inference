"""Test image retrieval through Dataset Viewer signed URLs, without Parquet images.

Requests one train and one validation row from the Dataset Viewer, fetches the
returned signed image URL, and fully decodes one image with Pillow. It does
not create corpus outputs or read the Parquet image binary on the Pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from urllib.parse import urljoin

import requests
from PIL import Image

DATASET_ID = "pixparse/docvqa-single-page-questions"
API_ROOT = "https://datasets-server.huggingface.co"


def get_json(path: str, params: dict[str, object]) -> tuple[dict, str]:
    url = API_ROOT + path
    response = requests.get(url, params=params, timeout=(10, 90))
    response.raise_for_status()
    return response.json(), response.url


def get_one_row(dataset_id: str, config: str, split: str) -> tuple[dict, str]:
    params = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "offset": 0,
        "length": 1,
    }
    try:
        payload, endpoint_url = get_json("/rows", params)
        rows = payload.get("rows", [])
        if rows:
            row = rows[0].get("row", rows[0])
            return row, endpoint_url
    except requests.HTTPError as exc:
        print(f"/rows failed for {split}: {exc}; trying /first-rows", flush=True)
    payload, endpoint_url = get_json("/first-rows", {
        "dataset": dataset_id, "config": config, "split": split
    })
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows returned by /first-rows for split={split}")
    first = rows[0]
    row = first.get("row", first)
    return row, endpoint_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    session = requests.Session()
    for split in args.splits:
        print(f"===== {split.upper()} VIEWER IMAGE TEST =====", flush=True)
        row, endpoint_url = get_one_row(args.dataset_id, args.config, split)
        required = {"question_id", "image", "other_metadata", "ocr_results"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{split} row missing fields: {sorted(missing)}")
        image_cell = row["image"]
        if not isinstance(image_cell, dict) or not image_cell.get("src"):
            raise TypeError(f"{split} image cell does not contain a signed src URL")
        image_url = urljoin(API_ROOT + "/", str(image_cell["src"]))
        response = session.get(image_url, timeout=(10, 90))
        response.raise_for_status()
        image_bytes = response.content
        digest = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            result = {
                "split": split,
                "question_id": str(row["question_id"]),
                "question": row["question"],
                "answers": row["answers"],
                "document_id": row["other_metadata"].get("ucsf_document_id"),
                "page_no": row["other_metadata"].get("ucsf_document_page_no"),
                "ocr_line_count": len(row["ocr_results"].get("lines", [])),
                "viewer_endpoint": endpoint_url,
                "image_url_host": requests.utils.urlparse(image_url).netloc,
                "image_http_status": response.status_code,
                "image_content_type": response.headers.get("content-type"),
                "image_bytes": len(image_bytes),
                "image_sha256": digest,
                "image_size": list(image.size),
                "image_mode": image.mode,
                "image_format": image.format,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        del image_bytes, response


if __name__ == "__main__":
    main()
