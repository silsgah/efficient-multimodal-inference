#!/usr/bin/env python3
"""Archive the frozen DocVQA research corpus in a private Hugging Face dataset repo.

The original corpus is never modified. A sibling staging tree shards the
11,472 image files by their SHA-256 filename prefix so that no repository
folder exceeds the Hub's 10,000-entry limit. Images must be hard-linked (not
copied), preventing a second 12 GiB local copy. JSON/JSONL metadata are copied
and image paths rewritten to the staged relative paths. Upload is resumable
and remote file count/bytes are verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CORPUS = Path(
    "/workspace/worktrees/emi-docvqa-corpus/data/docvqa/full_corpus_2026-11-22"
)
SOURCE_DATASET = "pixparse/docvqa-single-page-questions"
SOURCE_REVISION = "13619c00b32b50729b6e50b4bb0aa4b355698f3f"
PARTITIONS = {
    "policy_train": {"questions": 34092, "groups": 4734},
    "policy_validation": {"questions": 3612, "groups": 513},
    "historical_development": {"questions": 517, "groups": 45},
    "locked_evaluation": {"questions": 1784, "groups": 394},
    "cross_split_validation_excluded": {"questions": 3048, "groups": 499},
    "historical_overlap_train_excluded": {"questions": 1759, "groups": 24},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_destination(path: Path) -> str:
    stem = path.stem.lower()
    shard = stem[:2] if len(stem) >= 2 else "misc"
    return f"images/{shard}/{path.name}"


def collect_source_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {
            "ARCHIVE_MANIFEST.json",
            "README.md",
            "UPLOAD_VERIFICATION.json",
        }:
            continue
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        destination = image_destination(path) if relative.startswith("images/") else relative
        files.append({
            "source_path": relative,
            "path": destination,
            "size_bytes": path.stat().st_size,
            "source_sha256": sha256(path),
        })
    return files


def source_fingerprint(files: list[dict[str, Any]]) -> str:
    payload = [
        (item["source_path"], item["size_bytes"], item["source_sha256"])
        for item in files
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rewrite_json_value(value: Any, basename_to_dest: dict[str, str]) -> tuple[Any, int]:
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        # Do a constant-time basename lookup rather than scanning all image names.
        if normalized.startswith("images/") or "/images/" in normalized:
            basename = normalized.rsplit("/", 1)[-1]
            destination = basename_to_dest.get(basename)
            if destination is not None and normalized != destination:
                return destination, 1
        return value, 0

    if isinstance(value, list):
        updated = []
        replacements = 0
        for item in value:
            new_item, count = rewrite_json_value(item, basename_to_dest)
            updated.append(new_item)
            replacements += count
        return updated, replacements

    if isinstance(value, dict):
        updated = {}
        replacements = 0
        for key, item in value.items():
            new_item, count = rewrite_json_value(item, basename_to_dest)
            updated[key] = new_item
            replacements += count
        return updated, replacements

    return value, 0


def rewrite_staged_json_paths(stage: Path, basename_to_dest: dict[str, str]) -> int:
    total_replacements = 0
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.name == "ARCHIVE_MANIFEST.json":
            continue

        if path.suffix.lower() == ".jsonl":
            temp = path.with_suffix(path.suffix + ".rewrite-tmp")
            replacements = 0
            with path.open("r", encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
                for line in source:
                    if not line.strip():
                        target.write(line)
                        continue
                    item = json.loads(line)
                    updated, count = rewrite_json_value(item, basename_to_dest)
                    target.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
                    replacements += count
            temp.replace(path)
            total_replacements += replacements
        else:
            content = json.loads(path.read_text(encoding="utf-8"))
            updated, count = rewrite_json_value(content, basename_to_dest)
            if count:
                path.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            total_replacements += count

    return total_replacements


def dataset_card(manifest: dict[str, Any]) -> str:
    return f"""---
language:
- en
task_categories:
- document-question-answering
- question-answering
tags:
- docvqa
- document-understanding
- visual-question-answering
- research
pretty_name: DocVQA Research Corpus (document-grouped partitions)
size_categories:
- 10K<n<100K
---

# DocVQA research corpus — document-grouped partitions

This is a research-prepared archival copy of the
PixParse DocVQA single-page questions dataset ({SOURCE_DATASET}) and its page
images/OCR, pinned to source revision {SOURCE_REVISION}.

## Corpus snapshot

- Questions: 44,812
- Unique page images: {manifest["image_file_count"]:,}
- Source train questions: 39,463
- Source validation questions: 5,349
- Official test split: not loaded
- Files before archive metadata: {manifest["file_count"]:,}
- Bytes before archive metadata: {manifest["total_bytes"]:,}
- Images are stored under images/xx/ hash-prefix folders to respect Hub folder limits.

## Partitions

| Partition | Questions | Document groups |
|---|---:|---:|
| policy_train | 34,092 | 4,734 |
| policy_validation | 3,612 | 513 |
| historical_development | 517 | 45 |
| locked_evaluation | 1,784 | 394 |
| cross_split_validation_excluded | 3,048 | 499 |
| historical_overlap_train_excluded | 1,759 | 24 |

The exclusions are intentional. Source counts reconcile to 39,463 train and
5,349 validation questions. Five protected leakage comparisons were audited
with zero shared document groups and zero shared page-image hashes.

## Provenance, license and upstream terms

The source Hugging Face dataset card labels its dataset MIT. The official
DocVQA challenge page separately instructs downloaders to read and follow the
challenge terms and conditions. The original images are pages from documents
in the UCSF Industry Documents Library. This archive does not override,
expand, or replace any upstream license, rights, or access conditions.
The repository should remain private unless redistribution rights have been
independently confirmed.

Source dataset: https://huggingface.co/datasets/{SOURCE_DATASET}
Official terms page: https://site.docvqa.org/datasets

Citation:

@InProceedings{{docvqa_wacv,
  author    = {{Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, C.V.}},
  title     = {{DocVQA: A Dataset for VQA on Document Images}},
  booktitle = {{WACV}},
  year      = {{2021}},
  pages     = {{2200-2209}}
}}

## Integrity

ARCHIVE_MANIFEST.json records each staged archive path, source path, file
size, source SHA-256, and staged SHA-256. The manifest excludes itself and
this README. The uploader verifies the remote file count and byte total.

## Structure

The image subfolder layout is a storage-only change. JSON and JSONL image
references in this staged archive point to sharded relative paths. The
original source corpus directory is left unchanged.
"""


def verify_existing_stage(
    stage: Path,
    marker_path: Path,
    files: list[dict[str, Any]],
    fingerprint: str,
) -> dict[str, Any]:
    if not marker_path.is_file():
        raise FileExistsError(
            f"Staging directory exists without a verification marker: {stage}. "
            "Inspect it manually; refusing to overwrite."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("source_fingerprint") != fingerprint:
        raise RuntimeError("Existing staging directory was built from different source bytes.")

    manifest_path = stage / "ARCHIVE_MANIFEST.json"
    readme_path = stage / "README.md"
    if not manifest_path.is_file() or not readme_path.is_file():
        raise RuntimeError("Existing staging tree is missing its archive manifest or README.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest.get("files", [])}
    if len(expected) != len(files):
        raise RuntimeError("Staging manifest file count does not match source corpus.")

    actual_paths = set()
    for path in stage.rglob("*"):
        if not path.is_file() or path.name in {"ARCHIVE_MANIFEST.json", "README.md"}:
            continue
        relative = path.relative_to(stage).as_posix()
        actual_paths.add(relative)
        if relative not in expected:
            raise RuntimeError(f"Unexpected file in staging tree: {relative}")
        if path.stat().st_size != expected[relative]["size_bytes"]:
            raise RuntimeError(f"Staged file size changed: {relative}")
    if actual_paths != set(expected):
        raise RuntimeError("Staging tree is incomplete; missing files detected.")
    return marker


def make_staging(
    root: Path,
    stage: Path,
    files: list[dict[str, Any]],
    repo_id: str,
    is_public: bool,
) -> dict[str, Any]:
    basename_to_dest = {
        Path(item["source_path"]).name: item["path"]
        for item in files if item["source_path"].startswith("images/")
    }
    image_source_count = sum(
        1 for item in files if item["source_path"].startswith("images/")
    )
    if len(basename_to_dest) != image_source_count:
        raise RuntimeError("Duplicate image basenames found; refusing ambiguous path rewriting.")

    marker_path = stage.parent / (stage.name + ".source_manifest.json")
    fingerprint = source_fingerprint(files)
    if stage.exists():
        return verify_existing_stage(stage, marker_path, files, fingerprint)

    stage.mkdir(parents=True)
    bucket_counts: Counter[str] = Counter()
    expected_path_map = {item["path"]: item for item in files}

    try:
        for item in files:
            source = root / item["source_path"]
            destination = stage / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)

            if item["source_path"].startswith("images/"):
                bucket_counts[destination.parent.name] += 1
                try:
                    os.link(source, destination)
                except OSError as exc:
                    raise RuntimeError(
                        "Hard-link staging failed. Source corpus was not copied. "
                        "The staging and source directories must be on the same filesystem."
                    ) from exc
            else:
                # Structured JSON files are copied so their path references can be rewritten.
                if source.suffix.lower() in {".json", ".jsonl"}:
                    import shutil
                    shutil.copy2(source, destination)
                else:
                    try:
                        os.link(source, destination)
                    except OSError as exc:
                        raise RuntimeError(
                            f"Hard-link staging failed for {item['source_path']}; refusing duplicate storage."
                        ) from exc

        path_replacements = rewrite_staged_json_paths(stage, basename_to_dest)
        staged_entries = []
        staged_bytes = 0
        actual_count = 0
        actual_buckets: Counter[str] = Counter()

        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            expected = expected_path_map.get(relative)
            if expected is None:
                raise RuntimeError(f"Unexpected staged file before archive metadata: {relative}")

            actual_count += 1
            staged_size = path.stat().st_size
            staged_bytes += staged_size
            if relative.startswith("images/"):
                actual_buckets[path.parent.name] += 1
                if sha256(path) != expected["source_sha256"]:
                    raise RuntimeError(f"Hard-linked image hash mismatch: {relative}")

            staged_entries.append({
                "path": relative,
                "source_path": expected["source_path"],
                "size_bytes": staged_size,
                "source_sha256": expected["source_sha256"],
                "staged_sha256": sha256(path),
            })

        if actual_count != len(files):
            raise RuntimeError(f"Staged file count {actual_count} != source count {len(files)}")
        if any(count >= 10000 for count in actual_buckets.values()):
            raise RuntimeError("An image subfolder reaches 10,000 files; refusing upload.")

        manifest = {
            "archive_created_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": repo_id,
            "repo_type": "dataset",
            "visibility": "public" if is_public else "private",
            "source_corpus_directory": str(root),
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_test_split_loaded": False,
            "partition_counts": PARTITIONS,
            "image_file_count": image_source_count,
            "image_directory_shards": dict(sorted(actual_buckets.items())),
            "json_image_path_replacements": path_replacements,
            "file_count": actual_count,
            "total_bytes": staged_bytes,
            "files": staged_entries,
        }
        (stage / "ARCHIVE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (stage / "README.md").write_text(dataset_card(manifest), encoding="utf-8")

        marker = {
            "source_fingerprint": fingerprint,
            "path_replacements": path_replacements,
            "bucket_counts": dict(sorted(actual_buckets.items())),
            "staged_file_count": actual_count,
            "staged_total_bytes": staged_bytes,
        }
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        return marker
    except Exception:
        # Partial staging is kept for inspection; original corpus is untouched.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo in namespace/repository-name form",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--public", action="store_true", help="Create/use a public repo. Default is private.")
    parser.add_argument("--dry-run", action="store_true", help="Audit only. No repository or files changed.")
    args = parser.parse_args()

    root = args.corpus_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if "/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/"):
        raise ValueError("--repo-id must be namespace/repository-name.")
    if not (root / "pages.jsonl").is_file():
        raise FileNotFoundError(root / "pages.jsonl")

    image_dir = root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    image_files = [path for path in image_dir.iterdir() if path.is_file()]
    if len(image_files) != 11472:
        raise RuntimeError(f"Found {len(image_files)} images; expected exactly 11,472.")

    files = collect_source_files(root)
    total_bytes = sum(item["size_bytes"] for item in files)
    if sum(1 for item in files if item["source_path"].startswith("images/")) != len(image_files):
        raise RuntimeError("Corpus inventory and image directory count disagree.")

    stage = root.parent / (root.name + "_hf_upload_staging")
    shard_counts = Counter(image_destination(path).split("/")[1] for path in image_files)

    print("========== CORPUS AUDIT ==========")
    print("Source:", root)
    print("Images:", len(image_files))
    print("Files:", len(files))
    print("Bytes:", total_bytes)
    print("GiB:", round(total_bytes / (1024 ** 3), 3))
    print("Image shards:", len(shard_counts))
    print("Largest image shard:", max(shard_counts.values(), default=0))
    print("Upload staging:", stage)
    print("Target:", f"https://huggingface.co/datasets/{args.repo_id}")
    print("Visibility:", "public" if args.public else "private")

    if args.dry_run:
        print("DRY RUN: no files changed, no staging created, and no upload attempted.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is unavailable in this Python environment.") from exc

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname") or "authenticated account"
    owner = args.repo_id.split("/", 1)[0]
    organizations = identity.get("orgs", []) or identity.get("auth", {}).get("orgs", [])
    org_names = {item.get("name") for item in organizations if isinstance(item, dict)}
    if owner != username and owner not in org_names:
        raise PermissionError(
            f"Authenticated HF identity {username!r} cannot be verified as owner of {owner!r}."
        )

    marker = make_staging(root, stage, files, args.repo_id, not args.public)
    manifest = json.loads((stage / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    expected_remote_count = int(manifest["file_count"]) + 2
    expected_remote_bytes = int(manifest["total_bytes"]) + sum(
        (stage / name).stat().st_size
        for name in ("ARCHIVE_MANIFEST.json", "README.md")
    )

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    if not args.public:
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        if not getattr(repo_info, "private", False):
            api.update_repo_settings(
                repo_id=args.repo_id,
                repo_type="dataset",
                private=True,
            )
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        if not getattr(repo_info, "private", False):
            raise RuntimeError("Hugging Face repository is not private; refusing to upload.")
    
    try:
        print("[upload] Staging verified. Uploading via resumable upload_folder().", flush=True)
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(stage),
            path_in_repo="full_corpus_2026-11-22",
            commit_message="Archive frozen DocVQA research corpus 2026-11-22",
        )
        print("[upload] Corpus commit:", getattr(result, "oid", str(result)), flush=True)

        api.upload_file(
            path_or_fileobj=stage / "README.md",
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add DocVQA research corpus dataset card",
        )

        remote = list(api.list_repo_tree(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo="full_corpus_2026-11-22",
            recursive=True,
            expand=False,
        ))
        remote_files = [
            entry for entry in remote if getattr(entry, "size", None) is not None
        ]
        remote_count = len(remote_files)
        remote_bytes = sum(int(entry.size) for entry in remote_files)

        report = {
            "status": (
                "verified"
                if remote_count == expected_remote_count
                and remote_bytes == expected_remote_bytes
                else "verification_mismatch"
            ),
            "repo_id": args.repo_id,
            "visibility": "public" if args.public else "private",
            "source_files": len(files),
            "source_bytes": total_bytes,
            "image_file_count": len(image_files),
            "image_shard_count": len(marker.get("bucket_counts", {})),
            "max_files_in_image_shard": max(
                marker.get("bucket_counts", {}).values(), default=0
            ),
            "json_image_path_replacements": marker.get("path_replacements", 0),
            "expected_remote_files_under_corpus": expected_remote_count,
            "remote_files_under_corpus": remote_count,
            "expected_remote_bytes_under_corpus": expected_remote_bytes,
            "remote_bytes_under_corpus": remote_bytes,
            "source_corpus_unchanged": True,
            "remote_url": (
                f"https://huggingface.co/datasets/{args.repo_id}/tree/main/"
                "full_corpus_2026-11-22"
            ),
        }
        report_path = root.parent / (root.name + "_UPLOAD_VERIFICATION.json")
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        if report["status"] != "verified":
            raise RuntimeError(
                "Remote verification did not match expected file count and byte total."
            )
    except Exception:
        print(
            "Upload did not fully verify. Source corpus remains unchanged. "
            "Re-run the same command to resume; upload_folder skips committed files.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
