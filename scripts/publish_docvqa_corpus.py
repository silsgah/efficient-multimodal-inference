#!/usr/bin/env python3
"""Publish the frozen DocVQA research corpus to a private Hugging Face dataset repo.

The default is private. The script writes an archive manifest, uploads a
dataset card and the corpus under full_corpus_2026-11-22/, then verifies the
remote file count and byte total. It never prints or stores an access token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def collect_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"ARCHIVE_MANIFEST.json", "README.md", "UPLOAD_VERIFICATION.json"}:
            continue
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        stat = path.stat()
        files.append({
            "path": relative,
            "size_bytes": stat.st_size,
            "sha256": sha256(path),
        })
    return files


def dataset_card(repo_id: str, manifest: dict[str, Any]) -> str:
    total_q = 44812
    return f"""---
license: mit
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

This repository contains a reproducible research-prepared copy of the
[PixParse DocVQA single-page questions dataset](https://huggingface.co/datasets/{SOURCE_DATASET})
and its page images/OCR, pinned to source revision {SOURCE_REVISION}.

## Purpose

Prepared for experiments on resource-constrained visual evidence acquisition
for document question answering. It preserves page-level images and OCR and
uses document-grouped partitions to reduce document/image leakage between
policy development and held-out evaluation.

## Corpus snapshot

- Questions: {total_q:,}
- Unique page images: 11,472
- Source train questions: 39,463
- Source validation questions: 5,349
- Official test split: **not loaded**
- Files before archive metadata: {manifest["file_count"]:,}
- Bytes before archive metadata: {manifest["total_bytes"]:,}

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

## Provenance and licensing

The source dataset is {SOURCE_DATASET}, pinned to commit {SOURCE_REVISION}.
Its Hugging Face dataset card declares the MIT license. The DocVQA project
site also directs users to read the original challenge download terms and
conditions. Users must comply with those upstream terms and the conditions
governing the underlying UCSF Industry Documents Library material. This
repository does not supersede or expand any upstream rights.

Citation:

@InProceedings{{docvqa_wacv,
  author    = {{Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, C.V.}},
  title     = {{DocVQA: A Dataset for VQA on Document Images}},
  booktitle = {{WACV}},
  year      = {{2021}},
  pages     = {{2200-2209}}
}}

## Integrity

ARCHIVE_MANIFEST.json records each uploaded corpus file's relative path,
size in bytes, and SHA-256 digest. The manifest excludes itself and this
README to avoid recursive hashing. The upload tool verifies the remote file
count and total corpus bytes after transfer.

## Structure

The archive preserves the original corpus builder output directory structure.
Do not move files within the archive without regenerating the manifest.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo, e.g. username/docvqa-research-corpus")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--public", action="store_true", help="Create/use a public repo. Default is private.")
    parser.add_argument("--dry-run", action="store_true", help="Audit corpus and print planned target without uploading.")
    args = parser.parse_args()

    root = args.corpus_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if "/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/"):
        raise ValueError("--repo-id must be namespace/repository-name.")
    for required in ("pages.jsonl",):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Expected corpus artifact missing: {root / required}")
    image_dir = root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Expected image directory missing: {image_dir}")

    image_count = sum(1 for p in image_dir.iterdir() if p.is_file())
    if image_count < 10000:
        raise RuntimeError(f"Only {image_count} image files found; expected around 11,472. Refusing upload.")

    files = collect_files(root)
    total_bytes = sum(item["size_bytes"] for item in files)
    manifest = {
        "archive_created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "visibility": "public" if args.public else "private",
        "corpus_directory": str(root),
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_test_split_loaded": False,
        "partition_counts": PARTITIONS,
        "image_file_count": image_count,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }

    print("========== CORPUS AUDIT ==========")
    print("Path:", root)
    print("Image files:", image_count)
    print("Files:", len(files))
    print("Bytes:", total_bytes)
    print("GiB:", round(total_bytes / (1024 ** 3), 3))
    print("Target:", f"https://huggingface.co/datasets/{args.repo_id}")
    print("Visibility:", "public" if args.public else "private")

    if args.dry_run:
        print("DRY RUN: no files changed and no upload attempted.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install/upgrade huggingface_hub in the data environment.") from exc

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname") or "authenticated account"
    owner = args.repo_id.split("/", 1)[0]
    organizations = identity.get("auth", {}).get("orgs", [])
    org_names = {
        item.get("name") for item in organizations if isinstance(item, dict)
    }
    if owner != username and owner not in org_names:
        raise PermissionError(
            f"Authenticated HF identity is {username!r}, but target owner is {owner!r}; "
            "use a namespace where this account has write access."
        )

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    manifest_path = root / "ARCHIVE_MANIFEST.json"
    readme_path = root / "README.md"
    if manifest_path.exists() or readme_path.exists():
        raise FileExistsError(
            "README.md or ARCHIVE_MANIFEST.json already exists in corpus root. "
            "Review existing files; refusing to overwrite."
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(dataset_card(args.repo_id, manifest), encoding="utf-8")

    try:
        print("[upload] Uploading corpus folder. Re-run after interruption to resume.", flush=True)
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo="full_corpus_2026-11-22",
            commit_message="Archive frozen DocVQA research corpus 2026-11-22",
        )
        print("[upload] Corpus commit:", getattr(result, "oid", str(result)), flush=True)

        api.upload_file(
            path_or_fileobj=readme_path,
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
        remote_files = [entry for entry in remote if getattr(entry, "size", None) is not None]
        remote_count = len(remote_files)
        remote_bytes = sum(int(entry.size) for entry in remote_files)
        report = {
            "status": "verified" if remote_count == len(files) + 2 and remote_bytes >= total_bytes else "verification_mismatch",
            "repo_id": args.repo_id,
            "visibility": "public" if args.public else "private",
            "local_corpus_files_excluding_archive_metadata": len(files),
            "local_corpus_bytes_excluding_archive_metadata": total_bytes,
            "local_archive_metadata_added": 2,
            "remote_files_under_corpus": remote_count,
            "remote_bytes_under_corpus": remote_bytes,
            "expected_remote_file_count": len(files) + 2,
            "expected_minimum_remote_bytes": total_bytes,
            "remote_url": f"https://huggingface.co/datasets/{args.repo_id}/tree/main/full_corpus_2026-11-22",
        }
        report_path = root / "UPLOAD_VERIFICATION.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if report["status"] != "verified":
            raise RuntimeError("Remote verification did not match expected file count and bytes.")
    except Exception:
        print(
            "Upload did not fully verify. The source corpus and its files are preserved. "
            "Re-run the same command after resolving the issue; upload_folder is resumable.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
