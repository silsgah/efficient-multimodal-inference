"""Build full DocVQA training/validation manifests with document-level isolation.

This builder uses the pinned Hub Parquet shards directly via PyArrow. It does
not use datasets.load_dataset(), does not load the official test split, reads
Parquet in small batches, stores each canonical page image once, and retains
all accepted answers plus page-level OCR/layout data.

Partition rules:
- Official train is grouped by ucsf_document_id (and merged when identical
  canonical image bytes occur under different document IDs), then split by
  document group into policy_train and policy_validation.
- Every official validation document group touched by the 50 historical
  question IDs becomes historical_development.
- Official validation groups overlapping official train by document ID or
  page-image hash are excluded from locked evaluation.
- Historical document groups are excluded from policy training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any


DATASET_ID = "pixparse/docvqa-single-page-questions"
EXPECTED_COUNTS = {"train": 39463, "validation": 5349}
SOURCE_PREFIXES = {"train": "data/train-", "validation": "data/validation-"}
PARQUET_COLUMNS = [
    "image", "question_id", "question", "answers", "data_split",
    "ocr_results", "other_metadata",
]


class UnionFind:
    """Disjoint-set union for source documents linked by identical page bytes."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

    def stable_groups(self) -> dict[str, str]:
        members_by_root: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            members_by_root[self.find(item)].append(item)
        result: dict[str, str] = {}
        for members in members_by_root.values():
            canonical = "\n".join(sorted(members)).encode("utf-8")
            group_id = "docgrp-" + hashlib.sha256(canonical).hexdigest()[:20]
            for member in members:
                result[member] = group_id
        return result


def document_key(metadata: dict[str, Any]) -> str:
    value = metadata.get("ucsf_document_id")
    if value is None or not str(value).strip():
        raise ValueError(
            "Missing other_metadata.ucsf_document_id. Refusing to use page-level "
            "doc_id as a substitute for source-document identity."
        )
    return "ucsf:" + str(value).strip()


def policy_partition(group_id: str, validation_fraction: float, seed: int) -> str:
    """Deterministically assign whole document groups without random state."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    value = int(
        hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(), 16
    ) / float(1 << 256)
    return "policy_validation" if value < validation_fraction else "policy_train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def canonical_png(raw_bytes: bytes) -> bytes:
    """Match prepare_docvqa.py: decode, convert RGB, and encode PNG."""
    from PIL import Image

    with Image.open(BytesIO(raw_bytes)) as image:
        image.load()
        converted = image.convert("RGB")
        buffer = BytesIO()
        converted.save(buffer, format="PNG")
        return buffer.getvalue()


def get_raw_image_bytes(image_record: Any, repo_root: Path) -> bytes:
    if not isinstance(image_record, dict):
        raise TypeError(
            f"Parquet image cell must be a struct, got {type(image_record).__name__}"
        )
    raw = image_record.get("bytes")
    if raw:
        return bytes(raw)
    path_value = image_record.get("path")
    if path_value:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            return path.read_bytes()
        raise FileNotFoundError(
            f"Parquet image struct has no embedded bytes and path is unavailable: {path}"
        )
    raise ValueError("Parquet image cell has neither bytes nor a usable path.")


def write_jsonl_line(file, record: dict[str, Any]) -> None:
    file.write(json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) + "\n")


def read_historical_inputs(
    csv_path: Path, pilot_manifest: Path, repo_root: Path
) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "request_id" not in (reader.fieldnames or []):
            raise ValueError("Historical CSV must contain request_id.")
        for row in reader:
            request_id = str(row.get("request_id", "")).strip()
            if request_id:
                ids.add(request_id)
    if not ids:
        raise ValueError("No historical request IDs found.")

    image_hashes: dict[str, str] = {}
    with pilot_manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in ids:
                continue
            image_value = Path(str(row.get("image", "")))
            candidates = [
                image_value if image_value.is_absolute() else repo_root / image_value,
                image_value if image_value.is_absolute()
                else pilot_manifest.parent / image_value,
            ]
            image_path = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                None,
            )
            if image_path is None:
                raise FileNotFoundError(
                    f"Cannot resolve pilot image for historical question {sample_id} "
                    f"(manifest line {line_number})."
                )
            image_hashes[sample_id] = sha256_file(image_path)

    missing = ids - set(image_hashes)
    if missing:
        raise ValueError(
            "Historical question IDs absent from pilot manifest: "
            + ", ".join(sorted(missing)[:20])
        )
    return ids, image_hashes


def ensure_new_build_paths(output_dir: Path) -> Path:
    building_dir = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed corpus directory: {output_dir}"
        )
    if building_dir.exists():
        raise FileExistsError(
            f"An incomplete build exists at {building_dir}. Inspect it before "
            "removing it or choose a new --output-dir."
        )
    building_dir.mkdir(parents=True, exist_ok=False)
    return building_dir


def write_page_image(
    image_dir: Path,
    image_hash: str,
    png_bytes: bytes,
) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    name = image_hash + ".png"
    destination = image_dir / name
    if destination.exists():
        if sha256_file(destination) != image_hash:
            raise RuntimeError(f"Existing image cache file hash mismatch: {destination}")
        return "images/" + name

    temporary = destination.with_suffix(".png.tmp")
    with temporary.open("wb") as file:
        file.write(png_bytes)
        file.flush()
        os.fsync(file.fileno())
    if sha256_file(temporary) != image_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Image failed post-write SHA-256 verification: {temporary}")
    temporary.replace(destination)
    return "images/" + name


def process_split(
    split: str,
    shard_names: list[str],
    *,
    dataset_id: str,
    revision: str,
    cache_dir: Path,
    repo_root: Path,
    building_dir: Path,
    union_find: UnionFind,
    page_owner: dict[str, str],
    page_meta: dict[str, dict[str, Any]],
    raw_to_canonical: dict[str, tuple[str, str]],
    seen_question_ids: set[str],
    historical_hashes: dict[str, str],
    historical_ids: set[str],
    shard_records: list[dict[str, Any]],
) -> int:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the isolated data dependencies with "
            "pip install -r requirements/data.txt"
        ) from exc

    staging_dir = building_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{split}.jsonl"
    image_dir = building_dir / "images"
    question_count = 0

    with staging_path.open("w", encoding="utf-8") as questions_file, \
         (staging_dir / "pages.jsonl").open("a", encoding="utf-8") as pages_file:

        for shard_index, filename in enumerate(shard_names, start=1):
            print(
                f"[{split}] shard {shard_index}/{len(shard_names)}: {filename}",
                flush=True,
            )
            local_path = Path(hf_hub_download(
                repo_id=dataset_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir),
            ))
            shard_rows = 0
            parquet_file = pq.ParquetFile(local_path)
            available = set(parquet_file.schema_arrow.names)
            missing_columns = set(PARQUET_COLUMNS) - available
            if missing_columns:
                raise ValueError(
                    f"{filename} missing columns: {sorted(missing_columns)}"
                )

            for batch in parquet_file.iter_batches(
                batch_size=2,
                columns=PARQUET_COLUMNS,
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    question_count += 1
                    shard_rows += 1
                    question_id_value = row.get("question_id")
                    if question_id_value is None:
                        raise ValueError(f"{filename}: missing question_id")
                    sample_id = str(question_id_value)
                    if sample_id in seen_question_ids:
                        raise ValueError(
                            f"Duplicate question_id across source data: {sample_id}"
                        )
                    seen_question_ids.add(sample_id)

                    question = row.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"{split}/{sample_id}: invalid question text")
                    answers = row.get("answers")
                    if not isinstance(answers, list) or not answers:
                        raise ValueError(f"{split}/{sample_id}: missing answer list")
                    answers = [str(answer) for answer in answers]
                    if any(answer == "" for answer in answers):
                        raise ValueError(f"{split}/{sample_id}: empty reference answer")

                    metadata = row.get("other_metadata")
                    if not isinstance(metadata, dict):
                        raise ValueError(f"{split}/{sample_id}: invalid other_metadata")
                    doc_key = document_key(metadata)
                    page_no = metadata.get("ucsf_document_page_no")
                    if page_no is None:
                        raise ValueError(f"{split}/{sample_id}: missing page number")
                    union_find.add(doc_key)

                    raw_bytes = get_raw_image_bytes(row.get("image"), repo_root)
                    raw_hash = sha256_bytes(raw_bytes)
                    cached = raw_to_canonical.get(raw_hash)
                    if cached is None:
                        png_bytes = canonical_png(raw_bytes)
                        image_hash = sha256_bytes(png_bytes)
                        image_relpath = write_page_image(
                            image_dir, image_hash, png_bytes
                        )
                        raw_to_canonical[raw_hash] = (image_hash, image_relpath)
                    else:
                        image_hash, image_relpath = cached

                    if split == "validation" and sample_id in historical_ids:
                        expected = historical_hashes[sample_id]
                        if image_hash != expected:
                            raise RuntimeError(
                                f"Historical page hash differs for question {sample_id}: "
                                f"pilot={expected}, upstream={image_hash}. "
                                "Check dataset revision or image canonicalization."
                            )

                    prior_owner = page_owner.get(image_hash)
                    if prior_owner is None:
                        page_owner[image_hash] = doc_key
                    elif prior_owner != doc_key:
                        union_find.union(prior_owner, doc_key)

                    ocr_results = row.get("ocr_results")
                    if not isinstance(ocr_results, dict):
                        raise ValueError(f"{split}/{sample_id}: invalid OCR structure")
                    ocr_hash = json_fingerprint(ocr_results)
                    page_entry = page_meta.get(image_hash)
                    if page_entry is None:
                        page_meta[image_hash] = {
                            "document_keys": {doc_key},
                            "source_splits": {split},
                            "ocr_sha256": ocr_hash,
                        }
                        write_jsonl_line(pages_file, {
                            "image_sha256": image_hash,
                            "image": image_relpath,
                            "document_keys": [doc_key],
                            "source_splits": [split],
                            "ocr_results": ocr_results,
                            "other_metadata": metadata,
                        })
                    else:
                        if page_entry["ocr_sha256"] != ocr_hash:
                            raise RuntimeError(
                                f"OCR differs for identical canonical page hash {image_hash}."
                            )
                        page_entry["document_keys"].add(doc_key)
                        page_entry["source_splits"].add(split)

                    write_jsonl_line(questions_file, {
                        "sample_id": sample_id,
                        "question_id": question_id_value,
                        "question": question,
                        "answer": answers[0],
                        "answers": answers,
                        "image": image_relpath,
                        "image_sha256": image_hash,
                        "document_key": doc_key,
                        "ucsf_document_id": str(metadata["ucsf_document_id"]),
                        "ucsf_document_page_no": str(page_no),
                        "doc_id": metadata.get("doc_id"),
                        "source_split": split,
                        "data_split": row.get("data_split"),
                    })

            del parquet_file
            shard_records.append({
                "source_split": split,
                "filename": filename,
                "rows_read": shard_rows,
                "file_bytes": local_path.stat().st_size,
                "file_sha256": sha256_file(local_path),
            })
            print(
                f"[{split}] completed {filename}: {shard_rows} rows "
                f"(total {question_count})",
                flush=True,
            )

    if question_count != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[split]} {split} questions, read {question_count}."
        )
    return question_count


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc


def finalize_partitions(
    building_dir: Path,
    train_count: int,
    validation_count: int,
    document_groups: dict[str, str],
    page_meta: dict[str, dict[str, Any]],
    historical_ids: set[str],
    historical_hashes: dict[str, str],
    validation_fraction: float,
    seed: int,
    shard_records: list[dict[str, Any]],
    manifest_path: Path,
    historical_csv: Path,
    revision: str,
    dataset_id: str = DATASET_ID,
) -> dict[str, Any]:
    staging = building_dir / ".staging"
    train_path = staging / "train.jsonl"
    validation_path = staging / "validation.jsonl"

    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    historical_groups: set[str] = set()
    historical_seen: set[str] = set()

    # First pass: calculate whole-document group membership and historical IDs.
    for row in iter_jsonl(train_path):
        train_groups.add(document_groups[row["document_key"]])
    for row in iter_jsonl(validation_path):
        group_id = document_groups[row["document_key"]]
        validation_groups.add(group_id)
        if row["sample_id"] in historical_ids:
            historical_groups.add(group_id)
            historical_seen.add(row["sample_id"])

    missing_historical = historical_ids - historical_seen
    if missing_historical:
        raise ValueError(
            "Historical question IDs not found in official validation: "
            + ", ".join(sorted(missing_historical)[:20])
        )
    if len(historical_seen) != len(historical_ids):
        raise RuntimeError("Historical validation ID accounting mismatch.")

    training_excluded_groups = train_groups & historical_groups
    validation_cross_overlap_groups = (
        validation_groups & train_groups
    ) - historical_groups

    policy_groups = train_groups - historical_groups
    assignments = {
        group_id: policy_partition(group_id, validation_fraction, seed)
        for group_id in policy_groups
    }

    partition_names = [
        "policy_train",
        "policy_validation",
        "historical_development",
        "locked_evaluation",
        "cross_split_validation_excluded",
        "historical_overlap_train_excluded",
    ]
    output_handles = {
        name: (building_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in partition_names
    }
    counts: Counter[str] = Counter()
    group_sets: dict[str, set[str]] = {name: set() for name in partition_names}
    page_sets: dict[str, set[str]] = {name: set() for name in partition_names}
    document_question_counts: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        for source_name, source_path in (
            ("train", train_path),
            ("validation", validation_path),
        ):
            for row in iter_jsonl(source_path):
                group_id = document_groups[row["document_key"]]
                row["document_group_id"] = group_id
                document_question_counts[group_id][source_name] += 1

                if source_name == "train":
                    if group_id in historical_groups:
                        destination = "historical_overlap_train_excluded"
                    elif assignments[group_id] == "policy_train":
                        destination = "policy_train"
                    else:
                        destination = "policy_validation"
                else:
                    if group_id in historical_groups:
                        destination = "historical_development"
                    elif group_id in validation_cross_overlap_groups:
                        destination = "cross_split_validation_excluded"
                    else:
                        destination = "locked_evaluation"

                write_jsonl_line(output_handles[destination], row)
                counts[destination] += 1
                group_sets[destination].add(group_id)
                page_sets[destination].add(row["image_sha256"])
    finally:
        for handle in output_handles.values():
            handle.close()

    # Verify no document/page appears in both policy partitions or locked eval.
    protected_pairs = [
        ("policy_train", "policy_validation"),
        ("policy_train", "locked_evaluation"),
        ("policy_validation", "locked_evaluation"),
        ("historical_development", "policy_train"),
        ("historical_development", "policy_validation"),
    ]
    overlap_report = {}
    for first, second in protected_pairs:
        group_overlap = group_sets[first] & group_sets[second]
        page_overlap = page_sets[first] & page_sets[second]
        overlap_report[f"{first}__{second}"] = {
            "shared_document_groups": len(group_overlap),
            "shared_image_hashes": len(page_overlap),
        }
        if group_overlap or page_overlap:
            raise RuntimeError(
                f"Leakage between {first} and {second}: "
                f"{len(group_overlap)} groups, {len(page_overlap)} page hashes"
            )

    # Write enriched page records, preserving OCR once per unique page.
    pages_staging = staging / "pages.jsonl"
    pages_output = building_dir / "pages.jsonl"
    page_count = 0
    with pages_output.open("w", encoding="utf-8") as output_file:
        for row in iter_jsonl(pages_staging):
            hashes = row["image_sha256"]
            meta = page_meta[hashes]
            doc_keys = sorted(meta["document_keys"])
            group_ids = sorted({document_groups[key] for key in doc_keys})
            if len(group_ids) != 1:
                raise RuntimeError(
                    f"Identical page {hashes} did not resolve to one document group."
                )
            row["document_keys"] = doc_keys
            row["document_group_id"] = group_ids[0]
            row["source_splits"] = sorted(meta["source_splits"])
            write_jsonl_line(output_file, row)
            page_count += 1

    # Document catalog records pages/questions by source split and final partition.
    pages_per_group: dict[str, set[str]] = defaultdict(set)
    source_keys_per_group: dict[str, set[str]] = defaultdict(set)
    for image_hash, meta in page_meta.items():
        for doc_key in meta["document_keys"]:
            group_id = document_groups[doc_key]
            pages_per_group[group_id].add(image_hash)
            source_keys_per_group[group_id].add(doc_key)

    docs_output = building_dir / "documents.jsonl"
    with docs_output.open("w", encoding="utf-8") as file:
        for group_id in sorted(document_question_counts):
            # A source document may appear in both an exclusion partition and
            # its retained partition (e.g., train kept for policy learning,
            # validation copy excluded; historical validation retained while
            # any train copy is excluded). Preserve every membership instead
            # of overwriting it with a misleading single partition label.
            memberships = [
                name for name in partition_names if group_id in group_sets[name]
            ]
            write_jsonl_line(file, {
                "document_group_id": group_id,
                "source_document_keys": sorted(source_keys_per_group[group_id]),
                "unique_page_count": len(pages_per_group[group_id]),
                "question_counts_by_official_split": dict(
                    document_question_counts[group_id]
                ),
                "partition_memberships": memberships,
            })

    # Hash final manifests/artifacts incrementally.
    artifact_names = [f"{name}.jsonl" for name in partition_names] + [
        "pages.jsonl", "documents.jsonl"
    ]
    artifact_hashes = {
        name: sha256_file(building_dir / name) for name in artifact_names
    }

    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "official_test_split_loaded": False,
        "source_question_counts": {
            "train": train_count,
            "validation": validation_count,
        },
        "historical_question_ids": len(historical_ids),
        "historical_image_hash_checks": len(historical_hashes),
        "historical_validation_document_groups": len(historical_groups),
        "train_document_groups": len(train_groups),
        "validation_document_groups": len(validation_groups),
        "policy_document_groups": len(policy_groups),
        "policy_validation_fraction_requested": validation_fraction,
        "seed": seed,
        "partition_question_counts": dict(counts),
        "partition_document_group_counts": {
            name: len(group_sets[name]) for name in partition_names
        },
        "partition_unique_page_counts": {
            name: len(page_sets[name]) for name in partition_names
        },
        "unique_pages": page_count,
        "train_groups_excluded_for_historical_overlap": len(training_excluded_groups),
        "validation_groups_excluded_for_train_overlap": len(
            validation_cross_overlap_groups
        ),
        "leakage_checks": overlap_report,
        "source_shards": shard_records,
        "input_sha256": {
            "historical_pilot_manifest": sha256_file(manifest_path),
            "historical_request_csv": sha256_file(historical_csv),
        },
        "artifact_sha256": artifact_hashes,
        "notes": [
            "All accepted reference answers are preserved in the answers list.",
            "Page OCR/layout is stored once per canonical image in pages.jsonl.",
            "Images are canonicalized as RGB PNG using the same process as the historical pilot preparation.",
            "Document groups use ucsf_document_id; identical canonical page hashes across document IDs merge groups.",
            "Historical validation documents are isolated from policy training and locked evaluation.",
            "Validation documents that overlap official train are excluded from locked evaluation.",
            "The official test split is not loaded and is left untouched.",
            "The original 200-question pilot manifest, images, historical CSV, and frozen pilot split are read-only inputs.",
        ],
    }
    (building_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face dataset commit SHA.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/hf-cache"))
    parser.add_argument("--historical-csv", type=Path,
                        default=Path("publication/energy_measurement_2026-11-18/energy_requests.csv"))
    parser.add_argument("--pilot-manifest", type=Path,
                        default=Path("data/docvqa/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/docvqa/full_corpus_2026-11-21"))
    parser.add_argument("--policy-validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.policy_validation_fraction < 1.0:
        raise ValueError("--policy-validation-fraction must be strictly between 0 and 1.")

    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    cache_dir = resolve(args.cache_dir)
    historical_csv = resolve(args.historical_csv)
    pilot_manifest = resolve(args.pilot_manifest)
    output_dir = resolve(args.output_dir)

    if not historical_csv.is_file():
        raise FileNotFoundError(historical_csv)
    if not pilot_manifest.is_file():
        raise FileNotFoundError(pilot_manifest)

    historical_ids, historical_hashes = read_historical_inputs(
        historical_csv, pilot_manifest, repo_root
    )
    print(
        f"Historical questions: {len(historical_ids)}; "
        f"verified pilot page hashes: {len(historical_hashes)}",
        flush=True,
    )

    building_dir = ensure_new_build_paths(output_dir)
    staging_dir = building_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    image_dir = building_dir / "images"

    try:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "Install isolated data dependencies using requirements/data.txt."
            ) from exc

        api = HfApi()
        files = api.list_repo_files(
            args.dataset_id, repo_type="dataset", revision=args.revision
        )
        source_shards = {
            split: sorted(
                name for name in files
                if name.startswith(prefix) and name.endswith(".parquet")
            )
            for split, prefix in SOURCE_PREFIXES.items()
        }
        for split, shards in source_shards.items():
            if not shards:
                raise FileNotFoundError(f"No {split} Parquet shards at revision {args.revision}")

        print(json.dumps({
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "train_shards": len(source_shards["train"]),
            "validation_shards": len(source_shards["validation"]),
            "test_shards": "not loaded",
            "output_dir": str(output_dir),
        }, indent=2), flush=True)

        union_find = UnionFind()
        page_owner: dict[str, str] = {}
        page_meta: dict[str, dict[str, Any]] = {}
        raw_to_canonical: dict[str, tuple[str, str]] = {}
        seen_question_ids: set[str] = set()
        shard_records: list[dict[str, Any]] = []

        train_count = process_split(
            "train", source_shards["train"],
            dataset_id=args.dataset_id, revision=args.revision,
            cache_dir=cache_dir, repo_root=repo_root, building_dir=building_dir,
            union_find=union_find, page_owner=page_owner, page_meta=page_meta,
            raw_to_canonical=raw_to_canonical, seen_question_ids=seen_question_ids,
            historical_hashes=historical_hashes, historical_ids=historical_ids,
            shard_records=shard_records,
        )
        validation_count = process_split(
            "validation", source_shards["validation"],
            dataset_id=args.dataset_id, revision=args.revision,
            cache_dir=cache_dir, repo_root=repo_root, building_dir=building_dir,
            union_find=union_find, page_owner=page_owner, page_meta=page_meta,
            raw_to_canonical=raw_to_canonical, seen_question_ids=seen_question_ids,
            historical_hashes=historical_hashes, historical_ids=historical_ids,
            shard_records=shard_records,
        )

        document_groups = union_find.stable_groups()
        summary = finalize_partitions(
            building_dir,
            train_count,
            validation_count,
            document_groups,
            page_meta,
            historical_ids,
            historical_hashes,
            args.policy_validation_fraction,
            args.seed,
            shard_records,
            pilot_manifest,
            historical_csv,
            args.revision,
        )

        # Remove internal staging files only after all integrity checks pass.
        import shutil
        shutil.rmtree(staging_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        building_dir.rename(output_dir)

        print(json.dumps({
            "status": "complete",
            "output_dir": str(output_dir),
            "source_question_counts": summary["source_question_counts"],
            "partition_question_counts": summary["partition_question_counts"],
            "partition_document_group_counts": summary["partition_document_group_counts"],
            "partition_unique_page_counts": summary["partition_unique_page_counts"],
            "unique_pages": summary["unique_pages"],
            "train_groups_excluded_for_historical_overlap": summary["train_groups_excluded_for_historical_overlap"],
            "validation_groups_excluded_for_train_overlap": summary["validation_groups_excluded_for_train_overlap"],
            "leakage_checks": summary["leakage_checks"],
            "artifact_sha256": summary["artifact_sha256"],
        }, indent=2), flush=True)
    except Exception:
        print(
            f"BUILD FAILED. Partial files are preserved for diagnosis at {building_dir}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
