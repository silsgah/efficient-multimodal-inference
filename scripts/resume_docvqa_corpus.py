#!/usr/bin/env python3
"""Repair the interrupted DocVQA train shard and resume the corpus build.

This script is intentionally conservative:
- preserves the original .building directory and source manifests;
- repairs the known missing suffix of train-00020 using the pinned cached shard;
- verifies all completed train prefix question IDs against the source Parquet shards;
- processes remaining train and all validation shards in a separate delta directory;
- flushes/fsyncs JSONL outputs periodically and at every completed shard;
- merges into the original staging tree only after the delta build succeeds;
- runs the original partition/leakage finalization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_ID = "pixparse/docvqa-single-page-questions"
REVISION = "13619c00b32b50729b6e50b4bb0aa4b355698f3f"
EXPECTED_TRAIN = 39463
EXPECTED_VALIDATION = 5349
COMPLETED_TRAIN_SHARDS = 21


def log(message: str) -> None:
    print(message, flush=True)


def write_jsonl(file, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def fsync_file(path: Path) -> None:
    with path.open("ab") as f:
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}, line {n}") from exc


def atomic_copy_plus_rows(source: Path, target_tmp: Path, additions: list[dict[str, Any]]) -> None:
    with source.open("r", encoding="utf-8") as src, target_tmp.open("w", encoding="utf-8") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        for row in additions:
            write_jsonl(dst, row)
        dst.flush()
        os.fsync(dst.fileno())


def atomic_copy_plus_file(source: Path, append_source: Path, target_tmp: Path) -> None:
    with source.open("rb") as src, target_tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        with append_source.open("rb") as delta:
            shutil.copyfileobj(delta, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def validate_jsonl_unique(path: Path, key: str) -> tuple[int, set[str]]:
    seen: set[str] = set()
    count = 0
    for row in iter_jsonl(path):
        value = str(row[key])
        if value in seen:
            raise RuntimeError(f"Duplicate {key}={value} in {path}")
        seen.add(value)
        count += 1
    return count, seen


def load_builder(source_path: Path, runtime_path: Path):
    text = source_path.read_text(encoding="utf-8")
    needle = "            metadata_stage.unlink()\n            shard_records.append({"
    durable = (
        "            # Make every completed shard durable before recording completion.\n"
        "            questions_file.flush()\n"
        "            os.fsync(questions_file.fileno())\n"
        "            pages_file.flush()\n"
        "            os.fsync(pages_file.fileno())\n"
        "            metadata_stage.unlink()\n"
        "            shard_records.append({"
    )
    if durable not in text:
        if needle not in text:
            raise RuntimeError("Could not locate the process_split shard-completion point to add durable flushes.")
        text = text.replace(needle, durable, 1)
    runtime_path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("docvqa_resume_builder", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load runtime builder module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Periodic durability during a shard, plus the explicit fsync at shard end.
    original_write = module.write_jsonl_line
    counters: dict[int, int] = defaultdict(int)

    def durable_write(file, record):
        original_write(file, record)
        key = id(file)
        counters[key] += 1
        if counters[key] % 250 == 0:
            file.flush()
            os.fsync(file.fileno())

    module.write_jsonl_line = durable_write
    return module


def fetch_parquet(hf_hub_download, filename: str, cache_dir: Path) -> Path:
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        revision=REVISION,
        cache_dir=str(cache_dir),
        local_files_only=filename.startswith("data/train-0000") and int(filename.split("train-")[1].split("-")[0]) < COMPLETED_TRAIN_SHARDS,
    )
    return Path(path)


def source_question_ids(pq, path: Path) -> list[str]:
    return [str(x) for x in pq.read_table(path, columns=["question_id"])["question_id"].to_pylist()]


def repair_interrupted_shard(builder, pq, hf_hub_download, repo_root: Path, cache_dir: Path, building: Path) -> int:
    staging = building / ".staging"
    train_path = staging / "train.jsonl"
    pages_path = staging / "pages.jsonl"
    image_dir = building / "images"
    shard_name = "data/train-00020-of-00036.parquet"
    shard_path = fetch_parquet(hf_hub_download, shard_name, cache_dir)

    log("[repair] validating the complete 21-shard prefix against the pinned source")
    source_prefix_ids: list[str] = []
    train_shard_names = [f"data/train-{i:05d}-of-00036.parquet" for i in range(36)]
    for name in train_shard_names[:COMPLETED_TRAIN_SHARDS]:
        p = fetch_parquet(hf_hub_download, name, cache_dir)
        ids = source_question_ids(pq, p)
        source_prefix_ids.extend(ids)

    existing_rows = list(iter_jsonl(train_path))
    existing_ids = [str(row["sample_id"]) for row in existing_rows]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("Existing train staging contains duplicate sample IDs.")
    if len(existing_ids) > len(source_prefix_ids) or existing_ids != source_prefix_ids[:len(existing_ids)]:
        raise RuntimeError("Existing train staging is not an exact source-order prefix of the first 21 train shards.")

    source_ids_last = source_question_ids(pq, shard_path)
    prior_count = len(source_prefix_ids) - len(source_ids_last)
    if len(existing_ids) < prior_count:
        raise RuntimeError("Staging is missing records from a shard earlier than the known interrupted shard.")
    last_existing_ids = set(existing_ids[prior_count:])
    if existing_ids[:prior_count] != source_prefix_ids[:prior_count]:
        raise RuntimeError("Earlier completed train shards do not match the pinned source.")
    present_positions = [i for i, qid in enumerate(source_ids_last) if qid in last_existing_ids]
    if present_positions != list(range(len(present_positions))):
        raise RuntimeError("Persisted records from the interrupted shard are not a contiguous prefix.")
    missing_ids = source_ids_last[len(present_positions):]
    if len(present_positions) != 976 or len(missing_ids) != 43:
        raise RuntimeError(
            f"Expected the diagnosed 976+43 suffix pattern; found {len(present_positions)} present and {len(missing_ids)} missing. Refusing automatic repair."
        )
    if len(existing_ids) != 23429:
        raise RuntimeError(f"Expected 23,429 persisted questions before repair, found {len(existing_ids)}.")
    log(f"[repair] verified {len(existing_ids)} persisted questions; restoring {len(missing_ids)} missing suffix rows")

    page_rows_existing = list(iter_jsonl(pages_path))
    page_by_hash: dict[str, dict[str, Any]] = {}
    for row in page_rows_existing:
        image_hash = row["image_sha256"]
        if image_hash in page_by_hash:
            raise RuntimeError(f"Duplicate page hash already present: {image_hash}")
        page_by_hash[image_hash] = row

    missing_set = set(missing_ids)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=1, columns=builder.QUESTION_OCR_COLUMNS, use_threads=False):
        row = batch.to_pylist()[0]
        qid = str(row.get("question_id"))
        if qid not in missing_set:
            continue
        metadata = row.get("other_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid metadata for missing question {qid}")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers:
            raise RuntimeError(f"Invalid answers for missing question {qid}")
        ocr = row.get("ocr_results")
        if not isinstance(ocr, dict):
            raise RuntimeError(f"Invalid OCR for missing question {qid}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"Invalid question text for missing question {qid}")
        doc_key = builder.document_key(metadata)
        page_no = metadata.get("ucsf_document_page_no")
        if page_no is None:
            raise RuntimeError(f"Missing page number for question {qid}")
        metadata_by_id[qid] = {
            "sample_id": qid,
            "question_id": row["question_id"],
            "question": question,
            "answers": [str(a) for a in answers],
            "data_split": row.get("data_split"),
            "document_key": doc_key,
            "ucsf_document_id": str(metadata["ucsf_document_id"]),
            "ucsf_document_page_no": str(page_no),
            "doc_id": metadata.get("doc_id"),
            "other_metadata": metadata,
            "ocr_results": ocr,
        }
    if set(metadata_by_id) != missing_set:
        raise RuntimeError("Could not recover all missing metadata rows from the pinned shard.")

    image_by_id: dict[str, dict[str, str]] = {}
    for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=1, columns=builder.IMAGE_COLUMNS, use_threads=False):
        row = batch.to_pylist()[0]
        qid = str(row.get("question_id"))
        if qid not in missing_set:
            continue
        raw = builder.get_raw_image_bytes(row.get("image"), repo_root)
        png, pixel_hash = builder.canonical_image(raw)
        image_hash = builder.sha256_bytes(png)
        image_rel = "images/" + image_hash + ".png"
        image_path = image_dir / (image_hash + ".png")
        if not image_path.is_file():
            builder.write_page_image(image_dir, image_hash, png)
        elif sha256_file(image_path) != image_hash:
            raise RuntimeError(f"Existing canonical image failed hash verification: {image_path}")
        image_by_id[qid] = {
            "image_sha256": image_hash,
            "image_pixel_sha256": pixel_hash,
            "image": image_rel,
        }
    if set(image_by_id) != missing_set:
        raise RuntimeError("Could not recover all missing image records from the pinned shard.")

    question_additions: list[dict[str, Any]] = []
    new_page_rows: dict[str, dict[str, Any]] = {}
    for qid in missing_ids:
        meta = metadata_by_id[qid]
        image = image_by_id[qid]
        image_hash = image["image_sha256"]
        ocr = meta["ocr_results"]
        ocr_hash = builder.json_fingerprint(ocr)
        known_page = page_by_hash.get(image_hash)
        if known_page is not None:
            if known_page["image_pixel_sha256"] != image["image_pixel_sha256"]:
                raise RuntimeError(f"Pixel hash mismatch for already-staged page {image_hash}")
            if builder.json_fingerprint(known_page["ocr_results"]) != ocr_hash:
                raise RuntimeError(f"OCR hash mismatch for already-staged page {image_hash}")
        else:
            pending = new_page_rows.get(image_hash)
            if pending is None:
                new_page_rows[image_hash] = {
                    "image_sha256": image_hash,
                    "image_pixel_sha256": image["image_pixel_sha256"],
                    "image": image["image"],
                    "document_keys": [meta["document_key"]],
                    "source_splits": ["train"],
                    "ocr_results": ocr,
                    "other_metadata": meta["other_metadata"],
                }
            else:
                if pending["image_pixel_sha256"] != image["image_pixel_sha256"] or builder.json_fingerprint(pending["ocr_results"]) != ocr_hash:
                    raise RuntimeError(f"Duplicate page image has inconsistent OCR/pixels: {image_hash}")
                if meta["document_key"] not in pending["document_keys"]:
                    pending["document_keys"].append(meta["document_key"])
        question_additions.append({
            "sample_id": qid,
            "question_id": meta["question_id"],
            "question": meta["question"],
            "answer": meta["answers"][0],
            "answers": meta["answers"],
            "image": image["image"],
            "image_sha256": image_hash,
            "image_pixel_sha256": image["image_pixel_sha256"],
            "document_key": meta["document_key"],
            "ucsf_document_id": meta["ucsf_document_id"],
            "ucsf_document_page_no": meta["ucsf_document_page_no"],
            "doc_id": meta["doc_id"],
            "source_split": "train",
            "data_split": meta["data_split"],
        })

    if [r["sample_id"] for r in question_additions] != missing_ids:
        raise RuntimeError("Recovered question rows are not in source order.")

    train_tmp = staging / "train.jsonl.recovery.tmp"
    pages_tmp = staging / "pages.jsonl.recovery.tmp"
    if train_tmp.exists() or pages_tmp.exists():
        raise RuntimeError("A previous recovery temporary file exists; inspect it before proceeding.")
    atomic_copy_plus_rows(train_path, train_tmp, question_additions)
    atomic_copy_plus_rows(pages_path, pages_tmp, list(new_page_rows.values()))

    repaired_count, repaired_ids = validate_jsonl_unique(train_tmp, "sample_id")
    repaired_page_count, repaired_page_hashes = validate_jsonl_unique(pages_tmp, "image_sha256")
    if repaired_count != len(source_prefix_ids):
        raise RuntimeError(f"Repaired train row count {repaired_count} != expected prefix count {len(source_prefix_ids)}")
    if repaired_ids != set(source_prefix_ids):
        raise RuntimeError("Repaired train manifest IDs do not exactly match the first 21 pinned source shards.")
    question_hashes = {row["image_sha256"] for row in iter_jsonl(train_tmp)}
    if question_hashes != repaired_page_hashes:
        raise RuntimeError(
            f"Page/question image hash mismatch after repair: question-only={len(question_hashes-repaired_page_hashes)}, page-only={len(repaired_page_hashes-question_hashes)}"
        )
    log(f"[repair] prepared {repaired_count} questions and {repaired_page_count} page records; {len(new_page_rows)} new OCR page records")

    train_backup = staging / "train.jsonl.before_shard21_repair"
    pages_backup = staging / "pages.jsonl.before_shard21_repair"
    if train_backup.exists() or pages_backup.exists():
        raise RuntimeError("Recovery backup already exists; refusing to overwrite it.")
    shutil.copy2(train_path, train_backup)
    shutil.copy2(pages_path, pages_backup)
    os.replace(train_tmp, train_path)
    os.replace(pages_tmp, pages_path)
    fsync_file(train_path)
    fsync_file(pages_path)
    log("[repair] committed recovered manifests; original pre-repair backups retained")
    return len(question_additions)


def reconstruct_state(builder, staging: Path):
    pages_path = staging / "pages.jsonl"
    train_path = staging / "train.jsonl"
    page_meta: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(pages_path):
        image_hash = row["image_sha256"]
        if image_hash in page_meta:
            raise RuntimeError(f"Duplicate page staging row {image_hash}")
        page_meta[image_hash] = {
            "document_keys": set(str(x) for x in row.get("document_keys", [])),
            "source_splits": set(str(x) for x in row.get("source_splits", [])),
            "ocr_sha256": builder.json_fingerprint(row["ocr_results"]),
            "pixel_sha256": row["image_pixel_sha256"],
        }
    union_find = builder.UnionFind()
    page_owner: dict[str, str] = {}
    seen_ids: set[str] = set()
    question_hashes: set[str] = set()
    for row in iter_jsonl(train_path):
        qid = str(row["sample_id"])
        if qid in seen_ids:
            raise RuntimeError(f"Duplicate staged train question ID {qid}")
        seen_ids.add(qid)
        image_hash = row["image_sha256"]
        question_hashes.add(image_hash)
        page = page_meta.get(image_hash)
        if page is None:
            raise RuntimeError(f"No OCR/page record for staged train question {qid} ({image_hash})")
        if page["pixel_sha256"] != row["image_pixel_sha256"]:
            raise RuntimeError(f"Page pixel hash mismatch for {qid}")
        key = row["document_key"]
        union_find.add(key)
        page["document_keys"].add(key)
        page["source_splits"].add("train")
        owner = page_owner.get(image_hash)
        if owner is None:
            page_owner[image_hash] = key
        elif owner != key:
            union_find.union(owner, key)
    if question_hashes != set(page_meta):
        raise RuntimeError(
            f"Staged page records are not in one-to-one coverage with train pages: question-only={len(question_hashes-set(page_meta))}, page-only={len(set(page_meta)-question_hashes)}"
        )
    return union_find, page_owner, page_meta, seen_ids


def merge_delta(builder, building: Path, delta: Path, train_delta_count: int, validation_count: int) -> None:
    staging = building / ".staging"
    delta_staging = delta / ".staging"
    original_train = staging / "train.jsonl"
    original_pages = staging / "pages.jsonl"
    delta_train = delta_staging / "train.jsonl"
    delta_pages = delta_staging / "pages.jsonl"
    delta_validation = delta_staging / "validation.jsonl"
    if (staging / "validation.jsonl").exists():
        raise RuntimeError("A validation staging file already exists; refusing to overwrite.")

    # The delta image directory may be a symlink to the original canonical image store.
    image_source = delta / "images"
    image_target = building / "images"
    image_target.mkdir(parents=True, exist_ok=True)
    if not image_source.is_symlink() and image_source.exists():
        for src in image_source.glob("*.png"):
            dst = image_target / src.name
            if dst.exists():
                if sha256_file(dst) != sha256_file(src):
                    raise RuntimeError(f"Image collision while merging: {src.name}")
            else:
                tmp = dst.with_suffix(".png.merge.tmp")
                shutil.copy2(src, tmp)
                if sha256_file(tmp) != src.stem:
                    raise RuntimeError(f"New image failed content hash: {tmp}")
                os.replace(tmp, dst)

    train_tmp = staging / "train.jsonl.merged.tmp"
    pages_tmp = staging / "pages.jsonl.merged.tmp"
    validation_tmp = staging / "validation.jsonl.merged.tmp"
    if any(p.exists() for p in (train_tmp, pages_tmp, validation_tmp)):
        raise RuntimeError("A merge temporary file already exists; inspect before proceeding.")
    atomic_copy_plus_file(original_train, delta_train, train_tmp)
    atomic_copy_plus_file(original_pages, delta_pages, pages_tmp)
    shutil.copy2(delta_validation, validation_tmp)
    fsync_file(validation_tmp)

    train_count, train_ids = validate_jsonl_unique(train_tmp, "sample_id")
    val_count, val_ids = validate_jsonl_unique(validation_tmp, "sample_id")
    page_count, page_hashes = validate_jsonl_unique(pages_tmp, "image_sha256")
    if train_count != EXPECTED_TRAIN or train_delta_count != EXPECTED_TRAIN - 23472:
        raise RuntimeError(f"Merged train count is {train_count}; delta={train_delta_count}")
    if val_count != EXPECTED_VALIDATION or val_count != validation_count:
        raise RuntimeError(f"Merged validation count is {val_count}; delta={validation_count}")
    if train_ids & val_ids:
        raise RuntimeError(f"Train/validation question ID overlap: {len(train_ids & val_ids)}")
    q_hashes = {row["image_sha256"] for row in iter_jsonl(train_tmp)} | {row["image_sha256"] for row in iter_jsonl(validation_tmp)}
    if q_hashes != page_hashes:
        raise RuntimeError(f"Merged question/page image coverage mismatch: q-only={len(q_hashes-page_hashes)}, pages-only={len(page_hashes-q_hashes)}")
    log(f"[merge] verified train={train_count}, validation={val_count}, unique_pages={page_count}")

    # Keep rollback copies of the repaired staging data before atomic promotion.
    for original in (original_train, original_pages):
        backup = original.with_name(original.name + ".before_resume_merge")
        if backup.exists():
            raise RuntimeError(f"Merge backup exists already: {backup}")
        shutil.copy2(original, backup)
    os.replace(train_tmp, original_train)
    os.replace(pages_tmp, original_pages)
    os.replace(validation_tmp, staging / "validation.jsonl")
    for p in (original_train, original_pages, staging / "validation.jsonl"):
        fsync_file(p)
    log("[merge] delta manifests committed; rollback copies retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=Path("/workspace/worktrees/emi-docvqa-corpus"))
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/projects/efficient-multimodal-inference"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/hf-cache"))
    parser.add_argument("--building-dir", type=Path, default=Path("/workspace/worktrees/emi-docvqa-corpus/data/docvqa/full_corpus_2026-11-22.building"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/worktrees/emi-docvqa-corpus/data/docvqa/full_corpus_2026-11-22"))
    args = parser.parse_args()
    wt = args.worktree.resolve()
    root = args.repo_root.resolve()
    cache = args.cache_dir.resolve()
    building = args.building_dir.resolve()
    output = args.output_dir.resolve()
    staging = building / ".staging"
    if output.exists():
        raise RuntimeError(f"Final output already exists: {output}")
    if not building.is_dir() or not staging.is_dir():
        raise RuntimeError(f"Expected existing interrupted build not found: {building}")
    if (building / ".resume_delta_20261123").exists():
        raise RuntimeError("Resume delta directory already exists; inspect it before retrying.")
    if (staging / "validation.jsonl").exists():
        raise RuntimeError("Validation staging already exists; this resume script expects the known pre-validation interruption.")

    source_script = wt / "scripts/build_docvqa_corpus.py"
    if not source_script.is_file():
        raise FileNotFoundError(source_script)
    delta = building / ".resume_delta_20261123"
    delta.mkdir(parents=False, exist_ok=False)
    runtime_builder = delta / "build_docvqa_corpus_runtime.py"
    builder = load_builder(source_script, runtime_builder)
    # Share the canonical image store with the interrupted build to avoid
    # duplicating tens of gigabytes in the delta workspace.
    os.symlink(building / "images", delta / "images", target_is_directory=True)

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("The isolated data environment must provide pyarrow and huggingface_hub.") from exc

    log("=== DOCVQA STAGED CORPUS RECOVERY ===")
    log(f"Building directory: {building}")
    log(f"Pinned revision: {REVISION}")

    # 1. Restore the known 43-row tail and three missing OCR page rows.
    repaired = repair_interrupted_shard(builder, pq, hf_hub_download, root, cache, building)
    log(f"[repair] restored question records: {repaired}")

    # 2. Rebuild all in-memory global grouping/dedup state from durable staging.
    union_find, page_owner, page_meta, seen_ids = reconstruct_state(builder, staging)
    train_count_before = len(seen_ids)
    if train_count_before != 23472:
        raise RuntimeError(f"Expected 23,472 repaired train questions; found {train_count_before}.")
    log(f"[resume] reconstructed {len(union_find.parent)} document keys, {len(page_meta)} pages, {len(seen_ids)} question IDs")

    historical_csv = root / "publication/energy_measurement_2026-11-18/energy_requests.csv"
    pilot_manifest = root / "data/docvqa/manifest.jsonl"
    historical_ids, historical_hashes = builder.read_historical_inputs(historical_csv, pilot_manifest, root)
    if len(historical_ids) != 50 or len(historical_hashes) != 50:
        raise RuntimeError("Historical input verification did not return the expected 50 question/image pairs.")

    # 3. Resolve pinned source inventory and retain source-level hashes/counts for completed shards.
    api = HfApi()
    files = api.list_repo_files(DATASET_ID, repo_type="dataset", revision=REVISION)
    source_shards = {
        "train": sorted(n for n in files if n.startswith("data/train-") and n.endswith(".parquet")),
        "validation": sorted(n for n in files if n.startswith("data/validation-") and n.endswith(".parquet")),
    }
    if len(source_shards["train"]) != 36 or len(source_shards["validation"]) != 4:
        raise RuntimeError(f"Unexpected shard inventory: train={len(source_shards['train'])}, validation={len(source_shards['validation'])}")
    if source_shards["train"][:COMPLETED_TRAIN_SHARDS] != [f"data/train-{i:05d}-of-00036.parquet" for i in range(COMPLETED_TRAIN_SHARDS)]:
        raise RuntimeError("Unexpected train shard ordering from pinned revision.")

    shard_records = []
    for name in source_shards["train"][:COMPLETED_TRAIN_SHARDS]:
        p = fetch_parquet(hf_hub_download, name, cache)
        meta = pq.ParquetFile(p).metadata
        shard_records.append({
            "source_split": "train", "filename": name,
            "rows_read": meta.num_rows, "file_bytes": p.stat().st_size,
            "file_sha256": sha256_file(p),
        })
    prefix_rows = sum(r["rows_read"] for r in shard_records)
    if prefix_rows != train_count_before:
        raise RuntimeError(f"Completed train shard row total {prefix_rows} != staged count {train_count_before}")

    # 4. Run the remaining 15 train shards and all four validation shards in isolation.
    remaining_train = source_shards["train"][COMPLETED_TRAIN_SHARDS:]
    remaining_expected = EXPECTED_TRAIN - train_count_before
    builder.EXPECTED_COUNTS["train"] = remaining_expected
    log(f"[resume] processing remaining train shards: {len(remaining_train)}; expected additional questions={remaining_expected}")
    train_delta_count = builder.process_split(
        "train", remaining_train,
        dataset_id=DATASET_ID, revision=REVISION,
        cache_dir=cache, repo_root=root, building_dir=delta,
        union_find=union_find, page_owner=page_owner, page_meta=page_meta,
        seen_question_ids=seen_ids,
        historical_hashes=historical_hashes, historical_ids=historical_ids,
        shard_records=shard_records,
    )
    if train_delta_count != remaining_expected:
        raise RuntimeError(f"Remaining train split produced {train_delta_count}, expected {remaining_expected}")

    builder.EXPECTED_COUNTS["train"] = EXPECTED_TRAIN
    builder.EXPECTED_COUNTS["validation"] = EXPECTED_VALIDATION
    log("[resume] processing all four official validation shards (official test remains untouched)")
    validation_count = builder.process_split(
        "validation", source_shards["validation"],
        dataset_id=DATASET_ID, revision=REVISION,
        cache_dir=cache, repo_root=root, building_dir=delta,
        union_find=union_find, page_owner=page_owner, page_meta=page_meta,
        seen_question_ids=seen_ids,
        historical_hashes=historical_hashes, historical_ids=historical_ids,
        shard_records=shard_records,
    )
    if validation_count != EXPECTED_VALIDATION:
        raise RuntimeError(f"Validation split produced {validation_count}, expected {EXPECTED_VALIDATION}")

    # 5. Merge only after the complete delta build has succeeded and validate the combined manifests.
    merge_delta(builder, building, delta, train_delta_count, validation_count)
    shutil.rmtree(delta)

    # All train/validation IDs have been checked for cross-split duplication in process_split.
    if len(seen_ids) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise RuntimeError(f"Combined seen question IDs={len(seen_ids)}, expected {EXPECTED_TRAIN + EXPECTED_VALIDATION}")
    document_groups = union_find.stable_groups()
    summary = builder.finalize_partitions(
        building,
        EXPECTED_TRAIN,
        EXPECTED_VALIDATION,
        document_groups,
        page_meta,
        historical_ids,
        historical_hashes,
        0.1,
        42,
        shard_records,
        pilot_manifest,
        historical_csv,
        REVISION,
        dataset_id=DATASET_ID,
    )

    # Publish only after source/partition/leakage checks in finalize_partitions pass.
    shutil.rmtree(staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    building.rename(output)
    log(json.dumps({
        "status": "complete",
        "output_dir": str(output),
        "source_question_counts": summary["source_question_counts"],
        "partition_question_counts": summary["partition_question_counts"],
        "partition_document_group_counts": summary["partition_document_group_counts"],
        "partition_unique_page_counts": summary["partition_unique_page_counts"],
        "unique_pages": summary["unique_pages"],
        "train_groups_excluded_for_historical_overlap": summary["train_groups_excluded_for_historical_overlap"],
        "validation_groups_excluded_for_train_overlap": summary["validation_groups_excluded_for_train_overlap"],
        "leakage_checks": summary["leakage_checks"],
        "artifact_sha256": summary["artifact_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
