import json
from pathlib import Path

from PIL import Image

from scripts.build_docvqa_corpus import (
    UnionFind,
    canonical_png,
    document_key,
    finalize_partitions,
    policy_partition,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_document_key_requires_source_document_identity():
    assert document_key({"ucsf_document_id": "doc-17"}) == "ucsf:doc-17"
    try:
        document_key({"doc_id": 42})
        assert False, "Expected missing ucsf_document_id to fail"
    except ValueError as exc:
        assert "ucsf_document_id" in str(exc)


def test_union_find_merges_document_ids_linked_by_same_page():
    groups = UnionFind()
    groups.add("ucsf:A")
    groups.add("ucsf:B")
    groups.add("ucsf:C")
    groups.union("ucsf:A", "ucsf:B")
    mapping = groups.stable_groups()
    assert mapping["ucsf:A"] == mapping["ucsf:B"]
    assert mapping["ucsf:C"] != mapping["ucsf:A"]


def test_policy_split_is_deterministic_for_whole_document_group():
    first = policy_partition("docgrp-abc", 0.1, 42)
    second = policy_partition("docgrp-abc", 0.1, 42)
    assert first == second
    assert first in {"policy_train", "policy_validation"}


def test_canonical_png_converts_images_to_rgb_and_is_deterministic():
    import io
    source = Image.new("RGBA", (4, 3), (10, 20, 30, 100))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    first = canonical_png(buffer.getvalue())
    second = canonical_png(buffer.getvalue())
    assert first == second
    with Image.open(io.BytesIO(first)) as result:
        assert result.mode == "RGB"
        assert result.size == (4, 3)


def test_finalize_partitions_enforces_document_and_page_isolation(tmp_path):
    building = tmp_path / "corpus.building"
    staging = building / ".staging"
    staging.mkdir(parents=True)

    train_rows = [
        {"sample_id": "train-historical", "document_key": "ucsf:A", "image_sha256": "page-A"},
        {"sample_id": "train-overlap", "document_key": "ucsf:B", "image_sha256": "page-B"},
        {"sample_id": "train-policy-1", "document_key": "ucsf:C", "image_sha256": "page-C"},
        {"sample_id": "train-policy-2", "document_key": "ucsf:C", "image_sha256": "page-C"},
    ]
    validation_rows = [
        {"sample_id": "historical-question", "document_key": "ucsf:A", "image_sha256": "page-A"},
        {"sample_id": "validation-overlap", "document_key": "ucsf:B", "image_sha256": "page-B"},
        {"sample_id": "locked-question", "document_key": "ucsf:D", "image_sha256": "page-D"},
    ]
    _write_jsonl(staging / "train.jsonl", train_rows)
    _write_jsonl(staging / "validation.jsonl", validation_rows)

    page_meta = {}
    page_staging_rows = []
    for doc_key, page_hash, splits in (
        ("ucsf:A", "page-A", {"train", "validation"}),
        ("ucsf:B", "page-B", {"train", "validation"}),
        ("ucsf:C", "page-C", {"train"}),
        ("ucsf:D", "page-D", {"validation"}),
    ):
        page_meta[page_hash] = {
            "document_keys": {doc_key},
            "source_splits": set(splits),
            "ocr_sha256": "ocr-" + page_hash,
        }
        page_staging_rows.append({
            "image_sha256": page_hash,
            "image": "images/" + page_hash + ".png",
            "document_keys": [doc_key],
            "source_splits": sorted(splits),
            "ocr_results": {"lines": []},
            "other_metadata": {"ucsf_document_id": doc_key.removeprefix("ucsf:")},
        })
    _write_jsonl(staging / "pages.jsonl", page_staging_rows)

    document_groups = {
        "ucsf:A": "group-A",
        "ucsf:B": "group-B",
        "ucsf:C": "group-C",
        "ucsf:D": "group-D",
    }
    pilot_manifest = tmp_path / "pilot.jsonl"
    historical_csv = tmp_path / "historical.csv"
    pilot_manifest.write_text("{}\n", encoding="utf-8")
    historical_csv.write_text("request_id\nhistorical-question\n", encoding="utf-8")

    summary = finalize_partitions(
        building_dir=building,
        train_count=len(train_rows),
        validation_count=len(validation_rows),
        document_groups=document_groups,
        page_meta=page_meta,
        historical_ids={"historical-question"},
        historical_hashes={"historical-question": "page-A"},
        validation_fraction=0.5,
        seed=42,
        shard_records=[],
        manifest_path=pilot_manifest,
        historical_csv=historical_csv,
        revision="test-revision",
    )

    def ids(name: str) -> set[str]:
        path = building / f"{name}.jsonl"
        return {row["sample_id"] for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )}

    assert ids("historical_development") == {"historical-question"}
    assert ids("locked_evaluation") == {"locked-question"}
    assert ids("cross_split_validation_excluded") == {"validation-overlap"}
    assert ids("historical_overlap_train_excluded") == {"train-historical"}
    assert ids("policy_train") | ids("policy_validation") == {
        "train-overlap", "train-policy-1", "train-policy-2"
    }
    assert ("train-policy-1" in ids("policy_train")) == (
        "train-policy-2" in ids("policy_train")
    )
    assert ("train-policy-1" in ids("policy_validation")) == (
        "train-policy-2" in ids("policy_validation")
    )
    assert sum(summary["partition_question_counts"].values()) == 7
    assert summary["train_groups_excluded_for_historical_overlap"] == 1
    assert summary["validation_groups_excluded_for_train_overlap"] == 1
    assert summary["unique_pages"] == 4
    assert all(
        check["shared_document_groups"] == 0 and check["shared_image_hashes"] == 0
        for check in summary["leakage_checks"].values()
    )
    assert (building / "pages.jsonl").is_file()
    assert (building / "documents.jsonl").is_file()

    documents = {
        row["document_group_id"]: row
        for row in (
            json.loads(line)
            for line in (building / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert set(documents["group-A"]["partition_memberships"]) == {
        "historical_development",
        "historical_overlap_train_excluded",
    }
    group_b_memberships = set(documents["group-B"]["partition_memberships"])
    assert "cross_split_validation_excluded" in group_b_memberships
    assert len(group_b_memberships & {"policy_train", "policy_validation"}) == 1

