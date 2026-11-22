from scripts.freeze_validation_holdout import (
    ensure_output_paths_available,
    partition_by_document,
)


def test_partition_moves_every_question_on_development_images_to_development():
    samples = [
        {"sample_id": "q1", "image": "page-a.png"},
        {"sample_id": "q2", "image": "page-a-copy.png"},
        {"sample_id": "q3", "image": "page-b.png"},
        {"sample_id": "q4", "image": "page-c.png"},
    ]
    hashes = {
        "q1": "hash-a",
        "q2": "hash-a",
        "q3": "hash-b",
        "q4": "hash-c",
    }

    development, holdout, dev_hashes = partition_by_document(
        samples,
        historical_development_ids={"q1"},
        image_hashes=hashes,
    )

    assert [s["sample_id"] for s in development] == ["q1", "q2"]
    assert [s["sample_id"] for s in holdout] == ["q3", "q4"]
    assert dev_hashes == {"hash-a"}


def test_partition_rejects_missing_historical_ids():
    samples = [{"sample_id": "q1", "image": "a.png"}]
    try:
        partition_by_document(
            samples,
            historical_development_ids={"missing"},
            image_hashes={"q1": "hash-a"},
        )
        assert False, "Expected missing historical ID to fail"
    except ValueError as exc:
        assert "absent from manifest" in str(exc)


def test_partition_rejects_empty_holdout():
    samples = [
        {"sample_id": "q1", "image": "a.png"},
        {"sample_id": "q2", "image": "a-copy.png"},
    ]
    hashes = {"q1": "same-page", "q2": "same-page"}

    try:
        partition_by_document(
            samples,
            historical_development_ids={"q1"},
            image_hashes=hashes,
        )
        assert False, "Expected an empty holdout to fail"
    except ValueError as exc:
        assert "No holdout samples remain" in str(exc)



def test_split_artifacts_are_not_overwritten_without_explicit_opt_in(tmp_path):
    existing = tmp_path / "development.jsonl"
    existing.write_text("{\"frozen\": true}\\n", encoding="utf-8")

    try:
        ensure_output_paths_available(
            [existing, tmp_path / "holdout.jsonl"],
            overwrite=False,
        )
        assert False, "Expected existing output to block replacement"
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)

    # The existing artifact remains byte-for-byte unchanged.
    assert existing.read_text(encoding="utf-8") == "{\"frozen\": true}\\n"


def test_split_artifacts_can_be_overwritten_only_with_explicit_opt_in(tmp_path):
    existing = tmp_path / "development.jsonl"
    existing.write_text("old\\n", encoding="utf-8")

    ensure_output_paths_available([existing], overwrite=True)
