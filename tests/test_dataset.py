import json
import pytest

from src.data.dataset import BenchmarkDataset


def create_dataset(tmp_path, records):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "sample.jpg"
    image.touch()

    manifest = tmp_path / "manifest.jsonl"

    with manifest.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    return manifest, image


def test_dataset_loads_valid_manifest(tmp_path):
    manifest, image = create_dataset(
        tmp_path,
        [{
            "sample_id": "sample_001",
            "image": "images/sample.jpg",
            "question": "What is shown?",
            "answer": "A document",
        }],
    )

    dataset = BenchmarkDataset(manifest)

    assert len(dataset) == 1
    assert dataset[0].sample_id == "sample_001"
    assert dataset[0].image == image.resolve()
    assert dataset[0].answer == "A document"


def test_dataset_resolves_docvqa_repo_relative_image_paths(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "configs").mkdir()
    image = project / "data/docvqa/images/page.png"
    image.parent.mkdir(parents=True)
    image.touch()
    manifest = project / "data/docvqa/manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "sample_id": "docvqa-1",
            "image": "data/docvqa/images/page.png",
            "question": "What is the invoice total?",
            "answer": ["100.00"],
        }) + "\n",
        encoding="utf-8",
    )

    dataset = BenchmarkDataset(manifest)

    assert dataset[0].image == image.resolve()


def test_dataset_creates_inference_request(tmp_path):
    manifest, _ = create_dataset(
        tmp_path,
        [{
            "sample_id": "sample_001",
            "image": "images/sample.jpg",
            "question": "Read this document.",
        }],
    )

    dataset = BenchmarkDataset(manifest)

    request = dataset.to_request(0, max_pixels=401408)

    assert request.request_id == "sample_001"
    assert request.question == "Read this document."
    assert request.metadata["max_pixels"] == 401408


def test_dataset_rejects_missing_image(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id":"x","image":"missing.jpg","question":"Read it"}\n',
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        BenchmarkDataset(manifest)


def test_dataset_rejects_duplicate_ids(tmp_path):
    record = {
        "sample_id": "duplicate",
        "image": "images/sample.jpg",
        "question": "Read it",
    }

    manifest, _ = create_dataset(tmp_path, [record, record])

    with pytest.raises(ValueError, match="Duplicate"):
        BenchmarkDataset(manifest)


def test_dataset_rejects_empty_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no samples"):
        BenchmarkDataset(manifest)
