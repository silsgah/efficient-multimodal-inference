import json

from src.benchmark.runner import BenchmarkRunner
from src.data.dataset import BenchmarkDataset
from src.models.base import (
    GenerationConfig,
    InferenceResponse,
    ModelBackend,
)


class FakeBackend(ModelBackend):
    def __init__(self):
        self.loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.generate_count = 0

    @property
    def backend_name(self):
        return "fake"

    def load(self):
        self.loaded = True
        self.load_count += 1

    def generate(self, request, config):
        assert self.loaded
        self.generate_count += 1

        return InferenceResponse(
            answer="Test answer",
            request_id=request.request_id,
            latency_seconds=0.1,
            input_tokens=20,
            output_tokens=5,
        )

    def unload(self):
        self.loaded = False
        self.unload_count += 1


def create_dataset(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").touch()

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id":"sample_001",'
        '"image":"images/sample.jpg",'
        '"question":"What is shown?"}\n',
        encoding="utf-8",
    )

    return BenchmarkDataset(manifest)


def test_runner_executes_and_records_requests(tmp_path):
    dataset = create_dataset(tmp_path)
    backend = FakeBackend()

    runner = BenchmarkRunner(
        backend=backend,
        dataset=dataset,
        generation_config=GenerationConfig(),
        max_pixels=401408,
        warmup_requests=2,
        measured_requests=3,
        repetitions=2,
        results_dir=tmp_path / "results",
    )

    summary = runner.run()

    assert summary["total_measured_requests"] == 6
    assert summary["total_output_tokens"] == 30
    assert backend.generate_count == 8
    assert backend.load_count == 1
    assert backend.unload_count == 1
    assert backend.loaded is False


def test_runner_saves_json_and_csv(tmp_path):
    dataset = create_dataset(tmp_path)

    runner = BenchmarkRunner(
        backend=FakeBackend(),
        dataset=dataset,
        generation_config=GenerationConfig(),
        max_pixels=401408,
        warmup_requests=0,
        measured_requests=2,
        repetitions=1,
        results_dir=tmp_path / "results",
    )

    summary = runner.run()

    json_files = list((tmp_path / "results").glob("*.json"))
    csv_files = list((tmp_path / "results").glob("*.csv"))

    assert len(json_files) == 1
    assert len(csv_files) == 1

    saved = json.loads(json_files[0].read_text())
    assert saved["total_measured_requests"] == 2
    assert saved["records"][0]["ttft_seconds"] is None


def test_runner_rejects_invalid_pixel_budget(tmp_path):
    dataset = create_dataset(tmp_path)

    try:
        BenchmarkRunner(
            backend=FakeBackend(),
            dataset=dataset,
            generation_config=GenerationConfig(),
            max_pixels=200000,
            results_dir=tmp_path / "results",
        )
        assert False, "Expected invalid pixel budget to fail"
    except ValueError:
        pass
