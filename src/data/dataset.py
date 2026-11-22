import json
from dataclasses import dataclass
from pathlib import Path

from src.models.base import InferenceRequest


@dataclass
class BenchmarkSample:
    sample_id: str
    image: Path
    question: str
    answer: str | None = None


class BenchmarkDataset:
    """Load image-question samples from a JSONL manifest."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = self._find_project_root()
        self.samples = self._load()

    def _find_project_root(self) -> Path | None:
        """Find a repository root when the manifest stores repo-relative paths."""
        for parent in self.manifest_path.parents:
            if (parent / "src").is_dir() and (parent / "configs").is_dir():
                return parent
        return None

    def _resolve_image_path(self, value: str, sample_id: str) -> Path:
        image_path = Path(value)
        if image_path.is_absolute():
            resolved = image_path.resolve()
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(
                f"Image not found for {sample_id}: {resolved}"
            )

        # Preserve the original manifest-relative contract first.
        candidates = [(self.manifest_path.parent / image_path).resolve()]

        # The DocVQA preparation script historically writes paths relative to
        # the repository root (for example, data/docvqa/images/<hash>.png).
        if self.project_root is not None:
            root_candidate = (self.project_root / image_path).resolve()
            if root_candidate not in candidates:
                candidates.append(root_candidate)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        attempted = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Image not found for {sample_id}; checked: {attempted}"
        )

    def _load(self) -> list[BenchmarkSample]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        samples = []
        seen_ids = set()

        with self.manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from exc

                required = ("sample_id", "image", "question")

                for field in required:
                    if not record.get(field):
                        raise ValueError(
                            f"Missing {field} on line {line_number}"
                        )

                sample_id = str(record["sample_id"])

                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")

                image_path = self._resolve_image_path(
                    str(record["image"]),
                    sample_id,
                )

                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        image=image_path,
                        question=str(record["question"]),
                        answer=(
                            str(record["answer"])
                            if record.get("answer") is not None
                            else None
                        ),
                    )
                )

                seen_ids.add(sample_id)

        if not samples:
            raise ValueError("Dataset manifest contains no samples.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BenchmarkSample:
        return self.samples[index]

    def to_request(
        self,
        index: int,
        max_pixels: int,
        min_pixels: int | None = None,
    ) -> InferenceRequest:
        sample = self.samples[index]

        metadata = {"max_pixels": max_pixels}

        if min_pixels is not None:
            metadata["min_pixels"] = min_pixels

        return InferenceRequest(
            question=sample.question,
            image=str(sample.image),
            request_id=sample.sample_id,
            metadata=metadata,
        )
