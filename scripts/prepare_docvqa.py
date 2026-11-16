import hashlib
import json
import random
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image

SEED = 42
SAMPLE_SIZE = 200

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/docvqa/images"
MANIFEST = ROOT / "data/docvqa/manifest.jsonl"
CACHE_DIR = Path("/workspace/hf-cache/datasets")


def image_to_png(image_value):
    """Normalize Hugging Face image values to PNG bytes."""
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict):
        if image_value.get("bytes"):
            image = Image.open(BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            image = Image.open(image_value["path"])
        else:
            raise ValueError("Image dictionary has neither bytes nor path")
    else:
        raise TypeError(f"Unsupported image type: {type(image_value)}")

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    print("Loading DocVQA validation split...")
    dataset = load_dataset(
        "pixparse/docvqa-single-page-questions",
        split="validation",
        cache_dir=str(CACHE_DIR),
    )

    print(f"Available questions: {len(dataset)}")

    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:min(SAMPLE_SIZE, len(indices))]

    image_cache = {}
    records = []

    for position, index in enumerate(indices, start=1):
        item = dataset[index]
        png_bytes = image_to_png(item["image"])
        digest = hashlib.sha256(png_bytes).hexdigest()
        image_name = f"{digest}.png"
        image_path = IMAGE_DIR / image_name

        if digest not in image_cache:
            if not image_path.exists():
                image_path.write_bytes(png_bytes)
            image_cache[digest] = image_path

        records.append({
            "sample_id": str(item["question_id"]),
            "image": str(image_path.relative_to(ROOT)),
            "question": item["question"],
            "answer": item["answers"],
        })

        if position % 25 == 0:
            print(f"Prepared {position}/{len(indices)} questions")

    with MANIFEST.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Questions: {len(records)}")
    print(f"Unique images: {len(image_cache)}")
    print(f"Manifest: {MANIFEST}")
    print(f"Images: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
