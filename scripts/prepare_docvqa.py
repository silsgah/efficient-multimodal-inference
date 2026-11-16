import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from io import BytesIO

SEED = 42
SAMPLE_SIZE = 200

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/docvqa/images"
MANIFEST = ROOT / "data/docvqa/manifest.jsonl"

def main():
    print("Loading DocVQA validation split...")
    dataset = load_dataset(
        "pixparse/docvqa-single-page-questions",
        split="validation",
    )

    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:min(SAMPLE_SIZE, len(indices))]

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_cache = {}
    records = []

    for index in indices:
        item = dataset[index]
        image_data = item["image"]["bytes"]

        digest = hashlib.sha256(image_data).hexdigest()
        image_name = f"{digest}.png"
        image_path = IMAGE_DIR / image_name

        if digest not in image_cache:
            with Image.open(BytesIO(image_data)) as image:
                image.convert("RGB").save(image_path, format="PNG")
            image_cache[digest] = image_path

        records.append({
            "sample_id": str(item["question_id"]),
            "image": str(image_path.relative_to(ROOT)),
            "question": item["question"],
            "answer": item["answers"],
        })

    with MANIFEST.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Questions: {len(records)}")
    print(f"Unique images: {len(image_cache)}")
    print(f"Manifest: {MANIFEST}")
    print(f"Images: {IMAGE_DIR}")

if __name__ == "__main__":
    main()
