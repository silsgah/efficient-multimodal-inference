import json
import time
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MANIFEST = ROOT / "data/docvqa/manifest.jsonl"

with MANIFEST.open(encoding="utf-8") as f:
    sample = json.loads(next(f))

image_path = ROOT / sample["image"]
question = sample["question"]

print("Model:", MODEL)
print("Question:", question)
print("Expected answer:", sample["answer"])
print("Image:", image_path)

image = Image.open(image_path).convert("RGB")

print("Image dimensions:", image.size)
print("Loading model onto GPU...")

start = time.perf_counter()

llm = LLM(
    model=MODEL,
    max_model_len=8192,
    max_num_seqs=1,
    gpu_memory_utilization=0.80,
    limit_mm_per_prompt={"image": 1},
)

load_seconds = time.perf_counter() - start
print(f"Model load time: {load_seconds:.2f}s")

prompt = (
    "<|im_start|>system\n"
    "You are a document question-answering assistant. "
    "Answer using the information visible in the document. "
    "Be concise.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    f"{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)

start = time.perf_counter()

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }],
    sampling_params=sampling,
)

latency = time.perf_counter() - start
answer = outputs[0].outputs[0].text

print("\n========== INFERENCE RESULT ==========")
print("Question:", question)
print("Model answer:", answer)
print("Reference answer:", sample["answer"])
print(f"Inference latency: {latency:.3f}s")
print("Prompt tokens:", len(outputs[0].prompt_token_ids))
print("Generated tokens:", len(outputs[0].outputs[0].token_ids))
print("=====================================")
