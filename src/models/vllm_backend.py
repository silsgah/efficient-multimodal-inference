import gc
import time
from pathlib import Path
from typing import Any

from src.models.base import (
    GenerationConfig,
    InferenceRequest,
    InferenceResponse,
    ModelBackend,
)


class VLLMBackend(ModelBackend):
    """vLLM backend for single-image Qwen2.5-VL inference.

    vLLM and Transformers are imported lazily so CPU-side tests and controller
    development do not initialize CUDA or download model weights.
    """

    backend_name = "vllm"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 1,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_id:
            raise ValueError("model_id is required.")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive.")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1].")
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive.")

        self.model_id = model_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.trust_remote_code = trust_remote_code
        self.engine: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def build_image_content(
        image: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        content: dict[str, Any] = {"image": image}

        for key in ("min_pixels", "max_pixels"):
            value = metadata.get(key)
            if value is not None:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{key} must be a positive integer.")
                if value % (28 * 28) != 0:
                    raise ValueError(f"{key} must be a multiple of 28 * 28.")
                content[key] = value

        if (
            content.get("min_pixels") is not None
            and content.get("max_pixels") is not None
            and content["min_pixels"] > content["max_pixels"]
        ):
            raise ValueError("min_pixels cannot exceed max_pixels.")

        return content

    def load(self) -> None:
        if self.engine is not None:
            return

        try:
            from vllm import LLM
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "vLLM and Transformers must be installed in the active environment."
            ) from exc

        self.engine = LLM(
            model=self.model_id,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=self.trust_remote_code,
            limit_mm_per_prompt={"image": 1},
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

    def generate(
        self,
        request: InferenceRequest,
        generation_config: GenerationConfig,
    ) -> InferenceResponse:
        if self.engine is None:
            raise RuntimeError("Backend is not loaded.")

        try:
            from PIL import Image
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Pillow and vLLM must be installed to generate responses."
            ) from exc

        image_path = Path(request.image)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        image_content = self.build_image_content(
            str(image_path),
            request.metadata,
        )

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        prompt = (
            "<|im_start|>system\n"
            "You are a document question-answering assistant. "
            "Answer using the information visible in the document. "
            "Be concise and do not invent unreadable details.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{request.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        sampling = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=generation_config.max_new_tokens,
        )

        prompt_item: dict[str, Any] = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        processor_kwargs = {
            key: image_content[key]
            for key in ("min_pixels", "max_pixels")
            if key in image_content
        }
        if processor_kwargs:
            prompt_item["mm_processor_kwargs"] = processor_kwargs

        started = time.perf_counter()
        outputs = self.engine.generate(
            [prompt_item],
            sampling_params=sampling,
            use_tqdm=False,
        )
        latency = time.perf_counter() - started

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no generation.")

        request_output = outputs[0]
        generation = request_output.outputs[0]
        prompt_ids = getattr(request_output, "prompt_token_ids", None) or []
        output_ids = getattr(generation, "token_ids", None) or []

        visual_tokens = None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if isinstance(image_pad_id, int):
                visual_tokens = sum(
                    token_id == image_pad_id for token_id in prompt_ids
                )

        return InferenceResponse(
            answer=str(generation.text).strip(),
            request_id=request.request_id,
            latency_seconds=latency,
            input_tokens=len(prompt_ids) if prompt_ids else None,
            output_tokens=len(output_ids) if output_ids else None,
            visual_tokens=visual_tokens,
        )

    def unload(self) -> None:
        self.engine = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
