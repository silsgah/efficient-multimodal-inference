import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.adaptive.types import AcquisitionAction, BoundingBox, DecisionContext


@dataclass(frozen=True)
class AcquiredInput:
    request_id: str
    question: str
    image_path: Path
    max_pixels: int
    min_pixels: int
    action: AcquisitionAction


class ImageAcquirer:
    """Materializes a full-image resolution escalation or an ROI crop.

    FULL_UPSCALE retains the original image and raises the processor pixel
    budget; it does not create an interpolated image file. ROI_CROP writes a
    new crop file while leaving the source image untouched.
    """

    def __init__(
        self,
        output_dir: str | Path,
        minimum_pixels: int = 200704,
        upscale_max_pixels: int = 802816,
    ) -> None:
        if minimum_pixels <= 0 or minimum_pixels % 784:
            raise ValueError("minimum_pixels must be a positive multiple of 784.")
        if upscale_max_pixels <= 0 or upscale_max_pixels % 784:
            raise ValueError("upscale_max_pixels must be a positive multiple of 784.")
        if minimum_pixels > upscale_max_pixels:
            raise ValueError("minimum_pixels cannot exceed upscale_max_pixels.")

        self.output_dir = Path(output_dir)
        self.minimum_pixels = minimum_pixels
        self.upscale_max_pixels = upscale_max_pixels

    def acquire(
        self,
        context: DecisionContext,
        action: AcquisitionAction,
        roi_bbox: BoundingBox | None = None,
    ) -> AcquiredInput:
        if action is AcquisitionAction.STOP:
            raise ValueError("STOP does not acquire additional visual evidence.")

        target_pixels = min(
            context.maximum_max_pixels,
            self.upscale_max_pixels,
        )
        if target_pixels < self.minimum_pixels:
            raise ValueError("Target pixel budget is below the minimum pixel budget.")

        if action is AcquisitionAction.FULL_UPSCALE:
            if target_pixels <= context.current_max_pixels:
                raise ValueError("Full-image escalation requires a higher pixel budget.")
            return AcquiredInput(
                request_id=context.request_id,
                question=context.question,
                image_path=Path(context.image_path),
                max_pixels=target_pixels,
                min_pixels=self.minimum_pixels,
                action=action,
            )

        if action is not AcquisitionAction.ROI_CROP:
            raise ValueError(f"Unsupported acquisition action: {action}")

        bbox = roi_bbox or context.evidence.roi_bbox
        if bbox is None:
            raise ValueError("ROI_CROP requires a bounding-box proposal.")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to materialize ROI crops.") from exc

        source_path = Path(context.image_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            bbox.validate_image_bounds(*image.size)
            crop = image.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1))

        if crop.width <= 0 or crop.height <= 0:
            raise ValueError("ROI crop is empty.")

        safe_id = hashlib.sha256(context.request_id.encode("utf-8")).hexdigest()[:16]
        request_dir = self.output_dir / safe_id
        request_dir.mkdir(parents=True, exist_ok=True)
        crop_path = request_dir / "roi.png"
        crop.save(crop_path, format="PNG")

        return AcquiredInput(
            request_id=context.request_id,
            question=context.question,
            image_path=crop_path,
            max_pixels=target_pixels,
            min_pixels=self.minimum_pixels,
            action=action,
        )
