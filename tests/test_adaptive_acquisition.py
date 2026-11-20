from pathlib import Path

import pytest
from PIL import Image

from src.adaptive.acquisition import ImageAcquirer
from src.adaptive.types import (
    AcquisitionAction,
    BoundingBox,
    DecisionContext,
    EvidenceSignals,
)


def make_context(tmp_path, bbox=None):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (300, 200), color="white").save(image_path)
    return DecisionContext(
        request_id="sample/../unsafe",
        question="Read the date.",
        image_path=image_path,
        current_max_pixels=200704,
        maximum_max_pixels=802816,
        evidence=EvidenceSignals(
            evidence_sufficiency=0.1,
            dense_information_risk=0.1,
            roi_bbox=bbox,
            source="test",
        ),
    )


def test_full_upscale_reuses_original_without_mutation(tmp_path):
    context = make_context(tmp_path)
    acquirer = ImageAcquirer(tmp_path / "crops")
    result = acquirer.acquire(context, AcquisitionAction.FULL_UPSCALE)

    assert result.image_path == context.image_path
    assert result.max_pixels == 802816
    assert context.image_path.exists()


def test_roi_crop_materializes_bounded_crop_and_keeps_source(tmp_path):
    bbox = BoundingBox(10, 20, 110, 120)
    context = make_context(tmp_path, bbox)
    acquirer = ImageAcquirer(tmp_path / "crops")
    result = acquirer.acquire(context, AcquisitionAction.ROI_CROP)

    assert result.image_path.exists()
    assert result.image_path != context.image_path
    with Image.open(result.image_path) as crop:
        assert crop.size == (100, 100)
    assert context.image_path.exists()
    assert ".." not in str(result.image_path)


def test_roi_requires_bbox(tmp_path):
    context = make_context(tmp_path)
    acquirer = ImageAcquirer(tmp_path / "crops")
    with pytest.raises(ValueError, match="bounding-box"):
        acquirer.acquire(context, AcquisitionAction.ROI_CROP)


def test_bbox_cannot_exceed_image_dimensions():
    with pytest.raises(ValueError):
        BoundingBox(0, 0, 500, 500).validate_image_bounds(300, 200)
