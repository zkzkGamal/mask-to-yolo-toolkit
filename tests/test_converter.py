"""
This module contains unit tests for bounding box coordinate normalization, denormalization, contour calculation, and integration tests for MaskToYoloConverter and YoloToMaskConverter.
"""
import os
import tempfile
import pytest
import numpy as np
import cv2

from segment_toolkit import (
    normalize_coordinates,
    denormalize_coordinates,
    calculate_bbox,
    MaskToYoloConverter,
    YoloToMaskConverter,
)


def test_normalization():
    """
    Test bounding box coordinate normalization.
    """
    x, y, w, h = normalize_coordinates(320.0, 240.0, 100.0, 80.0, 640, 480)
    assert x == pytest.approx(320.0 / 640.0)
    assert y == pytest.approx(240.0 / 480.0)
    assert w == pytest.approx(100.0 / 640.0)
    assert h == pytest.approx(80.0 / 480.0)

    # Check boundaries clamping
    x_clamped, _, _, _ = normalize_coordinates(-50.0, 20.0, 10.0, 10.0, 100, 100)
    assert x_clamped == 0.0


def test_denormalization():
    """
    Test denormalization from YOLO format back to pixel coordinates.
    """
    x1, y1, x2, y2 = denormalize_coordinates(0.5, 0.5, 0.2, 0.2, 640, 640)
    # Center = (320, 320), width = 128, height = 128
    # x1 = 320 - 64 = 256
    # x2 = 320 + 64 = 384
    assert x1 == 256
    assert y1 == 256
    assert x2 == 384
    assert y2 == 384


def test_calculate_bbox():
    """
    Test coordinate calculations from contour geometries.
    """
    # Simple axis-aligned rectangle contour
    contour = np.array([
        [[10, 10]],
        [[30, 10]],
        [[30, 30]],
        [[10, 30]]
    ], dtype=np.int32)

    xc, yc, w, h = calculate_bbox(contour, bbox_type="standard")
    assert xc == 20.5
    assert yc == 20.5
    assert w == 21.0
    assert h == 21.0


def test_conversions():
    """
    Integration test validating end-to-end mask-to-YOLO-to-mask conversions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock image file
        img_path = os.path.join(tmpdir, "mock_img.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        # Create mock binary mask file (50x50 square in the center)
        mask_path = os.path.join(tmpdir, "mock_mask.png")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(mask_path, mask)

        # 1. Mask to YOLO conversion
        out_lbl = os.path.join(tmpdir, "mock_label.txt")
        converter_yolo = MaskToYoloConverter(target_size=(100, 100), bbox_type="standard")
        success = converter_yolo.convert_single(img_path, mask_path, out_lbl, class_id=3)
        assert success
        assert os.path.exists(out_lbl)

        # Parse and assert YOLO output values
        with open(out_lbl, "r", encoding="utf-8") as f:
            content = f.read().strip()
            parts = content.split()
            assert len(parts) == 5
            class_id, x, y, w, h = map(float, parts)
            assert int(class_id) == 3
            assert x == pytest.approx(0.5)
            assert y == pytest.approx(0.5)
            assert w == pytest.approx(0.5)
            assert h == pytest.approx(0.5)

        # 2. YOLO to Mask conversion
        out_mask = os.path.join(tmpdir, "mock_reconstructed.png")
        converter_mask = YoloToMaskConverter(target_size=(100, 100))
        success = converter_mask.convert_single(out_lbl, out_mask)
        assert success
        assert os.path.exists(out_mask)

        # Verify pixels on reconstructed mask
        reconstructed = cv2.imread(out_mask, cv2.IMREAD_GRAYSCALE)
        assert reconstructed[50, 50] == 255
        assert reconstructed[10, 10] == 0


def test_visualize_mask():
    """
    Test mask overlay visualization.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "mock_img.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        mask_path = os.path.join(tmpdir, "mock_mask.png")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        cv2.imwrite(mask_path, mask)

        out_vis = os.path.join(tmpdir, "mock_mask_vis.png")
        converter = YoloToMaskConverter(target_size=(100, 100))
        success = converter.visualize_mask(img_path, mask_path, out_vis)
        assert success
        assert os.path.exists(out_vis)

        visualized = cv2.imread(out_vis, cv2.IMREAD_COLOR)
        assert visualized[50, 50].any()
        assert not visualized[10, 10].any()
