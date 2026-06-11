"""
This module contains unit tests for MaskToPolygonConverter and PolygonToMaskConverter, validating correct point scaling and reconstruction.
"""
import os
import tempfile
import numpy as np
import cv2
from segment_toolkit import MaskToPolygonConverter, PolygonToMaskConverter

def test_polygon_conversions():
    """
    Test polygon mask-to-polygon and polygon-to-mask conversions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        mask_path = os.path.join(tmpdir, "mask.png")
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        cv2.imwrite(mask_path, mask)

        # 1. Mask to Polygon
        out_lbl = os.path.join(tmpdir, "label.txt")
        m2p = MaskToPolygonConverter(target_size=(100, 100))
        success = m2p.convert_single(img_path, mask_path, out_lbl, class_id=2)
        assert success
        assert os.path.exists(out_lbl)

        # 2. Check content
        with open(out_lbl, "r") as f:
            content = f.read().strip()
            parts = content.split()
            assert parts[0] == "2"
            assert len(parts) >= 7

        # 3. Polygon to Mask
        out_mask = os.path.join(tmpdir, "reconstructed.png")
        p2m = PolygonToMaskConverter(target_size=(100, 100))
        success = p2m.convert_single(out_lbl, out_mask)
        assert success
        assert os.path.exists(out_mask)

        reconstructed = cv2.imread(out_mask, cv2.IMREAD_GRAYSCALE)
        assert reconstructed[50, 50] == 255
        assert reconstructed[5, 5] == 0
