"""
This script performs validation of the polygon instance segmentation converter pipeline on colored multi-class masks and binary masks.
"""
import os
import sys
import json
import logging
import shutil
import cv2
import numpy as np

# Ensure local module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from segment_toolkit import MaskToPolygonConverter, PolygonToMaskConverter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validation_polygon")

PREFIX = "files/" if os.path.exists("files") else ""
OUT_DIR = "validate_data/polygon_val"
os.makedirs(OUT_DIR, exist_ok=True)


def test_single_polygon_to_mask():
    logger.info("========================================")
    logger.info("Test: Polygon Label to Mask (Single Conversion)")
    logger.info("========================================")

    label_path = f"{PREFIX}polygon_test/labels/ISIC_0024306_jpg.rf.MPOiUQ78Nr6PFTQiVBw2.txt"
    out_mask = os.path.join(OUT_DIR, "val_single_mask.png")

    converter = PolygonToMaskConverter(target_size=(640, 640))
    success = converter.convert_single(label_path, out_mask)
    if not success:
        logger.error("Failed polygon-to-mask conversion.")
        return False

    if not os.path.exists(out_mask):
        logger.error(f"Mask file {out_mask} was not created.")
        return False

    logger.info("✅ Polygon to mask single conversion successful!")
    return True


def test_single_mask_to_polygon():
    logger.info("========================================")
    logger.info("Test: Mask to Polygon Label (Single Conversion)")
    logger.info("========================================")

    img_path = f"{PREFIX}polygon_test/images/ISIC_0024306_jpg.rf.MPOiUQ78Nr6PFTQiVBw2.jpg"
    mask_path = os.path.join(OUT_DIR, "val_single_mask.png")
    out_lbl = os.path.join(OUT_DIR, "val_single_polygon.txt")

    converter = MaskToPolygonConverter(target_size=(640, 640))
    success = converter.convert_single(img_path, mask_path, out_lbl, class_id=1)
    if not success:
        logger.error("Failed mask-to-polygon conversion.")
        return False

    if not os.path.exists(out_lbl):
        logger.error(f"Label file {out_lbl} was not created.")
        return False

    with open(out_lbl, "r") as f:
        content = f.read().strip()
        logger.info(f"Generated polygon label sample: '{content[:100]}...'")
        parts = content.split()
        if len(parts) < 6:
            logger.error("Generated label is too short to be a valid polygon.")
            return False
        if parts[0] != "1":
            logger.error(f"Expected class_id 1, got {parts[0]}")
            return False

    logger.info("✅ Mask to polygon single conversion successful!")
    return True


def test_dataset_conversions():
    logger.info("========================================")
    logger.info("Test: Dataset batch conversion & visualization")
    logger.info("========================================")

    labels_dir = f"{PREFIX}polygon_test/labels"
    out_masks_dir = os.path.join(OUT_DIR, "dataset_masks")

    p2m = PolygonToMaskConverter(target_size=(640, 640))
    count_masks = p2m.convert_dataset(labels_dir, out_masks_dir)
    logger.info(f"Successfully batch converted {count_masks} masks from labels.")
    if count_masks != 4:
        logger.error(f"Expected 4 masks, got {count_masks}")
        return False

    m2p = MaskToPolygonConverter(target_size=(640, 640))
    out_labels_dir = os.path.join(OUT_DIR, "dataset_labels")
    count_labels = m2p.convert_dataset(
        images_dir=f"{PREFIX}polygon_test/images",
        masks_dir=out_masks_dir,
        output_labels_dir=out_labels_dir,
        default_class_id=0
    )
    logger.info(f"Successfully batch converted {count_labels} labels from masks.")
    if count_labels != 4:
        logger.error(f"Expected 4 labels, got {count_labels}")
        return False

    # Test visualization overlay
    img_path = f"{PREFIX}polygon_test/images/ISIC_0024306_jpg.rf.MPOiUQ78Nr6PFTQiVBw2.jpg"
    label_path = os.path.join(out_labels_dir, "ISIC_0024306_jpg.rf.MPOiUQ78Nr6PFTQiVBw2.txt")
    out_vis = os.path.join(OUT_DIR, "val_polygon_vis.png")

    success = p2m.visualize_polygon(img_path, label_path, out_vis)
    if not success or not os.path.exists(out_vis):
        logger.error("Failed to generate polygon visualization overlay.")
        return False

    logger.info("✅ Dataset conversion and visualization test successful!")
    return True


def test_multiclass_colored_masks():
    logger.info("========================================")
    logger.info("Test: Multi-class colored mask mode")
    logger.info("========================================")

    # 1. Create a mock colored mask (RGB)
    # Class 0: Red (255, 0, 0)
    # Class 1: Green (0, 255, 0)
    H, W = 640, 640
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (200, 200), (0, 0, 255), -1)      # Red in BGR is (0, 0, 255)
    cv2.rectangle(mask, (300, 300), (500, 500), (0, 255, 0), -1)   # Green in BGR is (0, 255, 0)

    mock_mask_path = os.path.join(OUT_DIR, "mock_colored_mask.png")
    cv2.imwrite(mock_mask_path, mask)

    # 2. Define classes mapping
    # structure: [((R, G, B), class_name), ...]
    classes = [
        ((255, 0, 0), "lesion_red"),
        ((0, 255, 0), "lesion_green")
    ]

    mock_img_path = os.path.join(OUT_DIR, "mock_image.png")
    cv2.imwrite(mock_img_path, np.zeros((H, W, 3), dtype=np.uint8)) # blank image

    # 3. Convert colored mask to polygon label
    m2p = MaskToPolygonConverter(target_size=(640, 640))
    out_txt = os.path.join(OUT_DIR, "mock_colored_label.txt")
    success = m2p.convert_single(mock_img_path, mock_mask_path, out_txt, classes=classes)
    if not success or not os.path.exists(out_txt):
        logger.error("Failed multi-class color mask-to-polygon conversion.")
        return False

    # 4. Check label file contents
    with open(out_txt, "r") as f:
        lines = f.readlines()
        logger.info(f"Generated multi-class colored label contents:\n" + "".join(lines))
        if len(lines) != 2:
            logger.error(f"Expected 2 lines in label file, got {len(lines)}")
            return False

    # 5. Convert polygon label back to colored mask
    p2m = PolygonToMaskConverter(target_size=(640, 640))
    out_reconstructed = os.path.join(OUT_DIR, "mock_reconstructed_mask.png")
    success = p2m.convert_single(out_txt, out_reconstructed, classes=classes)
    if not success or not os.path.exists(out_reconstructed):
        logger.error("Failed multi-class color polygon-to-mask conversion.")
        return False

    logger.info("✅ Multi-class colored mask test successful!")
    return True


if __name__ == "__main__":
    t1 = test_single_polygon_to_mask()
    t2 = test_single_mask_to_polygon()
    t3 = test_dataset_conversions()
    t4 = test_multiclass_colored_masks()

    if t1 and t2 and t3 and t4:
        logger.info("\n🎉 All polygon pipeline tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some polygon pipeline tests failed.")
        sys.exit(1)
