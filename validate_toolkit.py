"""
Validation script for segment_toolkit.
Runs the toolkit against sample images from ISIC and plant datasets, testing both CSV and JSON ground truths.
Saves all output validation files under the 'validate_data' directory.
"""

import os
import sys
import json
import logging
import shutil

# Ensure local module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from segment_toolkit import MaskToYoloConverter, YoloToMaskConverter

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("validation")

# Auto-detect folder prefix
PREFIX = "files/" if os.path.exists("files") else ""
logger.info(f"Using directory path prefix: '{PREFIX}'")

# Establish output directory for validation artifacts
OUT_DIR = "validate_data"
os.makedirs(OUT_DIR, exist_ok=True)
logger.info(f"All validation outputs will be saved to: '{OUT_DIR}/'")


def validate_isic():
    logger.info("========================================")
    logger.info("Validating on ISIC Dataset (Single Conversion)")
    logger.info("========================================")

    img_path = f"{PREFIX}images/ISIC_0024310.jpg"
    mask_path = f"{PREFIX}mask/ISIC_0024310_segmentation.png"
    out_lbl = os.path.join(OUT_DIR, "scratch_val_isic.txt")
    out_mask = os.path.join(OUT_DIR, "scratch_val_isic_mask.png")
    out_vis = os.path.join(OUT_DIR, "scratch_val_isic_vis.png")

    # 1. Mask to YOLO
    converter_yolo = MaskToYoloConverter(target_size=(640, 640), bbox_type="standard")
    success = converter_yolo.convert_single(img_path, mask_path, out_lbl, class_id=4)
    if not success:
        logger.error("ISIC: Failed mask-to-yolo conversion.")
        return False

    # Assert file exists
    if not os.path.exists(out_lbl):
        logger.error(f"ISIC: Label file {out_lbl} was not created.")
        return False

    # Check contents
    with open(out_lbl, "r") as f:
        content = f.read().strip()
        logger.info(f"Generated ISIC YOLO label content: '{content}'")
        parts = content.split()
        if len(parts) != 5:
            logger.error("ISIC: Label format is incorrect.")
            return False
        class_id, x, y, w, h = parts
        if class_id != "4":
            logger.error(f"ISIC: Expected class_id 4, got {class_id}")
            return False

    # 2. YOLO to Mask
    converter_mask = YoloToMaskConverter(target_size=(640, 640))
    success = converter_mask.convert_single(out_lbl, out_mask)
    if not success:
        logger.error("ISIC: Failed yolo-to-mask conversion.")
        return False

    if not os.path.exists(out_mask):
        logger.error(f"ISIC: Output mask file {out_mask} was not created.")
        return False

    # 3. Visualize
    success = converter_mask.visualize_label(img_path, out_lbl, out_vis)
    if not success:
        logger.error("ISIC: Failed visualization generation.")
        return False

    if not os.path.exists(out_vis):
        logger.error(f"ISIC: Visualization file {out_vis} was not created.")
        return False

    logger.info("✅ ISIC validation successful!")
    return True


def validate_isic_json():
    logger.info("========================================")
    logger.info("Validating JSON Ground Truth Parsing")
    logger.info("========================================")

    # Prepare temp JSON ground truth file (Format B: nested indicator structure)
    json_path = os.path.join(OUT_DIR, "scratch_val_truth.json")
    dummy_data = {
        "ISIC_0024310": {
            "MEL": 1,
            "NV": 0,
            "BCC": 0
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    # Let's verify batch convert using this JSON file
    # We will copy ISIC_0024310.jpg and mask to a scratch directory to run batch mode
    scratch_images = os.path.join(OUT_DIR, "scratch_test_images")
    scratch_masks = os.path.join(OUT_DIR, "scratch_test_masks")
    scratch_labels = os.path.join(OUT_DIR, "scratch_test_labels")

    os.makedirs(scratch_images, exist_ok=True)
    os.makedirs(scratch_masks, exist_ok=True)

    shutil.copy2(f"{PREFIX}images/ISIC_0024310.jpg", os.path.join(scratch_images, "ISIC_0024310.jpg"))
    shutil.copy2(f"{PREFIX}mask/ISIC_0024310_segmentation.png", os.path.join(scratch_masks, "ISIC_0024310_segmentation.png"))

    converter = MaskToYoloConverter()
    converted_count = converter.convert_dataset(
        images_dir=scratch_images,
        masks_dir=scratch_masks,
        output_labels_dir=scratch_labels,
        ground_truth=json_path
    )

    if converted_count != 1:
        logger.error(f"JSON validation failed: expected 1 file converted, got {converted_count}")
        return False

    label_file = os.path.join(scratch_labels, "ISIC_0024310.txt")
    if not os.path.exists(label_file):
        logger.error("JSON validation failed: label file was not created.")
        return False

    with open(label_file, "r") as f:
        content = f.read().strip()
        parts = content.split()
        if not parts or parts[0] != "4":  # MEL maps to 4
            logger.error(f"JSON validation failed: class_id is incorrect. Content: '{content}'")
            return False

    # Clean up scratch files
    shutil.rmtree(scratch_images, ignore_errors=True)
    shutil.rmtree(scratch_masks, ignore_errors=True)
    shutil.rmtree(scratch_labels, ignore_errors=True)
    if os.path.exists(json_path):
        os.remove(json_path)

    logger.info("✅ JSON ground truth parsing validation successful!")
    return True


def validate_plant():
    logger.info("========================================")
    logger.info("Validating on Plant Dataset")
    logger.info("========================================")

    img_path = f"{PREFIX}images_plant/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL-0555_JPG.rf.c888c66fe2fef4355b3f2acc8a381ae0.jpg"
    mask_path = f"{PREFIX}masks_plant/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL-0555_JPG.rf.c888c66fe2fef4355b3f2acc8a381ae0.png"
    out_lbl = os.path.join(OUT_DIR, "scratch_val_plant.txt")
    out_mask = os.path.join(OUT_DIR, "scratch_val_plant_mask.png")
    out_vis = os.path.join(OUT_DIR, "scratch_val_plant_vis.png")

    # 1. Mask to YOLO
    converter_yolo = MaskToYoloConverter(target_size=(640, 640), bbox_type="standard")
    success = converter_yolo.convert_single(img_path, mask_path, out_lbl, class_id=0)
    if not success:
        logger.error("Plant: Failed mask-to-yolo conversion.")
        return False

    # Assert file exists
    if not os.path.exists(out_lbl):
        logger.error(f"Plant: Label file {out_lbl} was not created.")
        return False

    # Check contents
    with open(out_lbl, "r") as f:
        content = f.read().strip()
        logger.info(f"Generated Plant YOLO label content: '{content}'")
        parts = content.split()
        if len(parts) != 5:
            logger.error("Plant: Label format is incorrect.")
            return False
        class_id, x, y, w, h = parts
        if class_id != "0":
            logger.error(f"Plant: Expected class_id 0, got {class_id}")
            return False

    # 2. YOLO to Mask
    converter_mask = YoloToMaskConverter(target_size=(640, 640))
    success = converter_mask.convert_single(out_lbl, out_mask)
    if not success:
        logger.error("Plant: Failed yolo-to-mask conversion.")
        return False

    if not os.path.exists(out_mask):
        logger.error(f"Plant: Output mask file {out_mask} was not created.")
        return False

    # 3. Visualize
    success = converter_mask.visualize_label(img_path, out_lbl, out_vis)
    if not success:
        logger.error("Plant: Failed visualization generation.")
        return False

    if not os.path.exists(out_vis):
        logger.error(f"Plant: Visualization file {out_vis} was not created.")
        return False

    logger.info("✅ Plant validation successful!")
    return True


if __name__ == "__main__":
    isic_ok = validate_isic()
    json_ok = validate_isic_json()
    plant_ok = validate_plant()

    if isic_ok and json_ok and plant_ok:
        logger.info("\n🎉 All validation tests (including JSON parsing) passed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Validation failed.")
        sys.exit(1)
