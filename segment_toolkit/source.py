"""
Core conversion classes for segment_toolkit.
Includes MaskToYoloConverter and YoloToMaskConverter.
"""

import os
import shutil
import random
import logging
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from .helpers import (
    safe_read_image,
    preprocess_image,
    get_largest_contour,
    calculate_bbox,
    normalize_coordinates,
    denormalize_coordinates,
    draw_bbox_on_image,
)

logger = logging.getLogger(__name__)


class MaskToYoloConverter:
    """
    Converts binary segmentation masks into YOLO bounding box labels.
    Supports single file conversion, full dataset directory batch conversion,
    ground truth CSV classification mapping, and train/test dataset splitting.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640), bbox_type: str = "standard"):
        """
        Initialize the converter.

        Args:
            target_size: Dimensions (width, height) to resize images and masks to.
            bbox_type: Bounding box style, either 'standard' or 'rotated'.
        """
        self.target_size = target_size
        self.bbox_type = bbox_type

    def convert_single(
        self, image_path: str, mask_path: str, output_txt_path: str, class_id: int = 0
    ) -> bool:
        """
        Convert a single image and mask pair into a YOLO bounding box label file.

        Args:
            image_path: Path to the input image.
            mask_path: Path to the input binary mask.
            output_txt_path: Path to save the output YOLO text file.
            class_id: The class ID to write to the label file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Read image to obtain original shape
            image = safe_read_image(image_path)
            orig_h, orig_w = image.shape[:2]

            # Read and process mask
            mask = safe_read_image(mask_path)
            
            # Preprocess image and mask to the target size
            resized_image = preprocess_image(image, self.target_size, is_mask=False)
            resized_mask = preprocess_image(mask, self.target_size, is_mask=True)

            # Get contour from the resized mask
            contour = get_largest_contour(resized_mask)
            if contour is None:
                logger.warning(f"No contour/objects found in mask: {mask_path}")
                return False

            # Calculate bbox on resized coordinates
            x_center, y_center, w, h = calculate_bbox(contour, bbox_type=self.bbox_type)

            # Normalize coordinates using the target dimensions
            x_norm, y_norm, w_norm, h_norm = normalize_coordinates(
                x_center, y_center, w, h, self.target_size[0], self.target_size[1]
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)

            # Write to YOLO label format (class_id center_x center_y width height)
            # Use 'w' to overwrite or write fresh, or 'a+' depending on use case.
            # Here we write single bounding box per file, so 'w' is appropriate.
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            logger.info(f"Successfully generated label: {output_txt_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert mask {mask_path} to YOLO label: {str(e)}")
            return False

    def _load_ground_truth(self, file_path: str) -> Dict[str, int]:
        """
        Parses a CSV or JSON file and maps image filename/ID to class ID (int).
        """
        import json
        class_lookup = {}
        isic_mapping = {
            "AKIEC": 0,
            "BCC": 1,
            "BKL": 2,
            "DF": 3,
            "MEL": 4,
            "NV": 5,
            "VASC": 6,
        }

        # Helper to extract class ID from nested indicators
        def get_class_id_from_indicators(indicators: dict) -> int:
            for col, val in indicators.items():
                if val == 1 or val == 1.0 or str(val).strip() == "1":
                    if col in isic_mapping:
                        return isic_mapping[col]
            for key in ["class_id", "class", "label"]:
                if key in indicators:
                    v = indicators[key]
                    if isinstance(v, int):
                        return v
                    elif str(v).isdigit():
                        return int(v)
                    elif str(v) in isic_mapping:
                        return isic_mapping[str(v)]
            return 0

        if file_path.endswith(".json"):
            logger.info(f"Parsing ground truth metadata JSON: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for img_id, val in data.items():
                    img_id = str(img_id).strip()
                    if isinstance(val, dict):
                        class_lookup[img_id] = get_class_id_from_indicators(val)
                    elif isinstance(val, int):
                        class_lookup[img_id] = val
                    elif str(val).isdigit():
                        class_lookup[img_id] = int(val)
                    elif str(val) in isic_mapping:
                        class_lookup[img_id] = isic_mapping[str(val)]
                    else:
                        class_lookup[img_id] = 0
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    keys = list(item.keys())
                    if not keys:
                        continue
                    id_key = keys[0]
                    for k in keys:
                        if any(x in k.lower() for x in ["image", "file", "name", "id"]):
                            id_key = k
                            break
                    img_id = str(item[id_key]).strip()
                    remaining = {k: v for k, v in item.items() if k != id_key}
                    class_lookup[img_id] = get_class_id_from_indicators(remaining)
        else:
            logger.info(f"Parsing ground truth metadata CSV: {file_path}")
            df = pd.read_csv(file_path)
            id_col = df.columns[0]
            class_cols = list(df.columns[1:])

            col_to_class = {}
            for col in class_cols:
                if col in isic_mapping:
                    col_to_class[col] = isic_mapping[col]
                else:
                    col_to_class[col] = len(col_to_class)

            for _, row in df.iterrows():
                img_id = str(row[id_col]).strip()
                for col in class_cols:
                    if row[col] == 1 or row[col] == 1.0 or str(row[col]).strip() == "1":
                        class_lookup[img_id] = col_to_class[col]
                        break
        return class_lookup

    def convert_dataset(
        self,
        images_dir: str,
        masks_dir: str,
        output_labels_dir: str,
        default_class_id: int = 0,
        ground_truth: Optional[str] = None,
    ) -> int:
        """
        Batch convert a folder of masks into YOLO labels.

        Args:
            images_dir: Directory containing original images.
            masks_dir: Directory containing binary masks.
            output_labels_dir: Directory where YOLO labels will be written.
            default_class_id: Class ID to use if no ground truth metadata is provided or matches.
            ground_truth: Optional path to GroundTruth CSV or JSON mapping filenames to classes.

        Returns:
            int: Number of successfully generated label files.
        """
        try:
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(masks_dir):
                raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

            os.makedirs(output_labels_dir, exist_ok=True)

            # Parse Ground Truth CSV or JSON if provided
            class_lookup = {}
            if ground_truth:
                class_lookup = self._load_ground_truth(ground_truth)

            # Process files
            success_count = 0
            image_files = [
                f
                for f in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            logger.info(f"Found {len(image_files)} images to convert.")

            for img_file in image_files:
                base_name, ext = os.path.splitext(img_file)

                # Attempt to find the mask file with common extension variants
                mask_file = None
                for mask_ext in [".png", ".jpg", ".jpeg", "_segmentation.png", "_mask.png"]:
                    # Try matching base_name directly or with suffixes
                    candidate1 = os.path.join(masks_dir, f"{base_name}{mask_ext}")
                    candidate2 = os.path.join(masks_dir, f"{base_name}_segmentation{mask_ext}")
                    candidate3 = os.path.join(masks_dir, f"{base_name.replace('_segmentation', '')}{mask_ext}")
                    
                    if os.path.exists(candidate1):
                        mask_file = candidate1
                        break
                    elif os.path.exists(candidate2):
                        mask_file = candidate2
                        break
                    elif os.path.exists(candidate3):
                        mask_file = candidate3
                        break

                # If no direct match, check if image_file base name is related to any mask files
                if not mask_file:
                    # Search directory
                    for m in os.listdir(masks_dir):
                        if m.startswith(base_name) and m.lower().endswith((".png", ".jpg", ".jpeg")):
                            mask_file = os.path.join(masks_dir, m)
                            break

                if not mask_file:
                    logger.warning(f"No matching mask found for image: {img_file}")
                    continue

                # Determine Class ID
                class_id = default_class_id
                if ground_truth:
                    # Match exact base name or try parts of the prefix
                    if base_name in class_lookup:
                        class_id = class_lookup[base_name]
                    else:
                        # Try prefix mapping (e.g. if CSV/JSON has ISIC_0024310 but filename is ISIC_0024310.jpg)
                        matched = False
                        for k, v in class_lookup.items():
                            if base_name.startswith(k) or k.startswith(base_name):
                                class_id = v
                                matched = True
                                break
                        if not matched:
                            logger.debug(f"Metadata not found in CSV/JSON for {base_name}. Using default Class ID {default_class_id}")

                output_txt = os.path.join(output_labels_dir, f"{base_name}.txt")
                if self.convert_single(
                    os.path.join(images_dir, img_file), mask_file, output_txt, class_id=class_id
                ):
                    success_count += 1

            logger.info(f"Dataset conversion completed. Generated {success_count} label files.")
            return success_count

        except Exception as e:
            logger.error(f"Error during dataset batch conversion: {str(e)}")
            raise

    def split_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_dataset_dir: str,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> Dict[str, int]:
        """
        Splits images and labels into train/test splits under the output directory.
        Generates standard YOLO data.yaml configuration.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing generated text labels.
            output_dataset_dir: Root output directory for structured dataset.
            split_ratio: Percentage of data allocated for training (default: 0.8).
            seed: Random seed for reproducibility.

        Returns:
            Dict[str, int]: Dictionary with counts of train and test images.
        """
        try:
            random.seed(seed)

            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

            # Collect valid image files that have matching label files
            valid_pairs = []
            image_files = os.listdir(images_dir)
            for img_file in image_files:
                base_name, _ = os.path.splitext(img_file)
                label_file = f"{base_name}.txt"
                if os.path.exists(os.path.join(labels_dir, label_file)):
                    valid_pairs.append((img_file, label_file))
                else:
                    logger.warning(f"Skipping split for {img_file}: label file {label_file} not found.")

            if not valid_pairs:
                raise ValueError("No matching image and label file pairs found to split.")

            random.shuffle(valid_pairs)
            split_idx = int(len(valid_pairs) * split_ratio)
            train_pairs = valid_pairs[:split_idx]
            test_pairs = valid_pairs[split_idx:]

            # Establish folders
            for split in ["train", "test"]:
                os.makedirs(os.path.join(output_dataset_dir, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(output_dataset_dir, split, "labels"), exist_ok=True)

            def copy_pairs(pairs: List[Tuple[str, str]], split: str):
                for img, lbl in pairs:
                    shutil.copy2(os.path.join(images_dir, img), os.path.join(output_dataset_dir, split, "images", img))
                    shutil.copy2(os.path.join(labels_dir, lbl), os.path.join(output_dataset_dir, split, "labels", lbl))

            copy_pairs(train_pairs, "train")
            copy_pairs(test_pairs, "test")

            # Collect unique classes to build configuration yaml
            # In YOLO format, we can read classes to set nc and names automatically
            unique_classes = set()
            for _, lbl in valid_pairs:
                try:
                    with open(os.path.join(labels_dir, lbl), "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                unique_classes.add(int(parts[0]))
                except Exception:
                    pass

            num_classes = max(unique_classes) + 1 if unique_classes else 1
            # Standard class names
            class_names = [f"class_{i}" for i in range(num_classes)]
            # If standard ISIC classes, rename them
            if num_classes == 7:
                class_names = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

            yaml_path = os.path.join(output_dataset_dir, "data.yaml")
            with open(yaml_path, "w", encoding="utf-8") as yaml_file:
                yaml_file.write(f"train: ./train/images\n")
                yaml_file.write(f"val: ./test/images\n")
                yaml_file.write(f"test: ./test/images\n\n")
                yaml_file.write(f"nc: {num_classes}\n")
                yaml_file.write(f"names: {class_names}\n")

            logger.info(f"Splitting done. Train: {len(train_pairs)}, Test: {len(test_pairs)}.")
            return {"train": len(train_pairs), "test": len(test_pairs)}

        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise


class YoloToMaskConverter:
    """
    Converts YOLO format bounding box label files back into binary segmentation masks.
    Supports single file generation, batch directory conversion, and bbox overlay visualization.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the converter.

        Args:
            target_size: Target mask dimensions (width, height) to generate.
        """
        self.target_size = target_size

    def convert_single(self, label_path: str, output_mask_path: str) -> bool:
        """
        Create a binary mask image from a YOLO label file.

        Args:
            label_path: Path to the input YOLO label text file.
            output_mask_path: Path to save the generated PNG mask.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")

            # Create empty black canvas mask
            mask = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
            img_w, img_h = self.target_size

            has_drawn = False
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(f"Malformed label line in {label_path}: '{line}'")
                        continue

                    # Parse values
                    class_id = int(parts[0])
                    x_norm, y_norm, w_norm, h_norm = map(float, parts[1:])

                    # Denormalize coordinates to top-left and bottom-right corners
                    x1, y1, x2, y2 = denormalize_coordinates(
                        x_norm, y_norm, w_norm, h_norm, img_w, img_h
                    )

                    # Draw solid rectangle on mask canvas
                    if x1 < x2 and y1 < y2:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
                        has_drawn = True
                    else:
                        logger.warning(f"Invalid bounding box in {label_path}: ({x1}, {y1}) to ({x2}, {y2})")

            # Save mask file
            os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)), exist_ok=True)
            mask_img = Image.fromarray(mask)
            mask_img.save(output_mask_path)

            if not has_drawn:
                logger.warning(f"No bounding boxes were drawn to mask: {output_mask_path}")

            logger.info(f"Successfully generated mask: {output_mask_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert label {label_path} to mask: {str(e)}")
            return False

    def convert_dataset(self, labels_dir: str, output_masks_dir: str) -> int:
        """
        Batch convert a folder of YOLO label text files into binary mask images.

        Args:
            labels_dir: Directory containing YOLO text label files.
            output_masks_dir: Directory where binary masks will be saved.

        Returns:
            int: Number of masks successfully converted.
        """
        try:
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

            os.makedirs(output_masks_dir, exist_ok=True)

            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
            success_count = 0

            for lbl in label_files:
                base_name = os.path.splitext(lbl)[0]
                output_mask = os.path.join(output_masks_dir, f"{base_name}.png")
                if self.convert_single(os.path.join(labels_dir, lbl), output_mask):
                    success_count += 1

            logger.info(f"Dataset conversion completed. Generated {success_count} masks.")
            return success_count

        except Exception as e:
            logger.error(f"Error during label dataset conversion: {str(e)}")
            raise

    def visualize_label(self, image_path: str, label_path: str, output_image_path: str) -> bool:
        """
        Overlays bounding box coordinates from a YOLO label onto the original image.

        Args:
            image_path: Path to input image.
            label_path: Path to YOLO label file.
            output_image_path: Path to save the visualized image output.

        Returns:
            bool: True if success, False otherwise.
        """
        try:
            image = safe_read_image(image_path)
            # Standardize image size for visualization matching target size
            resized_image = preprocess_image(image, self.target_size, is_mask=False)
            img_w, img_h = self.target_size

            if not os.path.exists(label_path):
                logger.warning(f"Label file not found for visualization: {label_path}")
                return False

            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_norm, y_norm, w_norm, h_norm = map(float, parts[1:])

                    # Convert coordinates
                    x1, y1, x2, y2 = denormalize_coordinates(
                        x_norm, y_norm, w_norm, h_norm, img_w, img_h
                    )

                    # Draw bounding box
                    cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # Label background
                    label_text = f"Class {class_id}"
                    cv2.putText(
                        resized_image,
                        label_text,
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

            os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
            cv2.imwrite(output_image_path, resized_image)
            logger.info(f"Saved visualization overlay to: {output_image_path}")
            return True

        except Exception as e:
            logger.error(f"Error visualizing label {label_path}: {str(e)}")
            return False
