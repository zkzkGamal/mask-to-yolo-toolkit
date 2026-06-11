"""
This module provides the MaskToYoloConverter class, which converts binary segmentation masks to YOLO bounding box coordinates.
It supports standard axis-aligned bounding boxes, minimum-area rotated bounding boxes, dynamic class mapping using ground truth metadata, and dataset splitting.
"""
import os
import random
import shutil
import pandas as pd
import json
from typing import Tuple, Optional, Dict, List
from .base_converter import BaseConverter
from ..utils.io import safe_read_image
from ..utils.helpers import (
    get_largest_contour,
    calculate_bbox,
    normalize_coordinates,
    ensure_dir,
)
from ..transforms.geometric import Resize
from ..utils.logger import logger

class MaskToYoloConverter(BaseConverter):
    """
    Converts binary segmentation masks into YOLO bounding box labels.
    Supports single file conversion, full dataset directory batch conversion,
    ground truth CSV classification mapping, and train/test dataset splitting.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640), bbox_type: str = "standard"):
        self.target_size = target_size
        self.bbox_type = bbox_type
        self.resize_transform = Resize(target_size)

    def convert_single(
        self, image_path: str, mask_path: str, output_txt_path: str, class_id: int = 0
    ) -> bool:
        try:
            # Read image to obtain original shape
            image = safe_read_image(image_path)

            # Read and process mask
            mask = safe_read_image(mask_path)
            
            # Preprocess image and mask to the target size
            resized_image, resized_mask = self.resize_transform(image, mask)

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
            ensure_dir(os.path.dirname(os.path.abspath(output_txt_path)))

            # Write to YOLO label format
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            logger.info(f"Successfully generated label: {output_txt_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert mask {mask_path} to YOLO label: {str(e)}")
            return False

    def _load_ground_truth(self, file_path: str) -> Dict[str, int]:
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
        try:
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(masks_dir):
                raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

            ensure_dir(output_labels_dir)

            class_lookup = {}
            if ground_truth:
                class_lookup = self._load_ground_truth(ground_truth)

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

                # Find matching mask
                mask_file = None
                for mask_ext in [".png", ".jpg", ".jpeg", "_segmentation.png", "_mask.png"]:
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

                if not mask_file:
                    for m in os.listdir(masks_dir):
                        if m.startswith(base_name) and m.lower().endswith((".png", ".jpg", ".jpeg")):
                            mask_file = os.path.join(masks_dir, m)
                            break

                if not mask_file:
                    logger.warning(f"No matching mask found for image: {img_file}")
                    continue

                class_id = default_class_id
                if ground_truth:
                    if base_name in class_lookup:
                        class_id = class_lookup[base_name]
                    else:
                        matched = False
                        for k, v in class_lookup.items():
                            if base_name.startswith(k) or k.startswith(base_name):
                                class_id = v
                                matched = True
                                break

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
        try:
            random.seed(seed)

            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

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

            for split in ["train", "test"]:
                ensure_dir(os.path.join(output_dataset_dir, split, "images"))
                ensure_dir(os.path.join(output_dataset_dir, split, "labels"))

            def copy_pairs(pairs: List[Tuple[str, str]], split: str):
                for img, lbl in pairs:
                    shutil.copy2(os.path.join(images_dir, img), os.path.join(output_dataset_dir, split, "images", img))
                    shutil.copy2(os.path.join(labels_dir, lbl), os.path.join(output_dataset_dir, split, "labels", lbl))

            copy_pairs(train_pairs, "train")
            copy_pairs(test_pairs, "test")

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
            class_names = [f"class_{i}" for i in range(num_classes)]
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
