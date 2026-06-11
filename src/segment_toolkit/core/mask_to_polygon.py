"""
This module provides the MaskToPolygonConverter class, which converts segmentation masks (binary or multi-class color-coded) to YOLO polygon format annotations.
It extracts contours, normalizes coordinates to polygon points, and maps colors to class IDs.
"""
import os
import random
import shutil
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from .base_converter import BaseConverter
from ..utils.io import safe_read_image
from ..utils.helpers import ensure_dir
from ..utils.logger import logger
from ..transforms.geometric import Resize

class MaskToPolygonConverter(BaseConverter):
    """
    Converts segmentation masks (binary or multi-class RGB) into YOLO polygon label files.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.resize_transform = Resize(target_size)

    def convert_single(
        self,
        image_path: str,
        mask_path: str,
        output_txt_path: str,
        class_id: int = 0,
        classes: Optional[List[Tuple[Tuple[int, int, int], str]]] = None,
    ) -> bool:
        try:
            # Read image to obtain shape
            image = safe_read_image(image_path)
            orig_h, orig_w = image.shape[:2]

            polygons = []

            if classes:
                # Multi-class colored mask mode
                mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if mask is None:
                    raise ValueError(f"Unable to read mask image: {mask_path}")
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                H, W, _ = mask.shape
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # Find unique RGB tuples efficiently
                unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
                unique_rgb_tuples = [tuple(c) for c in unique_colors]

                # Filter out tuples not present in classes list
                valid_rgb_tuples = [rgb_tuple for rgb_tuple in unique_rgb_tuples if rgb_tuple in [c[0] for c in classes]]

                for rgb_tuple in valid_rgb_tuples:
                    class_name = [c[1] for c in classes if c[0] == rgb_tuple][0]
                    lower_bound = np.array([rgb_tuple[2], rgb_tuple[1], rgb_tuple[0]], dtype=np.uint8)
                    upper_bound = np.array([rgb_tuple[2], rgb_tuple[1], rgb_tuple[0]], dtype=np.uint8)
                    mask_binary = cv2.inRange(mask, lower_bound, upper_bound)
                    mask_contour, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in mask_contour:
                        if cv2.contourArea(contour) > 200:
                            polygon = []
                            for point in contour:
                                x, y = point[0]
                                polygon.append(x / W)
                                polygon.append(y / H)
                            polygons.append((polygon, class_name))
            else:
                # Grayscale binary mask mode
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Unable to read mask image: {mask_path}")
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                H, W = mask.shape
                _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask_contour, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in mask_contour:
                    if cv2.contourArea(contour) > 200:
                        polygon = []
                        for point in contour:
                            x, y = point[0]
                            polygon.append(x / W)
                            polygon.append(y / H)
                        polygons.append((polygon, class_id))

            if polygons:
                ensure_dir(os.path.dirname(os.path.abspath(output_txt_path)))
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    for polygon, cls in polygons:
                        if classes:
                            class_number = [c[1] for c in classes].index(cls)
                        else:
                            class_number = cls
                        f.write(f"{class_number} " + " ".join(f"{p:.6f}" for p in polygon) + "\n")
                logger.info(f"Successfully generated polygon label: {output_txt_path}")
                return True
            else:
                logger.warning(f"No polygons/contours found in mask: {mask_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to convert mask {mask_path} to polygon label: {str(e)}")
            return False

    def convert_dataset(
        self,
        images_dir: str,
        masks_dir: str,
        output_labels_dir: str,
        default_class_id: int = 0,
        classes: Optional[List[Tuple[Tuple[int, int, int], str]]] = None,
    ) -> int:
        try:
            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(masks_dir):
                raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

            ensure_dir(output_labels_dir)

            success_count = 0
            image_files = [
                f
                for f in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            logger.info(f"Found {len(image_files)} images to convert to polygons.")

            for img_file in image_files:
                base_name, ext = os.path.splitext(img_file)

                # Find mask
                mask_file = None
                for mask_ext in [".png", ".jpg", ".jpeg", "_segmentation.png", "_mask.png"]:
                    candidate1 = os.path.join(masks_dir, f"{base_name}{mask_ext}")
                    candidate2 = os.path.join(masks_dir, f"{base_name}_segmentation{mask_ext}")
                    if os.path.exists(candidate1):
                        mask_file = candidate1
                        break
                    elif os.path.exists(candidate2):
                        mask_file = candidate2
                        break

                if not mask_file:
                    for m in os.listdir(masks_dir):
                        if m.startswith(base_name) and m.lower().endswith((".png", ".jpg", ".jpeg")):
                            mask_file = os.path.join(masks_dir, m)
                            break

                if not mask_file:
                    logger.warning(f"No matching mask found for image: {img_file}")
                    continue

                output_txt = os.path.join(output_labels_dir, f"{base_name}.txt")
                if self.convert_single(
                    os.path.join(images_dir, img_file), mask_file, output_txt, class_id=default_class_id, classes=classes
                ):
                    success_count += 1

            logger.info(f"Dataset conversion completed. Generated {success_count} polygon label files.")
            return success_count

        except Exception as e:
            logger.error(f"Error during polygon dataset batch conversion: {str(e)}")
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
