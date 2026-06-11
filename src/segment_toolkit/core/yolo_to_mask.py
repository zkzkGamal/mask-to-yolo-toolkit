"""
This module provides the YoloToMaskConverter class, which converts YOLO format bounding box annotations back to binary segmentation masks.
It supports single file conversions, batch processing of directories, and visualizing bounding box overlays on source images.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from .base_converter import BaseConverter
from ..utils.io import safe_read_image
from ..utils.helpers import denormalize_coordinates, ensure_dir
from ..transforms.geometric import Resize
from ..utils.logger import logger

class YoloToMaskConverter(BaseConverter):
    """
    Converts YOLO format bounding box label files back into binary segmentation masks.
    Supports single file generation, batch directory conversion, bbox overlay visualization,
    and mask overlay visualization.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.resize_transform = Resize(target_size)

    def convert_single(self, label_path: str, output_mask_path: str) -> bool:
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
            ensure_dir(os.path.dirname(os.path.abspath(output_mask_path)))
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
        try:
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

            ensure_dir(output_masks_dir)

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
        try:
            image = safe_read_image(image_path)
            # Standardize image size for visualization matching target size
            resized_image = self.resize_transform(image)
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

            ensure_dir(os.path.dirname(os.path.abspath(output_image_path)))
            cv2.imwrite(output_image_path, resized_image)
            logger.info(f"Saved visualization overlay to: {output_image_path}")
            return True

        except Exception as e:
            logger.error(f"Error visualizing label {label_path}: {str(e)}")
            return False

    def visualize_mask(self, image_path: str, mask_path: str, output_image_path: str) -> bool:
        try:
            image = safe_read_image(image_path)
            resized_image = self.resize_transform(image)

            if not os.path.exists(mask_path):
                logger.warning(f"Mask file not found for visualization: {mask_path}")
                return False

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Unable to read mask image: {mask_path}")

            resized_mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST,
            )

            mask_binary = resized_mask > 0
            overlay = resized_image.copy()
            overlay[mask_binary] = (0, 0, 255)
            blended = cv2.addWeighted(overlay, 0.4, resized_image, 0.6, 0)

            ensure_dir(os.path.dirname(os.path.abspath(output_image_path)))
            cv2.imwrite(output_image_path, blended)
            logger.info(f"Saved mask visualization overlay to: {output_image_path}")
            return True

        except Exception as e:
            logger.error(f"Error visualizing mask {mask_path}: {str(e)}")
            return False
