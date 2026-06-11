"""
This module provides the PolygonToMaskConverter class, which converts YOLO polygon annotations back to binary/colored segmentation masks.
It parses polygon points, reconstructs mask shapes, and renders overlay visualizations.
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from .base_converter import BaseConverter
from ..utils.io import safe_read_image
from ..utils.helpers import ensure_dir
from ..utils.logger import logger
from ..transforms.geometric import Resize

class PolygonToMaskConverter(BaseConverter):
    """
    Converts YOLO segmentation polygon label files back into binary segmentation masks.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.resize_transform = Resize(target_size)

    def convert_single(
        self,
        label_path: str,
        output_mask_path: str,
        classes: Optional[List[Tuple[Tuple[int, int, int], str]]] = None,
    ) -> bool:
        try:
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")

            # Prepare canvas
            W, H = self.target_size
            if classes:
                mask = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                mask = np.zeros((H, W), dtype=np.uint8)

            has_drawn = False

            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(f"Malformed polygon label line in {label_path}: '{line}'")
                        continue

                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # Handle case where line is a standard bounding box (5 elements total)
                    if len(parts) == 5:
                        x_n, y_n, w_n, h_n = coords
                        x1 = x_n - w_n / 2
                        y1 = y_n - h_n / 2
                        x2 = x_n + w_n / 2
                        y2 = y_n + h_n / 2
                        pts = np.array([
                            [int(x1 * W), int(y1 * H)],
                            [int(x2 * W), int(y1 * H)],
                            [int(x2 * W), int(y2 * H)],
                            [int(x1 * W), int(y2 * H)]
                        ], dtype=np.int32)
                    else:
                        # Reconstruct polygon points
                        pts = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * W)
                            py = int(coords[i+1] * H)
                            pts.append([px, py])
                        pts = np.array(pts, dtype=np.int32)

                    # Draw filled polygon
                    if len(pts) >= 3:
                        if classes:
                            if class_id < len(classes):
                                rgb_tuple = classes[class_id][0]
                                color = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0]) # BGR
                            else:
                                color = (255, 255, 255)
                            cv2.fillPoly(mask, [pts], color)
                        else:
                            cv2.fillPoly(mask, [pts], 255)
                        has_drawn = True

            # Save mask file
            ensure_dir(os.path.dirname(os.path.abspath(output_mask_path)))
            mask_img = Image.fromarray(mask)
            mask_img.save(output_mask_path)

            if not has_drawn:
                logger.warning(f"No polygons were drawn to mask: {output_mask_path}")

            logger.info(f"Successfully generated mask from polygon: {output_mask_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert label {label_path} to mask: {str(e)}")
            return False

    def convert_dataset(
        self,
        labels_dir: str,
        output_masks_dir: str,
        classes: Optional[List[Tuple[Tuple[int, int, int], str]]] = None,
    ) -> int:
        try:
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

            ensure_dir(output_masks_dir)

            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
            success_count = 0

            for lbl in label_files:
                base_name = os.path.splitext(lbl)[0]
                output_mask = os.path.join(output_masks_dir, f"{base_name}.png")
                if self.convert_single(os.path.join(labels_dir, lbl), output_mask, classes=classes):
                    success_count += 1

            logger.info(f"Dataset conversion completed. Generated {success_count} masks.")
            return success_count

        except Exception as e:
            logger.error(f"Error during label dataset conversion: {str(e)}")
            raise

    def visualize_polygon(
        self,
        image_path: str,
        label_path: str,
        output_image_path: str,
        classes: Optional[List[Tuple[Tuple[int, int, int], str]]] = None,
    ) -> bool:
        try:
            image = safe_read_image(image_path)
            resized_image = self.resize_transform(image)
            W, H = self.target_size

            if not os.path.exists(label_path):
                logger.warning(f"Label file not found for visualization: {label_path}")
                return False

            overlay = resized_image.copy()

            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # Parse points
                    if len(parts) == 5:
                        x_n, y_n, w_n, h_n = coords
                        x1 = x_n - w_n / 2
                        y1 = y_n - h_n / 2
                        x2 = x_n + w_n / 2
                        y2 = y_n + h_n / 2
                        pts = np.array([
                            [int(x1 * W), int(y1 * H)],
                            [int(x2 * W), int(y1 * H)],
                            [int(x2 * W), int(y2 * H)],
                            [int(x1 * W), int(y2 * H)]
                        ], dtype=np.int32)
                    else:
                        pts = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * W)
                            py = int(coords[i+1] * H)
                            pts.append([px, py])
                        pts = np.array(pts, dtype=np.int32)

                    if len(pts) >= 3:
                        if classes:
                            if class_id < len(classes):
                                rgb_tuple = classes[class_id][0]
                                color = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])
                                class_name = classes[class_id][1]
                            else:
                                color = (0, 255, 255)
                                class_name = f"Class {class_id}"
                        else:
                            color = (0, 255, 255)
                            class_name = f"Class {class_id}"

                        # Draw filled polygon on overlay
                        cv2.fillPoly(overlay, [pts], color)
                        # Draw outline on main image
                        cv2.polylines(resized_image, [pts], isClosed=True, color=color, thickness=2)

                        # Label class name
                        top_pt = pts[np.argmin(pts[:, 1])]
                        tx, ty = int(top_pt[0]), int(top_pt[1])
                        cv2.putText(
                            resized_image,
                            class_name,
                            (tx, max(ty - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            # Blend overlay
            blended = cv2.addWeighted(overlay, 0.4, resized_image, 0.6, 0)

            ensure_dir(os.path.dirname(os.path.abspath(output_image_path)))
            cv2.imwrite(output_image_path, blended)
            logger.info(f"Saved polygon visualization overlay to: {output_image_path}")
            return True

        except Exception as e:
            logger.error(f"Error visualizing polygon {label_path}: {str(e)}")
            return False
