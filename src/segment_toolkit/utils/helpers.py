"""
This module provides helper utilities for geometric computations, contour detection, bounding box calculations, and coordinate normalization/denormalization.
"""
import os
from typing import Tuple, Optional
import cv2
import numpy as np

def ensure_dir(path: str):
    """
    Ensures that a directory exists.
    """
    os.makedirs(os.path.abspath(path), exist_ok=True)

def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Finds the contours of the binary mask and returns the largest one by area.
    """
    try:
        # Ensure mask is grayscale / single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Threshold to ensure binary mask
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Return the contour with maximum area
        return max(contours, key=cv2.contourArea)
    except Exception as e:
        raise RuntimeError(f"Error extracting contour from mask: {str(e)}") from e

def calculate_bbox(contour: np.ndarray, bbox_type: str = "standard") -> Tuple[float, float, float, float]:
    """
    Calculates center-based bounding box from a contour.
    """
    try:
        if bbox_type == "rotated":
            rect = cv2.minAreaRect(contour)
            (x_center, y_center), (w, h), _ = rect
            return float(x_center), float(y_center), float(w), float(h)
        else:
            x_min, y_min, w, h = cv2.boundingRect(contour)
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0
            return float(x_center), float(y_center), float(w), float(h)
    except Exception as e:
        raise RuntimeError(f"Error calculating bounding box: {str(e)}") from e

def normalize_coordinates(
    x: float, y: float, w: float, h: float, img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Normalizes pixel bounding box coordinates to [0, 1] relative to image dimensions.
    """
    x_norm = x / img_width
    y_norm = y / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_norm, y_norm, w_norm, h_norm

def denormalize_coordinates(
    x_norm: float, y_norm: float, w_norm: float, h_norm: float, img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Converts normalized center-based coordinates to pixel-space top-left and bottom-right corners.
    """
    x_center = x_norm * img_width
    y_center = y_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    x1 = max(0, min(img_width - 1, x1))
    y1 = max(0, min(img_height - 1, y1))
    x2 = max(0, min(img_width - 1, x2))
    y2 = max(0, min(img_height - 1, y2))

    return x1, y1, x2, y2
