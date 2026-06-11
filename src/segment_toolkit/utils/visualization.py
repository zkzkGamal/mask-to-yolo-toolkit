"""
This module provides visualization utilities to draw annotations, overlays, and labels onto source images for validation.
"""
import cv2
import numpy as np
from typing import Tuple

def draw_bbox_on_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a bounding box on the image (BGR format).
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image
