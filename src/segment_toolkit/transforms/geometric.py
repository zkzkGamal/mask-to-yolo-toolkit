"""
This module provides geometric image and mask transformations, specifically the Resize transform, which maintains mask border sharpness using nearest-neighbor interpolation.
"""
import cv2
import numpy as np
from typing import Tuple, Union
from .base import BaseTransform

class Resize(BaseTransform):
    """
    Resizes image and optional mask.
    Uses linear interpolation for images, nearest-neighbor for masks to keep mask borders sharp.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            resized_mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            return resized_image, resized_mask
        return resized_image
