"""
This module provides intensity-based pixel transformations, specifically the Normalize transform, which scales pixel values of images.
"""
import numpy as np
from typing import Union, Tuple
from .base import BaseTransform

class Normalize(BaseTransform):
    """
    Normalizes image pixel intensities to [0, 1]. Does not modify binary masks.
    """
    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        norm_img = image.astype(np.float32) / 255.0
        if mask is not None:
            return norm_img, mask
        return norm_img
