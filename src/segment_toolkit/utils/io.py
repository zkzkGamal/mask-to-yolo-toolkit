"""
This module provides safe input/output utilities for reading images and handling directories.
"""
import os
import cv2
import numpy as np
from typing import Union
from .logger import logger

def safe_read_image(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Safely reads an image using OpenCV and handles errors.
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image file not found: {path_str}")

    try:
        image = cv2.imread(path_str)
        if image is None:
            raise ValueError(f"Failed to decode image: {path_str}")
        return image
    except Exception as e:
        raise ValueError(f"Error reading image {path_str}: {str(e)}") from e
