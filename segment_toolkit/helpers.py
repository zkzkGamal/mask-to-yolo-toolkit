"""
Helper utilities for segment_toolkit.
Provides image preprocessing, contour extraction, bounding box computation,
coordinate normalization, and visualization helpers.
"""

import os
from typing import Tuple, Optional, Union
import cv2
import numpy as np


def safe_read_image(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Safely reads an image using OpenCV and handles errors.

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: The loaded image array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid image.
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


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """
    Resizes image/mask to target size.
    Uses linear interpolation for images and nearest-neighbor for masks to keep values binary.

    Args:
        image: Input image array.
        target_size: Target (width, height).
        is_mask: Flag indicating if input is a binary mask.

    Returns:
        np.ndarray: The preprocessed image/mask.
    """
    try:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(image, target_size, interpolation=interpolation)
        return resized
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess/resize image: {str(e)}") from e


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Finds the contours of the binary mask and returns the largest one by area.

    Args:
        mask: Grayscale binary mask array.

    Returns:
        Optional[np.ndarray]: The largest contour array, or None if no contours are found.
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

    Args:
        contour: Contour points array.
        bbox_type: Type of bounding box, either 'standard' or 'rotated'.

    Returns:
        Tuple[float, float, float, float]: (x_center, y_center, w, h) in pixel coordinates.
    """
    try:
        if bbox_type == "rotated":
            # minAreaRect returns ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(contour)
            (x_center, y_center), (w, h), _ = rect
            return float(x_center), float(y_center), float(w), float(h)
        else:
            # boundingRect returns (x_min, y_min, w, h)
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

    Args:
        x: Center x pixel.
        y: Center y pixel.
        w: Width pixel.
        h: Height pixel.
        img_width: Image width.
        img_height: Image height.

    Returns:
        Tuple[float, float, float, float]: (x_norm, y_norm, w_norm, h_norm).
    """
    x_norm = x / img_width
    y_norm = y / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clamp to [0, 1] to stay within image bounds
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

    Args:
        x_norm: Normalized center x.
        y_norm: Normalized center y.
        w_norm: Normalized width.
        h_norm: Normalized height.
        img_width: Target image width.
        img_height: Target image height.

    Returns:
        Tuple[int, int, int, int]: (x1, y1, x2, y2) pixel corners.
    """
    x_center = x_norm * img_width
    y_center = y_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    # Clamp coordinates to image boundaries
    x1 = max(0, min(img_width - 1, x1))
    y1 = max(0, min(img_height - 1, y1))
    x2 = max(0, min(img_width - 1, x2))
    y2 = max(0, min(img_height - 1, y2))

    return x1, y1, x2, y2


def draw_bbox_on_image(
    image: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 2
) -> np.ndarray:
    """
    Draws a bounding box on the image.

    Args:
        image: Image array.
        bbox: (x1, y1, x2, y2) pixel coordinates.
        color: Box color in BGR/RGB (default: yellow).
        thickness: Box border thickness.

    Returns:
        np.ndarray: Image with box drawn.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

