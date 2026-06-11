"""
This module initializes the utilities package of Segment Toolkit, exposing file I/O, directory management, image math, metrics, and visual overlays.
"""
from .logger import logger
from .io import safe_read_image
from .helpers import (
    ensure_dir,
    get_largest_contour,
    calculate_bbox,
    normalize_coordinates,
    denormalize_coordinates,
)
from .visualization import draw_bbox_on_image
from .metrics import compute_iou, compute_dice

__all__ = [
    "logger",
    "safe_read_image",
    "ensure_dir",
    "get_largest_contour",
    "calculate_bbox",
    "normalize_coordinates",
    "denormalize_coordinates",
    "draw_bbox_on_image",
    "compute_iou",
    "compute_dice",
]
