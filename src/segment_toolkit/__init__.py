"""
This package initializes the Segment Toolkit and exposes all primary classes, transforms, and utility functions at the top level.
It automatically checks and ensures that required external dependencies (numpy, opencv-python, pillow, pandas, matplotlib, typer) are installed.
The components exposed in this package include:
- Converters: BaseConverter, MaskToYoloConverter, YoloToMaskConverter, MaskToPolygonConverter, PolygonToMaskConverter
- Transforms: BaseTransform, Compose, Resize, Normalize
- Utilities: logger, safe_read_image, ensure_dir, get_largest_contour, calculate_bbox, normalize_coordinates, denormalize_coordinates, draw_bbox_on_image, compute_iou, compute_dice
"""
from .__version__ import __version__

import sys
import subprocess

def _ensure_dependencies():
    """
    Checks for required external modules and attempts to install them via pip if missing.
    """
    dependencies = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "typer": "typer[all]"
    }
    missing = []
    for module, pip_name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
            
    if missing:
        print(f"[segment_toolkit] Missing required package(s): {', '.join(missing)}", file=sys.stderr)
        print("[segment_toolkit] Attempting to auto-install dependencies...", file=sys.stderr)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("[segment_toolkit] Dependencies installed successfully.", file=sys.stderr)
        except Exception as err:
            print(f"[segment_toolkit] Error: Auto-installation of dependencies failed: {err}", file=sys.stderr)
            print("[segment_toolkit] Please install them manually using: pip install -r requirements.txt", file=sys.stderr)

# Check and install dependencies before importing other submodules
_ensure_dependencies()

from .core import (
    BaseConverter,
    MaskToYoloConverter,
    YoloToMaskConverter,
    MaskToPolygonConverter,
    PolygonToMaskConverter,
)
from .transforms import (
    BaseTransform,
    Compose,
    Resize,
    Normalize,
)
from .utils import (
    logger,
    safe_read_image,
    ensure_dir,
    get_largest_contour,
    calculate_bbox,
    normalize_coordinates,
    denormalize_coordinates,
    draw_bbox_on_image,
    compute_iou,
    compute_dice,
)

__all__ = [
    "__version__",
    "BaseConverter",
    "MaskToYoloConverter",
    "YoloToMaskConverter",
    "MaskToPolygonConverter",
    "PolygonToMaskConverter",
    "BaseTransform",
    "Compose",
    "Resize",
    "Normalize",
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
