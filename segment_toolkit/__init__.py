"""
Segment Toolkit: A library and CLI tool for converting binary segmentation masks to YOLO labels and vice versa.
"""

__version__ = "1.0.1"

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
        "matplotlib": "matplotlib"
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

from .source import MaskToYoloConverter, YoloToMaskConverter
from .helpers import (
    safe_read_image,
    preprocess_image,
    get_largest_contour,
    calculate_bbox,
    normalize_coordinates,
    denormalize_coordinates,
)

__all__ = [
    "MaskToYoloConverter",
    "YoloToMaskConverter",
    "safe_read_image",
    "preprocess_image",
    "get_largest_contour",
    "calculate_bbox",
    "normalize_coordinates",
    "denormalize_coordinates",
]
