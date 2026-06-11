"""
This module initializes the core package of Segment Toolkit, exposing the base converter and all specialized format converters.
The converters exposed here are:
- BaseConverter: The abstract base class defining the converter interface.
- MaskToYoloConverter: For converting binary segmentation masks to YOLO bounding boxes.
- YoloToMaskConverter: For reconstructing binary masks from YOLO bounding boxes.
- MaskToPolygonConverter: For converting segmentation masks to YOLO polygon coordinates.
- PolygonToMaskConverter: For reconstructing masks from YOLO polygon coordinates.
"""
from .base_converter import BaseConverter
from .mask_to_yolo import MaskToYoloConverter
from .yolo_to_mask import YoloToMaskConverter
from .mask_to_polygon import MaskToPolygonConverter
from .polygon_to_mask import PolygonToMaskConverter

__all__ = [
    "BaseConverter",
    "MaskToYoloConverter",
    "YoloToMaskConverter",
    "MaskToPolygonConverter",
    "PolygonToMaskConverter",
]
