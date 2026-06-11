"""
This module contains the command-line interface (CLI) implementation for Segment Toolkit using Typer.
It registers subcommands such as mask-to-yolo, yolo-to-mask, mask-to-polygon, polygon-to-mask, split, and visualization tools.
It includes detailed argument verification to handle missing inputs and supply diagnostic error logs.
"""
"""
This module contains the command-line interface (CLI) for the Segment Toolkit. 
It provides a way for users to interact with the toolkit through the terminal, 
    allowing them to execute various commands and operations related to segmentation tasks.
The CLI is built using a framework that simplifies the creation of command-line applications, 
    making it easy to define commands, options, and arguments.
Commands include converting between binary masks and YOLO format labels, converting between masks and polygon labels, splitting datasets, and visualizing annotations. The CLI supports both single file operations and batch processing of directories.
Commands to run :
- `python -m segment_toolkit.cli mask-to-yolo --image path/to/image.jpg --mask path/to/mask.png --output-txt path/to/label.txt`
- `python -m segment_toolkit.cli yolo-to-mask --label path/to/label.txt --output-mask path/to/mask.png`
- `python -m segment_toolkit.cli mask-to-polygon --image path/to/image.jpg --mask path/to/mask.png --output-txt path/to/polygon_label.txt`
- `python -m segment_toolkit.cli polygon-to-mask --label path/to/polygon_label.txt --output-mask path/to/mask.png`
- `python -m segment_toolkit.cli split --images path/to/images/ --labels path/to/labels/ --output path/to/output/ --ratio 0.8 --seed 42`
- `python -m segment_toolkit.cli visualize --image path/to/image.jpg --label path/to/label.txt --output path/to/output.jpg`
"""
import typer
from typing import Optional, List
import os
import json
from ..core.mask_to_yolo import MaskToYoloConverter
from ..core.yolo_to_mask import YoloToMaskConverter
from ..core.mask_to_polygon import MaskToPolygonConverter
from ..core.polygon_to_mask import PolygonToMaskConverter
from ..utils.logger import logger

app = typer.Typer(help="Segment Toolkit: Convert segmentation masks to/from YOLO format annotations.")

def _load_classes_from_json(classes_path: Optional[str]):
    if not classes_path:
        return None
    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = []
        for item in data:
            if isinstance(item, list) and len(item) == 2:
                classes.append((tuple(item[0]), item[1]))
            else:
                logger.warning(f"Skipping invalid class configuration item: {item}")
        return classes
    except Exception as e:
        logger.error(f"Failed to load classes JSON file from {classes_path}: {e}")
        return None

def _log_error_and_exit(
    command_name: str,
    provided_args: dict,
    missing_args: list,
    mode: str,  # "single", "directory", or "either"
    required_single: dict,
    required_dir: dict
):
    logger.error(f"Error: Missing required arguments for '{command_name}' command.")
    logger.error("\n[Provided Arguments (Real Input Data Taken)]")
    for name, val in provided_args.items():
        logger.error(f"  {name:<35}: {repr(val)}")
        
    if mode == "single":
        logger.error("\n[Required Missing Arguments for Single Conversion]")
        for opt in missing_args:
            desc = required_single.get(opt, "")
            logger.error(f"  {opt:<35}: {desc}")
    elif mode == "directory":
        logger.error("\n[Required Missing Arguments for Folder/Batch Conversion]")
        for opt in missing_args:
            desc = required_dir.get(opt, "")
            logger.error(f"  {opt:<35}: {desc}")
    else:  # "either"
        logger.error("\n[Required Arguments for Single Conversion]")
        for opt, desc in required_single.items():
            logger.error(f"  {opt:<35}: {desc}")
        logger.error("\n[Required Arguments for Folder/Batch Conversion]")
        for opt, desc in required_dir.items():
            logger.error(f"  {opt:<35}: {desc}")
            
    raise typer.Exit(code=1)

@app.command("mask-to-yolo")
def mask_to_yolo(
    image: Optional[str] = typer.Option(None, "--image", "--image-path", help="Path to a single input image."),
    mask: Optional[str] = typer.Option(None, "--mask", "--mask-path", help="Path to a single input binary mask."),
    output_txt: Optional[str] = typer.Option(None, "--output-txt", "--output-txt-path", help="Output file path for single YOLO label txt."),
    image_dir: Optional[str] = typer.Option(None, "--image-dir", help="Directory containing input images."),
    mask_dir: Optional[str] = typer.Option(None, "--mask-dir", help="Directory containing input masks."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to save generated YOLO label files."),
    class_id: int = typer.Option(0, help="Default Class ID to write."),
    ground_truth: Optional[str] = typer.Option(None, help="Path to GroundTruth.csv/json file."),
    rotated: bool = typer.Option(False, "--rotated", help="Use rotated minimum area bounding box."),
    resize_width: int = typer.Option(640, help="Resize target width."),
    resize_height: int = typer.Option(640, help="Resize target height."),
):
    """
    Convert binary mask(s) to YOLO format bounding box labels.
    """
    converter = MaskToYoloConverter(
        target_size=(resize_width, resize_height),
        bbox_type="rotated" if rotated else "standard",
    )
    
    provided = {
        "--image / --image-path": image,
        "--mask / --mask-path": mask,
        "--output-txt / --output-txt-path": output_txt,
        "--image-dir": image_dir,
        "--mask-dir": mask_dir,
        "--output-dir": output_dir,
    }
    
    req_single = {
        "--image / --image-path": "Path of a single input image.",
        "--mask / --mask-path": "Path of a single input binary mask.",
        "--output-txt / --output-txt-path": "Output file path for single YOLO label txt."
    }
    req_dir = {
        "--image-dir": "Directory containing input images.",
        "--mask-dir": "Directory containing input masks.",
        "--output-dir": "Directory to save generated YOLO label files."
    }

    if image or mask or output_txt:
        if not (image and mask and output_txt):
            missing = []
            if not image:
                missing.append("--image / --image-path")
            if not mask:
                missing.append("--mask / --mask-path")
            if not output_txt:
                missing.append("--output-txt / --output-txt-path")
            _log_error_and_exit("mask-to-yolo", provided, missing, "single", req_single, req_dir)
            
        success = converter.convert_single(image, mask, output_txt, class_id=class_id)
        if not success:
            raise typer.Exit(code=1)
    elif image_dir or mask_dir or output_dir:
        if not (image_dir and mask_dir and output_dir):
            missing = []
            if not image_dir:
                missing.append("--image-dir")
            if not mask_dir:
                missing.append("--mask-dir")
            if not output_dir:
                missing.append("--output-dir")
            _log_error_and_exit("mask-to-yolo", provided, missing, "directory", req_single, req_dir)
            
        converter.convert_dataset(
            image_dir,
            mask_dir,
            output_dir,
            default_class_id=class_id,
            ground_truth=ground_truth,
        )
    else:
        _log_error_and_exit("mask-to-yolo", provided, [], "either", req_single, req_dir)

@app.command("yolo-to-mask")
def yolo_to_mask(
    label: Optional[str] = typer.Option(None, "--label", "--label-path", help="Path to a single input YOLO label txt file."),
    output_mask: Optional[str] = typer.Option(None, "--output-mask", "--output-mask-path", help="Output path for single binary mask png."),
    label_dir: Optional[str] = typer.Option(None, "--label-dir", help="Directory containing YOLO label txt files."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to save generated binary mask png files."),
    resize_width: int = typer.Option(640, help="Output mask width."),
    resize_height: int = typer.Option(640, help="Output mask height."),
):
    """
    Convert YOLO format bounding box label(s) to binary mask(s).
    """
    converter = YoloToMaskConverter(target_size=(resize_width, resize_height))
    
    provided = {
        "--label / --label-path": label,
        "--output-mask / --output-mask-path": output_mask,
        "--label-dir": label_dir,
        "--output-dir": output_dir,
    }
    
    req_single = {
        "--label / --label-path": "Path to a single input YOLO label txt file.",
        "--output-mask / --output-mask-path": "Output path for single binary mask png."
    }
    req_dir = {
        "--label-dir": "Directory containing YOLO label txt files.",
        "--output-dir": "Directory to save generated binary mask png files."
    }

    if label or output_mask:
        if not (label and output_mask):
            missing = []
            if not label:
                missing.append("--label / --label-path")
            if not output_mask:
                missing.append("--output-mask / --output-mask-path")
            _log_error_and_exit("yolo-to-mask", provided, missing, "single", req_single, req_dir)
            
        success = converter.convert_single(label, output_mask)
        if not success:
            raise typer.Exit(code=1)
    elif label_dir or output_dir:
        if not (label_dir and output_dir):
            missing = []
            if not label_dir:
                missing.append("--label-dir")
            if not output_dir:
                missing.append("--output-dir")
            _log_error_and_exit("yolo-to-mask", provided, missing, "directory", req_single, req_dir)
            
        converter.convert_dataset(label_dir, output_dir)
    else:
        _log_error_and_exit("yolo-to-mask", provided, [], "either", req_single, req_dir)

@app.command("mask-to-polygon")
def mask_to_polygon(
    image: Optional[str] = typer.Option(None, "--image", "--image-path", help="Path to a single input image."),
    mask: Optional[str] = typer.Option(None, "--mask", "--mask-path", help="Path to a single input mask."),
    output_txt: Optional[str] = typer.Option(None, "--output-txt", "--output-txt-path", help="Output file path for single YOLO polygon label."),
    image_dir: Optional[str] = typer.Option(None, "--image-dir", help="Directory containing input images."),
    mask_dir: Optional[str] = typer.Option(None, "--mask-dir", help="Directory containing input masks."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to save generated polygon label files."),
    class_id: int = typer.Option(0, help="Default Class ID to write."),
    classes: Optional[str] = typer.Option(None, help="Path to JSON file mapping classes to RGB color tuples."),
    resize_width: int = typer.Option(640, help="Resize target width."),
    resize_height: int = typer.Option(640, help="Resize target height."),
):
    """
    Convert binary/multi-class mask(s) to YOLO polygon labels.
    """
    classes_list = _load_classes_from_json(classes)
    converter = MaskToPolygonConverter(target_size=(resize_width, resize_height))
    
    provided = {
        "--image / --image-path": image,
        "--mask / --mask-path": mask,
        "--output-txt / --output-txt-path": output_txt,
        "--image-dir": image_dir,
        "--mask-dir": mask_dir,
        "--output-dir": output_dir,
    }
    
    req_single = {
        "--image / --image-path": "Path to a single input image.",
        "--mask / --mask-path": "Path to a single input mask.",
        "--output-txt / --output-txt-path": "Output file path for single YOLO polygon label."
    }
    req_dir = {
        "--image-dir": "Directory containing input images.",
        "--mask-dir": "Directory containing input masks.",
        "--output-dir": "Directory to save generated polygon label files."
    }

    if image or mask or output_txt:
        if not (image and mask and output_txt):
            missing = []
            if not image:
                missing.append("--image / --image-path")
            if not mask:
                missing.append("--mask / --mask-path")
            if not output_txt:
                missing.append("--output-txt / --output-txt-path")
            _log_error_and_exit("mask-to-polygon", provided, missing, "single", req_single, req_dir)
            
        success = converter.convert_single(
            image, mask, output_txt, class_id=class_id, classes=classes_list
        )
        if not success:
            raise typer.Exit(code=1)
    elif image_dir or mask_dir or output_dir:
        if not (image_dir and mask_dir and output_dir):
            missing = []
            if not image_dir:
                missing.append("--image-dir")
            if not mask_dir:
                missing.append("--mask-dir")
            if not output_dir:
                missing.append("--output-dir")
            _log_error_and_exit("mask-to-polygon", provided, missing, "directory", req_single, req_dir)
            
        converter.convert_dataset(
            image_dir,
            mask_dir,
            output_dir,
            default_class_id=class_id,
            classes=classes_list,
        )
    else:
        _log_error_and_exit("mask-to-polygon", provided, [], "either", req_single, req_dir)

@app.command("polygon-to-mask")
def polygon_to_mask(
    label: Optional[str] = typer.Option(None, "--label", "--label-path", help="Path to a single input YOLO polygon label txt file."),
    output_mask: Optional[str] = typer.Option(None, "--output-mask", "--output-mask-path", help="Output path for single binary/colored mask png."),
    label_dir: Optional[str] = typer.Option(None, "--label-dir", help="Directory containing YOLO polygon labels."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to save generated mask png files."),
    classes: Optional[str] = typer.Option(None, help="Path to JSON file mapping classes to RGB color tuples."),
    resize_width: int = typer.Option(640, help="Output mask width."),
    resize_height: int = typer.Option(640, help="Output mask height."),
):
    """
    Convert YOLO polygon label(s) to mask images.
    """
    classes_list = _load_classes_from_json(classes)
    converter = PolygonToMaskConverter(target_size=(resize_width, resize_height))
    
    provided = {
        "--label / --label-path": label,
        "--output-mask / --output-mask-path": output_mask,
        "--label-dir": label_dir,
        "--output-dir": output_dir,
    }
    
    req_single = {
        "--label / --label-path": "Path to a single input YOLO polygon label txt file.",
        "--output-mask / --output-mask-path": "Output path for single binary/colored mask png."
    }
    req_dir = {
        "--label-dir": "Directory containing YOLO polygon labels.",
        "--output-dir": "Directory to save generated mask png files."
    }

    if label or output_mask:
        if not (label and output_mask):
            missing = []
            if not label:
                missing.append("--label / --label-path")
            if not output_mask:
                missing.append("--output-mask / --output-mask-path")
            _log_error_and_exit("polygon-to-mask", provided, missing, "single", req_single, req_dir)
            
        success = converter.convert_single(label, output_mask, classes=classes_list)
        if not success:
            raise typer.Exit(code=1)
    elif label_dir or output_dir:
        if not (label_dir and output_dir):
            missing = []
            if not label_dir:
                missing.append("--label-dir")
            if not output_dir:
                missing.append("--output-dir")
            _log_error_and_exit("polygon-to-mask", provided, missing, "directory", req_single, req_dir)
            
        converter.convert_dataset(label_dir, output_dir, classes=classes_list)
    else:
        _log_error_and_exit("polygon-to-mask", provided, [], "either", req_single, req_dir)

@app.command("split")
def split(
    images: str = typer.Option(..., help="Directory of source images."),
    labels: str = typer.Option(..., help="Directory of source YOLO label txt files."),
    output: str = typer.Option(..., help="Root directory for split dataset outputs."),
    ratio: float = typer.Option(0.8, help="Split ratio for training partition."),
    seed: int = typer.Option(42, help="Seed value for reproduction."),
):
    """
    Randomly split a dataset of images and labels into train/test subfolders.
    """
    converter = MaskToYoloConverter()
    converter.split_dataset(
        images_dir=images,
        labels_dir=labels,
        output_dataset_dir=output,
        split_ratio=ratio,
        seed=seed,
    )

@app.command("visualize")
def visualize(
    image: str = typer.Option(..., help="Path to the source image."),
    label: str = typer.Option(..., help="Path to the YOLO label file."),
    output: str = typer.Option(..., help="Path to save output visualization image."),
    resize_width: int = typer.Option(640, help="Resize target width."),
    resize_height: int = typer.Option(640, help="Resize target height."),
):
    """
    Draw bounding boxes from a YOLO label file onto the source image.
    """
    converter = YoloToMaskConverter(target_size=(resize_width, resize_height))
    success = converter.visualize_label(image, label, output)
    if not success:
        raise typer.Exit(code=1)

@app.command("visualize-mask")
def visualize_mask(
    image: str = typer.Option(..., help="Path to the source image."),
    mask: str = typer.Option(..., help="Path to the binary mask image."),
    output: str = typer.Option(..., help="Path to save output visualization image."),
    resize_width: int = typer.Option(640, help="Resize target width."),
    resize_height: int = typer.Option(640, help="Resize target height."),
):
    """
    Overlay a binary mask onto the source image.
    """
    converter = YoloToMaskConverter(target_size=(resize_width, resize_height))
    success = converter.visualize_mask(image, mask, output)
    if not success:
        raise typer.Exit(code=1)

@app.command("visualize-polygon")
def visualize_polygon(
    image: str = typer.Option(..., help="Path to the source image."),
    label: str = typer.Option(..., help="Path to the YOLO polygon label file."),
    output: str = typer.Option(..., help="Path to save output visualization image."),
    classes: Optional[str] = typer.Option(None, help="Path to JSON file mapping classes to RGB color tuples."),
    resize_width: int = typer.Option(640, help="Resize target width."),
    resize_height: int = typer.Option(640, help="Resize target height."),
):
    """
    Draw polygon segmentation overlay from a YOLO label onto the source image.
    """
    classes_list = _load_classes_from_json(classes)
    converter = PolygonToMaskConverter(target_size=(resize_width, resize_height))
    success = converter.visualize_polygon(image, label, output, classes=classes_list)
    if not success:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
