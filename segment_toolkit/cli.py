"""
Command Line Interface (CLI) for segment_toolkit.
Exposes mask-to-yolo, yolo-to-mask, split, and visualize commands.
"""

import argparse
import sys
import logging
from typing import List, Optional

from .source import MaskToYoloConverter, YoloToMaskConverter

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Segment Toolkit: Convert segmentation masks to/from YOLO format annotations."
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # subcommand: mask-to-yolo
    m2y_parser = subparsers.add_parser(
        "mask-to-yolo", help="Convert binary mask(s) to YOLO format labels."
    )
    m2y_parser.add_argument("--image", type=str, help="Path to a single input image.")
    m2y_parser.add_argument("--mask", type=str, help="Path to a single input binary mask.")
    m2y_parser.add_argument("--output-txt", type=str, help="Output file path for single YOLO label txt.")
    
    m2y_parser.add_argument("--image-dir", type=str, help="Directory containing input images.")
    m2y_parser.add_argument("--mask-dir", type=str, help="Directory containing input masks.")
    m2y_parser.add_argument("--output-dir", type=str, help="Directory to save generated YOLO label files.")

    m2y_parser.add_argument("--class-id", type=int, default=0, help="Default Class ID to write (default: 0).")
    m2y_parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to GroundTruth.csv mapping images to multi-class columns.",
    )
    m2y_parser.add_argument(
        "--rotated",
        action="store_true",
        help="Use rotated minimum area rectangle (minAreaRect) instead of axis-aligned.",
    )
    m2y_parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Target dimensions for resizing (default: 640 640).",
    )

    # subcommand: yolo-to-mask
    y2m_parser = subparsers.add_parser(
        "yolo-to-mask", help="Convert YOLO format label(s) to binary mask(s)."
    )
    y2m_parser.add_argument("--label", type=str, help="Path to a single input YOLO label txt file.")
    y2m_parser.add_argument("--output-mask", type=str, help="Output path for single binary mask png.")

    y2m_parser.add_argument("--label-dir", type=str, help="Directory containing YOLO label txt files.")
    y2m_parser.add_argument("--output-dir", type=str, help="Directory to save generated binary mask png files.")

    y2m_parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Output mask dimensions (default: 640 640).",
    )

    # subcommand: split
    split_parser = subparsers.add_parser(
        "split", help="Randomly split a dataset of images and labels into train/test subfolders."
    )
    split_parser.add_argument("--images", type=str, required=True, help="Directory of source images.")
    split_parser.add_argument("--labels", type=str, required=True, help="Directory of source YOLO label txt files.")
    split_parser.add_argument("--output", type=str, required=True, help="Root directory for split dataset outputs.")
    split_parser.add_argument(
        "--ratio", type=float, default=0.8, help="Split ratio for training partition (default: 0.8)."
    )
    split_parser.add_argument("--seed", type=int, default=42, help="Seed value for reproduction (default: 42).")

    # subcommand: visualize
    vis_parser = subparsers.add_parser(
        "visualize", help="Draw bounding boxes from a YOLO label file onto the source image."
    )
    vis_parser.add_argument("--image", type=str, required=True, help="Path to the source image.")
    vis_parser.add_argument("--label", type=str, required=True, help="Path to the YOLO label file.")
    vis_parser.add_argument("--output", type=str, required=True, help="Path to save output visualization image.")
    vis_parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Resize image for visualization (default: 640 640).",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for command-line execution.
    """
    if args is None:
        args = sys.argv[1:]

    try:
        parsed = parse_args(args)

        if parsed.command == "mask-to-yolo":
            converter = MaskToYoloConverter(
                target_size=(parsed.resize[0], parsed.resize[1]),
                bbox_type="rotated" if parsed.rotated else "standard",
            )
            # Check if processing single file or folder
            if parsed.image or parsed.mask or parsed.output_txt:
                if not (parsed.image and parsed.mask and parsed.output_txt):
                    logger.error("Error: --image, --mask, and --output-txt must all be specified for single conversion.")
                    return 1
                success = converter.convert_single(
                    parsed.image, parsed.mask, parsed.output_txt, class_id=parsed.class_id
                )
                return 0 if success else 1
            elif parsed.image_dir or parsed.mask_dir or parsed.output_dir:
                if not (parsed.image_dir and parsed.mask_dir and parsed.output_dir):
                    logger.error("Error: --image-dir, --mask-dir, and --output-dir must all be specified for folder conversion.")
                    return 1
                converter.convert_dataset(
                    parsed.image_dir,
                    parsed.mask_dir,
                    parsed.output_dir,
                    default_class_id=parsed.class_id,
                    ground_truth=parsed.ground_truth,
                )
                return 0
            else:
                logger.error("Error: Must specify either single file arguments (--image, --mask, --output-txt) or directory arguments (--image-dir, --mask-dir, --output-dir).")
                return 1

        elif parsed.command == "yolo-to-mask":
            converter = YoloToMaskConverter(target_size=(parsed.resize[0], parsed.resize[1]))
            if parsed.label or parsed.output_mask:
                if not (parsed.label and parsed.output_mask):
                    logger.error("Error: Both --label and --output-mask must be specified for single conversion.")
                    return 1
                success = converter.convert_single(parsed.label, parsed.output_mask)
                return 0 if success else 1
            elif parsed.label_dir or parsed.output_dir:
                if not (parsed.label_dir and parsed.output_dir):
                    logger.error("Error: Both --label-dir and --output-dir must be specified for folder conversion.")
                    return 1
                converter.convert_dataset(parsed.label_dir, parsed.output_dir)
                return 0
            else:
                logger.error("Error: Must specify either single file arguments (--label, --output-mask) or directory arguments (--label-dir, --output-dir).")
                return 1

        elif parsed.command == "split":
            converter = MaskToYoloConverter()
            converter.split_dataset(
                images_dir=parsed.images,
                labels_dir=parsed.labels,
                output_dataset_dir=parsed.output,
                split_ratio=parsed.ratio,
                seed=parsed.seed,
            )
            return 0

        elif parsed.command == "visualize":
            converter = YoloToMaskConverter(target_size=(parsed.resize[0], parsed.resize[1]))
            success = converter.visualize_label(parsed.image, parsed.label, parsed.output)
            return 0 if success else 1

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
