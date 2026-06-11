# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-06-11

### Added
- **YOLO Polygon Segmentation Support**: 
  - Added `MaskToPolygonConverter` for extracting polygon outlines from binary and multi-class RGB masks.
  - Added `PolygonToMaskConverter` for reconstructing binary/colored masks from YOLO polygon labels.
  - Robust handling of mixed label formats (can read both standard bounding boxes and polygons in label files).
  - Visualization support for polygons (`visualize-polygon` command and overlay method).
- **Transform Pipeline**:
  - Implemented pipeline-first `Compose` design pattern.
  - Added `Resize` and `Normalize` transforms.
- **Metrics utility**:
  - Added mask validation metrics (`compute_iou` and `compute_dice`).
- **Modern CLI**:
  - Rewrote command line interface using `typer` for clean, colorized interface options.

### Changed
- **Packaging Structure Restructuring**:
  - Migrated codebase to standard `/src` layout (`src/segment_toolkit/`).
  - Switched packaging system from `setup.py` to `pyproject.toml`.
  - Moved original conversion functions from monolithic `source.py` to separate module files under `core/`.
  - Moved validation scripts to the `scripts/` folder.
  - Moved unit and integration tests into the `tests/` folder.

---

## [1.1.0] - 2026-06-10

### Added
- Standard and rotated minimum-area bounding box conversions.
- Dataset train/test splitting functionality.
- Automatic missing package installer on load.
- Overlay visualizers for bounding box and binary masks.
- Support for ISIC ground-truth CSV/JSON class mapping schemas.
