# YOLO Label Processing and Visualization Pipeline
This project provides a comprehensive Jupyter notebook-based pipeline for working with YOLO format annotations and their corresponding images. It consists of two main workflows:

1. **Forward Pipeline** (`segment_with_preprocess_v2.ipynb`): Generate YOLO labels from binary segmentation masks
2. **Reverse Pipeline** (`convert_label_to_mask.ipynb`): Convert YOLO labels back to binary masks for visualization and validation

## 📌 Features

- **Forward Direction**: Automatically detects object regions in masks and converts to YOLO format
  - Extracts bounding boxes from binary segmentation masks
  - Normalizes coordinates to YOLO format: `<class_id> <center_x> <center_y> <width> <height>`
  - Outputs YOLO `.txt` label files

- **Reverse Direction**: Visualize and validate YOLO annotations
  - Converts YOLO labels back to binary segmentation masks
  - Overlays bounding boxes on original images for verification
  - Identifies annotation errors early in the pipeline

- Supports multi-class classification with class IDs
- Dataset-agnostic – works with medical, industrial, satellite imagery, etc.
- Includes comprehensive error logging and validation

## 📂 Project Structure

```
LabelFile-for-yoloModel/
├── images/                              # Original images for training
├── labels/                              # YOLO format annotations (.txt files)
├── mask/                                # Binary segmentation masks (source data)
├── masks_v2/                            # Generated masks from YOLO labels
├── segment_with_preprocess_v2.ipynb     # Forward pipeline: masks → YOLO labels
├── convert_label_to_mask.ipynb          # Reverse pipeline: YOLO labels → masks + visualization
├── GroundTruth.csv                      # Class labels for multi-class classification
├── data.yaml                            # YOLO dataset configuration
└── README.md                            # This file
```

## 📂 Input Folder Structure

Place your files in the following structure:

```
project/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── mask/                    # or masks/
│   ├── image_001_segmentation.png
│   ├── image_002_segmentation.png
│   └── ...
└── GroundTruth.csv         # For multi-class classification
```

Ensure that mask filenames **correspond** to image filenames.

## 🚀 Usage

### Forward Pipeline: Masks → YOLO Labels

Use `segment_with_preprocess_v2.ipynb` to generate YOLO labels from binary masks:

1. **Preprocessing**: Resize images and masks to 640×640
2. **Segmentation**: Extract bounding boxes from mask contours using OpenCV
3. **Conversion**: Transform pixel coordinates to YOLO normalized format
4. **Classification**: Assign class IDs based on `GroundTruth.csv`
5. **Split**: Divide dataset into training (80%) and testing (20%) sets
6. **Output**: Generate `labels/` directory with YOLO `.txt` files

### Reverse Pipeline: YOLO Labels → Visualization

Use `convert_label_to_mask.ipynb` to validate YOLO labels:

1. **Load Labels**: Read YOLO `.txt` annotation files
2. **Denormalize**: Convert normalized coordinates to pixel space
   - Formula: $x_{pixel} = x_{norm} \times img_{width}$
3. **Create Masks**: Draw rectangles on binary masks
4. **Visualize**: Overlay bounding boxes on original images
5. **Validate**: Verify annotation accuracy before training

### Step-by-Step Execution

**For Forward Pipeline:**
1. Ensure your `images/` and `mask/` directories are populated
2. Update `GroundTruth.csv` with class labels (MEL, NV, BCC, AKIEC, BKL, DF, VASC)
3. Open `segment_with_preprocess_v2.ipynb`
4. Run all cells in order
5. Generated labels will appear in `labels/` directory

**For Reverse Pipeline:**
1. Ensure YOLO labels exist in `labels/` directory
2. Open `convert_label_to_mask.ipynb`
3. Run all cells in order
4. Generated masks will appear in `masks_v2/` directory
5. Images with bounding boxes will be displayed

## ✅ Requirements

Install the following Python packages:

```bash
pip install opencv-python numpy matplotlib pillow pandas
```

To run Jupyter notebooks:
```bash
pip install notebook jupyterlab
```

### Recommended Versions
- Python: 3.8+
- OpenCV: 4.5+
- NumPy: 1.19+
- Matplotlib: 3.3+
- Pandas: 1.2+
## 🧠 Technical Background

### YOLO Format Explanation

Each label file contains one line per detected object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Key Points:**
- All values are **normalized** between 0 and 1
- Normalized relative to image dimensions
- Example for 640×480 image:
  - Object at (320, 240) with size 100×80
  - YOLO format: `0 0.500 0.500 0.156 0.167`

### Mathematical Foundations

#### Normalization (Pixel → YOLO)
$$x_{norm} = \frac{x_{pixel}}{img\_width}$$
$$y_{norm} = \frac{y_{pixel}}{img\_height}$$

#### Denormalization (YOLO → Pixel)
$$x_{pixel} = x_{norm} \times img\_width$$
$$y_{pixel} = y_{norm} \times img\_height$$

#### Center to Bounding Box Conversion
Given center $(x_c, y_c)$ and dimensions $(w, h)$:

**Top-left corner:**
$$x_1 = x_c - \frac{w}{2}, \quad y_1 = y_c - \frac{h}{2}$$

**Bottom-right corner:**
$$x_2 = x_c + \frac{w}{2}, \quad y_2 = y_c + \frac{h}{2}$$

### Key Technologies

#### OpenCV Contour Detection
- **Purpose**: Extract object boundaries from binary masks
- **Algorithm**: Finds contours using Moore-Neighbor tracing
- **Benefits**: Robust to noise, handles multiple objects

#### Binary Masks
- **Format**: Grayscale PNG (0 = background, 255 = object)
- **Advantage**: Simple, efficient, supports multiple objects per image
- **Storage**: Compact representation compared to coordinate lists

### Workflow Integration

```
Source Data
    ↓
[Segmentation Masks]
    ↓
[Extract Bounding Boxes via Contours]
    ↓
[Normalize Coordinates]
    ↓
[Assign Class IDs from Ground Truth]
    ↓
[Generate YOLO Labels]
    ↓
[Use in YOLO Training (YOLOv5, YOLOv8)]
```

### Validation Workflow

```
YOLO Labels
    ↓
[Denormalize Coordinates]
    ↓
[Draw on Binary Masks]
    ↓
[Overlay on Original Images]
    ↓
[Visual Inspection]
    ↓
[Approve/Correct Annotations]
```


## 📋 Pipeline Components

### `segment_with_preprocess_v2.ipynb`

**Main Steps:**
1. **Preprocessing** - Resize all images and masks to 640×640 RGB format
2. **Segmentation** - Extract bounding boxes from masks using contour detection
3. **Coordinate Transformation** - Convert from pixel to normalized YOLO coordinates
4. **Multi-class Classification** - Map class IDs from `GroundTruth.csv`
5. **Dataset Split** - Partition into training (80%) and test (20%) sets
6. **YOLO Format Output** - Save labels in `.txt` files

**Key Functions:**
- `preprocess_images()` - Standardize image sizes
- `segment_img_yolo()` - Extract bounding box using minimum area rectangle
- `copy_files()` - Organize files into train/test directories

### `convert_label_to_mask.ipynb`

**Main Classes:**
- **PreProcessImage** - Converts YOLO labels to binary mask images
  - `_create_mask_from_label()` - Draws rectangles for each bounding box
  - `convert_label_to_mask()` - Batch processes all label files

- **ShowImageRect** - Visualizes annotations
  - `segmentImg()` - Extracts bounding box from mask using contour detection
  - `show_image_with_mask()` - Displays image with overlay
  - `processImages()` - Iterates through dataset for validation

### Data Files

- **GroundTruth.csv** - Multi-class labels mapping filenames to disease types
  - Columns: Filename, MEL, NV, BCC, AKIEC, BKL, DF, VASC
  - Binary indicators (1 = present, 0 = absent)

- **data.yaml** - YOLO training configuration
  - Specifies train/test paths
  - Defines number of classes and class names

## ⚠️ Important Notes & Best Practices

### Coordinate System Understanding
- **Normalized coordinates** are always in range [0, 1]
- **Pixel coordinates** depend on image dimensions
- Always verify image dimensions match between label generation and usage

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Bounding boxes out of image bounds | Dimension mismatch | Verify img_width and img_height match actual images |
| Empty masks generated | No valid contours | Check mask binary format and contrast |
| Label format errors | Malformed YOLO lines | Check label file structure has exactly 5 values |
| Class ID mismatch | GroundTruth.csv issues | Verify CSV has correct binary indicators |

### Quality Checklist

- ✅ Binary masks use proper contrast (0 = black, 255 = white)
- ✅ Image dimensions are consistent (640×640 recommended)
- ✅ Filenames match between images and masks (except extension)
- ✅ YOLO coordinates are normalized [0, 1]
- ✅ Label files have exactly 5 space-separated values per line
- ✅ Class IDs match your dataset schema

### Performance Tips

- Use consistent image sizes (640×640 standard for YOLO)
- Ensure sufficient contrast in binary masks
- Pre-validate masks visually before processing
- Use logging to identify problematic files
- Process in batches for large datasets

## 🧑‍💻 Author
Zakria Gamal
Computer Vision & AI Engineer
🧠 LinkedIn https://www.linkedin.com/in/zkaria-gamal-82b486267/
