# YOLO Label Generator from Masks

This project provides a Jupyter notebook-based pipeline to automatically generate YOLO-format annotation files (`.txt`) from binary segmentation masks and their corresponding images. It is designed to work with any kind of dataset, not limited to medical or skin-related imagery.

## 📌 Features

- Accepts image and binary mask pairs.
- Automatically detects object regions in the mask.
- Converts detected regions into YOLO format:
<class_id> <center_x> <center_y> <width> <height>

- Outputs YOLO `.txt` files for each image.
- Supports single-class datasets (default `class_id = 0`).
- Dataset-agnostic – works with any domain (medical, industrial, satellite, etc.).

## 📂 Input Folder Structure

Place your files in two separate folders:

dataset/
├── images/
│ ├── image_001.jpg
│ ├── image_002.jpg
│ └── ...
└── masks/
├── image_001.png # binary mask for image_001
├── image_002.png
└── ...



Ensure that mask filenames **exactly match** the image filenames (except extension if needed).

## 🚀 Usage

1. Open the notebook: `segment_with_preprocess_v2.ipynb`.

2. Set the paths to your `images` and `masks` folders inside the notebook.

3. Run all cells to:
   - Load and preprocess each image-mask pair.
   - Extract bounding boxes from the mask.
   - Convert each box to YOLO format.
   - Save `.txt` label files into a `labels/` directory.

4. You can now use the `images/` and generated `labels/` folders in any YOLO training framework (YOLOv5, YOLOv8, etc.).

## ✅ Requirements

Install the following Python packages:

```bash
pip install opencv-python numpy matplotlib
```

To run the notebook:
  pip install notebook
## 🧠 YOLO Format Refresher
Each label file contains one line per object:
<class_id> <x_center> <y_center> <width> <height>
All values are normalized between 0 and 1 relative to the image dimensions.

For example, if an object is centered in the middle of a 640x480 image and is 100x80 in size, its YOLO label might look like:
0 0.500 0.500 0.156 0.167


## 📌 Notes
  This version is for single-class detection. To support multi-class masks, adapt the mask processing logic to assign different class_ids.
  
  The pipeline assumes binary masks (white = object, black = background).
  
  Handles multiple objects per mask using contour detection.
  
  Outputs one .txt file per image, with the same base filename.
## 🧑‍💻 Author
Zakria Gamal
Computer Vision & AI Engineer
🧠 LinkedIn https://www.linkedin.com/in/zkaria-gamal-82b486267/
