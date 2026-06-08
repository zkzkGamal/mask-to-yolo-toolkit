import os
from setuptools import setup, find_packages

# Read the contents of README.md for the long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A Python toolkit to convert between binary segmentation masks and YOLO labels"

setup(
    name="segment_toolkit",
    version="1.0.2",
    description="A Python toolkit to convert between binary segmentation masks and YOLO labels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zakria Gamal",
    author_email="zekogml11@gmail.com",
    url="https://github.com/zkzkGamal/mask-to-yolo-toolkit",
    project_urls={
        "Source Code": "https://github.com/zkzkGamal/mask-to-yolo-toolkit",
        "Bug Tracker": "https://github.com/zkzkGamal/mask-to-yolo-toolkit/issues",
    },
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pillow",
        "pandas",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "segment-toolkit=segment_toolkit.cli:main",
        ],
    },
    python_requires=">=3.6",
)
