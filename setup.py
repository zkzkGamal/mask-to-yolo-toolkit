from setuptools import setup, find_packages

setup(
    name="segment_toolkit",
    version="1.0.0",
    description="A Python toolkit to convert between binary segmentation masks and YOLO labels",
    author="Antigravity",
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
