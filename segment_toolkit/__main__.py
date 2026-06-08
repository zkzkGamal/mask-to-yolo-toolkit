"""
Main execution entry point for segment_toolkit.
Enables running the package directly via `python3 -m segment_toolkit` or `python3 segment_toolkit`.
"""

import sys
import os

# Add the parent folder of segment_toolkit to sys.path to resolve absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from segment_toolkit.cli import main

if __name__ == "__main__":
    sys.exit(main())
