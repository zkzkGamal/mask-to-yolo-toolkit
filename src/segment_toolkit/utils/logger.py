"""
This module configures the standard logging configuration used throughout the segment-toolkit package.
"""
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("segment_toolkit")
