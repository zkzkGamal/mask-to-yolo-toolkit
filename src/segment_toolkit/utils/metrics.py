"""
This module provides metrics for evaluating segmentation mask overlap, including Intersection over Union (IoU) and Dice Coefficient.
"""
import numpy as np

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two binary masks.
    """
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Dice coefficient between two binary masks.
    """
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    total = m1.sum() + m2.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * float(intersection) / float(total)
