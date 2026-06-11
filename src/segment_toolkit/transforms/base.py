"""
This module defines the BaseTransform abstract class and the Compose transform chainer.
Compose allows sequential application of multiple transforms on images and optional masks simultaneously.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np

class BaseTransform(ABC):
    """
    Abstract Base Class for image & mask transformations.
    """
    @abstractmethod
    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        pass

class Compose(BaseTransform):
    """
    Applies a list of transforms sequentially.
    """
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        for t in self.transforms:
            if mask is not None:
                res = t(image, mask)
                if isinstance(res, tuple):
                    image, mask = res
                else:
                    image = res
            else:
                image = t(image)
        if mask is not None:
            return image, mask
        return image
