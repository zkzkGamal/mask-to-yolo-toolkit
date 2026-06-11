"""
This module defines the BaseConverter class, an abstract base class that establishes a common interface for all conversion modules in the segment-toolkit.
It enforces the implementation of `convert_single` and `convert_dataset` methods to ensure consistency across different converter types.
"""
from abc import ABC, abstractmethod

class BaseConverter(ABC):
    """
    Abstract Base Class for all converters in segment-toolkit.
    """

    @abstractmethod
    def convert_single(self, *args, **kwargs) -> bool:
        """
        Convert a single image/label/mask.
        """
        pass

    @abstractmethod
    def convert_dataset(self, *args, **kwargs) -> int:
        """
        Convert a batch directory.
        """
        pass
