"""
Resize ultrasound images to target size.
"""

from PIL import Image
from typing import Union
import numpy as np

class Resize:
    @staticmethod
    def resize(image: Union[np.ndarray, Image.Image], size: int = 224) -> Image.Image:
        """
        Resize image to (size, size).
        Args:
            image: Input image (numpy array or PIL Image).
            size: Target size.
        Returns:
            Resized PIL Image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.resize((size, size), Image.BILINEAR)
