"""
Intensity normalization for ultrasound images.
"""

import numpy as np
from PIL import Image
from typing import Union

class Normalization:
    @staticmethod
    def normalize(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Normalize image intensity to [0, 255].
        Args:
            image: Input image (numpy array or PIL Image).
        Returns:
            Normalized image as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        img = image.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return img.astype(np.uint8)
