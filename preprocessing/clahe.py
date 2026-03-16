"""
CLAHE contrast enhancement for ultrasound images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union

class CLAHE:
    @staticmethod
    def apply_clahe(image: Union[np.ndarray, Image.Image], clip_limit: float = 2.0, tile_grid_size: tuple = (8,8)) -> np.ndarray:
        """
        Apply CLAHE to enhance contrast.
        Args:
            image: Input image (numpy array or PIL Image).
            clip_limit: CLAHE clip limit.
            tile_grid_size: Size of grid for CLAHE.
        Returns:
            Contrast-enhanced image as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Convert bool or other types to uint8
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Ensure image is 2D (grayscale)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
