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
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
