"""
Adaptive thresholding for ultrasound images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union

class AdaptiveThresholding:
    @staticmethod
    def threshold(image: Union[np.ndarray, Image.Image], block_size: int = 11, C: int = 2) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
