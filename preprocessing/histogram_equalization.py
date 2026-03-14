"""
Histogram equalization for ultrasound images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union

class HistogramEqualization:
    @staticmethod
    def equalize(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        return cv2.equalizeHist(image)
