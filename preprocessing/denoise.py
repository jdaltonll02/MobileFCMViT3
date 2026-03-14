"""
Speckle noise reduction for ultrasound images using median filtering and anisotropic diffusion.
"""

import cv2
import numpy as np
from typing import Union
from PIL import Image

class Denoise:
    @staticmethod
    def median_filter(image: Union[np.ndarray, Image.Image], kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filtering to reduce speckle noise.
        Args:
            image: Input image (numpy array or PIL Image).
            kernel_size: Size of the median filter kernel.
        Returns:
            Denoised image as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def anisotropic_diffusion(image: Union[np.ndarray, Image.Image], num_iter: int = 5, kappa: float = 50.0, gamma: float = 0.1) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion for speckle noise reduction.
        Args:
            image: Input image (numpy array or PIL Image).
            num_iter: Number of iterations.
            kappa: Conductance coefficient.
            gamma: Step size.
        Returns:
            Denoised image as numpy array.
        """
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)
        img = image.astype(np.float32)
        for _ in range(num_iter):
            nablaN = np.roll(img, -1, axis=0) - img
            nablaS = np.roll(img, 1, axis=0) - img
            nablaE = np.roll(img, -1, axis=1) - img
            nablaW = np.roll(img, 1, axis=1) - img
            cN = np.exp(-(nablaN/kappa)**2)
            cS = np.exp(-(nablaS/kappa)**2)
            cE = np.exp(-(nablaE/kappa)**2)
            cW = np.exp(-(nablaW/kappa)**2)
            img += gamma * (cN*nablaN + cS*nablaS + cE*nablaE + cW*nablaW)
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
