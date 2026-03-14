"""
Ultrasound image preprocessing pipeline.
"""

from typing import Union
from PIL import Image
import numpy as np
from .dicom_converter import DICOMConverter
from .denoise import Denoise
from .clahe import CLAHE
from .normalization import Normalization
from .resize import Resize

class UltrasoundPreprocessingPipeline:
    def __init__(self, size: int = 224, denoise_method: str = 'median', apply_clahe: bool = True, augment: bool = False):
        """
        Initialize preprocessing pipeline.
        Args:
            size: Target image size.
            denoise_method: 'median' or 'anisotropic'.
            apply_clahe: Whether to apply CLAHE.
            augment: Whether to apply augmentation.
        """
        self.size = size
        self.denoise_method = denoise_method
        self.apply_clahe = apply_clahe
        self.augment = augment

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image: denoise, enhance, normalize, resize, augment.
        Args:
            image: Input image.
        Returns:
            Preprocessed image as numpy array.
        """
        # Denoise
        if self.denoise_method == 'median':
            image = Denoise.median_filter(image)
        elif self.denoise_method == 'anisotropic':
            image = Denoise.anisotropic_diffusion(image)
        # CLAHE
        if self.apply_clahe:
            image = CLAHE.apply_clahe(image)
        # Normalize
        image = Normalization.normalize(image)
        # Resize
        image = Resize.resize(image, self.size)
        # Augmentation (placeholder)
        if self.augment:
            image = self._augment(image)
        return np.array(image)

    def _augment(self, image: Image.Image) -> Image.Image:
        # Example: horizontal flip
        return image.transpose(Image.FLIP_LEFT_RIGHT)
