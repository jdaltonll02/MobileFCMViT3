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
    def save_preprocessed(self, class_image_dict, output_dir='data/preprocessed'):
            """
            Preprocess and save images to output_dir, organized by class.
            Args:
                class_image_dict: dict mapping class names to lists of image paths
                output_dir: directory to save preprocessed images
            """
            import os
            os.makedirs(output_dir, exist_ok=True)
            for cls, images in class_image_dict.items():
                cls_dir = os.path.join(output_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                for img_path in images:
                    img = Image.open(img_path)
                    pre_img = self.preprocess(img)
                    # Save as PNG
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    out_path = os.path.join(cls_dir, f'{base}_preprocessed.png')
                    Image.fromarray(pre_img).save(out_path)
                    
    def balance_classes(self, class_image_dict):
        """
        Perform targeted augmentation for minority classes to ensure class balance.
        Args:
            class_image_dict: dict mapping class names to lists of image paths
        Returns:
            dict with balanced lists of image paths
        """
        from random import choice
        max_count = max(len(images) for images in class_image_dict.values())
        balanced = {cls: list(images) for cls, images in class_image_dict.items()}
        for cls, images in balanced.items():
            while len(images) < max_count:
                img_path = choice(images)
                img = Image.open(img_path)
                aug_img = self._augment(img)
                # Optionally save augmented image to disk, here we keep in memory
                images.append(img_path) # Placeholder: replace with actual augmented image path if saving
            balanced[cls] = images
        return balanced
    
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
