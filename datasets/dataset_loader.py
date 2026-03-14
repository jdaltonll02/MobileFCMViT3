"""
Dataset loader for ultrasound images.
"""

from typing import List, Tuple
import os
import numpy as np
from PIL import Image
import pydicom

class DatasetLoader:
    @staticmethod
    def load_image(path: str) -> Image.Image:
        """
        Load image from PNG, JPG, or DICOM.
        Args:
            path: Image file path.
        Returns:
            PIL Image.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.dcm':
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            return Image.fromarray(img)
        else:
            return Image.open(path).convert('L')

    @staticmethod
    def split_dataset(image_paths: List[str], split: dict, random_seed: int) -> Tuple[List[str], List[str], List[str]]:
        """
        Split dataset into train, val, test.
        Args:
            image_paths: List of image paths.
            split: Dict with train/val/test ratios.
            random_seed: Seed for shuffling.
        Returns:
            train, val, test lists.
        """
        np.random.seed(random_seed)
        np.random.shuffle(image_paths)
        n_total = len(image_paths)
        n_train = int(split['train'] * n_total)
        n_val = int(split['val'] * n_total)
        train = image_paths[:n_train]
        val = image_paths[n_train:n_train+n_val]
        test = image_paths[n_train+n_val:]
        return train, val, test
