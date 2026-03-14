"""
Custom transforms for ultrasound dataset.
"""

from typing import Callable
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class UltrasoundTransforms:
    @staticmethod
    def get_transforms(image_size: int = 224, augment: bool = False) -> Callable:
        t = [transforms.Resize((image_size, image_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])]
        if augment:
            t.insert(0, transforms.RandomHorizontalFlip())
        return transforms.Compose(t)
