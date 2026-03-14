
"""
PyTorch Dataset for ultrasound images.
"""

from typing import List, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image

class UltrasoundDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform: Callable = None):
        """
        Args:
            image_paths: List of image file paths.
            labels: List of labels.
            transform: Optional transform function.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
