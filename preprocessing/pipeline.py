import os
import numpy as np
from PIL import Image
import pydicom
import cv2
from torchvision import transforms

class UltrasoundPreprocessingPipeline:
    def __init__(self, image_size=224, apply_augmentation=False):
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.transform = self._build_transform()

    def _build_transform(self):
        t = [transforms.Resize((self.image_size, self.image_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])]
        if self.apply_augmentation:
            t.insert(0, transforms.RandomHorizontalFlip())
        return transforms.Compose(t)

    def dicom_to_png(self, dicom_path, png_path):
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        img = Image.fromarray(img)
        img.save(png_path)

    def speckle_noise_reduction(self, img):
        img_np = np.array(img)
        filtered = cv2.medianBlur(img_np, 3)
        return Image.fromarray(filtered)

    def clahe_contrast(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced)

    def normalize_intensity(self, img):
        img_np = np.array(img).astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def preprocess(self, img):
        img = self.speckle_noise_reduction(img)
        img = self.clahe_contrast(img)
        img = self.normalize_intensity(img)
        img = img.resize((self.image_size, self.image_size))
        img = self.transform(img)
        return img
