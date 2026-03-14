"""
DICOM to PNG converter for ultrasound images.
"""

import pydicom
from PIL import Image
from typing import Union
import numpy as np
import os

class DICOMConverter:
    @staticmethod
    def dicom_to_png(dicom_path: str, png_path: str) -> None:
        """
        Convert a DICOM file to PNG format.
        Args:
            dicom_path: Path to the DICOM file.
            png_path: Path to save the PNG file.
        """
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        im = Image.fromarray(img)
        im.save(png_path)
