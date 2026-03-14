"""
Wavelet denoising for ultrasound images.
"""

import numpy as np
import pywt
from PIL import Image
from typing import Union

class WaveletDenoising:
    @staticmethod
    def denoise(image: Union[np.ndarray, Image.Image], wavelet: str = 'db1', level: int = 2) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        coeffs_filtered = list(coeffs)
        coeffs_filtered[1:] = [tuple(pywt.threshold(c, np.std(c)/2, mode='soft') for c in detail) for detail in coeffs[1:]]
        denoised = pywt.waverec2(coeffs_filtered, wavelet)
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        return denoised
