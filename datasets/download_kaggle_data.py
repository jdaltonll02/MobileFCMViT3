import os

os.environ["KAGGLEHUB_CACHE"] = "/home/jgibson2/projects/breast_cancer_detection/data"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")

print("Path to dataset files:", path)
