"""
Adapter for external data sources and APIs (PACS, EMR, cloud storage).
"""

import requests
import boto3
from typing import Any

class PACSAdapter:
    def fetch_dicom(self, url: str) -> bytes:
        response = requests.get(url)
        return response.content

class EMRAdapter:
    def fetch_patient_record(self, api_url: str, patient_id: str) -> Any:
        response = requests.get(f"{api_url}/patients/{patient_id}")
        return response.json()

class S3Adapter:
    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
    def download_file(self, key: str, dest: str):
        self.s3.download_file(self.bucket, key, dest)
