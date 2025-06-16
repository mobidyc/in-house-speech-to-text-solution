import os
from io import BytesIO
from typing import Optional

import boto3
from pydantic import BaseModel

from app.resources.Config import config


class TransferFileRequest(BaseModel):
    filename: str
    file_object: Optional[BytesIO] = None
    source_path: Optional[str] = None
    destination_key: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allow BytesIO type


class TransferFile:
    def __init__(
        self,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_bucket_name: Optional[str] = None,
    ):
        self.s3_endpoint = s3_endpoint or config.s3_endpoint
        self.s3_access_key = s3_access_key or config.s3_access_key
        self.s3_secret_key = s3_secret_key or config.s3_secret_key
        if not self.s3_endpoint or not self.s3_access_key or not self.s3_secret_key:
            raise ValueError("S3 configuration is incomplete. Please check your settings.")

        self.s3_bucket_name = s3_bucket_name or config.s3_bucket_name
        if not self.s3_bucket_name:
            raise ValueError("Bucket name must be provided.")

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
        )

    def upload_file(self, request: TransferFileRequest) -> bool:
        try:
            # Determine source path and destination key
            destination_key = request.destination_key or os.path.basename(request.filename)

            # Upload file to S3
            # Check if we have a file object or file path
            if request.file_object:
                # Upload from BytesIO object
                request.file_object.seek(0)  # Ensure we're at the beginning
                self.s3_client.upload_fileobj(request.file_object, self.s3_bucket_name, destination_key)
            elif request.source_path:
                # Upload from file path (existing functionality)
                self.s3_client.upload_file(request.source_path, self.s3_bucket_name, destination_key)
            else:
                raise ValueError("Either file_object or source_path must be provided")

            return True

        except Exception as e:
            raise Exception(f"Failed to upload file {request.filename}: {str(e)}") from e

    def download_file(self, request: TransferFileRequest) -> bool:
        try:
            # Determine source key and destination path
            destination_path = request.destination_key or os.path.basename(request.filename)
            source_key = request.source_path or request.filename

            # Download file from S3
            print(f"Downloading file from S3: {self.s3_bucket_name} {source_key} to {destination_path}")
            self.s3_client.download_file(Bucket=self.s3_bucket_name, Key=source_key, Filename=destination_path)

            return True

        except Exception as e:
            raise Exception(f"Failed to download file {request.filename}: {str(e)}") from e
