"""
S3 service for uploading models and artifacts.

Supports RunPod S3-compatible storage and AWS S3.
"""

import os
import boto3
import asyncio
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
from botocore.exceptions import ClientError


class S3Service:
    """Service for uploading models and files to S3-compatible storage."""

    def __init__(
        self,
        provider: str = "runpod",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize S3 service.

        Args:
            provider: Storage provider ("runpod", "aws", "custom")
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            endpoint_url: S3-compatible endpoint URL (for RunPod, etc.)
            region: AWS region
        """
        self.provider = provider

        # Get credentials from parameters or environment variables
        if provider == "runpod":
            self.access_key_id = access_key_id or os.getenv("RUNPOD_BUCKET_ACCESS_KEY_ID")
            self.secret_access_key = secret_access_key or os.getenv("RUNPOD_BUCKET_SECRET_ACCESS_KEY")
            self.endpoint_url = endpoint_url or os.getenv("RUNPOD_BUCKET_ENDPOINT_URL", "https://storage.runpod.io")
            self.region = "us-east-1"  # RunPod uses us-east-1
        elif provider == "aws":
            self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            self.endpoint_url = None  # Use AWS S3
            self.region = region or os.getenv("AWS_REGION", "us-east-1")
        else:
            # Custom S3-compatible storage
            self.access_key_id = access_key_id
            self.secret_access_key = secret_access_key
            self.endpoint_url = endpoint_url
            self.region = region

        if not self.access_key_id or not self.secret_access_key:
            raise ValueError(f"S3 credentials not provided for provider: {provider}")

        # Create S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region
        )

    async def upload_file(
        self,
        file_path: str,
        bucket: str,
        object_key: str,
        metadata: Optional[Dict[str, str]] = None,
        make_public: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a file to S3.

        Args:
            file_path: Local path to file
            bucket: S3 bucket name
            object_key: S3 object key (path in bucket)
            metadata: Optional metadata to attach to object
            make_public: Whether to make object publicly readable

        Returns:
            Dict with upload result including URL

        Raises:
            FileNotFoundError: If file doesn't exist
            ClientError: If upload fails
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare upload parameters
        extra_args = {}

        if metadata:
            extra_args['Metadata'] = metadata

        if make_public:
            extra_args['ACL'] = 'public-read'

        # Upload file asynchronously
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.upload_file(
                    file_path,
                    bucket,
                    object_key,
                    ExtraArgs=extra_args if extra_args else None
                )
            )

            # Generate URL
            if self.endpoint_url:
                # S3-compatible storage
                url = f"{self.endpoint_url}/{bucket}/{object_key}"
            else:
                # AWS S3
                url = f"https://{bucket}.s3.{self.region}.amazonaws.com/{object_key}"

            # Get file size
            file_size = Path(file_path).stat().st_size

            return {
                "success": True,
                "bucket": bucket,
                "key": object_key,
                "url": url,
                "size_bytes": file_size,
                "provider": self.provider,
                "metadata": metadata or {}
            }

        except ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.response.get('Error', {}).get('Code'),
                "message": e.response.get('Error', {}).get('Message')
            }

    async def upload_fileobj(
        self,
        file_obj: BinaryIO,
        bucket: str,
        object_key: str,
        metadata: Optional[Dict[str, str]] = None,
        make_public: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a file object to S3.

        Args:
            file_obj: File-like object to upload
            bucket: S3 bucket name
            object_key: S3 object key (path in bucket)
            metadata: Optional metadata to attach to object
            make_public: Whether to make object publicly readable

        Returns:
            Dict with upload result including URL

        Raises:
            ClientError: If upload fails
        """
        # Prepare upload parameters
        extra_args = {}

        if metadata:
            extra_args['Metadata'] = metadata

        if make_public:
            extra_args['ACL'] = 'public-read'

        # Upload file object asynchronously
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.upload_fileobj(
                    file_obj,
                    bucket,
                    object_key,
                    ExtraArgs=extra_args if extra_args else None
                )
            )

            # Generate URL
            if self.endpoint_url:
                # S3-compatible storage
                url = f"{self.endpoint_url}/{bucket}/{object_key}"
            else:
                # AWS S3
                url = f"https://{bucket}.s3.{self.region}.amazonaws.com/{object_key}"

            # Get file size (if available)
            file_obj.seek(0, 2)  # Seek to end
            file_size = file_obj.tell()
            file_obj.seek(0)  # Seek back to start

            return {
                "success": True,
                "bucket": bucket,
                "key": object_key,
                "url": url,
                "size_bytes": file_size,
                "provider": self.provider,
                "metadata": metadata or {}
            }

        except ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.response.get('Error', {}).get('Code'),
                "message": e.response.get('Error', {}).get('Message')
            }

    async def download_file(
        self,
        bucket: str,
        object_key: str,
        local_path: str
    ) -> Dict[str, Any]:
        """
        Download a file from S3.

        Args:
            bucket: S3 bucket name
            object_key: S3 object key (path in bucket)
            local_path: Local path to save file

        Returns:
            Dict with download result

        Raises:
            ClientError: If download fails
        """
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.download_file(bucket, object_key, local_path)
            )

            file_size = Path(local_path).stat().st_size

            return {
                "success": True,
                "bucket": bucket,
                "key": object_key,
                "local_path": local_path,
                "size_bytes": file_size
            }

        except ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.response.get('Error', {}).get('Code'),
                "message": e.response.get('Error', {}).get('Message')
            }

    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> Dict[str, Any]:
        """
        List objects in S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix to filter objects
            max_keys: Maximum number of keys to return

        Returns:
            Dict with list of objects

        Raises:
            ClientError: If listing fails
        """
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix or '',
                    MaxKeys=max_keys
                )
            )

            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    "key": obj['Key'],
                    "size_bytes": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "etag": obj['ETag'].strip('"')
                })

            return {
                "success": True,
                "bucket": bucket,
                "prefix": prefix,
                "objects": objects,
                "count": len(objects),
                "is_truncated": response.get('IsTruncated', False)
            }

        except ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.response.get('Error', {}).get('Code'),
                "message": e.response.get('Error', {}).get('Message')
            }

    async def delete_object(
        self,
        bucket: str,
        object_key: str
    ) -> Dict[str, Any]:
        """
        Delete an object from S3.

        Args:
            bucket: S3 bucket name
            object_key: S3 object key (path in bucket)

        Returns:
            Dict with deletion result

        Raises:
            ClientError: If deletion fails
        """
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.delete_object(Bucket=bucket, Key=object_key)
            )

            return {
                "success": True,
                "bucket": bucket,
                "key": object_key,
                "message": "Object deleted successfully"
            }

        except ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.response.get('Error', {}).get('Code'),
                "message": e.response.get('Error', {}).get('Message')
            }

    def generate_presigned_url(
        self,
        bucket: str,
        object_key: str,
        expiration: int = 3600,
        http_method: str = 'GET'
    ) -> str:
        """
        Generate a presigned URL for S3 object access.

        Args:
            bucket: S3 bucket name
            object_key: S3 object key (path in bucket)
            expiration: URL expiration time in seconds (default 1 hour)
            http_method: HTTP method for the URL (GET, PUT, etc.)

        Returns:
            Presigned URL string

        Raises:
            ClientError: If URL generation fails
        """
        try:
            client_method = 'get_object' if http_method == 'GET' else 'put_object'

            url = self.s3_client.generate_presigned_url(
                client_method,
                Params={'Bucket': bucket, 'Key': object_key},
                ExpiresIn=expiration
            )

            return url

        except ClientError as e:
            raise Exception(f"Failed to generate presigned URL: {str(e)}")


# Singleton instances for different providers
_s3_services: Dict[str, S3Service] = {}


def get_s3_service(provider: str = "runpod") -> S3Service:
    """
    Get or create S3 service instance for provider.

    Args:
        provider: Storage provider ("runpod", "aws", "custom")

    Returns:
        S3Service instance
    """
    global _s3_services

    if provider not in _s3_services:
        _s3_services[provider] = S3Service(provider=provider)

    return _s3_services[provider]
