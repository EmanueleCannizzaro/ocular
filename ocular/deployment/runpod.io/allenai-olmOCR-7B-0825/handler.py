"""
OLMocr Serverless Handler for RunPod
Handles OCR requests with S3 storage integration for allenai/olmOCR-7B-0825
"""

import runpod
import boto3
import base64
import os
import json
from io import BytesIO
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get('RUNPOD_BUCKET_ENDPOINT_URL'),
    aws_access_key_id=os.environ.get('RUNPOD_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('RUNPOD_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('RUNPOD_REGION', 'eur-is-1')
)

# Load OLMocr model
MODEL_NAME = os.environ.get('MODEL_NAME', "allenai/olmOCR-7B-0825")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"Model loaded successfully on {device}")


def upload_to_s3(file_path, s3_key):
    """Upload file to RunPod S3 bucket."""
    bucket_name = os.environ.get('RUNPOD_NETWORK_VOLUME_ID')
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        print(f"S3 upload error: {e}")
        return None


def download_from_s3(s3_key):
    """Download file from RunPod S3 bucket."""
    bucket_name = os.environ.get('RUNPOD_NETWORK_VOLUME_ID')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        print(f"S3 download error: {e}")
        return None


def process_image(image_data, prompt="Extract all text from this image."):
    """Process image with OLMocr model."""
    try:
        # Decode base64 image if needed
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        # Load image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Prepare inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate OCR output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048)

        # Decode result
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return {
            "success": True,
            "text": result,
            "model": MODEL_NAME,
            "device": device
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def handler(job):
    """
    RunPod handler function.

    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image_or_s3_key",
            "prompt": "Optional custom prompt",
            "upload_result": true,  # Upload result to S3
            "s3_input": false,      # If true, treat 'image' as S3 key
            "max_new_tokens": 2048  # Override max tokens
        }
    }
    """
    try:
        job_input = job["input"]

        # Get image data
        if job_input.get("s3_input", False):
            # Download from S3
            s3_key = job_input["image"]
            image_data = download_from_s3(s3_key)
            if not image_data:
                return {"error": "Failed to download image from S3"}
        else:
            # Use provided base64 image
            image_data = job_input["image"]

        # Get custom prompt (optional)
        prompt = job_input.get("prompt", "Extract all text from this image.")

        # Process image
        result = process_image(image_data, prompt)

        # Upload result to S3 if requested
        if job_input.get("upload_result", False) and result["success"]:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(result, f)
                temp_path = f.name

            s3_key = f"ocr_results/{job['id']}_result.json"
            s3_url = upload_to_s3(temp_path, s3_key)
            os.unlink(temp_path)

            if s3_url:
                result["s3_url"] = s3_url

        return result

    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
