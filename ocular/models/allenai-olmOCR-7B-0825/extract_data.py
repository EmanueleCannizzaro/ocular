import boto3
from botocore.exceptions import NoCredentialsError
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
import os
import sys
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# --- Environment and S3 Configuration ---
# Verify that all required environment variables are set
required_env_vars = [
    "S3_ENDPOINT_URL",
    "S3_BUCKET_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]

if not all(var in os.environ for var in required_env_vars):
    print(f"Error: Missing one or more required environment variables: {required_env_vars}", file=sys.stderr)
    sys.exit(1)

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# --- Model Configuration ---
MODEL_ID = "allenai/olmOCR-7B-0225-preview"
PROCESSOR_ID = "Qwen/Qwen2-VL-7B-Instruct"
LOCAL_MODEL_PATH = "/app/model_cache" # Local path inside the container
S3_MODEL_PREFIX = "olmocr_model"  # Path prefix in the S3 bucket

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def download_from_s3(bucket, s3_prefix, local_path):
    """Downloads a folder from S3."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            if "Contents" not in page:
                return False
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                local_file_path = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3_client.download_file(bucket, s3_key, local_file_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}", file=sys.stderr)
        return False

def upload_to_s3(bucket, local_path, s3_prefix):
    """Uploads a folder to S3."""
    try:
        for root, _, files in os.walk(local_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                s3_key = os.path.join(s3_prefix, os.path.relpath(local_file_path, local_path))
                print(f"Uploading {local_file_path} to {s3_key}...")
                s3_client.upload_file(local_file_path, bucket, s3_key)
        return True
    except NoCredentialsError:
        print("S3 credentials not available. Cannot upload.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error uploading to S3: {e}", file=sys.stderr)
        return False

def main():
    """Main function to run the data extraction."""
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    # Check if model exists locally; if not, try S3; if not, download from Hugging Face
    if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print(f"Model not found locally. Attempting to download from S3 bucket '{S3_BUCKET_NAME}'...")
        if not download_from_s3(S3_BUCKET_NAME, S3_MODEL_PREFIX, LOCAL_MODEL_PATH):
            print("Model not found in S3. Downloading from Hugging Face Hub...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            processor = AutoProcessor.from_pretrained(PROCESSOR_ID)

            print(f"Saving model to {LOCAL_MODEL_PATH}...")
            model.save_pretrained(LOCAL_MODEL_PATH)
            processor.save_pretrained(LOCAL_MODEL_PATH)

            print("Uploading model to S3 for future runs...")
            upload_to_s3(S3_BUCKET_NAME, LOCAL_MODEL_PATH, S3_MODEL_PREFIX)

    print("Loading model from local cache...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on device: {device}")

    # --- Run Inference ---
    pdf_path = "/app/document.pdf"  # Expect the document to be mounted at /app/document.pdf
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'. Mount a volume when running the container.", file=sys.stderr)
        sys.exit(1)

    page_number = 0
    images = render_pdf_to_base64png(pdf_path, page_number)

    prompt = build_finetuning_prompt({"instruction": "Extract all text from the document image."})
    messages = [{"role": "user", "content": [{"type": "image", "image_url": f"data:image/png;base64,{images[0]}"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[images[0]], return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    print("\n--- ✅ Extracted Text ---")
    print(generated_text)


if __name__ == "__main__":
    main()
