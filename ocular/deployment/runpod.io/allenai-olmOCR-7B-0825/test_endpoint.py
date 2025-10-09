#!/usr/bin/env python3
"""
Test OLMocr RunPod Endpoint
Simple script to test your deployed OLMocr endpoint
"""

import os
import sys
import base64
import json
import time
from pathlib import Path
from dotenv import load_dotenv

try:
    import runpod
except ImportError:
    print("Error: runpod package not installed")
    print("Install with: pip install runpod")
    sys.exit(1)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_endpoint(endpoint_id: str, image_path: Path = None, prompt: str = None):
    """Test OLMocr endpoint with sample image."""

    # Configure RunPod
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY not set in .env file")
        sys.exit(1)

    runpod.api_key = api_key

    # Find test image
    if image_path is None:
        # Try to find a test image
        test_images = [
            Path("../../data/olmocr_3pg_sample.pdf"),
            Path("../../data/test_document.pdf"),
            Path("test_image.png"),
        ]
        for img_path in test_images:
            if img_path.exists():
                image_path = img_path
                break

    if image_path is None or not image_path.exists():
        print(f"Error: Test image not found")
        print("Please provide a valid image path:")
        print(f"  python {sys.argv[0]} <endpoint_id> <image_path>")
        sys.exit(1)

    print("=" * 60)
    print("  Testing OLMocr RunPod Endpoint")
    print("=" * 60)
    print(f"Endpoint ID: {endpoint_id}")
    print(f"Test Image: {image_path}")
    print(f"Image Size: {image_path.stat().st_size / 1024:.1f} KB")
    print("=" * 60)
    print("")

    # Encode image
    print("Encoding image...")
    image_base64 = encode_image(image_path)
    print(f"✓ Image encoded ({len(image_base64)} bytes)")

    # Prepare request
    default_prompt = "Extract all text from this document, preserving layout and formatting."
    request_input = {
        "input": {
            "image": image_base64,
            "prompt": prompt or default_prompt,
            "upload_result": True
        }
    }

    print(f"Prompt: {request_input['input']['prompt']}")
    print("")

    # Run inference
    print("Sending request to endpoint...")
    print("(This may take 30-60 seconds for cold start)")
    print("")

    try:
        endpoint = runpod.Endpoint(endpoint_id)
        start_time = time.time()

        result = endpoint.run_sync(request_input, timeout=300)

        elapsed_time = time.time() - start_time

        print("=" * 60)
        print("  Result")
        print("=" * 60)
        print(f"Status: {'✓ Success' if result.get('success') else '✗ Failed'}")
        print(f"Processing Time: {elapsed_time:.2f}s")
        print("")

        if result.get("success"):
            print("Extracted Text:")
            print("-" * 60)
            print(result.get("text", "No text extracted"))
            print("-" * 60)
            print("")
            print(f"Model: {result.get('model', 'N/A')}")
            print(f"Device: {result.get('device', 'N/A')}")

            if "s3_url" in result:
                print(f"S3 URL: {result['s3_url']}")

            print("")
            print("✓ Test completed successfully!")

        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            print("")
            print("✗ Test failed!")

        print("=" * 60)

        return result.get("success", False)

    except Exception as e:
        print("=" * 60)
        print(f"✗ Error: {e}")
        print("=" * 60)
        print("")
        print("Troubleshooting:")
        print("1. Check endpoint is active in RunPod console")
        print("2. Verify RUNPOD_API_KEY is correct")
        print("3. Ensure endpoint has finished deploying")
        print("4. Check RunPod logs for errors")
        print("")
        return False


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_endpoint.py <endpoint_id> [image_path] [prompt]")
        print("")
        print("Examples:")
        print("  python test_endpoint.py abc123")
        print("  python test_endpoint.py abc123 test.png")
        print("  python test_endpoint.py abc123 test.png 'Extract invoice details'")
        print("")

        # Try to load endpoint ID from file
        endpoint_file = Path(__file__).parent / "endpoint_id.txt"
        if endpoint_file.exists():
            endpoint_id = endpoint_file.read_text().strip()
            print(f"Found saved endpoint ID: {endpoint_id}")
            print("Running test with saved endpoint...")
            print("")
            test_endpoint(endpoint_id)
        else:
            sys.exit(1)
    else:
        endpoint_id = sys.argv[1]
        image_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        prompt = sys.argv[3] if len(sys.argv) > 3 else None

        success = test_endpoint(endpoint_id, image_path, prompt)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
