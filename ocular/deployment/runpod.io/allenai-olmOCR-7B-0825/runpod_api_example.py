#!/usr/bin/env python3
"""
RunPod API Example for OLMocr vLLM Deployment
This script demonstrates the correct API format for RunPod serverless endpoints.

Based on the provided RunPod API format:
- Endpoint: https://api.runpod.ai/v2/{endpoint_id}/run
- Headers: Authorization Bearer token + Content-Type application/json
- Payload: {"input": {...}}
"""

import requests
import base64
import json
import os
from pathlib import Path

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "6dldf1jqc8frk8")  # Example from provided format

def create_test_image_base64():
    """Create a simple test image in base64 format"""
    # This is a minimal 1x1 PNG in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def call_runpod_olmocr(image_base64=None, prompt="Extract text from this document and convert to markdown format."):
    """
    Call RunPod serverless endpoint for OLMocr processing

    Args:
        image_base64 (str): Base64 encoded image data
        prompt (str): Text extraction prompt

    Returns:
        dict: API response
    """

    # RunPod API endpoint format
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

    # Headers as specified in the provided format
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }

    # Use test image if none provided
    if image_base64 is None:
        image_base64 = create_test_image_base64()

    # Payload format for vLLM with multimodal input
    data = {
        'input': {
            "model": "allenai/olmOCR-7B-0225-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
    }

    print(f"Calling RunPod endpoint: {url}")
    print(f"Payload preview: {json.dumps(data, indent=2)[:500]}...")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)
        response.raise_for_status()

        result = response.json()
        print(f"Success! Response status: {response.status_code}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error calling RunPod API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def call_runpod_simple_prompt(prompt="Hello, world!"):
    """
    Simple text-only prompt for testing (no image)

    Args:
        prompt (str): Text prompt

    Returns:
        dict: API response
    """

    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RUNPOD_API_KEY}'
    }

    # Simple text-only payload
    data = {
        'input': {
            "prompt": prompt
        }
    }

    print(f"Calling RunPod endpoint with simple prompt: {url}")
    print(f"Payload: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()
        print(f"Success! Response status: {response.status_code}")
        return result

    except requests.exceptions.RequestException as e:
        print(f"Error calling RunPod API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def load_image_from_file(file_path):
    """
    Load image file and convert to base64

    Args:
        file_path (str): Path to image file

    Returns:
        str: Base64 encoded image data
    """
    try:
        with open(file_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error loading image file: {e}")
        return None

def main():
    """Main example function"""

    print("RunPod OLMocr API Example")
    print("=" * 40)

    # Check configuration
    if RUNPOD_API_KEY == "YOUR_API_KEY":
        print("⚠️  Please set RUNPOD_API_KEY environment variable")
        print("   export RUNPOD_API_KEY=your_actual_api_key")
        return

    if ENDPOINT_ID == "6dldf1jqc8frk8":
        print("⚠️  Using example endpoint ID. Set RUNPOD_ENDPOINT_ID for your actual endpoint")
        print("   export RUNPOD_ENDPOINT_ID=your_actual_endpoint_id")

    print(f"API Key: {RUNPOD_API_KEY[:8]}...")
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print()

    # Example 1: Simple prompt (based on provided format)
    print("Example 1: Simple prompt test")
    print("-" * 30)
    result1 = call_runpod_simple_prompt("Hello from RunPod!")
    if result1:
        print(f"Result: {json.dumps(result1, indent=2)}")
    print()

    # Example 2: OLMocr with test image
    print("Example 2: OLMocr with test image")
    print("-" * 30)
    result2 = call_runpod_olmocr()
    if result2:
        print(f"Result: {json.dumps(result2, indent=2)}")
    print()

    # Example 3: Load actual image file (if provided)
    import sys
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        if Path(image_file).exists():
            print(f"Example 3: Processing image file: {image_file}")
            print("-" * 30)
            image_b64 = load_image_from_file(image_file)
            if image_b64:
                result3 = call_runpod_olmocr(
                    image_base64=image_b64,
                    prompt="Extract all text from this document and convert to clean markdown format."
                )
                if result3:
                    print(f"Result: {json.dumps(result3, indent=2)}")
            print()

if __name__ == "__main__":
    main()

# Example usage:
# python runpod_api_example.py
# python runpod_api_example.py /path/to/image.png

# Environment setup:
# export RUNPOD_API_KEY=your_api_key_here
# export RUNPOD_ENDPOINT_ID=your_endpoint_id_here
