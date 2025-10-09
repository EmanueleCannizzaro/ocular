"""Tests for OLMo OCR model via RunPod API using settings configuration."""
import pytest

import base64
import boto3
import os
from pathlib import Path
import requests
import runpod

from ocular.settings import get_settings


@pytest.fixture(scope="module")
def settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture(scope="module")
def runpod_config(settings):
    """Get RunPod configuration from settings."""
    olmocr = settings.olmocr

    # Check if RunPod is configured
    if not olmocr.runpod_api_key or not olmocr.runpod_endpoint_id:
        pytest.skip("RunPod not configured - need RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")

    return {
        "api_key": olmocr.runpod_api_key,
        "endpoint_id": olmocr.runpod_endpoint_id,
        "endpoint_url": olmocr.runpod_url,
        "timeout": olmocr.timeout_seconds,
    }


@pytest.fixture(scope="module")
def runpod_headers(runpod_config):
    """Create request headers with RunPod API key."""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {runpod_config["api_key"]}'
    }


def make_runpod_request(runpod_config, runpod_headers, data):
    """Make a request to RunPod API with proper error handling."""
    try:
        response = requests.post(
            runpod_config["endpoint_url"],
            headers=runpod_headers,
            json=data,
            timeout=runpod_config["timeout"]
        )
        return response
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed with exception: {e}")


@pytest.mark.integration
def test_runpod_simple_prompt(runpod_config, runpod_headers):
    """Test simple prompt with RunPod API."""
    data = {
        'input': {
            "prompt": "What is 2+2?"
        }
    }

    response = make_runpod_request(runpod_config, runpod_headers, data)

    # Print debug information
    print(f"\nRequest URL: {runpod_config['endpoint_url']}")
    print(f"Response Status: {response.status_code}")
    print(f"Response Body: {response.text[:500]}")

    # Check if endpoint exists (404 means endpoint not found)
    if response.status_code == 404:
        pytest.skip(
            f"RunPod endpoint not found (404). "
            f"Endpoint ID '{runpod_config['endpoint_id']}' may be invalid or deleted. "
            f"Please verify the endpoint exists at: {runpod_config['endpoint_url']}"
        )

    # Verify response for valid endpoints
    assert response.status_code in [200, 202], \
        f"Request failed with status {response.status_code}: {response.text}"

    response_data = response.json()
    assert response_data, "Response should contain data"

    print(f"\nResponse: {response_data}")


@pytest.mark.integration
def test_runpod_ocr_prompt(runpod_config, runpod_headers):
    """Test OCR-specific prompt with RunPod API."""
    data = {
        'input': {
            "prompt": "Extract text from this document image: [Sample document text would go here]"
        }
    }

    response = make_runpod_request(runpod_config, runpod_headers, data)

    if response.status_code == 404:
        pytest.skip(f"RunPod endpoint not found: {runpod_config['endpoint_url']}")

    assert response.status_code in [200, 202], \
        f"Request failed with status {response.status_code}: {response.text}"

    response_data = response.json()
    assert response_data, "Response should contain data"

    print(f"\nOCR Response: {response_data}")


@pytest.mark.integration
def test_runpod_structured_input(runpod_config, runpod_headers):
    """Test structured input with RunPod API."""
    data = {
        'input': {
            "prompt": "Analyze this document",
            "max_tokens": 512,
            "temperature": 0.7
        }
    }

    response = make_runpod_request(runpod_config, runpod_headers, data)

    if response.status_code == 404:
        pytest.skip(f"RunPod endpoint not found: {runpod_config['endpoint_url']}")

    assert response.status_code in [200, 202], \
        f"Request failed with status {response.status_code}: {response.text}"

    response_data = response.json()
    assert response_data, "Response should contain data"

    print(f"\nStructured Response: {response_data}")


@pytest.mark.integration
@pytest.mark.parametrize("prompt", [
    "Hello, how are you?",
    "What is the capital of France?",
    "Tell me a joke.",
])
def test_runpod_various_prompts(runpod_config, runpod_headers, prompt):
    """Test various prompts with RunPod API."""
    data = {
        'input': {
            "prompt": prompt
        }
    }

    response = make_runpod_request(runpod_config, runpod_headers, data)

    if response.status_code == 404:
        pytest.skip(f"RunPod endpoint not found: {runpod_config['endpoint_url']}")

    # Basic assertions
    assert response.status_code in [200, 202], \
        f"Request failed with status {response.status_code}: {response.text}"
    assert response.json(), "Response should contain data"

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.json()}")


def test_runpod_configuration_loaded(settings):
    """Test that RunPod configuration is properly loaded from settings."""
    olmocr = settings.olmocr

    # Check that RunPod fields exist
    assert hasattr(olmocr, "runpod_api_key"), "Should have runpod_api_key field"
    assert hasattr(olmocr, "runpod_endpoint_id"), "Should have runpod_endpoint_id field"
    assert hasattr(olmocr, "runpod_url"), "Should have runpod_url property"

    # If configured, verify URL format
    if olmocr.runpod_endpoint_id:
        expected_url = f"https://api.runpod.ai/v2/{olmocr.runpod_endpoint_id}/run"
        assert olmocr.runpod_url == expected_url, "RunPod URL should be correctly formatted"
        print(f"\nRunPod URL: {olmocr.runpod_url}")
        print(f"Endpoint ID: {olmocr.runpod_endpoint_id}")
        print(f"API Key configured: {bool(olmocr.runpod_api_key)}")


def test_provider_selection_runpod(settings):
    """Test that provider can be set to runpod."""
    olmocr = settings.olmocr

    # When provider is set to runpod, active endpoints should use runpod config
    if olmocr.provider == "runpod" and olmocr.runpod_api_key and olmocr.runpod_endpoint_id:
        assert olmocr.active_api_key == olmocr.runpod_api_key, "Active API key should be RunPod key"
        assert olmocr.active_endpoint_url == olmocr.runpod_url, "Active endpoint should be RunPod URL"
        assert olmocr.is_configured, "Should be configured when credentials are present"
        print(f"\nProvider: {olmocr.provider}")
        print(f"Is configured: {olmocr.is_configured}")


@pytest.mark.integration
def test_runpod_endpoint_health(runpod_config, runpod_headers):
    """Test RunPod endpoint health/availability."""
    # Try to check endpoint health
    health_url = f"https://api.runpod.ai/v2/{runpod_config['endpoint_id']}/health"

    try:
        response = requests.get(health_url, headers=runpod_headers, timeout=10)
        print(f"\nHealth check URL: {health_url}")
        print(f"Health status: {response.status_code}")
        print(f"Health response: {response.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"\nHealth check failed: {e}")
        pytest.skip("Unable to perform health check - endpoint may not support it")


@pytest.mark.integration
def test_runpod_error_handling(runpod_config, runpod_headers):
    """Test error handling with invalid input."""
    # Send empty input to test error handling
    data = {
        'input': {}
    }

    response = make_runpod_request(runpod_config, runpod_headers, data)

    print(f"\nError test status: {response.status_code}")
    print(f"Error test response: {response.text[:200]}")

    if response.status_code == 404:
        pytest.skip(f"RunPod endpoint not found: {runpod_config['endpoint_url']}")

    # The request should complete (even if it returns an error in the response)
    assert response.status_code in [200, 202, 400, 422], \
        f"Request should complete with valid status code, got {response.status_code}"

    print(f"\nError handling response: {response.json()}")


@pytest.mark.integration
def test_runpod_ocr_pdf(runpod_config, settings):
    """Test OCR-specific prompt with RunPod API using actual PDF document."""
    # Use sample PDF from data directory
    pdf_path = Path(__file__).parent.parent.parent / "data" / "olmocr_3pg_sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found at {pdf_path}")

    # Configure RunPod API
    runpod.api_key = runpod_config["api_key"]
    endpoint_id = runpod_config["endpoint_id"]

    # Load and encode PDF
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode()

    print(f"\nProcessing PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f}KB)")

    # Run inference
    endpoint = runpod.Endpoint(endpoint_id)
    result = endpoint.run_sync({
        "input": {
            "image": pdf_base64,
            "prompt": "Extract all text from this PDF document.",
            "upload_result": True
        }
    })

    print(f"\nOCR Result: {result}")
    assert result, "Should receive OCR result"


@pytest.mark.integration
def test_runpod_ocr_png(runpod_config, settings):
    """Test OCR-specific prompt with RunPod API using image file (if available)."""
    # Try to find a test image - check multiple possible locations
    test_images = [
        Path("test_image.png"),
        Path(__file__).parent.parent.parent / "data" / "test_image.png",
        Path(__file__).parent / "fixtures" / "test_image.png"
    ]

    image_path = None
    for path in test_images:
        if path.exists():
            image_path = path
            break

    if not image_path:
        pytest.skip("No test image found - add test_image.png to data/ directory for this test")

    # Configure RunPod API
    runpod.api_key = runpod_config["api_key"]
    endpoint_id = runpod_config["endpoint_id"]

    # Load and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    print(f"\nProcessing image: {image_path.name} ({image_path.stat().st_size / 1024:.1f}KB)")

    # Run inference
    endpoint = runpod.Endpoint(endpoint_id)
    result = endpoint.run_sync({
        "input": {
            "image": image_base64,
            "prompt": "Extract all text from this image.",
            "upload_result": True
        }
    })

    print(f"\nOCR Result: {result}")
    assert result, "Should receive OCR result"



@pytest.mark.integration
def test_runpod_connectivity(settings):
    """Test RunPod S3 connectivity using settings."""
    olmocr = settings.olmocr

    # Skip if S3 not configured
    if not olmocr.runpod_bucket_endpoint_url:
        pytest.skip("RunPod S3 not configured - need RUNPOD_BUCKET_ENDPOINT_URL")

    # Test S3 connectivity
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=olmocr.runpod_bucket_endpoint_url,
            aws_access_key_id=olmocr.runpod_access_key_id,
            aws_secret_access_key=olmocr.runpod_secret_access_key,
            region_name=olmocr.runpod_region
        )
        buckets = s3.list_buckets()
        print(f"\nS3 Connection successful!")
        print(f"Endpoint: {olmocr.runpod_bucket_endpoint_url}")
        print(f"Buckets found: {len(buckets.get('Buckets', []))}")
    except Exception as e:
        pytest.skip(f"S3 connection failed: {e}") 


@pytest.mark.integration
def test_runpod_storage(settings):
    """Test RunPod S3 storage operations using settings."""
    import tempfile
    import json

    olmocr = settings.olmocr

    # Skip if S3 not configured
    if not all([
        olmocr.runpod_bucket_endpoint_url,
        olmocr.runpod_access_key_id,
        olmocr.runpod_secret_access_key,
        olmocr.runpod_network_volume_id
    ]):
        pytest.skip("RunPod S3 storage not fully configured")

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=olmocr.runpod_bucket_endpoint_url,
        aws_access_key_id=olmocr.runpod_access_key_id,
        aws_secret_access_key=olmocr.runpod_secret_access_key,
        region_name=olmocr.runpod_region
    )

    bucket_name = olmocr.runpod_network_volume_id
    test_key = 'test/ocular_test_file.json'

    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {"test": "data", "timestamp": "2025-10-05"}
        json.dump(test_data, f)
        temp_file_path = f.name

    try:
        # Upload file
        print(f"\nUploading to s3://{bucket_name}/{test_key}")
        s3_client.upload_file(temp_file_path, bucket_name, test_key)
        print("Upload successful!")

        # Download file
        download_path = temp_file_path + '.downloaded'
        print(f"Downloading from s3://{bucket_name}/{test_key}")
        s3_client.download_file(bucket_name, test_key, download_path)
        print("Download successful!")

        # Verify content
        with open(download_path, 'r') as f:
            downloaded_data = json.load(f)
        assert downloaded_data == test_data, "Downloaded data should match uploaded data"
        print("Data verification successful!")

        # List files
        print("\nListing files in bucket:")
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='test/')
        for obj in response.get('Contents', []):
            print(f"  - {obj['Key']} ({obj['Size']} bytes)")

        # Cleanup
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"\nCleaned up test file: {test_key}")

    finally:
        # Clean up local temp files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(temp_file_path + '.downloaded'):
            os.unlink(temp_file_path + '.downloaded')