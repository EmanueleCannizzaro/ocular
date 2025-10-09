#!/usr/bin/env python3
"""
OLMocr RunPod API Deployment Script
Deploys OLMocr endpoint to RunPod using the REST API
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
DOCKER_USERNAME = os.getenv("DOCKER_USERNAME", "ocular")
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")
DOCKER_IMAGE = f"{DOCKER_USERNAME}/olmocr-runpod:{IMAGE_TAG}"

# Endpoint configuration
ENDPOINT_CONFIG = {
    "name": os.getenv("RUNPOD_ENDPOINT_NAME", "olmocr-endpoint"),
    "template_id": None,  # Custom template
    "gpu_ids": os.getenv("GPU_TYPE", "NVIDIA RTX 4090"),
    "network_volume_id": os.getenv("RUNPOD_NETWORK_VOLUME_ID"),
    "container_disk_in_gb": int(os.getenv("CONTAINER_DISK_GB", "30")),
    "volumes_in_gb": int(os.getenv("CONTAINER_DISK_GB", "30")),
    "min_workers": int(os.getenv("MIN_WORKERS", "0")),
    "max_workers": int(os.getenv("MAX_WORKERS", "3")),
    "idle_timeout": int(os.getenv("IDLE_TIMEOUT", "30")),
    "execution_timeout": int(os.getenv("EXECUTION_TIMEOUT", "600")),
    "docker_image": DOCKER_IMAGE,
    "env": {
        "RUNPOD_ACCESS_KEY_ID": os.getenv("RUNPOD_ACCESS_KEY_ID", ""),
        "RUNPOD_SECRET_ACCESS_KEY": os.getenv("RUNPOD_SECRET_ACCESS_KEY", ""),
        "RUNPOD_BUCKET_ENDPOINT_URL": os.getenv("RUNPOD_BUCKET_ENDPOINT_URL", ""),
        "RUNPOD_NETWORK_VOLUME_ID": os.getenv("RUNPOD_NETWORK_VOLUME_ID", ""),
        "RUNPOD_REGION": os.getenv("RUNPOD_REGION", "eur-is-1"),
        "MODEL_NAME": "allenai/olmOCR-7B-0825"
    }
}


def print_status(message, status="info"):
    """Print colored status message."""
    colors = {
        "info": "\033[0;34m",  # Blue
        "success": "\033[0;32m",  # Green
        "warning": "\033[1;33m",  # Yellow
        "error": "\033[0;31m",  # Red
    }
    reset = "\033[0m"
    prefix = {
        "info": "ℹ",
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
    }
    print(f"{colors[status]}{prefix[status]} {message}{reset}")


def deploy_endpoint():
    """Deploy endpoint to RunPod."""

    if not RUNPOD_API_KEY:
        print_status("RUNPOD_API_KEY not set in .env file", "error")
        print("\nTo deploy via API, you need to:")
        print("1. Get your API key from https://console.runpod.io/settings")
        print("2. Add RUNPOD_API_KEY=your_key to .env file")
        print("\nAlternatively, deploy manually via the console:")
        print("https://console.runpod.io/serverless")
        return False

    print_status("Deploying OLMocr endpoint to RunPod...", "info")
    print(f"  Docker Image: {DOCKER_IMAGE}")
    print(f"  Endpoint Name: {ENDPOINT_CONFIG['name']}")
    print(f"  GPU Type: {ENDPOINT_CONFIG['gpu_ids']}")
    print("")

    # API endpoint
    api_url = "https://api.runpod.io/graphql"

    # GraphQL mutation for creating endpoint
    mutation = """
    mutation CreateEndpoint($input: EndpointInput!) {
        createEndpoint(input: $input) {
            id
            name
            status
        }
    }
    """

    # Prepare GraphQL request
    payload = {
        "query": mutation,
        "variables": {
            "input": ENDPOINT_CONFIG
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        print_status("Sending deployment request...", "info")
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()

        # Check for errors
        if "errors" in result:
            print_status("Deployment failed!", "error")
            print(json.dumps(result["errors"], indent=2))
            return False

        # Success
        endpoint_data = result.get("data", {}).get("createEndpoint", {})
        endpoint_id = endpoint_data.get("id")

        print_status("Deployment successful!", "success")
        print(f"\nEndpoint Details:")
        print(f"  ID: {endpoint_id}")
        print(f"  Name: {endpoint_data.get('name')}")
        print(f"  Status: {endpoint_data.get('status')}")
        print(f"\nEndpoint URL:")
        print(f"  https://api.runpod.ai/v2/{endpoint_id}/run")
        print(f"\nManage endpoint:")
        print(f"  https://console.runpod.io/serverless")

        # Save endpoint ID
        endpoint_file = Path(__file__).parent / "endpoint_id.txt"
        endpoint_file.write_text(endpoint_id)
        print_status(f"Endpoint ID saved to {endpoint_file}", "success")

        return True

    except requests.exceptions.RequestException as e:
        print_status(f"API request failed: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        return False


def main():
    """Main deployment function."""
    print("=" * 50)
    print("  OLMocr RunPod API Deployment")
    print("  Model: allenai/olmOCR-7B-0825")
    print("=" * 50)
    print("")

    success = deploy_endpoint()

    if success:
        print("")
        print("=" * 50)
        print("  Next Steps:")
        print("=" * 50)
        print("1. Wait for endpoint to become active (2-5 minutes)")
        print("2. Test endpoint: python test_endpoint.py <endpoint_id>")
        print("3. Monitor logs in RunPod console")
        print("")
        sys.exit(0)
    else:
        print("")
        print_status("Deployment via API failed", "warning")
        print("\nDeploy manually:")
        print("1. Go to https://console.runpod.io/serverless")
        print("2. Click 'Deploy Endpoint'")
        print(f"3. Use Docker image: {DOCKER_IMAGE}")
        print("4. Configure environment variables from .env file")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
