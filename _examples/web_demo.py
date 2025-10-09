"""
Example demonstrating the Ocular OCR web application usage.
"""

import asyncio
import aiohttp
import json
from pathlib import Path

async def test_ocr_api():
    """Test the OCR API endpoints."""
    base_url = "http://localhost:8000"
    
    # Test health check
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(f"{base_url}/health") as response:
            health_data = await response.json()
            print("Health Check:")
            print(json.dumps(health_data, indent=2))
            print()
        
        # Get available providers
        async with session.get(f"{base_url}/providers") as response:
            provider_data = await response.json()
            print("Available Providers:")
            print(json.dumps(provider_data, indent=2))
            print()
        
        # Test file upload (if you have sample files)
        sample_files = [
            "sample_image.jpg",
            "sample_document.pdf"
        ]
        
        existing_files = [f for f in sample_files if Path(f).exists()]
        
        if existing_files:
            print(f"Testing file upload with: {existing_files}")
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('strategy', 'fallback')
            data.add_field('providers', 'mistral')
            data.add_field('prompt', 'Extract all text maintaining formatting')
            
            for file_path in existing_files:
                with open(file_path, 'rb') as f:
                    data.add_field('files', f, filename=Path(file_path).name)
            
            # Upload and process
            try:
                async with session.post(f"{base_url}/process", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print("Processing Results:")
                        print(json.dumps(result, indent=2))
                    else:
                        error_text = await response.text()
                        print(f"Error {response.status}: {error_text}")
                        
            except Exception as e:
                print(f"Upload failed: {e}")
        else:
            print("No sample files found. Place some PDFs or images in the examples directory to test.")

async def demonstrate_api_usage():
    """Demonstrate various API usage patterns."""
    print("Ocular OCR Web API Demonstration")
    print("=" * 40)
    
    # Wait for server to be ready
    print("Testing connection to web service...")
    
    try:
        await test_ocr_api()
    except aiohttp.ClientConnectorError:
        print("Error: Could not connect to the web service.")
        print("Please ensure the server is running:")
        print("  python run_server.py")
        print("or")
        print("  uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    asyncio.run(demonstrate_api_usage())