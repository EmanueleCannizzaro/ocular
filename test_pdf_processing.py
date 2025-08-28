#!/usr/bin/env python3
"""
Test PDF processing to diagnose the issue.
"""

import os
import asyncio
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

async def test_pdf_processing():
    """Test PDF processing capabilities."""
    print("=== Testing PDF Processing ===\n")
    
    # Check for pdf2image dependency
    print("1. Checking dependencies...")
    try:
        from pdf2image import convert_from_path
        print("   ✓ pdf2image is available")
    except ImportError:
        print("   ✗ pdf2image not available")
        print("   Note: PDF processing will require this library")
        print("   Install with: pip install pdf2image")
        # Continue without pdf2image for now
    
    # Test provider creation
    print("\n2. Testing Mistral provider with PDF support...")
    try:
        from ocular.providers.mistral import MistralProvider
        
        config = {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "model": "pixtral-12b-2409",
            "timeout": 30,
            "max_retries": 3
        }
        
        provider = MistralProvider(config)
        print(f"   ✓ Provider created: {provider.provider_name}")
        
        # Check supported formats
        formats = provider.get_supported_formats()
        print(f"   Supported formats: {formats}")
        
        # Test availability
        available = await provider.is_available()
        print(f"   Available: {'✓' if available else '✗'}")
        
        if available:
            print("   🎉 Mistral provider is ready for PDF processing!")
        else:
            print("   ⚠️  Mistral provider is not available")
            
        return provider if available else None
            
    except Exception as e:
        print(f"   ✗ Error creating provider: {e}")
        return None

async def test_simple_image_processing():
    """Test with a simple test image first."""
    print("\n3. Testing with simple image processing...")
    
    provider = await test_pdf_processing()
    if not provider:
        print("   ⚠️  Skipping image test - provider not available")
        return
    
    try:
        # Create a simple test image
        from PIL import Image
        import tempfile
        
        # Create a simple white image with black text
        img = Image.new('RGB', (400, 100), color='white')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, 'PNG')
            temp_path = Path(tmp.name)
        
        print(f"   Created test image: {temp_path}")
        
        # Test encoding
        base64_data = await provider._encode_image(temp_path)
        print(f"   ✓ Image encoded successfully (length: {len(base64_data)})")
        
        # Cleanup
        temp_path.unlink()
        
        print("   ✓ Basic image processing works!")
        
    except Exception as e:
        print(f"   ✗ Image processing failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pdf_processing())
    asyncio.run(test_simple_image_processing())