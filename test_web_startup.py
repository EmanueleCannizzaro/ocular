#!/usr/bin/env python3
"""
Test web app startup process to debug provider availability.
"""

import os
import asyncio
from pathlib import Path

# Load environment variables first
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from {env_path}")
    else:
        load_dotenv()
        print("✓ Loaded .env from default location")
except ImportError:
    print("Warning: python-dotenv not available")

async def test_web_startup():
    """Test the exact startup process the web app uses."""
    print("=== Testing Web App Startup Process ===\n")
    
    # Step 1: Create config (exactly like web app)
    try:
        from ocular.settings import OcularSettings
        config = OcularSettings()
        print(f"✓ Config created with Mistral API key: {'Set' if config.providers.mistral_api_key else 'Not set'}")
        print(f"   Enabled providers: {config.providers.enabled_providers}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return
    
    # Step 2: Create processor (exactly like web app)
    try:
        from ocular import UnifiedDocumentProcessor
        processor = UnifiedDocumentProcessor(config)
        print("✓ Processor created successfully")
    except Exception as e:
        print(f"✗ Processor creation failed: {e}")
        return
    
    # Step 3: Initialize providers (exactly like web app home route)
    try:
        print("\n--- Provider Initialization ---")
        await processor._ensure_providers_initialized()
        available_providers = processor.get_available_providers()
        provider_stats = processor.get_provider_stats()
        
        print(f"Available providers: {available_providers}")
        print(f"Provider stats: {provider_stats}")
        
        if available_providers:
            print(f"🎉 Web app will show {len(available_providers)} provider(s)!")
            for provider in available_providers:
                print(f"   - {provider}")
        else:
            print("⚠️  Web app will show 'No OCR providers available' message")
            
            # Debug why no providers are available
            print("\n--- Debugging Missing Providers ---")
            print(f"   Processor configs: {list(processor._configs.keys())}")
            print(f"   Initialized providers: {list(processor.providers.keys())}")
            
    except Exception as e:
        print(f"✗ Provider initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_web_startup())