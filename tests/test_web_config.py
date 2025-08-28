#!/usr/bin/env python3
"""
Test script to verify web app configuration is working.
"""

import os
import asyncio
from pathlib import Path

# Add the ocular package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def test_web_config():
    """Test web application configuration."""
    print("=== Testing Web App Configuration ===\n")
    
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
    
    # Test configuration loading
    try:
        from ocular.settings import OcularSettings
        config = OcularSettings()
        print(f"✓ OcularSettings loaded successfully")
        print(f"   Mistral API key: {'Set' if config.providers.mistral_api_key else 'Not set'}")
        print(f"   Enabled providers: {config.providers.enabled_providers}")
    except Exception as e:
        print(f"✗ Failed to load OcularSettings: {e}")
        return
    
    # Test UnifiedDocumentProcessor
    try:
        from ocular import UnifiedDocumentProcessor
        processor = UnifiedDocumentProcessor(config)
        print("✓ UnifiedDocumentProcessor created successfully")
        
        # Test provider initialization
        async def test_providers():
            print("   Testing individual provider configs...")
            from ocular.core.enums import ProviderType
            for provider_type in [ProviderType.MISTRAL, ProviderType.TESSERACT]:
                provider_config = config.get_provider_config(provider_type)
                print(f"   {provider_type.value}: {provider_config}")
            
            await processor._ensure_providers_initialized()
            available = processor.get_available_providers()
            print(f"   Available providers: {available}")
            
            # Check provider configs used by processor
            print(f"   Processor configs: {list(processor._configs.keys())}")
            return available
            
        available_providers = asyncio.run(test_providers())
        
        if available_providers:
            print(f"🎉 {len(available_providers)} provider(s) ready for web app!")
        else:
            print("⚠️  No providers available for web app")
            
    except Exception as e:
        print(f"✗ Failed to create processor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_web_config()