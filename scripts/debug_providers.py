#!/usr/bin/env python3
"""
Debug script to check provider availability and configuration.
"""

import os
import sys
from pathlib import Path

# Add the ocular package to the path
sys.path.insert(0, str(Path(__file__).parent / "ocular"))

def debug_providers():
    """Debug provider availability issues."""
    print("=== Debugging OCR Providers ===\n")
    
    # Check environment variables first
    print("1. Environment Variables:")
    env_vars = ["MISTRAL_API_KEY", "OLM_API_ENDPOINT", "OLM_API_KEY", "ROLM_API_ENDPOINT", "ROLM_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        print(f"   {var}: {'SET' if value else 'NOT SET'}")
    print()
    
    # Load .env manually
    print("2. Loading .env file manually...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✓ .env loaded successfully")
        
        # Check again after loading
        mistral_key = os.getenv("MISTRAL_API_KEY")
        print(f"   MISTRAL_API_KEY after .env load: {'SET' if mistral_key else 'NOT SET'}")
    except ImportError:
        print("   ✗ python-dotenv not available")
    print()
    
    # Test provider imports
    print("3. Testing Provider Imports:")
    try:
        from ocular.core.enums import ProviderType
        print("   ✓ ProviderType imported successfully")
        print(f"   Available provider types: {[p.value for p in ProviderType]}")
    except Exception as e:
        print(f"   ✗ Failed to import ProviderType: {e}")
        return
    
    try:
        from ocular.providers.factory import ProviderFactory
        print("   ✓ ProviderFactory imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import ProviderFactory: {e}")
        return
    print()
    
    # Test Mistral provider specifically
    print("4. Testing Mistral Provider:")
    try:
        mistral_config = {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "model": os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"),
            "timeout": 30,
            "max_retries": 3
        }
        
        print(f"   Config: api_key={'SET' if mistral_config['api_key'] else 'NOT SET'}, model={mistral_config['model']}")
        
        provider = ProviderFactory.create_provider(ProviderType.MISTRAL, mistral_config)
        print(f"   ✓ Mistral provider created: {provider.provider_name}")
        
        config_valid = provider.validate_config()
        print(f"   Config validation: {'✓' if config_valid else '✗'}")
        
        # Test availability
        import asyncio
        async def test_availability():
            available = await provider.is_available()
            print(f"   Availability: {'✓' if available else '✗'}")
            return available
        
        available = asyncio.run(test_availability())
        
        if available:
            print("   🎉 Mistral provider is ready to use!")
        else:
            print("   ⚠️  Mistral provider is not available")
            
    except Exception as e:
        print(f"   ✗ Error testing Mistral provider: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test factory methods
    print("5. Testing Factory Methods:")
    try:
        configs = {
            ProviderType.MISTRAL: {
                "api_key": os.getenv("MISTRAL_API_KEY"),
                "model": os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"),
                "timeout": 30,
                "max_retries": 3
            }
        }
        
        async def test_factory():
            available_providers = await ProviderFactory.get_available_providers(configs)
            print(f"   Available providers: {[p.value for p in available_providers]}")
            return available_providers
            
        available = asyncio.run(test_factory())
        
        if available:
            print("   🎉 Factory detected available providers!")
        else:
            print("   ⚠️  No providers available through factory")
            
    except Exception as e:
        print(f"   ✗ Error testing factory: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_providers()