#!/usr/bin/env python3
"""
Test script to diagnose provider availability issues.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_providers():
    """Test provider availability."""
    print("=== Testing Provider Availability ===\n")
    
    # Import provider factory
    from ocular.providers.factory import ProviderFactory
    from ocular.core.enums import ProviderType
    
    # Test configurations
    configs = {
        ProviderType.MISTRAL: {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "model": os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"),
            "timeout": int(os.getenv("TIMEOUT_SECONDS", "30")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3"))
        },
        ProviderType.OLM_OCR: {
            "api_endpoint": os.getenv("OLM_API_ENDPOINT"),
            "api_key": os.getenv("OLM_API_KEY"),
            "timeout": 60,
            "max_retries": 3
        },
        ProviderType.ROLM_OCR: {
            "api_endpoint": os.getenv("ROLM_API_ENDPOINT"),
            "api_key": os.getenv("ROLM_API_KEY"),
            "timeout": 90,
            "max_retries": 3
        },
        ProviderType.GOOGLE_VISION: {
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "project_id": os.getenv("GOOGLE_PROJECT_ID"),
            "timeout": 30,
            "max_retries": 3
        },
        ProviderType.AWS_TEXTRACT: {
            "region": os.getenv("AWS_REGION", "us-east-1"),
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "timeout": 60,
            "max_retries": 3
        },
        ProviderType.TESSERACT: {
            "language": os.getenv("TESSERACT_LANGUAGE", "eng"),
            "dpi": int(os.getenv("TESSERACT_DPI", "300")),
            "timeout": 30
        },
        ProviderType.AZURE_DOCUMENT_INTELLIGENCE: {
            "endpoint": os.getenv("AZURE_DOC_INTEL_ENDPOINT"),
            "api_key": os.getenv("AZURE_DOC_INTEL_API_KEY"),
            "timeout": 120,
            "max_retries": 3
        }
    }
    
    print("Environment variables:")
    for key in ["MISTRAL_API_KEY", "OLM_API_ENDPOINT", "OLM_API_KEY", "ROLM_API_ENDPOINT", "ROLM_API_KEY"]:
        value = os.getenv(key)
        print(f"  {key}: {'SET' if value else 'NOT SET'} ({'***' if value else 'None'})")
    print()
    
    # Test each provider
    for provider_type, config in configs.items():
        print(f"Testing {provider_type.value}:")
        print(f"  Config keys: {list(config.keys())}")
        
        try:
            # Create provider
            provider = ProviderFactory.create_provider(provider_type, config)
            print(f"  ✓ Provider created: {provider.provider_name}")
            
            # Test configuration validation
            config_valid = provider.validate_config()
            print(f"  Config valid: {'✓' if config_valid else '✗'}")
            
            # Test availability
            available = await provider.is_available()
            print(f"  Available: {'✓' if available else '✗'}")
            
            if available:
                print(f"  ✓ {provider.provider_name} is ready to use!")
            else:
                print(f"  ✗ {provider.provider_name} is not available")
                
            # Cleanup
            await provider.cleanup()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Test factory methods
    print("Testing factory methods:")
    try:
        available_providers = await ProviderFactory.get_available_providers(configs)
        print(f"Available providers: {[p.value for p in available_providers]}")
        
        validation_results = await ProviderFactory.validate_provider_configs(configs)
        print(f"Config validation results: {[(k.value, v) for k, v in validation_results.items()]}")
        
        health_results = await ProviderFactory.health_check_all(configs)
        print("Health check results:")
        for provider_name, health in health_results.items():
            print(f"  {provider_name}: {health.get('status', 'unknown')}")
            
    except Exception as e:
        print(f"Factory method error: {e}")

if __name__ == "__main__":
    asyncio.run(test_providers())