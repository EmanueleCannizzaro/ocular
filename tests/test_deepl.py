#!/usr/bin/env python3
"""
Test script to verify DeepL API integration.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, relying on environment variables")

import deepl

def test_deepl_connection():
    """Test DeepL API connection and basic translation."""

    # Get API key from environment
    api_key = os.getenv('DEEPL_API_KEY')

    if not api_key:
        print("❌ DEEPL_API_KEY not found in environment variables")
        print("   Please ensure DEEPL_API_KEY is set in your .env file")
        return False

    print(f"✓ Found DEEPL_API_KEY: {api_key[:10]}...{api_key[-4:]}")

    try:
        # Initialize translator
        translator = deepl.Translator(api_key)
        print("✓ DeepL Translator initialized successfully")

        # Test basic translation
        test_text = "Hello, world!"
        print(f"\n📝 Testing translation:")
        print(f"   Source: {test_text} (EN)")

        # Translate to multiple languages
        target_languages = {
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'zh': 'Chinese'
        }

        print("\n   Translations:")
        for lang_code, lang_name in target_languages.items():
            result = translator.translate_text(test_text, target_lang=lang_code.upper())
            print(f"   {lang_name} ({lang_code.upper()}): {result.text}")

        # Get usage information
        usage = translator.get_usage()
        if usage.character.limit:
            print(f"\n📊 API Usage:")
            print(f"   Characters used: {usage.character.count:,}")
            print(f"   Character limit: {usage.character.limit:,}")
            print(f"   Remaining: {usage.character.limit - usage.character.count:,}")
        else:
            print(f"\n📊 API Usage:")
            print(f"   Characters used: {usage.character.count:,}")
            print(f"   Plan: Unlimited")

        print("\n✅ DeepL API is working correctly!")
        return True

    except deepl.AuthorizationException as e:
        print(f"\n❌ Authentication failed: {e}")
        print("   Please check your DEEPL_API_KEY is valid")
        return False
    except deepl.DeepLException as e:
        print(f"\n❌ DeepL API error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supported_languages():
    """List all supported languages."""
    api_key = os.getenv('DEEPL_API_KEY')
    if not api_key:
        return

    try:
        translator = deepl.Translator(api_key)

        print("\n📚 Supported target languages:")
        target_langs = translator.get_target_languages()
        for lang in target_langs:
            print(f"   {lang.code}: {lang.name}")

    except Exception as e:
        print(f"Error getting supported languages: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("DeepL API Integration Test")
    print("=" * 60)

    success = test_deepl_connection()

    if success:
        test_supported_languages()

    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)
