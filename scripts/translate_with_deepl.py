#!/usr/bin/env python3
"""
Translate PO files using DeepL API.

This script translates all untranslated strings in PO files using the DeepL API
for professional-quality translations.

Usage:
    python3 scripts/translate_with_deepl.py
    python3 scripts/translate_with_deepl.py --languages it,es,fr,de
    python3 scripts/translate_with_deepl.py --source-lang it  # Use Italian as source
"""

import os
import sys
import argparse
import polib
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Try to import DeepL
try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False
    print("⚠️ DeepL library not installed. Install with: pip install deepl")


# Language mapping from our codes to DeepL codes
DEEPL_LANGUAGE_MAP = {
    'it': 'IT',      # Italian
    'es': 'ES',      # Spanish
    'fr': 'FR',      # French
    'de': 'DE',      # German
    'ja': 'JA',      # Japanese
    'zh_CN': 'ZH',   # Chinese (Simplified)
}

# Reverse mapping
SOURCE_LANGUAGE_MAP = {
    'en': 'EN',      # English
    'it': 'IT',      # Italian (can be used as authoritative source)
}


def get_deepl_client() -> Optional[deepl.Translator]:
    """Get DeepL API client."""
    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv('DEEPL_API_KEY')

    if not api_key:
        print("❌ DEEPL_API_KEY not found in environment variables")
        print("   Add it to your .env file or export it:")
        print("   export DEEPL_API_KEY='your-api-key'")
        return None

    if not DEEPL_AVAILABLE:
        return None

    try:
        translator = deepl.Translator(api_key)
        # Test the connection
        usage = translator.get_usage()
        print(f"✅ DeepL API connected successfully")
        print(f"   Character usage: {usage.character.count} / {usage.character.limit}")
        return translator
    except Exception as e:
        print(f"❌ Failed to connect to DeepL API: {e}")
        return None


def translate_po_file(
    po_file_path: Path,
    target_lang: str,
    translator: deepl.Translator,
    source_lang: str = 'en',
    batch_size: int = 50
) -> int:
    """
    Translate a PO file using DeepL API.

    Args:
        po_file_path: Path to PO file
        target_lang: Target language code (it, es, fr, de, ja, zh_CN)
        translator: DeepL translator instance
        source_lang: Source language code (en or it)
        batch_size: Number of strings to translate per API call

    Returns:
        Number of strings translated
    """
    if not po_file_path.exists():
        print(f"❌ PO file not found: {po_file_path}")
        return 0

    # Load PO file
    po = polib.pofile(str(po_file_path))

    # Get untranslated entries
    untranslated = [entry for entry in po if not entry.msgstr or entry.fuzzy]

    if not untranslated:
        print(f"✅ {po_file_path.name}: All strings already translated")
        return 0

    print(f"🔄 Translating {len(untranslated)} strings to {target_lang}...")

    # Get DeepL language codes
    source_lang_code = SOURCE_LANGUAGE_MAP.get(source_lang, 'EN')
    target_lang_code = DEEPL_LANGUAGE_MAP.get(target_lang)

    if not target_lang_code:
        print(f"❌ Unsupported target language: {target_lang}")
        return 0

    translated_count = 0

    # Translate in batches
    for i in range(0, len(untranslated), batch_size):
        batch = untranslated[i:i + batch_size]
        texts_to_translate = [entry.msgid for entry in batch]

        try:
            # Translate batch
            results = translator.translate_text(
                texts_to_translate,
                source_lang=source_lang_code,
                target_lang=target_lang_code,
                preserve_formatting=True,
                tag_handling='html'  # Preserve HTML tags if any
            )

            # Update PO entries
            for entry, result in zip(batch, results):
                entry.msgstr = result.text
                entry.fuzzy = False  # Remove fuzzy flag
                translated_count += 1

            print(f"   Translated {min(i + batch_size, len(untranslated))}/{len(untranslated)} strings")

        except Exception as e:
            print(f"❌ Translation error: {e}")
            # Continue with next batch
            continue

    # Save updated PO file
    po.save(str(po_file_path))
    print(f"✅ Saved {translated_count} translations to {po_file_path.name}")

    return translated_count


def translate_all_languages(
    languages: Optional[List[str]] = None,
    source_lang: str = 'en'
):
    """
    Translate all PO files for specified languages.

    Args:
        languages: List of language codes to translate (None = all)
        source_lang: Source language (en or it)
    """
    # Get translator
    translator = get_deepl_client()
    if not translator:
        return

    # Get locale directory
    locale_dir = Path(__file__).parent.parent / 'ocular' / 'locales'

    if not locale_dir.exists():
        print(f"❌ Locale directory not found: {locale_dir}")
        return

    # Default to all languages if none specified
    if languages is None:
        languages = list(DEEPL_LANGUAGE_MAP.keys())

    print(f"🌐 Translating from {source_lang.upper()} to {len(languages)} languages")
    print(f"📁 Locale directory: {locale_dir}\n")

    total_translated = 0

    for lang in languages:
        # Get locale code (it -> it_IT)
        from ocular.services.i18n_service import SUPPORTED_LANGUAGES
        locale = SUPPORTED_LANGUAGES.get(lang)

        if not locale:
            print(f"⚠️ Skipping unsupported language: {lang}")
            continue

        po_file = locale_dir / locale / 'LC_MESSAGES' / 'messages.po'

        if not po_file.exists():
            print(f"⚠️ PO file not found: {po_file}")
            print(f"   Run 'pybabel init -i locales/messages.pot -d locales -l {locale}' first")
            continue

        # Translate
        count = translate_po_file(po_file, lang, translator, source_lang)
        total_translated += count

        print()  # Blank line between languages

    print(f"🎉 Translation complete! Translated {total_translated} strings total")
    print(f"\n📝 Next steps:")
    print(f"   1. Review translations in the PO files")
    print(f"   2. Compile translations: pybabel compile -d {locale_dir}")
    print(f"   3. Restart your application to load new translations")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Translate PO files using DeepL API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate all languages
  python3 scripts/translate_with_deepl.py

  # Translate specific languages
  python3 scripts/translate_with_deepl.py --languages it,es,fr

  # Use Italian as source language (recommended for higher quality)
  python3 scripts/translate_with_deepl.py --source-lang it

  # Translate only Italian and Spanish from English
  python3 scripts/translate_with_deepl.py --languages it,es --source-lang en
        """
    )

    parser.add_argument(
        '--languages',
        type=str,
        help='Comma-separated list of language codes (it,es,fr,de,ja,zh_CN)'
    )

    parser.add_argument(
        '--source-lang',
        type=str,
        default='en',
        choices=['en', 'it'],
        help='Source language for translation (default: en, recommended: it for quality)'
    )

    args = parser.parse_args()

    # Parse languages
    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(',')]

    # Run translation
    translate_all_languages(languages, args.source_lang)


if __name__ == '__main__':
    main()
