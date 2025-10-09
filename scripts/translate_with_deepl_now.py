#!/usr/bin/env python3
"""
Translate PO files using DeepL API
"""

import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DeepL API configuration
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
# Use paid API endpoint (api.deepl.com) instead of free endpoint (api-free.deepl.com)
DEEPL_API_URL = 'https://api.deepl.com/v2/translate'

print(f"DeepL API Key: {DEEPL_API_KEY[:20]}..." if DEEPL_API_KEY else "No API key found")

# Language mapping for DeepL
DEEPL_LANG_MAP = {
    'it_IT': 'IT',
    'es_ES': 'ES',
    'fr_FR': 'FR',
    'de_DE': 'DE',
    'ja_JP': 'JA',
    'zh_CN': 'ZH'
}

LOCALES_DIR = Path('ocular/locales')


def translate_with_deepl(text: str, target_lang: str) -> str:
    """Translate text using DeepL API."""
    if not DEEPL_API_KEY:
        print("Error: DEEPL_API_KEY not set!")
        return text

    try:
        response = requests.post(
            DEEPL_API_URL,
            data={
                'auth_key': DEEPL_API_KEY,
                'text': text,
                'target_lang': target_lang,
                'source_lang': 'EN'
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['translations'][0]['text']
        elif response.status_code == 456:
            print(f"  Quota exceeded!")
            return text
        else:
            print(f"  DeepL API error {response.status_code}: {response.text}")
            return text
    except Exception as e:
        print(f"  Translation error: {e}")
        return text


def translate_po_file(po_file: Path, target_lang: str) -> None:
    """Translate untranslated strings in a PO file using DeepL."""
    locale_name = po_file.parent.parent.name
    print(f"\nTranslating {locale_name} ({target_lang})...")

    content = po_file.read_text(encoding='utf-8')
    lines = content.split('\n')
    translated_lines = []
    current_msgid = None
    current_msgid_lines = []
    translations_count = 0
    in_multiline = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle msgid
        if line.startswith('msgid "'):
            if line == 'msgid ""':
                # Empty msgid (header)
                translated_lines.append(line)
                current_msgid = None
                current_msgid_lines = []
            else:
                # Start of message ID
                msgid_text = line[7:-1]  # Remove 'msgid "' and trailing '"'
                current_msgid_lines = [msgid_text]

                # Check for multiline msgid
                i += 1
                while i < len(lines) and lines[i].startswith('"') and not lines[i].startswith('msgstr'):
                    continuation = lines[i].strip()[1:-1]  # Remove quotes
                    current_msgid_lines.append(continuation)
                    i += 1
                i -= 1  # Step back one

                current_msgid = ''.join(current_msgid_lines)
                translated_lines.append(line)

        # Handle msgstr
        elif line.startswith('msgstr "'):
            if current_msgid and current_msgid != '':
                # Check if translation is empty
                msgstr_text = line[8:-1]  # Remove 'msgstr "' and trailing '"'

                # Check for multiline empty msgstr
                is_empty = msgstr_text == ''
                if is_empty and i + 1 < len(lines) and lines[i + 1].startswith('"'):
                    # Multiline but might still be empty
                    temp_i = i + 1
                    all_empty = True
                    while temp_i < len(lines) and lines[temp_i].startswith('"'):
                        if lines[temp_i].strip() != '""':
                            all_empty = False
                            break
                        temp_i += 1
                    is_empty = all_empty

                if is_empty:
                    # Translate it
                    print(f"  Translating: {current_msgid[:50]}...")
                    translated_text = translate_with_deepl(current_msgid, target_lang)
                    line = f'msgstr "{translated_text}"'
                    translations_count += 1
                    time.sleep(0.5)  # Rate limiting

            translated_lines.append(line)
            current_msgid = None
            current_msgid_lines = []

        else:
            translated_lines.append(line)

        i += 1

    if translations_count > 0:
        po_file.write_text('\n'.join(translated_lines), encoding='utf-8')
        print(f"✓ Translated {translations_count} strings for {locale_name}")
    else:
        print(f"  No untranslated strings found in {locale_name}")


def main():
    """Translate all PO files."""
    if not DEEPL_API_KEY:
        print("ERROR: DEEPL_API_KEY environment variable not set!")
        print("Please set it in your .env file")
        return 1

    print("=" * 60)
    print("DeepL Translation Tool")
    print("=" * 60)

    for locale_dir in sorted(LOCALES_DIR.iterdir()):
        if locale_dir.is_dir() and locale_dir.name in DEEPL_LANG_MAP:
            po_file = locale_dir / 'LC_MESSAGES' / 'messages.po'
            if po_file.exists():
                deepl_lang = DEEPL_LANG_MAP[locale_dir.name]
                translate_po_file(po_file, deepl_lang)

    print("\n" + "=" * 60)
    print("Translation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  pybabel compile -d ocular/locales")
    return 0


if __name__ == '__main__':
    sys.exit(main())
