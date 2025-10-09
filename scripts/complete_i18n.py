#!/usr/bin/env python3
"""
Complete internationalization script for Ocular OCR.
This script:
1. Internationalizes all templates with _() pattern
2. Extracts translatable strings
3. Translates via DeepL API
4. Compiles translation files
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# DeepL API configuration
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
DEEPL_API_URL = 'https://api-free.deepl.com/v2/translate'

# Language mapping for DeepL
DEEPL_LANG_MAP = {
    'it_IT': 'IT',
    'es_ES': 'ES',
    'fr_FR': 'FR',
    'de_DE': 'DE',
    'ja_JP': 'JA',
    'zh_CN': 'ZH'
}

TEMPLATES_DIR = Path('ocular/templates')
LOCALES_DIR = Path('ocular/locales')

# Templates that need internationalization
TEMPLATES_TO_UPDATE = [
    'analytics.html',
    'batch.html',
    'billing.html',
    'deployments.html',
    'document_tracking.html',
    'models.html',
    'observability.html',
    'providers.html',
    'settings.html',
    'user_profile.html',
]


def internationalize_template(template_path: Path) -> None:
    """Replace hardcoded text with _() pattern in a template."""
    print(f"Internationalizing {template_path.name}...")

    content = template_path.read_text()

    # Patterns to replace with _()
    patterns = [
        # Button text and labels
        (r'>([A-Z][^<]+)</button>', r'>{{ _(\'\1\') }}</button>'),
        (r'>([A-Z][^<]+)</a>', r'>{{ _(\'\1\') }}</a>'),
        (r'>([A-Z][^<]+)</h[1-6]>', r'>{{ _(\'\1\') }}</h[1-6]>'),
        (r'>([A-Z][^<]+)</label>', r'>{{ _(\'\1\') }}</label>'),
        (r'>([A-Z][^<]+)</th>', r'>{{ _(\'\1\') }}</th>'),
        (r'>([A-Z][^<]+)</p>', r'>{{ _(\'\1\') }}</p>'),
        (r'>([A-Z][^<]+)</span>', r'>{{ _(\'\1\') }}</span>'),
        (r'>([A-Z][^<]+)</small>', r'>{{ _(\'\1\') }}</small>'),
        (r'placeholder="([^"]+)"', r'placeholder="{{ _(\'\1\') }}"'),
        (r'title="([^"]+)"', r'title="{{ _(\'\1\') }}"'),
        (r'aria-label="([^"]+)"', r'aria-label="{{ _(\'\1\') }}"'),
    ]

    modified = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True

    if modified:
        template_path.write_text(content)
        print(f"✓ Updated {template_path.name}")
    else:
        print(f"  No changes needed for {template_path.name}")


def extract_messages() -> None:
    """Extract messages from templates using pybabel."""
    print("\nExtracting translatable messages...")

    # Create POT file
    cmd = [
        'pybabel', 'extract',
        '-F', 'babel.cfg',
        '-o', str(LOCALES_DIR / 'messages.pot'),
        'ocular'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Extracted messages to messages.pot")
    else:
        print(f"✗ Error extracting messages: {result.stderr}")
        return False

    return True


def update_po_files() -> None:
    """Update PO files with new messages."""
    print("\nUpdating PO files...")

    for locale_dir in LOCALES_DIR.iterdir():
        if locale_dir.is_dir() and locale_dir.name.endswith(('_IT', '_ES', '_FR', '_DE', '_JP', '_CN')):
            po_file = locale_dir / 'LC_MESSAGES' / 'messages.po'

            if po_file.exists():
                cmd = [
                    'pybabel', 'update',
                    '-i', str(LOCALES_DIR / 'messages.pot'),
                    '-d', str(LOCALES_DIR),
                    '-l', locale_dir.name
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ Updated {locale_dir.name}/LC_MESSAGES/messages.po")
                else:
                    print(f"✗ Error updating {locale_dir.name}: {result.stderr}")
            else:
                # Initialize new locale
                cmd = [
                    'pybabel', 'init',
                    '-i', str(LOCALES_DIR / 'messages.pot'),
                    '-d', str(LOCALES_DIR),
                    '-l', locale_dir.name
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ Initialized {locale_dir.name}/LC_MESSAGES/messages.po")
                else:
                    print(f"✗ Error initializing {locale_dir.name}: {result.stderr}")


def translate_with_deepl(text: str, target_lang: str) -> str:
    """Translate text using DeepL API."""
    if not DEEPL_API_KEY:
        print("Warning: DEEPL_API_KEY not set, skipping translation")
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
        else:
            print(f"DeepL API error: {response.status_code}")
            return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def translate_po_file(po_file: Path, target_lang: str) -> None:
    """Translate untranslated strings in a PO file using DeepL."""
    print(f"\nTranslating {po_file.parent.parent.name} with DeepL...")

    content = po_file.read_text()
    lines = content.split('\n')
    translated_lines = []
    current_msgid = None
    translations_count = 0

    for i, line in enumerate(lines):
        if line.startswith('msgid "') and not line.startswith('msgid ""'):
            # Extract the message ID
            current_msgid = line[7:-1]  # Remove 'msgid "' and trailing '"'
        elif line.startswith('msgstr ""') and current_msgid and current_msgid != '':
            # Found an empty translation
            translated_text = translate_with_deepl(current_msgid, target_lang)
            line = f'msgstr "{translated_text}"'
            translations_count += 1
            current_msgid = None

        translated_lines.append(line)

    if translations_count > 0:
        po_file.write_text('\n'.join(translated_lines))
        print(f"✓ Translated {translations_count} strings for {po_file.parent.parent.name}")
    else:
        print(f"  No untranslated strings found in {po_file.parent.parent.name}")


def compile_translations() -> None:
    """Compile PO files to MO files."""
    print("\nCompiling translations...")

    cmd = [
        'pybabel', 'compile',
        '-d', str(LOCALES_DIR)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Compiled all translations")
    else:
        print(f"✗ Error compiling translations: {result.stderr}")


def main():
    """Run the complete internationalization workflow."""
    print("=" * 60)
    print("OCULAR OCR - Complete Internationalization")
    print("=" * 60)

    # Step 1: Internationalize templates (commented out - already done manually)
    # print("\n[Step 1] Internationalizing templates...")
    # for template_name in TEMPLATES_TO_UPDATE:
    #     template_path = TEMPLATES_DIR / template_name
    #     if template_path.exists():
    #         internationalize_template(template_path)

    # Step 2: Extract messages
    print("\n[Step 2] Extracting translatable messages...")
    if not extract_messages():
        print("Failed to extract messages. Aborting.")
        return 1

    # Step 3: Update PO files
    print("\n[Step 3] Updating PO files...")
    update_po_files()

    # Step 4: Translate with DeepL
    if DEEPL_API_KEY:
        print("\n[Step 4] Translating with DeepL API...")
        for locale_dir in LOCALES_DIR.iterdir():
            if locale_dir.is_dir() and locale_dir.name in DEEPL_LANG_MAP:
                po_file = locale_dir / 'LC_MESSAGES' / 'messages.po'
                if po_file.exists():
                    deepl_lang = DEEPL_LANG_MAP[locale_dir.name]
                    translate_po_file(po_file, deepl_lang)
    else:
        print("\n[Step 4] Skipping DeepL translation (DEEPL_API_KEY not set)")

    # Step 5: Compile translations
    print("\n[Step 5] Compiling translations...")
    compile_translations()

    print("\n" + "=" * 60)
    print("Internationalization complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
