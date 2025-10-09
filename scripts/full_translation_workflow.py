#!/usr/bin/env python3
"""
Complete translation workflow:
1. Read POT file
2. For each locale, create fresh PO file with DeepL translations
3. Compile to MO files
"""

import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
DEEPL_API_URL = 'https://api-free.deepl.com/v2/translate'

# Language configurations
LANGUAGES = {
    'it_IT': {'deepl': 'IT', 'name': 'Italian'},
    'es_ES': {'deepl': 'ES', 'name': 'Spanish'},
    'fr_FR': {'deepl': 'FR', 'name': 'French'},
    'de_DE': {'deepl': 'DE', 'name': 'German'},
    'ja_JP': {'deepl': 'JA', 'name': 'Japanese'},
    'zh_CN': {'deepl': 'ZH', 'name': 'Chinese (Simplified)'}
}

LOC_DIR = Path('ocular/locales')
POT_FILE = LOC_DIR / 'messages.pot'


def translate_with_deepl(text, target_lang):
    """Translate text using DeepL API."""
    if not DEEPL_API_KEY:
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
            print(f"    DeepL error {response.status_code}")
            return text
    except Exception as e:
        print(f"    Translation error: {e}")
        return text


def parse_pot_file():
    """Parse POT file and extract all msgid strings."""
    content = POT_FILE.read_text()
    lines = content.split('\\n')

    strings = []
    current_msgid = None
    msgid_lines = []

    for line in lines:
        if line.startswith('msgid "'):
            if current_msgid:
                strings.append(current_msgid)
            msgid_text = line[7:-1]  # Remove 'msgid "' and trailing '"'
            if msgid_text:
                msgid_lines = [msgid_text]
                current_msgid = msgid_text
            else:
                current_msgid = None
                msgid_lines = []
        elif line.startswith('"') and current_msgid:
            # Continuation line
            continuation = line.strip()[1:-1]
            msgid_lines.append(continuation)
            current_msgid = ''.join(msgid_lines)
        elif line.startswith('msgstr'):
            if current_msgid:
                strings.append(current_msgid)
                current_msgid = None
                msgid_lines = []

    return [s for s in strings if s]  # Remove empty strings


def create_po_file(locale, strings):
    """Create a PO file with translated strings."""
    po_dir = LOC_DIR / locale / 'LC_MESSAGES'
    po_dir.mkdir(parents=True, exist_ok=True)
    po_file = po_dir / 'messages.po'

    deepl_lang = LANGUAGES[locale]['deepl']
    lang_name = LANGUAGES[locale]['name']

    print(f"\\nTranslating {locale} ({lang_name})...")

    header = f"""# {lang_name} translations for Ocular OCR
# Copyright (C) 2025 Ocular OCR
# This file is distributed under the same license as the Ocular package.
#
msgid ""
msgstr ""
"Project-Id-Version: Ocular OCR 1.0\\n"
"Report-Msgid-Bugs-To: support@ocular-ocr.com\\n"
"POT-Creation-Date: {datetime.now().strftime('%Y-%m-%d %H:%M%z')}\\n"
"PO-Revision-Date: {datetime.now().strftime('%Y-%m-%d %H:%M%z')}\\n"
"Last-Translator: DeepL API\\n"
"Language: {locale}\\n"
"Language-Team: {lang_name}\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=utf-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Generated-By: full_translation_workflow.py\\n"

"""

    entries = []
    for i, string in enumerate(strings, 1):
        print(f"  [{i}/{len(strings)}] Translating: {string[:50]}...")
        translated = translate_with_deepl(string, deepl_lang)
        entry = f'msgid "{string}"\\nmsgstr "{translated}"\\n'
        entries.append(entry)
        time.sleep(0.3)  # Rate limiting

    po_content = header + '\\n'.join(entries)
    po_file.write_text(po_content)

    print(f"✓ Created {po_file}")
    return po_file


def compile_mo_file(po_file):
    """Compile PO file to MO file."""
    import subprocess
    mo_file = po_file.with_suffix('.mo')

    cmd = ['msgfmt', '-o', str(mo_file), str(po_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Compiled {mo_file}")
        return True
    else:
        print(f"✗ Error compiling: {result.stderr}")
        return False


def main():
    """Run complete translation workflow."""
    if not DEEPL_API_KEY:
        print("ERROR: DEEPL_API_KEY not set!")
        return 1

    print("=" * 70)
    print("COMPLETE TRANSLATION WORKFLOW")
    print("=" * 70)

    # Step 1: Parse POT file
    print("\\n[Step 1] Parsing POT file...")
    strings = parse_pot_file()
    print(f"✓ Found {len(strings)} translatable strings")

    # Step 2: Translate for each locale
    print("\\n[Step 2] Translating to all languages...")
    po_files = []
    for locale in LANGUAGES:
        po_file = create_po_file(locale, strings)
        po_files.append(po_file)

    # Step 3: Compile MO files
    print("\\n[Step 3] Compiling MO files...")
    for po_file in po_files:
        compile_mo_file(po_file)

    print("\\n" + "=" * 70)
    print("TRANSLATION COMPLETE!")
    print("=" * 70)
    print(f"\\nTranslated {len(strings)} strings into {len(LANGUAGES)} languages")
    print("Ready for use!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
