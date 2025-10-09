#!/usr/bin/env python3
"""
Extract translatable strings from Jinja2 templates.
"""

import re
from pathlib import Path
from datetime import datetime

TEMPLATES_DIR = Path('ocular/templates')
OUTPUT_FILE = Path('ocular/locales/messages.pot')


def extract_strings_from_template(template_path):
    """Extract all _('...') strings from a template."""
    content = template_path.read_text()

    # Pattern to match {{ _('text') }} and {{ _("text") }}
    pattern = r"{{\s*_\(['\"](.+?)['\"]\)\s*}}"

    matches = re.findall(pattern, content)
    return matches


def create_pot_file(strings_dict):
    """Create a POT file from extracted strings."""
    header = f"""# Translations template for Ocular OCR.
# Copyright (C) 2025 Ocular OCR
# This file is distributed under the same license as the Ocular package.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: Ocular OCR 1.0\\n"
"Report-Msgid-Bugs-To: support@ocular-ocr.com\\n"
"POT-Creation-Date: {datetime.now().strftime('%Y-%m-%d %H:%M%z')}\\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"
"Language-Team: LANGUAGE <LL@li.org>\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=utf-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Generated-By: extract_strings.py\\n"

"""

    entries = []
    for string, locations in sorted(strings_dict.items()):
        location_comments = '\\n'.join([f'#: {loc}' for loc in sorted(locations)])
        entry = f'{location_comments}\\nmsgid "{string}"\\nmsgstr ""\\n'
        entries.append(entry)

    return header + '\\n'.join(entries)


def main():
    """Extract all translatable strings from templates."""
    print("Extracting translatable strings from templates...")

    # Dictionary to store strings and their locations
    strings_dict = {}

    # Process all HTML templates
    for template_path in sorted(TEMPLATES_DIR.glob('*.html')):
        print(f"  Processing {template_path.name}...")
        strings = extract_strings_from_template(template_path)

        for string in strings:
            if string not in strings_dict:
                strings_dict[string] = set()
            strings_dict[string].add(str(template_path))

    print(f"\\nFound {len(strings_dict)} unique translatable strings")

    # Create POT file
    pot_content = create_pot_file(strings_dict)
    OUTPUT_FILE.write_text(pot_content)

    print(f"✓ Created {OUTPUT_FILE}")
    print(f"  Total strings: {len(strings_dict)}")

    # Show sample
    print("\\nSample strings:")
    for i, string in enumerate(sorted(strings_dict.keys())[:10]):
        print(f"  - {string}")
        if i >= 9:
            break


if __name__ == '__main__':
    main()
