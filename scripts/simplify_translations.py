#!/usr/bin/env python3
"""
Simplify existing conditional translations in templates.

Changes {{ _('Text') if _ is defined else 'Text' }} to {{ _('Text') }}
"""

import re
import sys
from pathlib import Path


def simplify_template(template_path: Path) -> bool:
    """Simplify conditional translations in a template."""
    print(f"Processing: {template_path.name}")

    # Read content
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern for conditional translation
    pattern = r"\{\{\s*_\(['\"]([^'\"]+)['\"]\)\s+if\s+_\s+is\s+defined\s+else\s+['\"]([^'\"]+)['\"]\s*\}\}"

    # Replace conditional translations
    new_content = re.sub(pattern, r"{{ _('\1') }}", content)

    # Check if changed
    if content == new_content:
        print(f"  ✓ No changes needed")
        return False

    # Save backup
    backup_path = template_path.with_suffix(template_path.suffix + '.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Write new content
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✓ Simplified (backup: {backup_path.name})")
    return True


def main():
    # Get template directory
    script_dir = Path(__file__).parent
    template_dir = script_dir.parent / 'ocular' / 'templates'

    # Templates to process
    templates = ['index.html', 'analytics.html']

    print("Simplifying conditional translations...\n")

    modified = 0
    for name in templates:
        template_path = template_dir / name
        if template_path.exists():
            if simplify_template(template_path):
                modified += 1
        print()

    print(f"✅ Done! {modified} template(s) modified")
    return 0


if __name__ == '__main__':
    sys.exit(main())
