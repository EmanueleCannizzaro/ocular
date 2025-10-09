#!/usr/bin/env python3
"""
Automatically internationalize Jinja2 templates for translation.

This script:
1. Simplifies existing {{ _('Text') if _ is defined else 'Text' }} to {{ _('Text') }}
2. Wraps all user-visible text in {{ _() }} markers
3. Preserves HTML attributes, URLs, technical values, and icons

Usage:
    python3 scripts/internationalize_templates.py
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Patterns that should NOT be translated
SKIP_PATTERNS = [
    r'url_for\(',  # Flask/FastAPI URL generation
    r'class="[^"]*"',  # CSS classes
    r'id="[^"]*"',  # HTML IDs
    r'data-[a-z\-]+="[^"]*"',  # Data attributes
    r'aria-[a-z\-]+="[^"]*"',  # ARIA attributes
    r'role="[^"]*"',  # ARIA roles
    r'placeholder="[^"]*"',  # Already handled separately
    r'value="[^"]*"',  # Form values (often technical)
    r'href="[^"]*"',  # URLs
    r'src="[^"]*"',  # Image sources
    r'onclick="[^"]*"',  # JavaScript
    r'onchange="[^"]*"',  # JavaScript
    r'oninput="[^"]*"',  # JavaScript
    r'fas fa-[^"]*',  # FontAwesome icons
    r'fab fa-[^"]*',  # FontAwesome brand icons
    r'bi bi-[^"]*',  # Bootstrap icons
]


def should_skip_translation(text: str) -> bool:
    """Check if text should not be translated."""
    # Skip if already wrapped in translation
    if text.strip().startswith("{{ _('") or text.strip().startswith('{{ _("'):
        return True

    # Skip if it's a technical value
    if any(pattern in text for pattern in ['{{', '{%', 'http://', 'https://', 'javascript:']):
        return True

    # Skip if it matches any skip pattern
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Skip if it's only numbers, symbols, or whitespace
    if re.match(r'^[\d\s\.,\-_:;!?(){}[\]]+$', text):
        return True

    # Skip very short text that's likely technical
    if len(text.strip()) <= 2:
        return True

    return False


def simplify_conditional_translations(content: str) -> str:
    """
    Convert {{ _('Text') if _ is defined else 'Text' }} to {{ _('Text') }}
    """
    # Pattern for conditional translation
    pattern = r"\{\{\s*_\(['\"]([^'\"]+)['\"]\)\s+if\s+_\s+is\s+defined\s+else\s+['\"]([^'\"]+)['\"]\s*\}\}"

    def replace_conditional(match):
        text1 = match.group(1)
        text2 = match.group(2)
        # Both texts should be the same
        return f"{{{{ _('{text1}') }}}}"

    return re.sub(pattern, replace_conditional, content)


def wrap_text_in_translation(text: str) -> str:
    """Wrap plain text in translation markers."""
    if should_skip_translation(text):
        return text

    # Clean the text
    cleaned = text.strip()
    if not cleaned:
        return text

    # Preserve leading/trailing whitespace
    leading_ws = text[:len(text) - len(text.lstrip())]
    trailing_ws = text[len(text.rstrip()):]

    # Escape quotes in the text
    escaped = cleaned.replace("'", "\\'")

    return f"{leading_ws}{{{{ _('{escaped}') }}}}{trailing_ws}"


def internationalize_html_content(content: str) -> str:
    """
    Internationalize user-visible text in HTML templates.
    """
    # First, simplify existing conditional translations
    content = simplify_conditional_translations(content)

    # Patterns for text to internationalize

    # 1. Text between HTML tags (but not in special blocks)
    def wrap_tag_content(match):
        tag = match.group(1)
        text = match.group(2)
        closing = match.group(3)

        # Skip if already has translation or is a script/style tag
        if tag.lower() in ['script', 'style', 'code', 'pre']:
            return match.group(0)

        # Skip if text is only Jinja2 or empty
        if not text.strip() or text.strip().startswith('{{') or text.strip().startswith('{%'):
            return match.group(0)

        wrapped_text = wrap_text_in_translation(text)
        return f"{tag}{wrapped_text}{closing}"

    # Match text between tags: ><text></
    content = re.sub(
        r'(>[^<{%]*?)([^<{>]+?)(<[^>]*?>)',
        wrap_tag_content,
        content,
        flags=re.DOTALL
    )

    # 2. Title attributes
    def wrap_title(match):
        quote = match.group(1)
        text = match.group(2)
        if should_skip_translation(text):
            return match.group(0)
        escaped = text.replace("'", r"\'")
        return f"title={{{{ _('{escaped}') }}}}"

    content = re.sub(
        r'title=(["\'])([^"\']+)\1',
        wrap_title,
        content
    )

    # 3. Placeholder attributes
    def wrap_placeholder(match):
        quote = match.group(1)
        text = match.group(2)
        if should_skip_translation(text):
            return match.group(0)
        escaped = text.replace("'", r"\'")
        return f"placeholder={{{{ _('{escaped}') }}}}"

    content = re.sub(
        r'placeholder=(["\'])([^"\']+)\1',
        wrap_placeholder,
        content
    )

    # 4. Alt attributes
    def wrap_alt(match):
        quote = match.group(1)
        text = match.group(2)
        if should_skip_translation(text):
            return match.group(0)
        escaped = text.replace("'", r"\'")
        return f"alt={{{{ _('{escaped}') }}}}"

    content = re.sub(
        r'alt=(["\'])([^"\']+)\1',
        wrap_alt,
        content
    )

    # 5. Aria-label attributes
    def wrap_aria_label(match):
        quote = match.group(1)
        text = match.group(2)
        if should_skip_translation(text):
            return match.group(0)
        escaped = text.replace("'", r"\'")
        return f"aria-label={{{{ _('{escaped}') }}}}"

    content = re.sub(
        r'aria-label=(["\'])([^"\']+)\1',
        wrap_aria_label,
        content
    )

    return content


def process_template(template_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single template file.

    Args:
        template_path: Path to the template file
        dry_run: If True, only show what would be changed

    Returns:
        True if file was modified, False otherwise
    """
    print(f"Processing: {template_path.name}")

    # Read original content
    with open(template_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Internationalize content
    new_content = internationalize_html_content(original_content)

    # Check if content changed
    if original_content == new_content:
        print(f"  ✓ No changes needed for {template_path.name}")
        return False

    if dry_run:
        print(f"  ℹ Would modify {template_path.name} (dry-run mode)")
        return True

    # Backup original file
    backup_path = template_path.with_suffix(template_path.suffix + '.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

    # Write new content
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✓ Modified {template_path.name} (backup saved as {backup_path.name})")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Internationalize Jinja2 templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run to see what would be changed
  python3 scripts/internationalize_templates.py --dry-run

  # Process all templates
  python3 scripts/internationalize_templates.py

  # Process specific templates
  python3 scripts/internationalize_templates.py batch.html billing.html
        """
    )

    parser.add_argument(
        'templates',
        nargs='*',
        help='Specific template files to process (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    args = parser.parse_args()

    # Get template directory
    script_dir = Path(__file__).parent
    template_dir = script_dir.parent / 'ocular' / 'templates'

    if not template_dir.exists():
        print(f"❌ Template directory not found: {template_dir}")
        return 1

    # Get list of templates to process
    if args.templates:
        template_paths = [template_dir / name for name in args.templates]
        # Validate all exist
        for path in template_paths:
            if not path.exists():
                print(f"❌ Template not found: {path}")
                return 1
    else:
        template_paths = sorted(template_dir.glob('*.html'))

    if not template_paths:
        print("❌ No templates found to process")
        return 1

    print(f"🌐 Internationalizing {len(template_paths)} template(s)...")
    print()

    modified_count = 0
    for template_path in template_paths:
        if process_template(template_path, dry_run=args.dry_run):
            modified_count += 1
        print()

    print(f"✅ Completed! {modified_count} template(s) {'would be' if args.dry_run else 'were'} modified")

    if not args.dry_run and modified_count > 0:
        print()
        print("📝 Next steps:")
        print("   1. Review the changes in the modified templates")
        print("   2. Extract strings: pybabel extract -F babel.cfg -o ocular/locales/messages.pot .")
        print("   3. Update translations: pybabel update -i ocular/locales/messages.pot -d ocular/locales")
        print("   4. Translate: python3 scripts/translate_with_deepl.py")
        print("   5. Compile: pybabel compile -d ocular/locales")

    return 0


if __name__ == '__main__':
    sys.exit(main())
