#!/bin/bash
# Initialize translation files for all supported languages

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🌐 Initializing translations for Ocular OCR Service"
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check if pybabel is installed
if ! command -v pybabel &> /dev/null; then
    echo "❌ pybabel not found. Installing Babel..."
    pip install babel
fi

# Extract translatable strings
echo "📤 Extracting translatable strings..."
pybabel extract -F babel.cfg -o ocular/locales/messages.pot .

# Initialize PO files for each language
LOCALES=("it_IT" "es_ES" "fr_FR" "de_DE" "ja_JP" "zh_CN")

for LOCALE in "${LOCALES[@]}"; do
    PO_FILE="ocular/locales/$LOCALE/LC_MESSAGES/messages.po"

    if [ -f "$PO_FILE" ]; then
        echo "🔄 Updating $LOCALE..."
        pybabel update -i ocular/locales/messages.pot -d ocular/locales -l $LOCALE
    else
        echo "🆕 Initializing $LOCALE..."
        pybabel init -i ocular/locales/messages.pot -d ocular/locales -l $LOCALE
    fi
done

echo ""
echo "✅ Translation files initialized!"
echo ""
echo "📝 Next steps:"
echo "   1. Translate strings using DeepL:"
echo "      python3 scripts/translate_with_deepl.py"
echo ""
echo "   2. Or manually edit PO files in ocular/locales/"
echo ""
echo "   3. Compile translations:"
echo "      pybabel compile -d ocular/locales"
echo ""
