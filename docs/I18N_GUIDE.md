# Internationalization (i18n) Guide for Ocular OCR

This guide explains how to use the internationalization system in the Ocular OCR application.

## Table of Contents

1. [Overview](#overview)
2. [Supported Languages](#supported-languages)
3. [How It Works](#how-it-works)
4. [Using Translations in Templates](#using-translations-in-templates)
5. [Using Translations in Python Code](#using-translations-in-python-code)
6. [Adding New Translations](#adding-new-translations)
7. [Language Detection](#language-detection)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

## Overview

The Ocular OCR application supports 7 languages with professional translations using:

- **Babel** - String extraction and locale management
- **Gettext** - Industry-standard translation system
- **FastAPI Middleware** - Automatic language detection
- **Jinja2 Integration** - Seamless template translation

## Supported Languages

| Code    | Language             | Flag | Locale  |
|---------|---------------------|------|---------|
| `en`    | English             | 🇬🇧   | en_US   |
| `it`    | Italian (Italiano)  | 🇮🇹   | it_IT   |
| `es`    | Spanish (Español)   | 🇪🇸   | es_ES   |
| `fr`    | French (Français)   | 🇫🇷   | fr_FR   |
| `de`    | German (Deutsch)    | 🇩🇪   | de_DE   |
| `ja`    | Japanese (日本語)    | 🇯🇵   | ja_JP   |
| `zh_CN` | Chinese Simplified (简体中文) | 🇨🇳 | zh_CN |

## How It Works

### Middleware Architecture

The i18n system uses middleware to automatically detect the user's language preference:

```
Request → Language Detection → Translation Function Injection → Route Handler → Template Rendering
```

### Language Detection Priority

1. **Query Parameter**: `?lang=it` (highest priority)
2. **Cookie**: `language=it`
3. **Accept-Language Header**: Browser preferences
4. **Default**: English (en)

## Using Translations in Templates

### Basic Usage

The `_()` function is automatically available in all templates:

```html
<h1>{{ _('Welcome') }}</h1>
<p>{{ _('Extract text and data from PDFs and images') }}</p>
```

### With Fallback

For safety, use the `if _ is defined` pattern:

```html
<h1>{{ _('Welcome') if _ is defined else 'Welcome' }}</h1>
```

### In Attributes

```html
<button title="{{ _('Upload File') }}">{{ _('Upload') }}</button>
```

### Complete Example

```html
<!DOCTYPE html>
<html lang="{{ request.state.language if request.state.language is defined else 'en' }}">
<head>
    <title>{{ _('Ocular OCR Service') }}</title>
</head>
<body>
    <h1>{{ _('Upload') }}</h1>
    <p>{{ _('Drag and drop files here') }}</p>

    <button type="submit">{{ _('Upload File') }}</button>
</body>
</html>
```

## Using Translations in Python Code

### In Route Handlers

The translation function is available via `request.state._`:

```python
from fastapi import Request, APIRouter

router = APIRouter()

@router.get("/api/status")
async def get_status(request: Request):
    _ = request.state._  # Get translation function

    return {
        "status": "ok",
        "message": _("Processing your document")
    }
```

### Using Dependency Injection

```python
from fastapi import Depends
from ocular.middleware import get_translation_from_request, get_language_from_request

@router.get("/api/info")
async def get_info(
    _: Callable = Depends(get_translation_from_request),
    language: str = Depends(get_language_from_request)
):
    return {
        "message": _("Welcome"),
        "language": language
    }
```

### In Service Classes

```python
from ocular.services.i18n_service import get_translator

class MyService:
    def __init__(self, language: str = 'en'):
        self._ = get_translator(language)

    def process(self):
        return {
            "status": self._("Processing"),
            "message": self._("Please wait")
        }
```

## Adding New Translations

### Method 1: Using DeepL API (Recommended)

1. **Install DeepL library**:
   ```bash
   pip install deepl polib
   ```

2. **Add DeepL API key to `.env`**:
   ```bash
   DEEPL_API_KEY=your-deepl-api-key
   ```

3. **Add new strings to the template** (`ocular/locales/messages.pot`):
   ```
   msgid "New Feature"
   msgstr ""
   ```

4. **Update PO files**:
   ```bash
   pybabel update -i ocular/locales/messages.pot -d ocular/locales
   ```

5. **Translate using DeepL**:
   ```bash
   python3 scripts/translate_with_deepl.py
   ```

6. **Compile translations**:
   ```bash
   pybabel compile -d ocular/locales
   ```

### Method 2: Manual Translation

1. **Edit PO files directly**:
   ```bash
   # Edit Italian translations
   nano ocular/locales/it_IT/LC_MESSAGES/messages.po
   ```

2. **Add your translations**:
   ```
   msgid "New Feature"
   msgstr "Nuova Funzionalità"
   ```

3. **Compile translations**:
   ```bash
   pybabel compile -d ocular/locales
   ```

### Method 3: Generate New Translation Files

If you need to regenerate all translation files:

```bash
python3 scripts/generate_translations.py
pybabel compile -d ocular/locales
```

## Language Detection

### Switching Languages

Users can switch languages in three ways:

#### 1. Language Selector (UI)

The language dropdown in the navigation bar automatically reloads the page with the new language.

#### 2. Query Parameter

```
https://your-app.com/?lang=it
https://your-app.com/upload?lang=es
```

This sets a cookie and shows the page in the selected language.

#### 3. Browser Settings

The application automatically detects the user's preferred language from the `Accept-Language` header.

### Programmatic Language Change

```javascript
// JavaScript example
function changeLanguage(langCode) {
    const url = new URL(window.location.href);
    url.searchParams.set('lang', langCode);
    window.location.href = url.toString();
}

// Change to Italian
changeLanguage('it');
```

## Configuration

### Environment Variables

No special environment variables are required for basic i18n functionality. The DeepL API key is only needed for automated translations:

```bash
# Optional - only for automated translation
DEEPL_API_KEY=your-deepl-api-key
```

### Middleware Configuration

The i18n middleware is automatically configured in `ocular/app.py`:

```python
from ocular.middleware import I18nMiddleware

app.add_middleware(I18nMiddleware, default_language='en')
```

### Customizing Default Language

To change the default language:

```python
app.add_middleware(I18nMiddleware, default_language='it')
```

### Customizing Cookie/Query Parameter Names

```python
app.add_middleware(
    I18nMiddleware,
    default_language='en',
    cookie_name='user_language',
    query_param='locale'
)
```

## File Structure

```
ocular/
├── locales/                          # Translation files
│   ├── messages.pot                  # Translation template (source)
│   ├── it_IT/LC_MESSAGES/
│   │   ├── messages.po              # Italian translations (editable)
│   │   └── messages.mo              # Compiled Italian translations
│   ├── es_ES/LC_MESSAGES/
│   │   ├── messages.po              # Spanish translations
│   │   └── messages.mo              # Compiled Spanish translations
│   └── .../                         # Other languages
├── services/
│   └── i18n_service.py              # Translation service
├── middleware/
│   └── i18n_middleware.py           # FastAPI middleware
└── utils/
    └── template_helpers.py           # Template helper functions

scripts/
├── initialize_translations.sh        # Initialize all translation files
├── translate_with_deepl.py          # Automated translation with DeepL
└── generate_translations.py         # Generate pre-translated files

babel.cfg                            # Babel configuration
```

## Workflow for Adding New Strings

### Complete Workflow

```bash
# 1. Add translatable strings to your code/templates
echo '{{ _("New String") }}' >> ocular/templates/mypage.html

# 2. Extract all translatable strings
pybabel extract -F babel.cfg -o ocular/locales/messages.pot .

# 3. Update existing PO files with new strings
pybabel update -i ocular/locales/messages.pot -d ocular/locales

# 4. Translate using DeepL (or edit PO files manually)
python3 scripts/translate_with_deepl.py

# 5. Compile translations to binary format
pybabel compile -d ocular/locales

# 6. Restart your application
# Translations are now active!
```

### Quick Workflow (Using Pre-Generated Translations)

If you're using the pre-generated translations from `generate_translations.py`:

```bash
# 1. Add new string to generate_translations.py TRANSLATIONS dictionary
# 2. Regenerate all translation files
python3 scripts/generate_translations.py

# 3. Compile translations
pybabel compile -d ocular/locales

# 4. Restart application
```

## Troubleshooting

### Translations Not Showing

**Problem**: Translations are not appearing in the UI.

**Solutions**:

1. **Check if MO files are compiled**:
   ```bash
   ls -la ocular/locales/*/LC_MESSAGES/*.mo
   ```
   If missing, run: `pybabel compile -d ocular/locales`

2. **Verify middleware is installed**:
   Check `ocular/app.py` for:
   ```python
   app.add_middleware(I18nMiddleware, default_language='en')
   ```

3. **Restart the application**:
   ```bash
   # Kill and restart uvicorn
   pkill -f uvicorn
   uvicorn ocular.app:app --reload
   ```

4. **Check browser console** for JavaScript errors

### Wrong Language Displayed

**Problem**: Wrong language is displayed.

**Solutions**:

1. **Clear cookies**:
   ```javascript
   // In browser console
   document.cookie = 'language=; Max-Age=0';
   location.reload();
   ```

2. **Force language via URL**:
   ```
   https://your-app.com/?lang=en
   ```

3. **Check browser language settings**:
   The app respects `Accept-Language` headers

### Missing Translations

**Problem**: Some strings are in English even in other languages.

**Solutions**:

1. **Check PO file** for missing translations:
   ```bash
   grep -A1 "msgid \"Your String\"" ocular/locales/it_IT/LC_MESSAGES/messages.po
   ```

2. **Add missing translations** and recompile:
   ```bash
   # Edit PO file
   nano ocular/locales/it_IT/LC_MESSAGES/messages.po

   # Compile
   pybabel compile -d ocular/locales
   ```

3. **Use DeepL** to translate missing strings:
   ```bash
   python3 scripts/translate_with_deepl.py --languages it
   pybabel compile -d ocular/locales
   ```

### DeepL API Errors

**Problem**: Translation script fails with API errors.

**Solutions**:

1. **Check API key**:
   ```bash
   echo $DEEPL_API_KEY
   ```

2. **Verify API quota**:
   ```python
   import deepl
   translator = deepl.Translator(os.getenv('DEEPL_API_KEY'))
   usage = translator.get_usage()
   print(f"Used: {usage.character.count} / {usage.character.limit}")
   ```

3. **Use manual translation** if quota is exceeded

### Template Errors

**Problem**: Template rendering fails with undefined `_`.

**Solution**: Use the safe pattern:
```html
{{ _('Text') if _ is defined else 'Text' }}
```

## Best Practices

### 1. Always Use Translation Function

```html
<!-- Good -->
<h1>{{ _('Welcome') }}</h1>

<!-- Bad -->
<h1>Welcome</h1>
```

### 2. Keep Strings Simple

```html
<!-- Good -->
{{ _('Upload File') }}

<!-- Bad - hard to translate with context -->
{{ _('Upload') }} {{ _('File') }}
```

### 3. Use Placeholders for Dynamic Content

```python
# In Python
_("Welcome, {name}").format(name=user.name)
```

```html
<!-- In templates -->
{{ _('Welcome') }}, {{ user.name }}
```

### 4. Test All Languages

```bash
# Test each language manually
for lang in en it es fr de ja zh_CN; do
    echo "Testing $lang..."
    curl "http://localhost:8000/?lang=$lang"
done
```

### 5. Version Control

Always commit both `.po` and `.mo` files:

```bash
git add ocular/locales/*/LC_MESSAGES/*.po
git add ocular/locales/*/LC_MESSAGES/*.mo
git commit -m "Update translations"
```

## API Reference

### Translation Service

```python
from ocular.services.i18n_service import (
    get_translation_service,
    get_translator,
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAMES,
    LANGUAGE_FLAGS
)

# Get service instance
service = get_translation_service()

# Get translator for specific language
_ = get_translator('it')
translated = _("Hello")  # "Ciao"

# Get available languages
languages = service.get_available_languages()

# Detect language from header
lang = service.detect_language_from_header("en-US,en;q=0.9,it;q=0.8")
```

### Middleware Dependencies

```python
from ocular.middleware import (
    get_translation_from_request,
    get_language_from_request,
    get_locale_from_request
)

@router.get("/")
async def home(
    _: Callable = Depends(get_translation_from_request),
    language: str = Depends(get_language_from_request),
    locale: str = Depends(get_locale_from_request)
):
    return {
        "message": _("Welcome"),
        "language": language,  # "it"
        "locale": locale       # "it_IT"
    }
```

## Support

For questions or issues:

1. Check this documentation
2. Review `/mnt/developer/git/aecs4u.it/ocular/CLAUDE.md` for project setup
3. Examine example translations in `ocular/locales/`
4. Run the test scripts in `scripts/`

## License

Same as Ocular OCR Service project license.
