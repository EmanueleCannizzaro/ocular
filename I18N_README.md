# Ocular OCR - Internationalization (i18n) System

Complete internationalization support for the Ocular OCR Service with 7 languages and professional translations.

## Features

- ✅ **7 Languages Supported**: English, Italian, Spanish, French, German, Japanese, Chinese
- ✅ **Professional Translations**: Pre-generated with context-aware translations
- ✅ **Automatic Detection**: Language detection from query params, cookies, or browser settings
- ✅ **FastAPI Integration**: Seamless middleware integration with request-level translation
- ✅ **Template Support**: Jinja2 templates with `_()` translation function
- ✅ **API Support**: Translation functions available in all route handlers
- ✅ **Easy Extension**: Add new languages or strings in minutes
- ✅ **DeepL Integration**: Optional automated translation using DeepL API

## Quick Start

### 1. Using Translations in Templates

```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ _('Ocular OCR Service') }}</title>
</head>
<body>
    <h1>{{ _('Welcome') }}</h1>
    <p>{{ _('Extract text and data from PDFs and images') }}</p>
</body>
</html>
```

### 2. Using Translations in Python

```python
from fastapi import Request, APIRouter

router = APIRouter()

@router.get("/api/status")
async def get_status(request: Request):
    _ = request.state._  # Get translation function

    return {
        "message": _("Processing your document"),
        "status": _("Success")
    }
```

### 3. Switching Languages

```bash
# Via URL query parameter
http://localhost:8000/?lang=it

# Via cookie
document.cookie = 'language=it';

# Via Accept-Language header (automatic)
# Browser settings are automatically detected
```

## Supported Languages

| Language | Code | Locale | Flag |
|----------|------|--------|------|
| English | `en` | en_US | 🇬🇧 |
| Italian | `it` | it_IT | 🇮🇹 |
| Spanish | `es` | es_ES | 🇪🇸 |
| French | `fr` | fr_FR | 🇫🇷 |
| German | `de` | de_DE | 🇩🇪 |
| Japanese | `ja` | ja_JP | 🇯🇵 |
| Chinese (Simplified) | `zh_CN` | zh_CN | 🇨🇳 |

## Architecture

### Components

1. **Translation Service** (`ocular/services/i18n_service.py`)
   - Manages gettext translations
   - Provides translation functions
   - Handles language detection

2. **Middleware** (`ocular/middleware/i18n_middleware.py`)
   - Detects language preference
   - Injects translation function into request state
   - Sets language cookie

3. **Translation Files** (`ocular/locales/`)
   - `.pot` - Translation template (source)
   - `.po` - Editable translation files
   - `.mo` - Compiled binary translations

4. **Scripts** (`scripts/`)
   - `generate_translations.py` - Generate pre-translated files
   - `translate_with_deepl.py` - Automated translation with DeepL
   - `initialize_translations.sh` - Initialize translation structure

## File Structure

```
ocular/
├── locales/
│   ├── messages.pot              # Translation template
│   ├── it_IT/LC_MESSAGES/
│   │   ├── messages.po          # Italian translations (editable)
│   │   └── messages.mo          # Compiled Italian
│   ├── es_ES/LC_MESSAGES/       # Spanish
│   ├── fr_FR/LC_MESSAGES/       # French
│   ├── de_DE/LC_MESSAGES/       # German
│   ├── ja_JP/LC_MESSAGES/       # Japanese
│   └── zh_CN/LC_MESSAGES/       # Chinese
├── services/
│   └── i18n_service.py          # Translation service
├── middleware/
│   └── i18n_middleware.py       # FastAPI middleware
├── templates/
│   └── base.html                # Updated with translations
└── utils/
    └── template_helpers.py       # Template utilities

scripts/
├── initialize_translations.sh    # Setup script
├── translate_with_deepl.py      # DeepL integration
└── generate_translations.py     # Pre-generated translations

docs/
├── I18N_GUIDE.md                # Complete documentation
└── I18N_QUICKSTART.md           # Quick start guide

babel.cfg                        # Babel configuration
```

## How to Use

### In Jinja2 Templates

```html
<!-- Simple translation -->
<h1>{{ _('Upload') }}</h1>

<!-- With fallback (safe) -->
<p>{{ _('Welcome') if _ is defined else 'Welcome' }}</p>

<!-- In attributes -->
<button title="{{ _('Click to upload') }}">{{ _('Upload File') }}</button>

<!-- Language selector -->
{% set current_lang = request.state.language if request.state.language is defined else 'en' %}
<select id="language">
    <option value="en" {% if current_lang == 'en' %}selected{% endif %}>English</option>
    <option value="it" {% if current_lang == 'it' %}selected{% endif %}>Italiano</option>
</select>
```

### In FastAPI Routes

```python
# Method 1: Via request.state
@router.get("/")
async def home(request: Request):
    _ = request.state._
    return {"message": _("Welcome")}

# Method 2: Via dependency
from ocular.middleware import get_translation_from_request

@router.get("/info")
async def info(_: Callable = Depends(get_translation_from_request)):
    return {"status": _("Processing")}

# Method 3: Direct service usage
from ocular.services.i18n_service import get_translator

@router.get("/data")
async def data(language: str = 'en'):
    _ = get_translator(language)
    return {"message": _("Success")}
```

## Adding New Translations

### Method 1: Pre-Generated (Fastest)

```bash
# Edit scripts/generate_translations.py and add your strings
# Then run:
python3 scripts/generate_translations.py
pybabel compile -d ocular/locales
```

### Method 2: DeepL API (Professional Quality)

```bash
# 1. Get DeepL API key and add to .env
echo "DEEPL_API_KEY=your-key" >> .env

# 2. Add strings to templates/code
echo '{{ _("New Feature") }}' >> ocular/templates/mypage.html

# 3. Extract and translate
pybabel extract -F babel.cfg -o ocular/locales/messages.pot .
pybabel update -i ocular/locales/messages.pot -d ocular/locales
python3 scripts/translate_with_deepl.py

# 4. Compile
pybabel compile -d ocular/locales
```

### Method 3: Manual Translation

```bash
# 1. Edit PO file
nano ocular/locales/it_IT/LC_MESSAGES/messages.po

# 2. Add translation
# msgid "New Feature"
# msgstr "Nuova Funzionalità"

# 3. Compile
pybabel compile -d ocular/locales
```

## Language Detection

The middleware detects language in this priority order:

1. **Query Parameter**: `?lang=it` (highest priority)
   - Sets cookie and displays page in selected language
   - Cookie persists for 1 year

2. **Cookie**: `language=it`
   - Persists user's language preference
   - Set automatically when using query parameter

3. **Accept-Language Header**: Browser preferences
   - Automatically detects from browser settings
   - Parses quality factors (e.g., `en-US,en;q=0.9,it;q=0.8`)

4. **Default Language**: English (en)
   - Fallback if no preference detected

## Configuration

### Environment Variables

```bash
# Optional - only for automated translation
DEEPL_API_KEY=your-deepl-api-key
```

### Middleware Setup (Already Configured)

The middleware is automatically configured in `ocular/app.py`:

```python
from ocular.middleware import I18nMiddleware

app.add_middleware(I18nMiddleware, default_language='en')
```

### Custom Configuration

```python
# Change default language
app.add_middleware(I18nMiddleware, default_language='it')

# Customize cookie/query parameter names
app.add_middleware(
    I18nMiddleware,
    default_language='en',
    cookie_name='user_language',
    query_param='locale'
)
```

## Translation Coverage

All major UI elements are translated:

### Navigation
- Core Features, Infrastructure, Analytics & Monitoring, System
- Upload, Batch Processing, Document History
- Deployments, Models, Providers
- Analytics, Observability, Settings

### User Interface
- Profile, Billing, Logout, Notifications, Language
- Save, Cancel, Delete, Edit, Create, Update
- Search, Filter, Sort, Actions, Status

### Messages
- Processing, Completed, Failed, Error, Success, Warning
- New deployment completed, Batch job finished, API usage alert

### Application Info
- Ocular OCR Service
- Extract text and data from PDFs and images
- Powered by Mistral AI, OLM OCR, and RoLM OCR
- Version

### File Operations
- Select File, Upload File, Drag and drop files here
- Supported formats, Maximum file size
- Processing your document, Please wait
- Upload successful, Upload failed

### Validation & Errors
- An error occurred, Please try again
- Invalid input, Required field, Field cannot be empty
- Invalid email address, Invalid URL, Invalid date
- No results found, Loading, Refresh, Retry

## Testing

### Manual Testing

```bash
# Start application
uvicorn ocular.app:app --reload

# Test each language
curl "http://localhost:8000/?lang=en"
curl "http://localhost:8000/?lang=it"
curl "http://localhost:8000/?lang=es"
curl "http://localhost:8000/?lang=fr"
curl "http://localhost:8000/?lang=de"
curl "http://localhost:8000/?lang=ja"
curl "http://localhost:8000/?lang=zh_CN"
```

### Browser Testing

1. Open http://localhost:8000
2. Click language dropdown in navigation
3. Select different languages
4. Verify translations appear correctly
5. Check that language persists after page reload

### API Testing

```bash
# Test API endpoint with different languages
curl -H "Accept-Language: it-IT" http://localhost:8000/api/status
curl -H "Accept-Language: es-ES" http://localhost:8000/api/status
```

## Troubleshooting

### Translations not appearing

```bash
# Check if MO files exist
ls -la ocular/locales/*/LC_MESSAGES/*.mo

# Recompile if missing
pybabel compile -d ocular/locales

# Restart application
pkill -f uvicorn && uvicorn ocular.app:app --reload
```

### Wrong language displayed

```bash
# Clear language cookie
# In browser console:
document.cookie = 'language=; Max-Age=0';
location.reload();

# Or force language via URL
http://localhost:8000/?lang=en
```

### Adding new language

1. Add language to `ocular/services/i18n_service.py`:
   ```python
   SUPPORTED_LANGUAGES = {
       'en': 'en_US',
       'pt': 'pt_BR',  # Add Portuguese
   }
   ```

2. Initialize PO file:
   ```bash
   pybabel init -i ocular/locales/messages.pot -d ocular/locales -l pt_BR
   ```

3. Translate and compile:
   ```bash
   python3 scripts/translate_with_deepl.py --languages pt
   pybabel compile -d ocular/locales
   ```

## Performance

- **Translation Lookup**: O(1) - Constant time using gettext
- **Memory Footprint**: ~100KB per language (compiled .mo files)
- **Overhead**: <1ms per request (middleware processing)
- **Caching**: Translations are loaded once at startup and cached

## Best Practices

1. **Always use `_()` function** for user-facing strings
2. **Keep strings simple** and avoid concatenation
3. **Test all languages** before deploying
4. **Commit both .po and .mo files** to version control
5. **Use DeepL for professional quality** when possible
6. **Provide context** in comments for ambiguous strings
7. **Review translations** before production deployment

## Future Enhancements

- [ ] Add more languages (Portuguese, Russian, Arabic, etc.)
- [ ] Implement plural forms (ngettext)
- [ ] Add date/time localization
- [ ] Add number/currency formatting
- [ ] Create translation review workflow
- [ ] Add automated translation quality checks
- [ ] Implement translation memory for consistency

## Resources

- **Full Documentation**: [docs/I18N_GUIDE.md](docs/I18N_GUIDE.md)
- **Quick Start**: [docs/I18N_QUICKSTART.md](docs/I18N_QUICKSTART.md)
- **Babel Documentation**: https://babel.pocoo.org/
- **Gettext Manual**: https://www.gnu.org/software/gettext/manual/
- **DeepL API**: https://www.deepl.com/docs-api

## Support

For questions or issues:

1. Check the [I18N Guide](docs/I18N_GUIDE.md)
2. Review the [Quick Start](docs/I18N_QUICKSTART.md)
3. Examine example code in `ocular/templates/base.html`
4. Check the project documentation in `CLAUDE.md`

## Credits

- **Translation System**: Babel + Gettext
- **Automated Translation**: DeepL API
- **Framework Integration**: FastAPI + Jinja2
- **Languages**: 7 languages with professional translations

---

**Status**: ✅ Fully implemented and ready to use

**Last Updated**: 2025-10-06

**Version**: 1.0.0
