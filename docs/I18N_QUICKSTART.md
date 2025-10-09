# i18n Quick Start Guide

Get internationalization up and running in 5 minutes!

## Installation

```bash
# Install required packages
pip install babel polib deepl

# Or using uv
uv add babel polib deepl
```

## Setup (Already Done!)

The translation files are already generated and compiled. You're ready to use them!

## Using Translations

### In Templates

```html
<!-- base.html or any template -->
<h1>{{ _('Welcome') }}</h1>
<p>{{ _('Extract text and data from PDFs and images') }}</p>
```

### In Python Routes

```python
from fastapi import Request, APIRouter

router = APIRouter()

@router.get("/api/status")
async def status(request: Request):
    _ = request.state._  # Get translation function

    return {
        "message": _("Processing"),
        "status": _("Success")
    }
```

## Switching Languages

### Via URL
```
http://localhost:8000/?lang=it
http://localhost:8000/?lang=es
http://localhost:8000/?lang=fr
```

### Via Language Selector
Click the language dropdown in the navigation bar (🇬🇧 English ▼)

### Via Cookie
```javascript
document.cookie = 'language=it; path=/; max-age=31536000';
location.reload();
```

## Adding New Translations

### Option 1: Use Pre-Generated Translations

1. Edit `scripts/generate_translations.py`
2. Add your strings to the `TRANSLATIONS` dictionary
3. Run:
   ```bash
   python3 scripts/generate_translations.py
   pybabel compile -d ocular/locales
   ```

### Option 2: Use DeepL API (Professional Quality)

1. Get a DeepL API key from https://www.deepl.com/pro-api
2. Add to `.env`:
   ```bash
   DEEPL_API_KEY=your-api-key-here
   ```
3. Add new strings to your code/templates
4. Run:
   ```bash
   pybabel extract -F babel.cfg -o ocular/locales/messages.pot .
   pybabel update -i ocular/locales/messages.pot -d ocular/locales
   python3 scripts/translate_with_deepl.py
   pybabel compile -d ocular/locales
   ```

### Option 3: Manual Translation

1. Edit the `.po` files directly:
   ```bash
   nano ocular/locales/it_IT/LC_MESSAGES/messages.po
   ```

2. Find your string and add translation:
   ```
   msgid "New Feature"
   msgstr "Nuova Funzionalità"
   ```

3. Compile:
   ```bash
   pybabel compile -d ocular/locales
   ```

## Testing

```bash
# Start the app
uvicorn ocular.app:app --reload

# Test different languages
curl "http://localhost:8000/?lang=it"
curl "http://localhost:8000/?lang=es"
curl "http://localhost:8000/?lang=fr"
```

## Supported Languages

- 🇬🇧 English (en)
- 🇮🇹 Italian (it)
- 🇪🇸 Spanish (es)
- 🇫🇷 French (fr)
- 🇩🇪 German (de)
- 🇯🇵 Japanese (ja)
- 🇨🇳 Chinese Simplified (zh_CN)

## Troubleshooting

### Translations not showing?

```bash
# Recompile translations
pybabel compile -d ocular/locales

# Restart app
pkill -f uvicorn
uvicorn ocular.app:app --reload
```

### Need to force a language?

```bash
# Add ?lang=XX to any URL
http://localhost:8000/?lang=it
```

### Want to add more languages?

Edit `ocular/services/i18n_service.py` and add to `SUPPORTED_LANGUAGES` dictionary.

## Complete Workflow Example

```bash
# 1. Add new string to template
echo '<p>{{ _("New Feature Available") }}</p>' >> ocular/templates/upload.html

# 2. Extract, translate, compile
pybabel extract -F babel.cfg -o ocular/locales/messages.pot .
pybabel update -i ocular/locales/messages.pot -d ocular/locales
python3 scripts/translate_with_deepl.py  # or edit .po files manually
pybabel compile -d ocular/locales

# 3. Restart app
uvicorn ocular.app:app --reload

# 4. Test
curl "http://localhost:8000/?lang=it"
```

## Next Steps

- Read the full [I18N Guide](I18N_GUIDE.md) for detailed documentation
- Explore the translation files in `ocular/locales/`
- Check out the middleware code in `ocular/middleware/i18n_middleware.py`
- Review examples in the updated `base.html` template

That's it! Your app now speaks 7 languages! 🌐
