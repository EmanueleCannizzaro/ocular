"""
Internationalization (i18n) service for Ocular OCR.

Provides translation support using Babel and Gettext with DeepL API integration.
"""

import gettext
from pathlib import Path
from typing import Dict, Optional, Callable
from functools import lru_cache


# Supported languages with their locales
SUPPORTED_LANGUAGES = {
    'en': 'en_US',  # English (source language)
    'it': 'it_IT',  # Italian
    'es': 'es_ES',  # Spanish
    'fr': 'fr_FR',  # French
    'de': 'de_DE',  # German
    'ja': 'ja_JP',  # Japanese
    'zh_CN': 'zh_CN',  # Chinese Simplified
}

# Language names for display
LANGUAGE_NAMES = {
    'en': 'English',
    'it': 'Italiano',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'ja': '日本語',
    'zh_CN': '简体中文',
}

# Language flags (emoji or flag codes)
LANGUAGE_FLAGS = {
    'en': '🇬🇧',
    'it': '🇮🇹',
    'es': '🇪🇸',
    'fr': '🇫🇷',
    'de': '🇩🇪',
    'ja': '🇯🇵',
    'zh_CN': '🇨🇳',
}

# Default locale directory
DEFAULT_LOCALE_DIR = Path(__file__).parent.parent / 'locales'


class TranslationService:
    """
    Service for managing translations across the application.

    Provides translation functions for templates and API responses.
    """

    def __init__(self, locale_dir: Optional[Path] = None):
        """
        Initialize the translation service.

        Args:
            locale_dir: Path to locale directory containing .mo files
        """
        self.locale_dir = locale_dir or DEFAULT_LOCALE_DIR
        self._translations: Dict[str, gettext.GNUTranslations] = {}
        self._load_translations()

    def _load_translations(self):
        """Load all translation files from locale directory."""
        if not self.locale_dir.exists():
            print(f"⚠️ Locale directory not found: {self.locale_dir}")
            return

        for lang_code, locale in SUPPORTED_LANGUAGES.items():
            mo_file = self.locale_dir / locale / 'LC_MESSAGES' / 'messages.mo'

            if mo_file.exists():
                try:
                    with open(mo_file, 'rb') as f:
                        self._translations[lang_code] = gettext.GNUTranslations(f)
                    print(f"✅ Loaded translations for {lang_code} ({locale})")
                except Exception as e:
                    print(f"❌ Failed to load {lang_code} translations: {e}")
            else:
                # Fallback to NullTranslations for missing languages
                self._translations[lang_code] = gettext.NullTranslations()
                if lang_code != 'en':  # Don't warn for English (source language)
                    print(f"⚠️ No translation file found for {lang_code} at {mo_file}")

    def get_translator(self, language: str = 'en') -> Callable[[str], str]:
        """
        Get translation function for a specific language.

        Args:
            language: Language code (en, it, es, fr, de, ja, zh_CN)

        Returns:
            Translation function that takes a string and returns translated string
        """
        # Normalize language code
        language = self._normalize_language(language)

        # Get translation object
        translation = self._translations.get(language)

        if translation is None:
            # Fallback to English
            print(f"⚠️ Translation not found for {language}, using English")
            translation = self._translations.get('en', gettext.NullTranslations())

        return translation.gettext

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language code to supported format.

        Args:
            language: Language code in various formats (en, en_US, en-US, etc.)

        Returns:
            Normalized language code (en, it, es, etc.)
        """
        # Handle None or empty string
        if not language:
            return 'en'

        # Convert to lowercase and replace hyphens with underscores
        language = language.lower().replace('-', '_')

        # Extract base language code (en from en_us)
        base_lang = language.split('_')[0]

        # Special case for Chinese
        if language.startswith('zh'):
            if 'cn' in language or 'hans' in language or 'simplified' in language:
                return 'zh_CN'
            # Default to simplified Chinese
            return 'zh_CN'

        # Check if base language is supported
        if base_lang in SUPPORTED_LANGUAGES:
            return base_lang

        # Fallback to English
        return 'en'

    def get_available_languages(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of available languages with metadata.

        Returns:
            Dictionary of language codes with names and flags
        """
        return {
            code: {
                'name': LANGUAGE_NAMES.get(code, code),
                'flag': LANGUAGE_FLAGS.get(code, ''),
                'locale': SUPPORTED_LANGUAGES.get(code, code),
            }
            for code in SUPPORTED_LANGUAGES.keys()
        }

    def detect_language_from_header(self, accept_language: Optional[str]) -> str:
        """
        Detect preferred language from Accept-Language header.

        Args:
            accept_language: Accept-Language header value (e.g., "en-US,en;q=0.9,it;q=0.8")

        Returns:
            Detected language code
        """
        if not accept_language:
            return 'en'

        # Parse Accept-Language header
        # Format: "en-US,en;q=0.9,it;q=0.8,de;q=0.7"
        languages = []

        for lang_part in accept_language.split(','):
            lang_part = lang_part.strip()

            # Extract language and quality factor
            if ';q=' in lang_part:
                lang, quality = lang_part.split(';q=')
                try:
                    quality = float(quality)
                except ValueError:
                    quality = 1.0
            else:
                lang = lang_part
                quality = 1.0

            # Normalize language code
            normalized = self._normalize_language(lang)
            languages.append((quality, normalized))

        # Sort by quality factor (descending)
        languages.sort(reverse=True, key=lambda x: x[0])

        # Return first supported language
        for _, lang in languages:
            if lang in SUPPORTED_LANGUAGES:
                return lang

        # Fallback to English
        return 'en'


# Global translation service instance
@lru_cache(maxsize=1)
def get_translation_service() -> TranslationService:
    """Get singleton instance of translation service."""
    return TranslationService()


def get_translator(language: str = 'en') -> Callable[[str], str]:
    """
    Get translation function for a specific language.

    This is a convenience function for quick access to translations.

    Args:
        language: Language code (en, it, es, fr, de, ja, zh_CN)

    Returns:
        Translation function
    """
    service = get_translation_service()
    return service.get_translator(language)


# Alias for common usage
_ = get_translator('en')  # Default English translator
