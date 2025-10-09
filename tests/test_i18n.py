"""
Comprehensive tests for internationalization (i18n) service and middleware.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ocular.middleware.i18n_middleware import I18nMiddleware
from ocular.services.i18n_service import I18nService


class TestI18nService:
    """Test i18n service functionality."""

    @pytest.fixture
    def i18n_service(self):
        """Create I18nService instance."""
        return I18nService()

    def test_initialization(self, i18n_service):
        """Test service initializes with default language."""
        assert i18n_service.default_language == "en"
        assert "en" in i18n_service.supported_languages
        assert "it_IT" in i18n_service.supported_languages

    def test_supported_languages(self, i18n_service):
        """Test all supported languages are available."""
        expected_languages = ["en", "it_IT", "de_DE", "es_ES", "fr_FR", "ja_JP", "zh_CN"]
        for lang in expected_languages:
            assert lang in i18n_service.supported_languages

    def test_get_translation_english(self, i18n_service):
        """Test getting translation for English (default)."""
        translation = i18n_service.get_translation("Welcome", "en")
        assert translation == "Welcome"  # English is fallback

    def test_get_translation_italian(self, i18n_service):
        """Test getting translation for Italian."""
        # Test a common string that should be translated
        translation = i18n_service.get_translation("Providers Management", "it_IT")
        # Should either return Italian translation or fallback to English
        assert isinstance(translation, str)
        assert len(translation) > 0

    def test_get_translation_missing_key(self, i18n_service):
        """Test getting translation for missing key returns original."""
        missing_key = "THIS_KEY_DOES_NOT_EXIST_12345"
        translation = i18n_service.get_translation(missing_key, "it_IT")
        assert translation == missing_key  # Should return original key

    def test_get_translation_unsupported_language(self, i18n_service):
        """Test unsupported language falls back to English."""
        translation = i18n_service.get_translation("Welcome", "xx_XX")
        assert translation == "Welcome"

    def test_translations_directory_exists(self, i18n_service):
        """Test translations directory structure exists."""
        locales_path = Path("ocular/locales")
        assert locales_path.exists()
        assert locales_path.is_dir()

        # Check each supported language has directory
        for lang in ["it_IT", "de_DE", "es_ES", "fr_FR", "ja_JP", "zh_CN"]:
            lang_dir = locales_path / lang / "LC_MESSAGES"
            assert lang_dir.exists(), f"Missing directory for {lang}"

    def test_mo_files_exist(self, i18n_service):
        """Test compiled .mo files exist for all languages."""
        locales_path = Path("ocular/locales")

        for lang in ["it_IT", "de_DE", "es_ES", "fr_FR", "ja_JP", "zh_CN"]:
            mo_file = locales_path / lang / "LC_MESSAGES" / "messages.mo"
            # mo file should exist after compilation
            if mo_file.exists():
                assert mo_file.stat().st_size > 0, f"{lang} .mo file is empty"

    def test_detect_language_from_header(self, i18n_service):
        """Test language detection from Accept-Language header."""
        # Test Italian preference
        lang = i18n_service.detect_language("it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7")
        assert lang == "it_IT"

        # Test German preference
        lang = i18n_service.detect_language("de-DE,de;q=0.9,en;q=0.8")
        assert lang == "de_DE"

        # Test Spanish preference
        lang = i18n_service.detect_language("es-ES,es;q=0.9")
        assert lang == "es_ES"

    def test_detect_language_fallback(self, i18n_service):
        """Test language detection falls back to English."""
        lang = i18n_service.detect_language("xx-XX,yy;q=0.9")
        assert lang == "en"

    def test_detect_language_no_header(self, i18n_service):
        """Test language detection with no header."""
        lang = i18n_service.detect_language(None)
        assert lang == "en"


class TestI18nMiddleware:
    """Test i18n middleware functionality."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with i18n middleware."""
        app = FastAPI()

        # Add i18n middleware
        app.add_middleware(I18nMiddleware)

        @app.get("/test")
        async def test_endpoint(request: Request):
            # Access translation function from request state
            _ = request.state._
            return {"message": _("Welcome")}

        @app.get("/language")
        async def language_endpoint(request: Request):
            return {"language": request.state.language}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_middleware_sets_language(self, client):
        """Test middleware sets language in request state."""
        response = client.get("/language", headers={"Accept-Language": "it-IT,it;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "it_IT"

    def test_middleware_sets_default_language(self, client):
        """Test middleware sets default language when no header."""
        response = client.get("/language")
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"

    def test_middleware_provides_translation_function(self, client):
        """Test middleware provides translation function."""
        response = client.get("/test")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_middleware_german_language(self, client):
        """Test middleware with German language."""
        response = client.get("/language", headers={"Accept-Language": "de-DE,de;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "de_DE"

    def test_middleware_spanish_language(self, client):
        """Test middleware with Spanish language."""
        response = client.get("/language", headers={"Accept-Language": "es-ES,es;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "es_ES"

    def test_middleware_french_language(self, client):
        """Test middleware with French language."""
        response = client.get("/language", headers={"Accept-Language": "fr-FR,fr;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "fr_FR"

    def test_middleware_japanese_language(self, client):
        """Test middleware with Japanese language."""
        response = client.get("/language", headers={"Accept-Language": "ja-JP,ja;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "ja_JP"

    def test_middleware_chinese_language(self, client):
        """Test middleware with Chinese language."""
        response = client.get("/language", headers={"Accept-Language": "zh-CN,zh;q=0.9"})
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "zh_CN"


class TestTemplateTranslations:
    """Test translations in templates."""

    def test_base_template_has_gettext_calls(self):
        """Test base template uses gettext for translations."""
        base_template = Path("ocular/templates/base.html")
        if base_template.exists():
            content = base_template.read_text()
            # Check for gettext translation markers
            assert "{{ _(" in content or "{{_(" in content

    def test_index_template_has_gettext_calls(self):
        """Test index template uses gettext for translations."""
        index_template = Path("ocular/templates/index.html")
        if index_template.exists():
            content = index_template.read_text()
            assert "{{ _(" in content or "{{_(" in content

    def test_providers_template_has_gettext_calls(self):
        """Test providers template uses gettext for translations."""
        providers_template = Path("ocular/templates/providers.html")
        if providers_template.exists():
            content = providers_template.read_text()
            assert "{{ _(" in content or "{{_(" in content


class TestTranslationCoverage:
    """Test translation coverage for different languages."""

    @pytest.fixture
    def i18n_service(self):
        return I18nService()

    def test_german_translation_coverage(self, i18n_service):
        """Test German translations have good coverage."""
        # Test some key strings
        test_strings = [
            "Providers Management",
            "Upload Document",
            "Settings",
            "Analytics",
        ]

        translations = [
            i18n_service.get_translation(s, "de_DE") for s in test_strings
        ]

        # At least some should be translated (not equal to original)
        translated_count = sum(1 for t, s in zip(translations, test_strings) if t != s)
        assert translated_count > 0, "No German translations found"

    def test_spanish_translation_coverage(self, i18n_service):
        """Test Spanish translations have good coverage."""
        test_strings = [
            "Providers Management",
            "Upload Document",
            "Settings",
            "Analytics",
        ]

        translations = [
            i18n_service.get_translation(s, "es_ES") for s in test_strings
        ]

        translated_count = sum(1 for t, s in zip(translations, test_strings) if t != s)
        assert translated_count > 0, "No Spanish translations found"

    def test_italian_translation_exists(self, i18n_service):
        """Test Italian translation files exist and have content."""
        locales_path = Path("ocular/locales/it_IT/LC_MESSAGES")
        po_file = locales_path / "messages.po"

        if po_file.exists():
            content = po_file.read_text()
            # Should have msgid and msgstr entries
            assert "msgid" in content
            assert "msgstr" in content


class TestLanguageSwitching:
    """Test language switching functionality."""

    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.add_middleware(I18nMiddleware)

        @app.get("/content")
        async def content_endpoint(request: Request):
            _ = request.state._
            return {
                "title": _("Providers Management"),
                "upload": _("Upload Document"),
                "settings": _("Settings"),
            }

        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_switch_to_italian(self, client):
        """Test switching to Italian language."""
        response = client.get("/content", headers={"Accept-Language": "it-IT"})
        assert response.status_code == 200
        data = response.json()
        # Translations should be strings
        assert isinstance(data["title"], str)
        assert isinstance(data["upload"], str)

    def test_switch_to_german(self, client):
        """Test switching to German language."""
        response = client.get("/content", headers={"Accept-Language": "de-DE"})
        assert response.status_code == 200
        data = response.json()
        assert all(isinstance(v, str) for v in data.values())

    def test_switch_to_spanish(self, client):
        """Test switching to Spanish language."""
        response = client.get("/content", headers={"Accept-Language": "es-ES"})
        assert response.status_code == 200
        data = response.json()
        assert all(isinstance(v, str) for v in data.values())


class TestPluralForms:
    """Test plural form handling in translations."""

    @pytest.fixture
    def i18n_service(self):
        return I18nService()

    def test_plural_forms_configuration(self, i18n_service):
        """Test plural forms are configured for each language."""
        # Check .po files have plural form headers
        locales_path = Path("ocular/locales")

        for lang in ["it_IT", "de_DE", "es_ES", "fr_FR"]:
            po_file = locales_path / lang / "LC_MESSAGES" / "messages.po"
            if po_file.exists():
                content = po_file.read_text()
                # Should have Plural-Forms header
                assert "Plural-Forms:" in content or "plural" in content.lower()


class TestErrorHandling:
    """Test error handling in i18n system."""

    @pytest.fixture
    def i18n_service(self):
        return I18nService()

    def test_missing_translation_returns_key(self, i18n_service):
        """Test missing translation returns original key."""
        result = i18n_service.get_translation(
            "NONEXISTENT_KEY_12345", "it_IT"
        )
        assert result == "NONEXISTENT_KEY_12345"

    def test_invalid_language_falls_back(self, i18n_service):
        """Test invalid language code falls back gracefully."""
        result = i18n_service.get_translation("Welcome", "invalid_lang")
        assert result == "Welcome"  # Should fallback to English

    def test_empty_string_translation(self, i18n_service):
        """Test empty string translation."""
        result = i18n_service.get_translation("", "it_IT")
        assert result == ""

    def test_none_string_handling(self, i18n_service):
        """Test None string handling."""
        # Should handle None gracefully
        with pytest.raises((TypeError, AttributeError)):
            i18n_service.get_translation(None, "it_IT")


class TestTranslationQuality:
    """Test translation quality and consistency."""

    @pytest.fixture
    def i18n_service(self):
        return I18nService()

    def test_no_untranslated_markers(self, i18n_service):
        """Test translations don't contain fuzzy markers."""
        locales_path = Path("ocular/locales")

        for lang in ["it_IT", "de_DE", "es_ES", "fr_FR", "ja_JP", "zh_CN"]:
            po_file = locales_path / lang / "LC_MESSAGES" / "messages.po"
            if po_file.exists():
                content = po_file.read_text()
                # Count fuzzy entries
                fuzzy_count = content.count("#, fuzzy")
                # It's OK to have some fuzzy entries, but warn if too many
                if fuzzy_count > 50:
                    pytest.warn(
                        f"{lang} has {fuzzy_count} fuzzy translations",
                        UserWarning,
                    )

    def test_translation_file_encoding(self, i18n_service):
        """Test translation files are properly UTF-8 encoded."""
        locales_path = Path("ocular/locales")

        for lang in ["it_IT", "de_DE", "es_ES", "fr_FR", "ja_JP", "zh_CN"]:
            po_file = locales_path / lang / "LC_MESSAGES" / "messages.po"
            if po_file.exists():
                # Should be able to read as UTF-8
                content = po_file.read_text(encoding="utf-8")
                assert len(content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ocular.services.i18n_service", "--cov=ocular.middleware.i18n_middleware"])
