"""
Middleware components for Ocular OCR application.
"""

from ocular.middleware.i18n_middleware import (
    I18nMiddleware,
    get_translation_from_request,
    get_language_from_request,
    get_locale_from_request,
)

__all__ = [
    'I18nMiddleware',
    'get_translation_from_request',
    'get_language_from_request',
    'get_locale_from_request',
]
