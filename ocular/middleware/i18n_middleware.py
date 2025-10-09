"""
i18n Middleware for FastAPI.

Detects user's language preference and provides translation functions to templates.
"""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ocular.services.i18n_service import get_translation_service, SUPPORTED_LANGUAGES


class I18nMiddleware(BaseHTTPMiddleware):
    """
    Middleware for internationalization support.

    Detects language from:
    1. Query parameter (?lang=it)
    2. Cookie (language=it)
    3. Accept-Language header

    Provides translation function to request state.
    """

    def __init__(
        self,
        app: ASGIApp,
        default_language: str = 'en',
        cookie_name: str = 'language',
        query_param: str = 'lang'
    ):
        """
        Initialize i18n middleware.

        Args:
            app: ASGI application
            default_language: Default language code (default: 'en')
            cookie_name: Name of language cookie (default: 'language')
            query_param: Name of language query parameter (default: 'lang')
        """
        super().__init__(app)
        self.default_language = default_language
        self.cookie_name = cookie_name
        self.query_param = query_param
        self.translation_service = get_translation_service()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request and add translation support."""

        # Detect language preference
        language = self._detect_language(request)

        # Get translator function for this language
        translator = self.translation_service.get_translator(language)

        # Add translation function and metadata to request state
        request.state._ = translator  # Translation function
        request.state.gettext = translator  # Alias
        request.state.language = language  # Current language code
        request.state.locale = SUPPORTED_LANGUAGES.get(language, 'en_US')  # Full locale

        # Get available languages for language selector
        request.state.available_languages = self.translation_service.get_available_languages()

        # Process request
        response = await call_next(request)

        # If language was set via query parameter, set cookie
        if self.query_param in request.query_params:
            response.set_cookie(
                key=self.cookie_name,
                value=language,
                max_age=365 * 24 * 60 * 60,  # 1 year
                httponly=True,
                samesite='lax'
            )

        return response

    def _detect_language(self, request: Request) -> str:
        """
        Detect language from request.

        Priority:
        1. Query parameter (?lang=it)
        2. Cookie (language=it)
        3. Accept-Language header
        4. Default language

        Args:
            request: FastAPI request

        Returns:
            Detected language code
        """
        # 1. Check query parameter
        if self.query_param in request.query_params:
            lang = request.query_params[self.query_param]
            normalized = self.translation_service._normalize_language(lang)
            if normalized in SUPPORTED_LANGUAGES:
                return normalized

        # 2. Check cookie
        if self.cookie_name in request.cookies:
            lang = request.cookies[self.cookie_name]
            normalized = self.translation_service._normalize_language(lang)
            if normalized in SUPPORTED_LANGUAGES:
                return normalized

        # 3. Check Accept-Language header
        accept_language = request.headers.get('accept-language')
        if accept_language:
            detected = self.translation_service.detect_language_from_header(accept_language)
            if detected in SUPPORTED_LANGUAGES:
                return detected

        # 4. Default language
        return self.default_language


def get_translation_from_request(request: Request) -> Callable[[str], str]:
    """
    Get translation function from request state.

    This is a dependency that can be used in FastAPI routes.

    Args:
        request: FastAPI request

    Returns:
        Translation function

    Example:
        ```python
        @app.get("/")
        async def home(request: Request, _: Callable = Depends(get_translation_from_request)):
            return {"message": _("Welcome")}
        ```
    """
    return getattr(request.state, '_', lambda x: x)


def get_language_from_request(request: Request) -> str:
    """
    Get current language from request state.

    This is a dependency that can be used in FastAPI routes.

    Args:
        request: FastAPI request

    Returns:
        Current language code

    Example:
        ```python
        @app.get("/")
        async def home(language: str = Depends(get_language_from_request)):
            return {"language": language}
        ```
    """
    return getattr(request.state, 'language', 'en')


def get_locale_from_request(request: Request) -> str:
    """
    Get current locale from request state.

    Args:
        request: FastAPI request

    Returns:
        Current locale (e.g., 'en_US', 'it_IT')
    """
    return getattr(request.state, 'locale', 'en_US')
