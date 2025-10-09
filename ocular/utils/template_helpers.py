"""
Template helper utilities for Ocular OCR application.

Provides helper functions for use in Jinja2 templates.
"""

from typing import Dict, Any
from fastapi import Request
from fastapi.templating import Jinja2Templates


def prepare_template_context(
    request: Request,
    template_name: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Prepare template context with i18n support.

    This helper ensures that the translation function (_) is always
    available in templates, even if middleware hasn't run.

    Args:
        request: FastAPI request object
        template_name: Name of the template to render
        context: Additional context variables (optional)

    Returns:
        Complete context dictionary ready for template rendering

    Example:
        ```python
        @router.get("/")
        async def home(request: Request):
            context = prepare_template_context(
                request,
                "home.html",
                {"title": "Welcome", "user": user_obj}
            )
            return templates.TemplateResponse(context["template"], context)
        ```
    """
    # Start with base context
    base_context = {
        "request": request,
        "template": template_name,
    }

    # Add translation function from request state
    # Fallback to passthrough function if not available
    translation_func = getattr(request.state, '_', lambda x: x)
    base_context['_'] = translation_func

    # Add language metadata
    base_context['current_language'] = getattr(request.state, 'language', 'en')
    base_context['current_locale'] = getattr(request.state, 'locale', 'en_US')
    base_context['available_languages'] = getattr(request.state, 'available_languages', {})

    # Merge with provided context
    if context:
        base_context.update(context)

    return base_context


def setup_jinja_filters(templates: Jinja2Templates):
    """
    Add custom Jinja2 filters for i18n support.

    Args:
        templates: Jinja2Templates instance

    Example:
        ```python
        templates = Jinja2Templates(directory="templates")
        setup_jinja_filters(templates)
        ```
    """

    @templates.env.globals['translate']
    def translate_filter(text: str, request: Request = None) -> str:
        """
        Jinja2 filter for translating text.

        Usage in templates:
            {{ "Hello" | translate }}
        """
        if request and hasattr(request.state, '_'):
            return request.state._(text)
        return text

    @templates.env.globals['lang_name']
    def lang_name_filter(lang_code: str) -> str:
        """
        Get language name from code.

        Usage in templates:
            {{ "it" | lang_name }}  -> "Italiano"
        """
        from ocular.services.i18n_service import LANGUAGE_NAMES
        return LANGUAGE_NAMES.get(lang_code, lang_code)

    @templates.env.globals['lang_flag']
    def lang_flag_filter(lang_code: str) -> str:
        """
        Get language flag emoji from code.

        Usage in templates:
            {{ "it" | lang_flag }}  -> "🇮🇹"
        """
        from ocular.services.i18n_service import LANGUAGE_FLAGS
        return LANGUAGE_FLAGS.get(lang_code, '')


def inject_translation_to_response(
    templates: Jinja2Templates,
    template_name: str,
    request: Request,
    context: Dict[str, Any] = None
):
    """
    Convenience wrapper for TemplateResponse with i18n support.

    Args:
        templates: Jinja2Templates instance
        template_name: Name of template to render
        request: FastAPI request
        context: Additional context variables

    Returns:
        TemplateResponse with i18n support

    Example:
        ```python
        from ocular.utils.template_helpers import inject_translation_to_response

        @router.get("/")
        async def home(request: Request):
            return inject_translation_to_response(
                templates,
                "home.html",
                request,
                {"title": "Welcome"}
            )
        ```
    """
    full_context = prepare_template_context(request, template_name, context)
    return templates.TemplateResponse(template_name, full_context)
