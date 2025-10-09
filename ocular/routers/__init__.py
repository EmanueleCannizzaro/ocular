"""
Routers module for Ocular OCR application.
Contains modularized FastAPI routers for different providers and functionality.
"""

from fastapi.templating import Jinja2Templates

from ocular.settings import settings


# Setup templates with custom globals
templates = Jinja2Templates(directory=str(settings.app.templates_path))
