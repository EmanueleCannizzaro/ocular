"""
Google Cloud Functions entry point for Ocular OCR Service.

This module adapts the FastAPI application to run on Google Cloud Functions
using the Functions Framework.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path so imports work correctly
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the FastAPI app
from app.ocular_app import app

# Import functions framework
import functions_framework

# Create the Cloud Functions entry point
@functions_framework.http
def ocular_ocr(request):
    """
    Cloud Functions HTTP entry point for Ocular OCR service.
    
    This function serves as the entry point for Google Cloud Functions,
    routing all HTTP requests to the FastAPI application.
    
    Args:
        request: The HTTP request object from Cloud Functions
        
    Returns:
        HTTP response from the FastAPI application
    """
    # Import here to avoid issues during deployment
    from fastapi import Request as FastAPIRequest
    from starlette.applications import Starlette
    from starlette.responses import Response
    import asyncio
    
    # Create an ASGI adapter for the Cloud Functions request
    async def asgi_app(scope, receive, send):
        """ASGI application adapter for Cloud Functions"""
        await app(scope, receive, send)
    
    # Convert Cloud Functions request to ASGI format
    import json
    from urllib.parse import parse_qs, unquote
    
    # Build ASGI scope
    scope = {
        'type': 'http',
        'method': request.method,
        'path': request.path,
        'query_string': request.query_string.encode() if hasattr(request, 'query_string') else b'',
        'headers': [
            [key.lower().encode(), value.encode()]
            for key, value in request.headers.items()
        ],
        'server': ('localhost', 8080),
        'client': ('127.0.0.1', 0),
    }
    
    # Handle request body
    body = b''
    if hasattr(request, 'data'):
        body = request.data if isinstance(request.data, bytes) else request.data.encode()
    elif hasattr(request, 'get_data'):
        body = request.get_data()
    elif hasattr(request, 'form'):
        # Handle form data (for file uploads)
        import io
        body_io = io.BytesIO()
        # This is a simplified approach - Cloud Functions handles multipart differently
        body = body_io.getvalue()
    
    # ASGI receive callable
    async def receive():
        return {
            'type': 'http.request',
            'body': body,
            'more_body': False,
        }
    
    # ASGI send callable
    response_data = {}
    
    async def send(message):
        if message['type'] == 'http.response.start':
            response_data['status'] = message['status']
            response_data['headers'] = dict(message.get('headers', []))
        elif message['type'] == 'http.response.body':
            response_data['body'] = message.get('body', b'')
    
    # Run the ASGI app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asgi_app(scope, receive, send))
    finally:
        loop.close()
    
    # Return Cloud Functions compatible response
    from flask import Response as FlaskResponse
    
    headers = response_data.get('headers', {})
    return FlaskResponse(
        response_data.get('body', b''),
        status=response_data.get('status', 200),
        headers=headers
    )


# Alternative simpler approach using ASGI adapter
@functions_framework.http
def ocular_ocr_simple(request):
    """
    Simplified Cloud Functions entry point using ASGI adapter.
    
    This is an alternative approach that uses an ASGI-to-WSGI adapter
    for simpler integration with Cloud Functions.
    """
    from fastapi.middleware.wsgi import WSGIMiddleware
    from werkzeug.serving import WSGIRequestHandler
    
    # This approach requires additional dependencies and setup
    # Use the main ocular_ocr function above for standard deployment
    pass


# For local testing with functions-framework-python
if __name__ == "__main__":
    import functions_framework
    
    # Start the functions framework for local development
    functions_framework._http_view_func_registry["ocular_ocr"] = ocular_ocr
    
    # Run with: functions-framework --target=ocular_ocr --debug
    print("Cloud Function ready for local testing")
    print("Use: functions-framework --target=ocular_ocr --debug")