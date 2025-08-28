"""
Google Cloud Functions entry point for Ocular OCR Service.

This module adapts the FastAPI application to run on Google Cloud Functions
using Mangum ASGI adapter for reliable FastAPI integration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path so imports work correctly
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import functions framework
import functions_framework

# Import the FastAPI app
from app.ocular_app import app


@functions_framework.http
def ocular_ocr(request):
    """
    Cloud Functions HTTP entry point for Ocular OCR service.
    
    Uses Mangum ASGI adapter for proper FastAPI integration with Cloud Functions.
    
    Args:
        request: The HTTP request object from Cloud Functions
        
    Returns:
        HTTP response from the FastAPI application
    """
    from mangum import Mangum
    from flask import Response as FlaskResponse
    
    # Create Mangum handler for FastAPI (disable lifespan for Cloud Functions)
    handler = Mangum(app, lifespan="off")
    
    # Convert Cloud Functions request to Lambda-style event for Mangum
    # Get query string safely
    query_string_params = {}
    if hasattr(request, 'args') and request.args:
        query_string_params = dict(request.args)
    
    # Handle request body
    body = None
    is_base64 = False
    
    if hasattr(request, 'get_data'):
        data = request.get_data()
        if data:
            try:
                # Try to decode as text first
                body = data.decode('utf-8')
            except UnicodeDecodeError:
                # If it fails, it's binary data - encode as base64
                import base64
                body = base64.b64encode(data).decode('utf-8')
                is_base64 = True
    
    # Build Lambda-style event
    event = {
        "httpMethod": request.method,
        "path": request.path,
        "queryStringParameters": query_string_params or None,
        "multiValueQueryStringParameters": None,
        "headers": dict(request.headers),
        "multiValueHeaders": None,
        "body": body,
        "isBase64Encoded": is_base64,
        "requestContext": {
            "requestId": "cloud-functions-request",
            "stage": "prod",
            "httpMethod": request.method,
            "path": request.path,
        },
        "pathParameters": None,
        "stageVariables": None,
    }
    
    # Create minimal context
    class Context:
        def __init__(self):
            self.aws_request_id = "cloud-functions-request"
            self.function_name = "ocular-ocr-service"
            self.memory_limit_in_mb = "2048"
            self.invoked_function_arn = "arn:aws:lambda:region:account:function:ocular-ocr-service"
        
        def get_remaining_time_in_millis(self):
            return 540000  # 9 minutes
    
    context = Context()
    
    try:
        # Use Mangum to handle the request
        response = handler(event, context)
        
        # Convert Mangum response to Flask response
        headers = response.get("headers", {})
        status_code = response.get("statusCode", 200)
        body = response.get("body", "")
        
        # Handle base64 encoded responses
        if response.get("isBase64Encoded", False):
            import base64
            body = base64.b64decode(body)
        
        return FlaskResponse(
            body,
            status=status_code,
            headers=headers
        )
        
    except Exception as e:
        print(f"Error in Cloud Function: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error response
        return FlaskResponse(
            f"Internal Server Error: {str(e)}",
            status=500,
            headers={"Content-Type": "text/plain"}
        )


# For local testing with functions-framework-python
if __name__ == "__main__":
    # Run with: functions-framework --target=ocular_ocr --debug
    print("Cloud Function ready for local testing")
    print("Use: functions-framework --target=ocular_ocr --debug")