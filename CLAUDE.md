# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ocular is a comprehensive Python package that provides OCR (Optical Character Recognition) capabilities using multiple providers including AI vision models, cloud APIs, and local processing engines. The package supports data extraction from PDF files and scanned images with a unified interface and robust error handling.

### Current Implementation Status
- ✅ **Fully functional OCR system** with multiple provider support
- ✅ **Web application** with FastAPI backend and HTML frontend
- ✅ **Multiple OCR providers** implemented and working
- ✅ **Unified configuration management** with simplified settings architecture
- ✅ **Comprehensive error handling** and logging system
- ✅ **Streamlined codebase** with merged configuration classes

## Development Setup

### Git Workflow
**IMPORTANT**: Always create a new branch for any new code or changes:

```bash
# Create and switch to a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix

# Make your changes, then commit and push
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name

# Create a pull request for review before merging to main
```

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install the package in development mode
uv add -e .

# Install additional dependencies for specific providers
uv add pdf2image          # For PDF processing with Mistral
uv add google-cloud-vision  # For Google Vision API
uv add boto3             # For AWS Textract
uv add pytesseract pillow # For local Tesseract OCR
uv add azure-ai-documentintelligence  # For Azure Document Intelligence
```

### Running the Application
```bash
# Start the web application
source .venv/bin/activate
cd web
python app.py
# Or: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# The web interface will be available at http://localhost:8000
```

### Testing Commands
- `python -m pytest` - Run tests (when test suite is available)
- `python test_providers.py` - Test individual OCR providers
- `python test_web_config.py` - Test web app configuration
- `python debug_providers.py` - Debug provider availability issues

## Architecture Guidelines

### Current Package Structure
```
ocular/
├── core/                    # Core functionality
│   ├── enums.py            # Provider types and processing strategies
│   ├── interfaces.py       # Abstract base classes
│   ├── models.py           # Data models (OCRResult, etc.)
│   ├── exceptions.py       # Custom exception classes
│   ├── logging.py          # Logging system with Logfire integration
│   └── validation.py       # Input validation and security
├── providers/              # OCR provider implementations
│   ├── settings.py         # Unified settings with configuration management
│   ├── base.py            # Abstract base provider class
│   ├── factory.py         # Provider factory and manager
│   ├── mistral.py         # Mistral AI vision model provider
│   ├── google_vision.py   # Google Cloud Vision API provider
│   ├── aws_textract.py    # AWS Textract provider
│   ├── tesseract.py       # Local Tesseract OCR provider
│   ├── azure_document_intelligence.py  # Azure Document Intelligence
│   ├── olm.py             # OLM OCR provider (RunPod)
│   └── rolm.py            # RoLM OCR provider (RunPod)
├── services/               # Service layer
│   ├── cache_service.py   # Caching functionality
│   ├── document_service.py # Document processing services
│   ├── ocr_service.py     # OCR processing services
│   ├── processing_service.py # Processing coordination
│   └── validation_service.py # Request and data validation
├── config.py              # Configuration imports and legacy support
├── client.py              # Legacy client (MistralOCRClient)
├── parser.py              # Text parsing utilities
├── processor.py           # DocumentProcessor
├── pydantic_client.py     # PydanticOCRClient
├── unified_processor.py   # UnifiedDocumentProcessor (main interface)
└── __init__.py            # Package exports
web/                       # Web application
├── app.py                 # FastAPI web server
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Base template
│   └── upload.html        # File upload interface
└── static/                # Static assets (CSS, JS)
```

### Implemented Design Patterns
- ✅ **Factory Pattern**: `ProviderFactory` for creating OCR providers
- ✅ **Abstract Base Classes**: `BaseOCRProvider` for consistent interface
- ✅ **Strategy Pattern**: Multiple processing strategies (single, fallback, ensemble, best)
- ✅ **Unified Configuration**: Single `OcularSettings` class with integrated validation and management
- ✅ **Service Layer**: Dedicated services for validation, caching, and document processing
- ✅ **Dependency Injection**: Providers configured through factory
- ✅ **Error Handling**: Comprehensive exception hierarchy and logging
- ✅ **Async/Await**: Full async support for API calls and processing

### Available OCR Providers
1. **Mistral AI** (`mistral`) - Vision LLM with PDF-to-image conversion
2. **Google Cloud Vision** (`google_vision`) - Enterprise OCR API
3. **AWS Textract** (`aws_textract`) - Document analysis with form/table extraction
4. **Tesseract** (`tesseract`) - Local open-source OCR engine
5. **Azure Document Intelligence** (`azure_document_intelligence`) - Microsoft's document AI
6. **OLM OCR** (`olm_ocr`) - Custom vision model on RunPod
7. **RoLM OCR** (`rolm_ocr`) - Specialized OCR model on RunPod

## Configuration

### Environment Variables
Create a `.env` file in the project root with the following variables:

```bash
# Mistral AI Configuration (Primary provider)
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=pixtral-12b-2409

# Google Cloud Vision API
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_PROJECT_ID=your_project_id

# AWS Textract
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Azure Document Intelligence
AZURE_DOC_INTEL_ENDPOINT=https://your_resource.cognitiveservices.azure.com/
AZURE_DOC_INTEL_API_KEY=your_api_key

# Tesseract (local processing)
TESSERACT_LANGUAGE=eng
TESSERACT_DPI=300

# RunPod OCR Models (if using custom models)
OLM_API_ENDPOINT=https://api.runpod.ai/v2/your-olm-endpoint-id
OLM_API_KEY=your-runpod-api-key
ROLM_API_ENDPOINT=https://api.runpod.ai/v2/your-rolm-endpoint-id
ROLM_API_KEY=your-runpod-api-key

# Optional Configuration
MAX_FILE_SIZE_MB=10
TIMEOUT_SECONDS=30
MAX_RETRIES=3

# Logging
LOGFIRE_TOKEN=your_logfire_token  # Optional, for advanced logging
```

### Provider Priority
By default, providers are prioritized as follows:
1. Mistral AI (fastest, good quality)
2. Google Vision (enterprise-grade)  
3. AWS Textract (best for forms/tables)
4. Azure Document Intelligence (Microsoft ecosystem)
5. Tesseract (local, no API costs)
6. OLM/RoLM (custom models)

## Development Notes

### Security and Validation
- ✅ **Input validation** for all file uploads and API inputs
- ✅ **File type restrictions** with magic number verification
- ✅ **Size limits** and security scanning for uploaded files
- ✅ **API key validation** and secure storage in environment variables
- ✅ **Error handling** prevents information leakage

### Code Quality Standards
- ✅ **Type hints** throughout the codebase
- ✅ **Async/await patterns** for all API calls and I/O operations
- ✅ **Comprehensive logging** with structured data and error tracking
- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Unified configuration system** with integrated validation and management
- ✅ **Service-oriented design** with dedicated service classes
- ✅ **Simplified imports** with consolidated configuration management

### Known Limitations
- **PDF processing**: Currently processes only the first page for Mistral AI
- **Rate limiting**: Implement your own rate limiting for production use
- **Caching**: No built-in caching mechanism (can be added)
- **Batch processing**: Limited batch processing capabilities

### Troubleshooting
- **"No OCR providers available"**: Check `.env` file and API keys
- **PDF processing fails**: Ensure `pdf2image` is installed (`uv add pdf2image`)
- **Provider initialization errors**: Check network connectivity and API credentials
- **Web app not loading**: Verify all dependencies are installed and virtual environment is activated