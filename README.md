# Ocular

Ocular is a comprehensive Python package that provides OCR (Optical Character Recognition) capabilities using multiple providers including AI vision models, cloud APIs, and local processing engines. The package supports data extraction from PDF files and scanned images with a unified interface and robust error handling.

## Features

- 🚀 **Multiple OCR Providers**: Mistral AI, Google Vision, AWS Textract, Azure Document Intelligence, Tesseract, and custom RunPod models
- 🌐 **Web Interface**: FastAPI-based web application with file upload and processing
- ☁️ **Cloud Deployment**: Ready for Google Cloud Functions with automated CI/CD
- 🔧 **Flexible Processing**: Multiple strategies (single, fallback, ensemble, best)
- 🛡️ **Robust Error Handling**: Comprehensive exception hierarchy and validation
- ⚡ **Async Support**: Full async/await support for optimal performance
- 🎯 **Unified Configuration**: Single settings class for all providers

## Installation

```bash
# Install the package
uv pip install ocular-ocr

# Or for development
git clone https://github.com/your-repo/ocular.git
cd ocular
uv add -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from pathlib import Path
from ocular import UnifiedDocumentProcessor

async def basic_ocr():
    """Basic OCR example."""
    print("=== Basic OCR with Ocular ===")
    
    # Initialize processor (uses environment variables)
    processor = UnifiedDocumentProcessor()
    
    # Process a document
    file_path = Path("sample_document.pdf")  # or .jpg, .png
    
    if file_path.exists():
        result = await processor.process_document(file_path)
        print(f"Processed: {result.file_path.name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Provider used: {result.provider_used}")
        print(f"Extracted text:\n{result.get_full_text()}")
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    # Set MISTRAL_API_KEY in your environment
    asyncio.run(basic_ocr())
```

### Web Interface

Start the web application:

```bash
# From the project root
cd app
python ocular_app.py

# Or with uvicorn
uvicorn app.ocular_app:app --reload
```

Visit `http://localhost:8000` to use the web interface.

## Configuration

Create a `.env` file in the project root:

```bash
# Primary provider (required)
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional providers
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AZURE_DOC_INTEL_API_KEY=your_azure_key

# Provider settings
MISTRAL_MODEL=pixtral-12b-2409
TIMEOUT_SECONDS=30
MAX_RETRIES=3
```

## Deployment

### Google Cloud Functions

The project includes ready-to-deploy Google Cloud Functions configuration:

```bash
# Deploy to staging
gcloud functions deploy ocular-ocr-service-staging \
  --source . \
  --entry-point ocular_ocr \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars MISTRAL_API_KEY=your_key

# Or use the GitHub Actions workflow
git push origin main  # Auto-deploys to staging
```

### Local Development

```bash
# Install dependencies
uv add -r requirements.txt

# Run locally
python main.py

# Or with functions framework
functions-framework --target=ocular_ocr --debug
```

## Available Providers

1. **Mistral AI** - Vision LLM with PDF support (primary)
2. **Google Cloud Vision** - Enterprise OCR API
3. **AWS Textract** - Document analysis with forms/tables
4. **Azure Document Intelligence** - Microsoft's document AI
5. **Tesseract** - Local open-source OCR
6. **Custom RunPod Models** - OLM/RoLM OCR

## Processing Strategies

- **Single**: Use one specific provider
- **Fallback**: Try providers in order until success
- **Ensemble**: Use multiple providers and combine results
- **Best**: Select best result from multiple providers

## Examples

```python
from ocular import UnifiedDocumentProcessor, ProcessingStrategy, ProviderType

# Use specific provider
processor = UnifiedDocumentProcessor()
result = await processor.process_document(
    "document.pdf",
    strategy=ProcessingStrategy.SINGLE,
    providers=[ProviderType.MISTRAL]
)

# Fallback strategy
result = await processor.process_document(
    "document.pdf",
    strategy=ProcessingStrategy.FALLBACK,
    providers=[ProviderType.MISTRAL, ProviderType.TESSERACT]
)

# With custom prompt
result = await processor.process_document(
    "invoice.pdf",
    prompt="Extract the invoice number, date, and total amount"
)
```

## API Reference

### Core Classes

- `UnifiedDocumentProcessor`: Main processing interface
- `OCRResult`: Processing result with text and metadata
- `OcularSettings`: Configuration management
- `ProcessingStrategy`: Processing strategy enum
- `ProviderType`: Available provider types

### Web API Endpoints

- `GET /` - Web interface
- `POST /process` - Process files
- `GET /health` - Health check
- `GET /providers` - Available providers
- `GET /debug` - Debug information

## Contributing

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `python -m pytest`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: See `CLAUDE.md` for detailed development guide
- Issues: Report bugs via GitHub Issues
- Examples: Check the `examples/` directory for more use cases