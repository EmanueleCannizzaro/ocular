# Ocular OCR Documentation

Comprehensive documentation for the Ocular OCR system.

## Quick Links

- [User Guide](user-guide.md) - Getting started and basic usage
- [API Reference](api-reference.md) - Complete API documentation  
- [Developer Guide](developer-guide.md) - Contributing and extending Ocular
- [Configuration Guide](configuration.md) - Configuration options and environment setup
- [Examples](examples/) - Code examples and tutorials

## What is Ocular?

Ocular is a Python package that provides best-in-class OCR (Optical Character Recognition) capabilities using multiple AI providers including:

- **Mistral AI** - Advanced vision models for high-quality text extraction
- **OLM OCR** - Local Optical Language Models for offline processing
- **RoLM OCR** - Robust Optical Language Models with enhanced accuracy

## Key Features

### Multiple OCR Providers
- Support for cloud-based and local OCR models
- Automatic fallback between providers
- Ensemble processing for improved accuracy

### Flexible Processing Strategies
- **Single**: Use one provider
- **Fallback**: Try providers until success  
- **Ensemble**: Use multiple providers concurrently
- **Best**: Select best result by confidence score

### Document Support
- **Images**: JPG, PNG, BMP, TIFF, WEBP
- **PDFs**: Multi-page document processing
- **Batch Processing**: Handle multiple documents efficiently

### Advanced Features
- Structured data extraction (invoices, forms, etc.)
- Caching for improved performance
- Web interface for easy document upload
- RESTful API for integration
- Comprehensive error handling and logging

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   REST API      │    │   CLI Tools     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Service Layer          │
                    │  ┌─────────────────────┐  │
                    │  │  OCR Service       │  │
                    │  │  Document Service  │  │
                    │  │  Validation Svc    │  │
                    │  │  Cache Service     │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Provider Layer         │
                    │  ┌─────────────────────┐  │
                    │  │  Mistral Provider  │  │
                    │  │  OLM Provider     │  │
                    │  │  RoLM Provider    │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ocular.git
cd ocular
pip install -e .
```

### Basic Usage

```python
from ocular import OCRService
from ocular.core.models import ProcessingRequest
from ocular.core.enums import ProcessingStrategy, ProviderType
from pathlib import Path

# Initialize OCR service
service = OCRService.from_env()

# Create processing request
request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.FALLBACK,
    providers=[ProviderType.MISTRAL, ProviderType.OLM_OCR]
)

# Process document
result = await service.process_document(request)
print(f"Extracted text: {result.get_text()}")
```

### Web Interface

```bash
# Start web server
python run_server.py

# Open browser to http://localhost:8000
```

## Configuration

Create a `.env` file with your API keys:

```env
# Required for Mistral AI provider
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional configuration
MISTRAL_MODEL=pixtral-12b-2409
MAX_FILE_SIZE_MB=10
ENABLED_PROVIDERS=mistral,olm_ocr
```

## Examples

### Text Extraction

```python
import asyncio
from pathlib import Path
from ocular.services.ocr_service import OCRService
from ocular.core.models import ProcessingRequest

async def extract_text():
    service = OCRService.from_env()
    
    request = ProcessingRequest(
        file_path=Path("invoice.pdf"),
        prompt="Extract all text maintaining formatting"
    )
    
    result = await service.process_document(request)
    return result.get_text()

text = asyncio.run(extract_text())
print(text)
```

### Structured Data Extraction

```python
from ocular.core.models import InvoiceData

async def extract_invoice():
    service = OCRService.from_env()
    
    # Extract structured invoice data
    invoice = await service.extract_structured_data(
        Path("invoice.pdf"),
        InvoiceData,
        prompt="Extract invoice details including line items"
    )
    
    print(f"Invoice #{invoice.invoice_number}")
    print(f"Total: ${invoice.total_amount}")
    return invoice
```

### Batch Processing

```python
async def process_batch():
    service = OCRService.from_env()
    
    file_paths = [
        Path("doc1.pdf"),
        Path("doc2.jpg"), 
        Path("doc3.png")
    ]
    
    requests = [
        ProcessingRequest(file_path=path)
        for path in file_paths
    ]
    
    results = await service.process_batch(requests)
    
    for result in results:
        print(f"{result.file_path}: {len(result.get_text())} characters")
```

## Support and Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/ocular/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ocular/discussions)
- **Documentation**: [Full Documentation](https://yourusername.github.io/ocular/)

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Contributing

We welcome contributions! See our [Developer Guide](developer-guide.md) for details on:

- Setting up development environment
- Code style and testing requirements  
- Submitting pull requests
- Adding new OCR providers

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history and updates.