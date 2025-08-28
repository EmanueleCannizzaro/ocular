# User Guide

Complete guide for using Ocular OCR to extract text and data from documents.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Web Interface](#web-interface)
- [Command Line Usage](#command-line-usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.9 or higher
- For local OCR models: PyTorch (optional, install with `pip install torch`)
- For Mistral AI: API key from Mistral AI

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ocular.git
cd ocular

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"

# For local OCR models (optional)
pip install -e ".[torch]"
```

### Verify Installation

```bash
python -c "import ocular; print('Ocular installed successfully')"
```

## Quick Start

### 1. Set up Configuration

Create a `.env` file in your project directory:

```env
# Required for Mistral AI
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional settings
MISTRAL_MODEL=pixtral-12b-2409
MAX_FILE_SIZE_MB=10
ENABLED_PROVIDERS=mistral
```

### 2. Basic Text Extraction

```python
import asyncio
from pathlib import Path
from ocular.services.ocr_service import OCRService
from ocular.core.models import ProcessingRequest

async def extract_text():
    # Initialize service
    service = OCRService.from_env()
    
    # Create request
    request = ProcessingRequest(
        file_path=Path("your_document.pdf")
    )
    
    # Process document
    result = await service.process_document(request)
    
    # Get extracted text
    text = result.get_text()
    print(f"Extracted text:\n{text}")
    
    return text

# Run the extraction
text = asyncio.run(extract_text())
```

### 3. Start Web Interface

```bash
python run_server.py
```

Visit `http://localhost:8000` to use the web interface.

## Configuration

### Environment Variables

Ocular uses environment variables for configuration. Create a `.env` file or set them in your system:

#### Mistral AI Provider
```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=pixtral-12b-2409
MISTRAL_TIMEOUT=30
MISTRAL_MAX_RETRIES=3
```

#### File Settings
```env
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=.pdf,.jpg,.jpeg,.png,.bmp,.tiff,.webp
TEMP_DIR=/path/to/temp
UPLOAD_DIR=/path/to/uploads
```

#### Processing Settings
```env
DEFAULT_STRATEGY=fallback
MAX_CONCURRENT_REQUESTS=5
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
```

#### Local Model Settings (for OLM/RoLM)
```env
DEVICE=auto
OLM_MODEL_NAME=microsoft/trocr-base-printed
ROLM_MODEL_NAME=microsoft/trocr-large-printed
```

### Programmatic Configuration

```python
from ocular.settings import OcularSettings, ProviderSettings

settings = OcularSettings(
    environment="production",
    providers=ProviderSettings(
        mistral_api_key="your-key",
        enabled_providers=["mistral", "olm_ocr"],
        mistral_model="pixtral-12b-2409"
    )
)

service = OCRService(settings)
```

## Basic Usage

### Processing Single Documents

```python
from ocular.services.ocr_service import OCRService
from ocular.core.models import ProcessingRequest
from ocular.core.enums import ProcessingStrategy, ProviderType
from pathlib import Path

async def process_document():
    service = OCRService.from_env()
    
    # Simple processing
    request = ProcessingRequest(file_path=Path("document.pdf"))
    result = await service.process_document(request)
    
    print(f"Extracted text: {result.get_text()}")
    print(f"Processing time: {result.total_processing_time:.2f}s")
    print(f"Provider used: {result.primary_result.provider}")
```

### Processing with Custom Options

```python
async def process_with_options():
    service = OCRService.from_env()
    
    request = ProcessingRequest(
        file_path=Path("complex_document.pdf"),
        strategy=ProcessingStrategy.ENSEMBLE,  # Use multiple providers
        providers=[ProviderType.MISTRAL, ProviderType.OLM_OCR],
        prompt="Extract text preserving table structure and formatting",
        options={
            "enhance_image": True,
            "use_preprocessing": True
        }
    )
    
    result = await service.process_document(request)
    
    # Get results from all providers
    for provider, ocr_result in result.provider_results.items():
        print(f"\n{provider} (confidence: {ocr_result.confidence}):")
        print(ocr_result.text[:200] + "...")
```

### Batch Processing

```python
async def process_multiple_documents():
    service = OCRService.from_env()
    
    # List of documents to process
    documents = [
        Path("invoice1.pdf"),
        Path("receipt.jpg"),
        Path("form.png")
    ]
    
    # Create processing requests
    requests = [
        ProcessingRequest(
            file_path=doc,
            strategy=ProcessingStrategy.FALLBACK,
            prompt="Extract all text and maintain formatting"
        )
        for doc in documents
    ]
    
    # Process in parallel
    results = await service.process_batch(requests)
    
    # Display results
    for result in results:
        if isinstance(result, Exception):
            print(f"Error processing: {result}")
        else:
            print(f"\n{result.file_path.name}:")
            print(f"  Characters: {len(result.get_text())}")
            print(f"  Time: {result.total_processing_time:.2f}s")
            print(f"  Provider: {result.primary_result.provider}")
```

## Advanced Features

### Structured Data Extraction

Extract specific data types like invoices, forms, or tables:

```python
from ocular.core.models import InvoiceData

async def extract_invoice():
    service = OCRService.from_env()
    
    # Define the structure you want to extract
    invoice = await service.extract_structured_data(
        Path("invoice.pdf"),
        InvoiceData,
        prompt="Extract invoice details including all line items"
    )
    
    print(f"Invoice Number: {invoice.invoice_number}")
    print(f"Date: {invoice.date}")
    print(f"Vendor: {invoice.vendor_name}")
    print(f"Total: ${invoice.total_amount}")
    
    print("\nLine Items:")
    for item in invoice.line_items:
        print(f"  - {item['description']}: {item['quantity']} x ${item['price']}")
    
    return invoice
```

### Custom Data Structures

Define your own data structures for extraction:

```python
from pydantic import BaseModel
from typing import List, Optional

class ContractData(BaseModel):
    contract_number: Optional[str]
    parties: List[str] = []
    effective_date: Optional[str]
    expiration_date: Optional[str]
    key_terms: List[str] = []

async def extract_contract():
    service = OCRService.from_env()
    
    contract = await service.extract_structured_data(
        Path("contract.pdf"),
        ContractData,
        prompt="Extract contract details focusing on parties, dates, and key terms"
    )
    
    print(f"Contract: {contract.contract_number}")
    print(f"Parties: {', '.join(contract.parties)}")
    print(f"Term: {contract.effective_date} to {contract.expiration_date}")
    
    return contract
```

### Processing Strategies

Choose the right strategy for your needs:

#### Single Provider
```python
# Use only one provider (fastest)
request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.SINGLE,
    providers=[ProviderType.MISTRAL]
)
```

#### Fallback Processing
```python
# Try providers until one succeeds (reliable)
request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.FALLBACK,
    providers=[ProviderType.MISTRAL, ProviderType.OLM_OCR, ProviderType.ROLM_OCR]
)
```

#### Ensemble Processing
```python
# Use multiple providers and compare results (highest quality)
request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.ENSEMBLE,
    providers=[ProviderType.MISTRAL, ProviderType.ROLM_OCR]
)
```

#### Best Result Selection
```python
# Use ensemble then select best by confidence
request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.BEST,
    providers=[ProviderType.MISTRAL, ProviderType.OLM_OCR]
)
```

### Caching Results

Enable caching to avoid reprocessing identical documents:

```python
from ocular.settings import OcularSettings, ProcessingSettings

settings = OcularSettings(
    processing=ProcessingSettings(
        enable_caching=True,
        cache_ttl_seconds=3600  # Cache for 1 hour
    )
)

service = OCRService(settings)

# First processing - will be cached
result1 = await service.process_document(request)

# Second processing - will use cache (much faster)
result2 = await service.process_document(request)
```

## Web Interface

### Starting the Server

```bash
# Development server
python run_server.py

# Production server
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### Using the Interface

1. **Upload Files**: Drag and drop or click to select files
2. **Choose Strategy**: Select processing strategy
3. **Select Providers**: Choose which OCR providers to use
4. **Add Prompt**: Optional custom processing instructions
5. **Process**: Click to start processing
6. **View Results**: See extracted text and metadata

### API Endpoints

The web interface exposes REST API endpoints:

- `GET /` - Upload form
- `POST /process` - Process files
- `GET /health` - Health check
- `GET /providers` - List available providers

### API Usage

```python
import httpx
import asyncio

async def api_example():
    async with httpx.AsyncClient() as client:
        # Upload and process file
        with open("document.pdf", "rb") as f:
            files = {"files": ("document.pdf", f, "application/pdf")}
            data = {
                "strategy": "fallback",
                "providers": "mistral,olm_ocr",
                "prompt": "Extract all text"
            }
            
            response = await client.post(
                "http://localhost:8000/process",
                files=files,
                data=data
            )
            
        result = response.json()
        print(f"Success: {result['success']}")
        
        for doc_result in result['results']:
            print(f"File: {doc_result['original_filename']}")
            print(f"Text: {doc_result['primary_text'][:200]}...")

asyncio.run(api_example())
```

## Command Line Usage

### Basic CLI Script

Create a script for command-line processing:

```python
#!/usr/bin/env python3
"""
OCR command line tool.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from ocular.services.ocr_service import OCRService
from ocular.core.models import ProcessingRequest
from ocular.core.enums import ProcessingStrategy, ProviderType

async def main():
    parser = argparse.ArgumentParser(description="Ocular OCR CLI")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--strategy", default="fallback", 
                       choices=["single", "fallback", "ensemble", "best"])
    parser.add_argument("--providers", default="mistral",
                       help="Comma-separated list of providers")
    parser.add_argument("--prompt", help="Custom processing prompt")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize service
    service = OCRService.from_env()
    
    # Parse providers
    provider_names = args.providers.split(",")
    providers = []
    for name in provider_names:
        if name.strip() == "mistral":
            providers.append(ProviderType.MISTRAL)
        elif name.strip() == "olm_ocr":
            providers.append(ProviderType.OLM_OCR)
        elif name.strip() == "rolm_ocr":
            providers.append(ProviderType.ROLM_OCR)
    
    # Process files
    results = []
    for file_path in args.files:
        request = ProcessingRequest(
            file_path=Path(file_path),
            strategy=ProcessingStrategy(args.strategy),
            providers=providers,
            prompt=args.prompt
        )
        
        try:
            result = await service.process_document(request)
            results.append(result)
            
            print(f"✓ Processed {file_path}")
            print(f"  Provider: {result.primary_result.provider}")
            print(f"  Time: {result.total_processing_time:.2f}s")
            print(f"  Characters: {len(result.get_text())}")
            
        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")
    
    # Save results if requested
    if args.output:
        from ocular.utils import save_results_to_json
        save_results_to_json(results, args.output)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

Save as `ocr_cli.py` and use:

```bash
python ocr_cli.py document.pdf --strategy ensemble --output results.json
python ocr_cli.py *.jpg --providers mistral,olm_ocr --prompt "Extract invoice data"
```

## Best Practices

### File Preparation

1. **Image Quality**: Use high-resolution images (300+ DPI) for better results
2. **File Format**: 
   - PDFs: Native PDFs work better than scanned PDFs
   - Images: PNG or high-quality JPEG recommended
3. **Document Orientation**: Ensure documents are properly oriented
4. **Contrast**: High contrast between text and background improves accuracy

### Provider Selection

1. **Mistral AI**: Best for complex documents, requires API key
2. **OLM OCR**: Good balance of speed and accuracy, runs locally  
3. **RoLM OCR**: Most robust for difficult documents, slower processing

### Performance Optimization

1. **Use Caching**: Enable caching for repeated processing
2. **Batch Processing**: Process multiple documents together
3. **Appropriate Concurrency**: Don't exceed system capabilities
4. **Provider Prioritization**: Order providers by speed/accuracy needs

### Error Handling

```python
from ocular.core.exceptions import OcularError, OCRError, ValidationError

async def robust_processing(file_path):
    service = OCRService.from_env()
    
    try:
        request = ProcessingRequest(
            file_path=file_path,
            strategy=ProcessingStrategy.FALLBACK  # Try multiple providers
        )
        
        result = await service.process_document(request)
        return result.get_text()
        
    except ValidationError as e:
        print(f"Invalid input: {e.message}")
        return None
        
    except OCRError as e:
        print(f"OCR failed: {e.message}")
        return None
        
    except OcularError as e:
        print(f"Processing error: {e.message}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Memory Management

```python
# Use context managers for automatic cleanup
async with OCRService.from_env() as service:
    result = await service.process_document(request)
    # Service is automatically cleaned up

# Or manual cleanup
service = OCRService.from_env()
try:
    result = await service.process_document(request)
finally:
    await service.cleanup()
```

## Troubleshooting

### Common Issues

#### 1. Missing API Key
```
ConfigurationError: MISTRAL_API_KEY environment variable is required
```

**Solution**: Set your Mistral AI API key in `.env` file or environment variables.

#### 2. Large File Error
```
ValidationError: File size (15.2MB) exceeds limit of 10MB
```

**Solution**: Increase file size limit or compress the file:
```env
MAX_FILE_SIZE_MB=20
```

#### 3. Unsupported File Format
```
ValidationError: File extension .doc not allowed
```

**Solution**: Convert to supported format (PDF, JPG, PNG, etc.) or add to allowed extensions.

#### 4. Provider Not Available
```
ProviderError: Provider mistral not available
```

**Solution**: Check API key, network connection, or try different provider.

#### 5. Memory Issues with Local Models
```
RuntimeError: CUDA out of memory
```

**Solution**: 
- Use CPU instead: `DEVICE=cpu`
- Reduce batch size
- Use smaller models

### Debug Mode

Enable debug logging for more information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
OCULAR_LOG_LEVEL=DEBUG
```

### Health Check

Verify system health:

```python
async def health_check():
    service = OCRService.from_env()
    health = await service.health_check()
    
    print(f"Service status: {health['status']}")
    
    for provider, info in health['providers'].items():
        status = "✓" if info['available'] else "✗"
        print(f"{status} {provider}: {info.get('name', 'Unknown')}")

asyncio.run(health_check())
```

### Getting Help

1. **Check Logs**: Enable debug logging to see detailed error messages
2. **Verify Configuration**: Ensure all required settings are correct
3. **Test with Simple Files**: Start with small, clear documents
4. **Check Provider Status**: Verify API keys and model availability
5. **Community Support**: Check GitHub issues and discussions

### Performance Monitoring

```python
import time
from ocular.utils import calculate_processing_stats

async def monitor_performance():
    service = OCRService.from_env()
    
    start_time = time.time()
    results = await service.process_batch(requests)
    total_time = time.time() - start_time
    
    stats = calculate_processing_stats(results)
    
    print(f"Batch completed in {total_time:.2f}s")
    print(f"Documents: {stats['total_documents']}")
    print(f"Average time per document: {stats['average_processing_time']:.2f}s")
    print(f"Total characters extracted: {stats['total_text_characters']:,}")
```