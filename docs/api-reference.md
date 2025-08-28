# API Reference

Complete API reference for the Ocular OCR system.

## Core Classes

### OCRService

Main service for OCR processing operations.

```python
from ocular.services.ocr_service import OCRService
from ocular.settings import OcularSettings

# Initialize with default settings
service = OCRService.from_env()

# Initialize with custom settings
settings = OcularSettings(...)
service = OCRService(settings)
```

#### Methods

##### `process_document(request: ProcessingRequest) -> ProcessingResult`

Process a single document with OCR.

**Parameters:**
- `request`: Processing configuration and file details

**Returns:**
- `ProcessingResult`: Complete processing results with extracted text

**Example:**
```python
from ocular.core.models import ProcessingRequest
from pathlib import Path

request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.FALLBACK,
    providers=[ProviderType.MISTRAL]
)

result = await service.process_document(request)
text = result.get_text()
```

##### `process_batch(requests: List[ProcessingRequest]) -> List[ProcessingResult]`

Process multiple documents with concurrency control.

**Parameters:**
- `requests`: List of processing requests

**Returns:**
- `List[ProcessingResult]`: Results for each processed document

---

### ProcessingRequest

Configuration for document processing.

```python
from ocular.core.models import ProcessingRequest
from ocular.core.enums import ProcessingStrategy, ProviderType
from pathlib import Path

request = ProcessingRequest(
    file_path=Path("document.pdf"),
    strategy=ProcessingStrategy.FALLBACK,
    providers=[ProviderType.MISTRAL, ProviderType.OLM_OCR],
    prompt="Extract all text maintaining formatting",
    options={"enhance_image": True}
)
```

#### Fields

- `file_path: Path` - Path to document file
- `strategy: ProcessingStrategy` - Processing strategy to use  
- `providers: List[ProviderType]` - OCR providers to use
- `prompt: Optional[str]` - Custom processing prompt
- `options: Dict[str, Any]` - Additional processing options

---

### ProcessingResult

Results from document processing.

#### Fields

- `file_path: Path` - Original file path
- `document_type: DocumentType` - Type of document (IMAGE/PDF)
- `strategy: ProcessingStrategy` - Strategy used
- `status: ProcessingStatus` - Processing status
- `provider_results: Dict[str, OCRResult]` - Results from each provider
- `primary_result: Optional[OCRResult]` - Best result
- `total_processing_time: Optional[float]` - Total processing time

#### Methods

##### `get_text() -> str`

Get the primary extracted text.

##### `get_best_result() -> Optional[OCRResult]`

Get the result with highest confidence score.

##### `add_provider_result(provider: str, result: OCRResult)`

Add result from a specific provider.

---

### OCRResult

Result from OCR processing by a single provider.

```python
from ocular.core.models import OCRResult

result = OCRResult(
    text="Extracted text content",
    confidence=0.95,
    language="en",
    provider="Mistral AI",
    processing_time=1.5,
    metadata={"model": "pixtral-12b-2409"}
)
```

#### Fields

- `text: str` - Extracted text content
- `confidence: Optional[float]` - Confidence score (0.0-1.0)
- `language: Optional[str]` - Detected language
- `provider: str` - Provider that generated this result
- `processing_time: Optional[float]` - Processing time in seconds
- `metadata: Dict[str, Any]` - Additional metadata

---

## Enumerations

### ProcessingStrategy

Available processing strategies.

- `SINGLE` - Use single provider
- `FALLBACK` - Try providers until success
- `ENSEMBLE` - Use multiple providers concurrently
- `BEST` - Select best result by confidence

### ProviderType

Available OCR providers.

- `MISTRAL` - Mistral AI vision models
- `OLM_OCR` - Local Optical Language Models
- `ROLM_OCR` - Robust Optical Language Models

### DocumentType

Supported document types.

- `IMAGE` - Image files (JPG, PNG, etc.)
- `PDF` - PDF documents

### ProcessingStatus

Processing status values.

- `PENDING` - Waiting to be processed
- `IN_PROGRESS` - Currently processing
- `COMPLETED` - Successfully completed
- `FAILED` - Processing failed
- `CANCELLED` - Processing cancelled

---

## Configuration

### OcularSettings

Main configuration class.

```python
from ocular.settings import OcularSettings

# Load from environment
settings = OcularSettings.from_env_file(".env")

# Create programmatically
settings = OcularSettings(
    environment="production",
    providers=ProviderSettings(
        mistral_api_key="your-key",
        enabled_providers=["mistral", "olm_ocr"]
    )
)
```

#### Provider Configuration

Access provider-specific configuration:

```python
# Get Mistral configuration
mistral_config = settings.get_provider_config(ProviderType.MISTRAL)

# Check if provider is enabled
is_enabled = settings.is_provider_enabled(ProviderType.MISTRAL)

# Get provider priority
priority = settings.get_provider_priority(ProviderType.MISTRAL)
```

---

## Providers

### BaseOCRProvider

Abstract base class for OCR providers.

#### Methods

##### `initialize() -> None`

Initialize the provider (lazy loading).

##### `is_available() -> bool`

Check if provider is available and configured.

##### `extract_text(file_path: Path, prompt: Optional[str] = None) -> OCRResult`

Extract text from document.

##### `extract_structured_data(file_path: Path, schema: Dict[str, Any]) -> Dict[str, Any]`

Extract structured data based on schema.

##### `cleanup() -> None`

Clean up provider resources.

### MistralProvider

Mistral AI OCR provider.

```python
from ocular.providers.mistral import MistralProvider

config = {
    "api_key": "your-mistral-key",
    "model": "pixtral-12b-2409",
    "timeout": 30
}

provider = MistralProvider(config)
await provider.initialize()

result = await provider.extract_text(Path("document.pdf"))
```

### OLMProvider

Local Optical Language Model provider.

```python
from ocular.providers.olm import OLMProvider

config = {
    "model_name": "microsoft/trocr-base-printed",
    "device": "cuda"  # or "cpu"
}

provider = OLMProvider(config)
await provider.initialize()
```

### RoLMProvider

Robust Optical Language Model provider with enhanced features.

```python
from ocular.providers.rolm import RoLMProvider

config = {
    "model_name": "microsoft/trocr-large-printed", 
    "device": "cuda",
    "use_preprocessing": True,
    "ensemble_size": 3
}

provider = RoLMProvider(config)
```

---

## Services

### DocumentService

Service for document handling and validation.

```python
from ocular.services.document_service import DocumentService

doc_service = DocumentService(settings)

# Detect document type
doc_type = doc_service.detect_document_type(Path("file.pdf"))

# Validate file
is_valid = doc_service.validate_file(Path("file.pdf"))

# Get file information
info = doc_service.get_file_info(Path("file.pdf"))
```

### ValidationService

Service for validating requests and data.

```python
from ocular.services.validation_service import ValidationService

validator = ValidationService(settings)

# Validate processing request
await validator.validate_processing_request(request)

# Validate OCR result
validator.validate_ocr_result(result)

# Sanitize prompt
clean_prompt = validator.sanitize_prompt(user_prompt)
```

### CacheService

Service for caching OCR results.

```python
from ocular.services.cache_service import CacheService

cache = CacheService(settings)

# Generate cache key
key = cache.generate_cache_key(
    file_path=Path("doc.pdf"),
    strategy="fallback",
    providers=["mistral"]
)

# Cache result
await cache.cache_result(key, ocr_result, ttl=3600)

# Get cached result
cached = await cache.get_cached_result(key)
```

---

## Exceptions

### OcularError

Base exception for Ocular package.

```python
from ocular.core.exceptions import OcularError

try:
    result = await service.process_document(request)
except OcularError as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.error_code}")
    print(f"Details: {e.details}")
```

### OCRError

Raised when OCR processing fails.

### ConfigurationError

Raised when configuration is invalid.

### ValidationError

Raised when input validation fails.

### APIError

Raised when API calls fail.

### DocumentProcessingError

Raised when document processing fails.

---

## Structured Data Models

### InvoiceData

Model for invoice data extraction.

```python
from ocular.core.models import InvoiceData

# Extract invoice data
invoice = await service.extract_structured_data(
    Path("invoice.pdf"),
    InvoiceData,
    prompt="Extract invoice information"
)

print(f"Invoice: {invoice.invoice_number}")
print(f"Date: {invoice.date}")
print(f"Total: ${invoice.total_amount}")
print(f"Vendor: {invoice.vendor_name}")

for item in invoice.line_items:
    print(f"- {item['description']}: {item['quantity']} x ${item['price']}")
```

#### Fields

- `invoice_number: Optional[str]`
- `date: Optional[str]`
- `due_date: Optional[str]`
- `total_amount: Optional[float]`
- `tax_amount: Optional[float]`
- `vendor_name: Optional[str]`
- `vendor_address: Optional[str]`
- `customer_name: Optional[str]`
- `customer_address: Optional[str]`
- `line_items: List[Dict[str, Any]]`

---

## Utilities

### Helper Functions

```python
from ocular.utils import (
    save_results_to_json,
    save_text_to_file,
    create_markdown_report,
    calculate_processing_stats
)

# Save results to JSON
save_results_to_json(results, "output.json")

# Save text to file
save_text_to_file(results, "extracted_text.txt")

# Create markdown report
create_markdown_report(results, "report.md")

# Calculate statistics
stats = calculate_processing_stats(results)
print(f"Processed {stats['total_documents']} documents")
print(f"Average time: {stats['average_processing_time']:.2f}s")
```

---

## Error Handling

### Best Practices

```python
import logging
from ocular.core.exceptions import OcularError, OCRError, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_with_error_handling():
    try:
        result = await service.process_document(request)
        return result
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e.message}")
        # Handle validation errors (bad input)
        return None
        
    except OCRError as e:
        logger.error(f"OCR processing failed: {e.message}")
        # Handle OCR errors (provider issues)
        return None
        
    except OcularError as e:
        logger.error(f"Ocular error: {e.message}")
        # Handle other Ocular errors
        return None
        
    except Exception as e:
        logger.exception("Unexpected error")
        # Handle unexpected errors
        return None
```

---

## Performance Tips

### Caching

Enable caching for repeated processing:

```python
settings = OcularSettings(
    processing=ProcessingSettings(
        enable_caching=True,
        cache_ttl_seconds=3600  # 1 hour
    )
)
```

### Concurrency

Use batch processing for multiple documents:

```python
# Process up to 5 documents concurrently
results = await service.process_batch(
    requests,
    max_concurrent=5
)
```

### Provider Selection

Choose appropriate providers for your use case:

- **Mistral**: Best quality, requires API key
- **OLM**: Good balance, runs locally
- **RoLM**: Most robust, slower processing

### Memory Management

Clean up resources when done:

```python
async with OCRService.from_env() as service:
    result = await service.process_document(request)
    # Automatic cleanup
```