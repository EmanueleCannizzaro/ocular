# Ocular

Ocular is a python package that use best in class OCR LLM for data extraction from PDF files and scanned images.

## Installation

```bash
uv pip install ocular-ocr
```

## Usage

Here is a basic example of how to use Ocular to extract text from an image:

```python
import asyncio
from pathlib import Path

from ocular import DocumentProcessor


async def basic_image_ocr():
    """Basic OCR example for a single image."""
    print("=== Basic Image OCR ===")
    
    # Initialize processor (uses environment variables for configuration)
    processor = DocumentProcessor()
    
    # Process a single image
    image_path = Path("sample_image.jpg")  # Replace with your image path
    
    if image_path.exists():
        result = await processor.process_document(image_path)
        print(f"Processed: {result.file_path.name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Extracted text:\n{result.get_full_text()}")
    else:
        print(f"Sample image not found: {image_path}")

if __name__ == "__main__":
    # Make sure to set the MISTRAL_API_KEY environment variable
    asyncio.run(basic_image_ocr())
```

For more detailed examples, including PDF processing, batch processing, structured data extraction, and custom configurations, please see the `examples` directory.