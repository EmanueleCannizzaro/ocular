"""
Basic usage examples for Ocular OCR package.
"""

import asyncio
from pathlib import Path

from ocular import DocumentProcessor, OcularConfig
from ocular.utils import save_results_to_json, create_markdown_report


async def basic_image_ocr():
    """Basic OCR example for a single image."""
    print("=== Basic Image OCR ===")
    
    # Initialize processor (uses environment variables)
    processor = DocumentProcessor()
    
    # Process a single image
    image_path = Path("sample_image.jpg")  # Replace with actual path
    
    if image_path.exists():
        result = await processor.process_document(image_path)
        print(f"Processed: {result.file_path.name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Extracted text:\n{result.get_full_text()}")
    else:
        print(f"Sample image not found: {image_path}")


async def basic_pdf_ocr():
    """Basic OCR example for a PDF document."""
    print("\n=== Basic PDF OCR ===")
    
    processor = DocumentProcessor()
    
    pdf_path = Path("sample_document.pdf")  # Replace with actual path
    
    if pdf_path.exists():
        result = await processor.process_document(pdf_path)
        print(f"Processed: {result.file_path.name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Number of pages: {len(result.extracted_text)}")
        print(f"Full text preview (first 200 chars):\n{result.get_full_text()[:200]}...")
    else:
        print(f"Sample PDF not found: {pdf_path}")


async def batch_processing():
    """Example of batch processing multiple documents."""
    print("\n=== Batch Processing ===")
    
    processor = DocumentProcessor()
    
    # List of files to process
    file_paths = [
        "document1.pdf",
        "image1.jpg", 
        "image2.png"
    ]
    
    # Filter existing files
    existing_files = [Path(fp) for fp in file_paths if Path(fp).exists()]
    
    if existing_files:
        results = await processor.process_batch(existing_files, max_concurrent=2)
        
        print(f"Processed {len(results)} documents")
        for result in results:
            if isinstance(result, Exception):
                print(f"Error: {result}")
            else:
                print(f"- {result.file_path.name}: {len(result.get_full_text())} characters")
    else:
        print("No sample files found for batch processing")


async def structured_data_extraction():
    """Example of structured data extraction from an invoice or form."""
    print("\n=== Structured Data Extraction ===")
    
    processor = DocumentProcessor()
    
    # Define schema for extraction
    invoice_schema = {
        "invoice_number": "string",
        "date": "string", 
        "total_amount": "number",
        "vendor_name": "string",
        "items": [
            {
                "description": "string",
                "quantity": "number", 
                "price": "number"
            }
        ]
    }
    
    image_path = Path("sample_invoice.jpg")  # Replace with actual path
    
    if image_path.exists():
        result = await processor.extract_structured_data(
            image_path, 
            invoice_schema,
            prompt="Extract invoice data accurately, ensuring all numbers are parsed correctly."
        )
        
        print(f"Processed: {result.file_path.name}")
        print("Structured data:")
        import json
        print(json.dumps(result.structured_data, indent=2))
    else:
        print(f"Sample invoice not found: {image_path}")


async def directory_processing():
    """Example of processing all documents in a directory."""
    print("\n=== Directory Processing ===")
    
    processor = DocumentProcessor()
    
    docs_dir = Path("documents")  # Replace with actual directory
    
    if docs_dir.exists() and docs_dir.is_dir():
        results = await processor.process_directory(
            docs_dir, 
            recursive=True,
            prompt="Extract text maintaining original formatting",
            max_concurrent=3
        )
        
        print(f"Processed {len(results)} documents from {docs_dir}")
        
        # Save results
        save_results_to_json(results, "processing_results.json")
        create_markdown_report(results, "processing_report.md")
        
        print("Results saved to:")
        print("- processing_results.json")
        print("- processing_report.md")
    else:
        print(f"Documents directory not found: {docs_dir}")


async def custom_configuration():
    """Example using custom configuration."""
    print("\n=== Custom Configuration ===")
    
    # Create custom config
    config = OcularConfig(
        mistral_api_key="your-api-key-here",
        mistral_model="pixtral-12b-2409",
        max_file_size_mb=20,
        timeout_seconds=60,
        max_retries=5
    )
    
    processor = DocumentProcessor(config)
    
    print(f"Using model: {config.mistral_model}")
    print(f"Max file size: {config.max_file_size_mb}MB")
    print(f"Timeout: {config.timeout_seconds}s")


async def main():
    """Run all examples."""
    print("Ocular OCR Examples")
    print("==================")
    
    try:
        await basic_image_ocr()
        await basic_pdf_ocr()
        await batch_processing()
        await structured_data_extraction()
        await directory_processing()
        await custom_configuration()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure to:")
        print("1. Set MISTRAL_API_KEY environment variable")
        print("2. Place sample files in the examples directory")
        print("3. Install required dependencies: pip install -e .")


if __name__ == "__main__":
    asyncio.run(main())