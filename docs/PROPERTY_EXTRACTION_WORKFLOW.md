# Property Extraction Workflow

Complete workflow for extracting structured Italian property auction data from PDF appraisal documents using GPU-based OCR and AI-powered extraction.

## Overview

This workflow converts unstructured PDF property appraisal documents (perizie tecniche) into structured JSON data following the `AuctionPropertyProfile` schema.

### Workflow Steps

```
PDF Document → OCR Extraction → AI Structuring → Validated JSON
     ↓              ↓                  ↓              ↓
 Appraisal      GPU Server        Pydantic AI    Structured
   PDF          (OLM/RoLM)         (Claude)         Data
```

## Quick Start

### 1. Python API Usage

```python
from ocular.services.property_extraction_service import extract_property_from_pdf

# Simple extraction
property_data = await extract_property_from_pdf(
    pdf_path="appraisal.pdf",
    provider="olm_ocr"
)

print(f"Sale ID: {property_data.sale_id}")
print(f"Base Price: €{property_data.auction_details.base_price:,.2f}")
```

### 2. REST API Usage

```bash
# Extract property data from PDF
curl -X POST "http://localhost:8004/api/property-extraction/extract" \
  -F "file=@appraisal.pdf" \
  -F "provider=olm_ocr"
```

### 3. Command Line Usage

```bash
# Basic extraction
python _examples/property_extraction_workflow.py appraisal.pdf

# With specific provider
python _examples/property_extraction_workflow.py appraisal.pdf \
  --provider olm_ocr \
  --output extracted_data.json

# With portal data for enhanced extraction
python _examples/property_extraction_workflow.py appraisal.pdf \
  --portal-json portal_data.json \
  --output combined_data.json

# Batch processing
python _examples/property_extraction_workflow.py \
  --batch file1.pdf file2.pdf file3.pdf \
  --provider rolm_ocr
```

## Available OCR Providers

| Provider | Description | Best For | Quality |
|---|---|---|---|
| **olm_ocr** | OLMocr 7B on RunPod | Complex Italian documents | ⭐⭐⭐⭐⭐ |
| **rolm_ocr** | RoLM OCR on RunPod | Layout-preserving OCR | ⭐⭐⭐⭐⭐ |
| **mistral** | Mistral Vision API | Quick processing | ⭐⭐⭐⭐ |
| **google_vision** | Google Cloud Vision | Enterprise-grade | ⭐⭐⭐⭐ |
| **aws_textract** | AWS Textract | Forms and tables | ⭐⭐⭐⭐ |

**Recommendation:** Use `olm_ocr` or `rolm_ocr` for best results with Italian property documents.

## Data Structure

The workflow extracts data into the `AuctionPropertyProfile` schema:

```json
{
  "sale_id": 12345,
  "lot_number": "LOTTO 001",
  "appraisal_summary": {
    "court": "TRIBUNALE DI SPOLETO",
    "procedure_number": "R.G. 102/2022",
    "judge": {...},
    "expert_appraiser": {...}
  },
  "auction_details": {
    "sale_type": "Senza Incanto",
    "base_price": 180000.0,
    "minimum_bid": 135000.0,
    "auction_date": "2024-03-15 10:00",
    ...
  },
  "risk_assessment": {
    "compliance_issues": [...],
    "liens_and_encumbrances": [...],
    "costs_to_remedy": [...]
  },
  "valuation_details": {
    "total_market_value": 200000.0,
    "adjustment_percentage": -10.0,
    "final_auction_base_price": 180000.0
  },
  "assets": [
    {
      "asset_type": "Abitazione",
      "description": "Appartamento al secondo piano",
      "address": "Via Roma 15, Spoleto (PG)",
      "surface_sqm": 85.0,
      "cadastral_data": {...}
    }
  ],
  ...
}
```

## Complete Workflow Example

```python
import asyncio
from ocular.services.property_extraction_service import PropertyExtractionService

async def extract_property():
    # Initialize service
    service = PropertyExtractionService()

    # Step 1: Select OCR endpoint
    provider = await service.select_ocr_endpoint(
        preferred_provider="olm_ocr"
    )

    # Step 2: Extract text from PDF
    raw_text = await service.extract_text_from_pdf(
        pdf_path="appraisal.pdf",
        provider=provider
    )

    # Step 3: Convert to structured JSON
    structured_data = await service.convert_to_structured_json(
        extracted_text=raw_text
    )

    # Or use the all-in-one method
    result = await service.process_property_appraisal(
        pdf_path="appraisal.pdf",
        provider_name="olm_ocr",
        save_output=True
    )

    return result

# Run the workflow
result = asyncio.run(extract_property())
```

## API Endpoints

### Extract from Single PDF

**POST** `/api/property-extraction/extract`

**Form Data:**
- `file`: PDF file (required)
- `provider`: OCR provider name (default: "olm_ocr")
- `deployment_id`: RunPod deployment ID (optional)
- `additional_context`: JSON string with portal data (optional)
- `save_output`: Save to file (default: true)

**Response:**
```json
{
  "structured_data": {...},
  "raw_text": "...",
  "metadata": {
    "provider": "OLMocr (Runpod)",
    "processing_time_seconds": 12.5,
    "text_length": 15000,
    "num_assets": 2
  }
}
```

### Extract from Multiple PDFs (Batch)

**POST** `/api/property-extraction/extract-batch`

**Form Data:**
- `files`: Multiple PDF files (required)
- `provider`: OCR provider name (default: "olm_ocr")
- `save_output`: Save to file (default: true)

### Get Available Providers

**GET** `/api/property-extraction/providers`

Returns list of available OCR providers and recommendations.

### Get Schema

**GET** `/api/property-extraction/schema`

Returns the complete `AuctionPropertyProfile` JSON schema.

## Combining PDF with Portal Data

You can enhance extraction by providing additional JSON data from the auction portal:

```python
# Portal data from API
portal_data = {
    "idVendita": 12345,
    "codLotto": "LOTTO 001",
    "dataVendita": "2024-03-15",
    "impoBaseAsta": 180000.0,
    ...
}

# Extract with context
result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    additional_context=portal_data
)
```

## Batch Processing

Process multiple appraisal documents efficiently:

```python
from ocular.services.property_extraction_service import PropertyExtractionService

async def batch_extract():
    service = PropertyExtractionService()
    pdf_files = ["appraisal1.pdf", "appraisal2.pdf", "appraisal3.pdf"]

    results = []
    for pdf_path in pdf_files:
        result = await service.process_property_appraisal(
            pdf_path=pdf_path,
            provider_name="olm_ocr",
            save_output=True
        )
        results.append(result)

    return results
```

## Output Files

The workflow automatically saves results to JSON files:

```
extracted/
├── appraisal_extracted_20241009_123456.json
├── appraisal_extracted_20241009_123457.json
└── ...
```

Each file contains:
- `structured_data`: Complete AuctionPropertyProfile
- `raw_text`: Original OCR text
- `metadata`: Processing information

## Error Handling

The workflow includes comprehensive error handling:

```python
try:
    result = await service.process_property_appraisal(pdf_path="appraisal.pdf")
except FileNotFoundError:
    print("PDF file not found")
except OCRError as e:
    print(f"OCR extraction failed: {e}")
except ValidationError as e:
    print(f"Data validation failed: {e}")
```

## Performance

Typical processing times (per page):
- **OCR Extraction**: 2-5 seconds (GPU)
- **AI Structuring**: 8-15 seconds (Claude API)
- **Total**: ~10-20 seconds per document

## Deployment

### Using RunPod Endpoints

1. Deploy OLMocr or RoLM on RunPod
2. Get your endpoint ID
3. Configure in workflow:

```python
result = await service.process_property_appraisal(
    pdf_path="appraisal.pdf",
    provider_name="olm_ocr",
    deployment_id="your-runpod-endpoint-id"
)
```

### Environment Variables

Required environment variables:

```bash
# For OLM/RoLM OCR
OLM_API_ENDPOINT=https://api.runpod.ai/v2/your-endpoint-id
OLM_API_KEY=your-runpod-api-key

# For Anthropic (Pydantic AI)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional providers
MISTRAL_API_KEY=your-mistral-key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-creds.json
AWS_ACCESS_KEY_ID=your-aws-key
```

## Extracted Fields

The workflow extracts all fields defined in `AuctionPropertyProfile`:

### Court and Procedure
- Court name
- Procedure number
- Judge details
- Expert appraiser

### Auction Details
- Sale type and mode
- Publication date
- Offer deadline
- Auction date/time
- Base price
- Minimum bid
- Deposit requirements

### Property Assets
- Asset type (apartment, garage, land, etc.)
- Description
- Address
- Surface area (m²)
- Cadastral data (foglio, particella, etc.)

### Risk Assessment
- Seismic risk notes
- Compliance issues
- Remediation costs
- Liens and encumbrances

### Valuation
- Market value
- Adjustments
- Final base price

## Advanced Usage

### Custom Extraction Prompts

Modify the extraction prompt for specific requirements:

```python
service = PropertyExtractionService()
service.extraction_agent.system_prompt = """
Your custom extraction instructions here...
"""
```

### Streaming Results

For large batches, process with progress tracking:

```python
async def process_with_progress(pdf_files):
    service = PropertyExtractionService()

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"Processing {idx}/{len(pdf_files)}: {pdf_path}")

        result = await service.process_property_appraisal(pdf_path)
        yield result
```

## Troubleshooting

### Common Issues

**No OCR providers available**
- Ensure API keys are configured
- Check RunPod endpoint is running
- Verify network connectivity

**Validation errors**
- Check PDF is a valid appraisal document
- Ensure document is in Italian
- Verify all required fields are present in PDF

**Slow processing**
- Use GPU providers (OLM/RoLM) for better speed
- Consider batch processing for multiple files
- Check API rate limits

## Support

For issues or questions:
- Check the example scripts in `_examples/`
- Review the API documentation at `/docs`
- See the model schema at `/api/property-extraction/schema`
