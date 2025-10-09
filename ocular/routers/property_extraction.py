"""
Property Extraction API Router

FastAPI endpoints for extracting structured property auction data from PDFs.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import json
from datetime import datetime

from ..services.property_extraction_service import PropertyExtractionService
from ..services.llm_extraction_providers import LLMProvider, RECOMMENDED_FREE_MODELS
from ..models.property_assessment import AuctionPropertyProfile
from ..core.logging import get_logger


router = APIRouter(prefix="/api/property-extraction", tags=["Property Extraction"])
logger = get_logger(__name__)


@router.post("/extract", response_model=Dict[str, Any])
async def extract_property_from_pdf(
    file: UploadFile = File(..., description="Property appraisal PDF file"),
    provider: Optional[str] = Form(
        default="olm_ocr",
        description="OCR provider to use (olm_ocr, rolm_ocr, mistral, etc.)"
    ),
    llm_provider: Optional[str] = Form(
        default="mistral-7b",
        description="LLM provider for structure conversion (mistral-7b, phi-3-mini, llama-3-8b, anthropic, ollama) - Default: mistral-7b (FREE!)"
    ),
    deployment_id: Optional[str] = Form(
        default=None,
        description="Optional RunPod deployment ID for OCR"
    ),
    additional_context: Optional[str] = Form(
        default=None,
        description="Optional JSON string with additional context from auction portal"
    ),
    save_output: bool = Form(
        default=True,
        description="Whether to save the extraction result to file"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Extract structured property auction data from a PDF appraisal document.

    This endpoint orchestrates the complete workflow:
    1. Receives a PDF file upload
    2. Uses a selected GPU OCR endpoint to extract text
    3. Converts unstructured text to structured AuctionPropertyProfile JSON
    4. Returns the structured data

    **Workflow:**
    - Upload a property appraisal PDF (perizia tecnica)
    - Select OCR provider (OLM OCR, RoLM OCR, or Mistral)
    - Optionally provide additional context from auction portal
    - Receive structured JSON following AuctionPropertyProfile schema

    **Returns:**
    - structured_data: Complete AuctionPropertyProfile object
    - raw_text: Extracted OCR text
    - metadata: Processing information (provider, timing, etc.)
    """
    logger.info(
        "Property extraction request received",
        filename=file.filename,
        provider=provider,
        content_type=file.content_type
    )

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for property extraction"
        )

    # Parse additional context if provided
    context_data = None
    if additional_context:
        try:
            context_data = json.loads(additional_context)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in additional_context parameter"
            )

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Initialize extraction service with selected LLM provider
        service = PropertyExtractionService(llm_provider=llm_provider)

        # Process the document
        result = await service.process_property_appraisal(
            pdf_path=tmp_path,
            provider_name=provider,
            deployment_id=deployment_id,
            additional_context=context_data,
            save_output=save_output,
            output_dir=str(Path(tmp_path).parent / "property_extractions")
        )

        # Clean up temp file in background
        background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

        logger.info(
            "Property extraction completed successfully",
            sale_id=result["structured_data"]["sale_id"],
            processing_time=result["metadata"]["processing_time_seconds"]
        )

        return result

    except Exception as e:
        logger.error("Property extraction failed", error=str(e))

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Property extraction failed: {str(e)}"
        )


@router.post("/extract-batch", response_model=Dict[str, Any])
async def extract_properties_batch(
    files: list[UploadFile] = File(..., description="Multiple property appraisal PDFs"),
    provider: Optional[str] = Form(default="olm_ocr"),
    deployment_id: Optional[str] = Form(default=None),
    save_output: bool = Form(default=True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Extract structured data from multiple property appraisal PDFs in batch.

    **Workflow:**
    - Upload multiple PDF files
    - Process all files using the selected OCR provider
    - Return results for each file

    **Returns:**
    - results: List of extraction results for each file
    - summary: Batch processing summary
    """
    logger.info(
        "Batch property extraction request received",
        num_files=len(files),
        provider=provider
    )

    service = PropertyExtractionService()
    results = []
    errors = []
    temp_files = []

    try:
        # Process each file
        for idx, file in enumerate(files):
            try:
                # Validate file type
                if not file.filename.lower().endswith('.pdf'):
                    errors.append({
                        "filename": file.filename,
                        "error": "Only PDF files are supported"
                    })
                    continue

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                    temp_files.append(tmp_path)

                # Process
                result = await service.process_property_appraisal(
                    pdf_path=tmp_path,
                    provider_name=provider,
                    deployment_id=deployment_id,
                    save_output=save_output
                )

                results.append({
                    "filename": file.filename,
                    "success": True,
                    "data": result
                })

            except Exception as e:
                logger.error(f"Failed to process {file.filename}", error=str(e))
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })

        # Summary
        summary = {
            "total_files": len(files),
            "successful": len(results),
            "failed": len(errors),
            "processing_timestamp": datetime.now().isoformat()
        }

        # Clean up temp files in background
        for temp_path in temp_files:
            background_tasks.add_task(Path(temp_path).unlink, missing_ok=True)

        logger.info(
            "Batch property extraction completed",
            total=summary["total_files"],
            successful=summary["successful"],
            failed=summary["failed"]
        )

        return {
            "results": results,
            "errors": errors,
            "summary": summary
        }

    except Exception as e:
        # Clean up temp files
        for temp_path in temp_files:
            Path(temp_path).unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/providers", response_model=Dict[str, Any])
async def get_available_providers():
    """
    Get list of available OCR providers for property extraction.

    Returns information about which providers are configured and available.
    """
    service = PropertyExtractionService()

    try:
        # Try to get provider from auto-selection
        provider = await service.select_ocr_endpoint()

        return {
            "default_provider": provider.provider_type.value,
            "default_provider_name": provider.provider_name,
            "available_providers": [
                "olm_ocr",
                "rolm_ocr",
                "mistral",
                "google_vision",
                "aws_textract"
            ],
            "recommended": "olm_ocr",
            "recommendation_reason": "Best quality for Italian documents with complex layouts"
        }

    except Exception as e:
        logger.warning("Failed to get provider info", error=str(e))
        return {
            "error": str(e),
            "available_providers": [
                "olm_ocr",
                "rolm_ocr",
                "mistral",
                "google_vision",
                "aws_textract"
            ]
        }


@router.get("/llm-providers", response_model=Dict[str, Any])
async def get_llm_providers():
    """
    Get list of available LLM providers for structure conversion.

    Returns information about free and paid LLM options, including:
    - Open-source models via HuggingFace (FREE!)
    - Local models via Ollama (FREE!)
    - Commercial APIs (Anthropic, OpenAI)
    """
    return {
        "recommended": "mistral-7b",
        "recommendation_reason": "Free via HuggingFace Inference API, high quality for Italian documents",
        "models": RECOMMENDED_FREE_MODELS,
        "usage_examples": {
            "free_huggingface": {
                "provider": "mistral-7b",
                "description": "Mistral 7B Instruct via HuggingFace - 100% FREE",
                "api_key_required": False
            },
            "free_local": {
                "provider": "ollama",
                "model_name": "mistral:7b-instruct",
                "description": "Run Mistral locally with Ollama - 100% FREE",
                "setup": "Install Ollama from https://ollama.ai"
            },
            "paid_best_quality": {
                "provider": "anthropic",
                "description": "Claude Sonnet - Best quality, ~$3/1M tokens",
                "api_key_required": True
            }
        }
    }


@router.get("/schema", response_model=Dict[str, Any])
async def get_extraction_schema():
    """
    Get the JSON schema for the AuctionPropertyProfile structure.

    This endpoint returns the complete Pydantic model schema that defines
    the structure of extracted property data.
    """
    return {
        "schema": AuctionPropertyProfile.model_json_schema(),
        "description": "Schema for structured Italian property auction data",
        "example": _get_example_property_profile()
    }


def _get_example_property_profile() -> Dict[str, Any]:
    """Generate an example AuctionPropertyProfile."""
    return {
        "sale_id": 12345,
        "lot_number": "LOTTO 001",
        "appraisal_summary": {
            "court": "TRIBUNALE DI SPOLETO",
            "procedure_number": "R.G. 102/2022",
            "judge": {
                "full_name": "Dr. Mario Rossi",
                "role": "Judge",
                "email": None,
                "phone": None
            },
            "expert_appraiser": {
                "full_name": "Ing. Giuseppe Verdi",
                "role": "Expert Appraiser",
                "email": "g.verdi@example.it",
                "phone": "+39 0743 123456"
            }
        },
        "auction_details": {
            "sale_type": "Senza Incanto",
            "sale_mode": "Sincrona Mista",
            "publication_date": "2024-01-15",
            "offer_submission_deadline": "2024-03-01 13:00",
            "auction_date": "2024-03-15 10:00",
            "base_price": 180000.0,
            "minimum_bid": 135000.0,
            "minimum_bid_increase": 5000.0,
            "required_deposit_rule": "20% prezzo offerto"
        },
        "risk_assessment": {
            "seismic_risk_notes": "Edificio ristrutturato post-terremoto 2016",
            "compliance_issues": [
                "Mancanza certificato di agibilità",
                "Discrepanze catastali da regolarizzare"
            ],
            "costs_to_remedy": [
                {
                    "description": "Aggiornamento catastale",
                    "estimated_cost": 2000.0
                }
            ],
            "liens_and_encumbrances": [
                {
                    "lien_type": "Ipoteca Volontaria",
                    "registration_date": "2020-05-10",
                    "beneficiary": "Banca Nazionale",
                    "amount": 150000.0
                }
            ]
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
                "cadastral_data": {
                    "sheet": "12",
                    "section": "A",
                    "subsection": None,
                    "parcel": "456",
                    "subordinate": "2",
                    "category": "A/3",
                    "revenue": 550.0
                }
            }
        ],
        "occupancy_status": "Libero",
        "publication_websites": [
            {
                "site_name": "www.asteavvisi.it",
                "listing_url": "https://www.asteavvisi.it/dettaglio/12345"
            }
        ]
    }
