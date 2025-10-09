"""
Property Extraction Service

This service orchestrates the complete workflow for extracting structured property
auction data from PDF documents using GPU-based OCR and AI-powered extraction.

Workflow:
1. Select a deployed GPU OCR endpoint (OLM, RoLM, or Mistral)
2. Extract text from PDF using the OCR endpoint
3. Use Pydantic AI to convert unstructured text to AuctionPropertyProfile JSON
"""

import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from pydantic import ValidationError

from ..core.logging import get_logger
from ..core.exceptions import OCRError, ValidationError as OcularValidationError
from ..models.property_assessment import AuctionPropertyProfile
from ..providers.factory import ProviderFactory
from ..providers.base import BaseOCRProvider
from ..core.enums import ProviderType
from .llm_extraction_providers import (
    create_llm_extractor,
    LLMProvider
)


logger = get_logger(__name__)


class PropertyExtractionService:
    """
    Service for extracting structured auction property data from PDF documents.

    This service combines OCR extraction with AI-powered structured data extraction
    to convert PDF property appraisal documents into structured JSON.
    """

    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        llm_provider: LLMProvider = "mistral-7b",
        llm_model: Optional[str] = None
    ):
        """
        Initialize the property extraction service.

        Args:
            settings: Optional configuration settings
            llm_provider: LLM provider for structure conversion (default: mistral-7b - free!)
            llm_model: Optional specific model name
        """
        self.settings = settings or {}
        self.logger = logger
        self.llm_provider = llm_provider

        # Initialize LLM extractor (use free Mistral-7B by default)
        self.llm_extractor = create_llm_extractor(
            provider=llm_provider,
            model_name=llm_model
        )

    def _get_extraction_prompt(self) -> str:
        """Get the system prompt for the extraction agent."""
        return """You are an expert in Italian real estate auction documents and property appraisals.

Your task is to extract structured information from Italian property auction appraisal documents (perizie tecniche)
and convert them into a structured JSON format following the AuctionPropertyProfile schema.

Key extraction guidelines:

1. **Dates**: Extract in format "YYYY-MM-DD HH:MM" or "YYYY-MM-DD" as appropriate
2. **Numbers**: Extract prices and values as floats (Euro), remove currency symbols
3. **Addresses**: Provide complete addresses including street, number, city, province
4. **Cadastral Data**: Extract all available fields (Foglio, Particella, Subalterno, Categoria, Rendita)
5. **People**: Extract full names and roles (Judge, Delegate, Expert Appraiser, etc.)
6. **Assets**: Identify all properties in the lot (apartments, garages, land, etc.)
7. **Liens**: Extract all encumbrances (ipoteche, pignoramenti) with dates and amounts
8. **Compliance Issues**: List all non-conformities mentioned in the appraisal
9. **Valuation**: Extract market value, adjustments, and final base price

Common Italian terms:
- Tribunale = Court
- R.G. = Registro Generale (procedure number)
- CTU = Consulente Tecnico d'Ufficio (Expert Appraiser)
- Giudice dell'Esecuzione = Execution Judge
- Foglio = Sheet, Particella = Parcel
- Rendita Catastale = Cadastral Revenue
- Ipoteca = Mortgage, Pignoramento = Seizure
- Agibilità = Habitability Certificate
- Base d'asta = Auction Base Price

If information is not available in the document, use null for optional fields.
Be precise with numbers and dates. Maintain Italian text for descriptions where appropriate.
"""

    async def select_ocr_endpoint(
        self,
        preferred_provider: Optional[str] = None,
        deployment_id: Optional[str] = None
    ) -> BaseOCRProvider:
        """
        Select and initialize an OCR endpoint for processing.

        Args:
            preferred_provider: Preferred provider type ('olm_ocr', 'rolm_ocr', 'mistral', etc.)
            deployment_id: Optional specific deployment ID if using RunPod

        Returns:
            Initialized OCR provider

        Raises:
            OCRError: If no suitable provider is available
        """
        self.logger.info("Selecting OCR endpoint", preferred_provider=preferred_provider)

        # Determine which provider to use
        if preferred_provider:
            try:
                provider_type = ProviderType(preferred_provider.lower())
            except ValueError:
                raise OcularValidationError(
                    f"Invalid provider type: {preferred_provider}. "
                    f"Valid options: {[p.value for p in ProviderType]}"
                )
        else:
            # Auto-select: prefer GPU providers for better quality
            provider_type = await self._auto_select_provider()

        # Create and initialize provider
        config = self.settings.get(provider_type.value, {})
        if deployment_id and provider_type in [ProviderType.OLM_OCR, ProviderType.ROLM_OCR]:
            config['endpoint_id'] = deployment_id

        provider = ProviderFactory.create_provider(provider_type, config)
        await provider.initialize()

        if not await provider.is_available():
            raise OCRError(f"Provider {provider_type.value} is not available")

        self.logger.info(
            "OCR endpoint selected",
            provider=provider_type.value,
            provider_name=provider.provider_name
        )

        return provider

    async def _auto_select_provider(self) -> ProviderType:
        """
        Auto-select the best available provider.
        Priority: OLM > RoLM > Mistral > Others
        """
        priority_order = [
            ProviderType.OLM_OCR,
            ProviderType.ROLM_OCR,
            ProviderType.MISTRAL,
            ProviderType.GOOGLE_VISION,
            ProviderType.AWS_TEXTRACT,
        ]

        for provider_type in priority_order:
            try:
                config = self.settings.get(provider_type.value, {})
                provider = ProviderFactory.create_provider(provider_type, config)
                await provider.initialize()

                if await provider.is_available():
                    self.logger.info(f"Auto-selected provider: {provider_type.value}")
                    return provider_type
            except Exception as e:
                self.logger.debug(f"Provider {provider_type.value} not available: {e}")
                continue

        raise OCRError("No OCR providers available")

    async def extract_text_from_pdf(
        self,
        pdf_path: str,
        provider: BaseOCRProvider,
        **kwargs
    ) -> str:
        """
        Extract text from PDF using the selected OCR provider.

        Args:
            pdf_path: Path to the PDF file
            provider: Initialized OCR provider
            **kwargs: Additional provider-specific parameters

        Returns:
            Extracted text as string
        """
        self.logger.info("Extracting text from PDF", pdf_path=pdf_path)

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Process with OCR
        result = await provider.process_document(
            file_path=str(pdf_file),
            **kwargs
        )

        if not result.success:
            raise OCRError(f"OCR extraction failed: {result.error_message}")

        self.logger.info(
            "Text extraction completed",
            chars_extracted=len(result.text),
            confidence=result.confidence
        )

        return result.text

    async def convert_to_structured_json(
        self,
        extracted_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """
        Convert unstructured OCR text to structured AuctionPropertyProfile JSON.

        Uses the configured LLM provider (default: free Mistral-7B via HuggingFace).

        Args:
            extracted_text: The raw text extracted from the PDF
            additional_context: Optional additional data (e.g., from portal JSON)

        Returns:
            Structured AuctionPropertyProfile object

        Raises:
            ValidationError: If extraction fails validation
        """
        self.logger.info(
            "Converting text to structured JSON",
            text_length=len(extracted_text),
            llm_provider=self.llm_provider
        )

        # Run extraction with configured LLM
        try:
            structured_data = await self.llm_extractor.extract(
                ocr_text=extracted_text,
                additional_context=additional_context
            )

            self.logger.info(
                "Structured extraction completed",
                sale_id=structured_data.sale_id,
                lot_number=structured_data.lot_number,
                num_assets=len(structured_data.assets),
                llm_provider=self.llm_provider
            )

            return structured_data

        except ValidationError as e:
            self.logger.error("Validation failed during extraction", error=str(e))
            raise OcularValidationError(f"Failed to extract valid property data: {e}")

    async def process_property_appraisal(
        self,
        pdf_path: str,
        provider_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        save_output: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow: PDF -> OCR -> Structured JSON

        This is the main entry point for the property extraction workflow.

        Args:
            pdf_path: Path to the property appraisal PDF
            provider_name: Optional OCR provider to use
            deployment_id: Optional deployment ID for RunPod endpoints
            additional_context: Optional portal data to enhance extraction
            save_output: Whether to save the output JSON to file
            output_dir: Directory to save output files

        Returns:
            Dictionary containing:
                - structured_data: AuctionPropertyProfile object
                - raw_text: Extracted OCR text
                - metadata: Processing metadata
        """
        start_time = datetime.now()

        self.logger.info(
            "Starting property appraisal extraction workflow",
            pdf_path=pdf_path,
            provider_name=provider_name
        )

        try:
            # Step 1: Select OCR endpoint
            provider = await self.select_ocr_endpoint(provider_name, deployment_id)

            # Step 2: Extract text from PDF
            extracted_text = await self.extract_text_from_pdf(pdf_path, provider)

            # Step 3: Convert to structured JSON
            structured_data = await self.convert_to_structured_json(
                extracted_text,
                additional_context
            )

            # Prepare result
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "structured_data": structured_data.model_dump(),
                "raw_text": extracted_text,
                "metadata": {
                    "pdf_path": pdf_path,
                    "provider": provider.provider_name,
                    "provider_type": provider.provider_type.value,
                    "processing_time_seconds": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(extracted_text),
                    "num_assets": len(structured_data.assets),
                }
            }

            # Save output if requested
            if save_output:
                output_path = self._save_output(result, pdf_path, output_dir)
                result["metadata"]["output_path"] = str(output_path)
                self.logger.info("Output saved", output_path=output_path)

            self.logger.info(
                "Property extraction workflow completed successfully",
                processing_time=processing_time,
                sale_id=structured_data.sale_id
            )

            return result

        except Exception as e:
            self.logger.error("Property extraction workflow failed", error=str(e))
            raise

    def _save_output(
        self,
        result: Dict[str, Any],
        pdf_path: str,
        output_dir: Optional[str]
    ) -> Path:
        """Save the extraction result to a JSON file."""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(pdf_path).parent / "extracted"

        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        pdf_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"{pdf_name}_extracted_{timestamp}.json"

        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_file


# Convenience function for quick usage
async def extract_property_from_pdf(
    pdf_path: str,
    provider: str = "olm_ocr",
    **kwargs
) -> AuctionPropertyProfile:
    """
    Quick function to extract property data from a PDF.

    Args:
        pdf_path: Path to the PDF file
        provider: OCR provider to use (default: 'olm_ocr')
        **kwargs: Additional arguments passed to process_property_appraisal

    Returns:
        AuctionPropertyProfile object
    """
    service = PropertyExtractionService()
    result = await service.process_property_appraisal(
        pdf_path=pdf_path,
        provider_name=provider,
        **kwargs
    )
    return AuctionPropertyProfile(**result["structured_data"])
