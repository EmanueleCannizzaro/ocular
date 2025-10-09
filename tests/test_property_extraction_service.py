"""
Comprehensive tests for Property Extraction Service.
"""
import io
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ocular.core.models import OCRResult
from ocular.models.property_assessment import (
    AuctionPropertyProfile,
    AppraisalSummary,
    AuctionDetails,
)
from ocular.services.property_extraction_service import (
    PropertyExtractionService,
    PropertyExtractionError,
)


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF file bytes for testing."""
    # Create a minimal PDF structure
    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n211\n%%EOF"
    return pdf_content


@pytest.fixture
def sample_ocr_result():
    """Sample OCR result for testing."""
    return OCRResult(
        text="""
        PERIZIA TECNICA DI STIMA

        Immobile: Appartamento
        Indirizzo: Via Roma 123, Milano, 20100
        Superficie commerciale: 85 mq
        Vano: 3.5

        VALUTAZIONE ECONOMICA:
        Valore di mercato: € 250.000,00
        Valore di realizzo forzoso: € 200.000,00

        DATI CATASTALI:
        Foglio: 1234
        Particella: 567
        Subalterno: 8
        Categoria: A/2
        Rendita: € 850,00

        ASTA:
        Data asta: 15/11/2025
        Prezzo base d'asta: € 200.000,00
        Offerta minima: € 150.000,00

        STATO CONSERVAZIONE:
        Stato: Buono
        Necessità interventi: Minime manutenzioni ordinarie
        """,
        confidence=0.95,
        provider="mistral",
        processing_time=2.5,
    )


@pytest.fixture
def sample_auction_profile():
    """Sample auction property profile for testing."""
    return AuctionPropertyProfile(
        property_id="PROP-TEST-001",
        appraisal_summary=AppraisalSummary(
            market_value=250000.0,
            forced_sale_value=200000.0,
            auction_date="2025-11-15",
            property_type="Appartamento",
        ),
        auction_details=AuctionDetails(
            auction_date="2025-11-15",
            base_price=200000.0,
            minimum_bid=150000.0,
            location="Via Roma 123, Milano, 20100",
        ),
    )


@pytest.fixture
def mock_llm_extractor(sample_auction_profile):
    """Mock LLM extractor."""
    mock = AsyncMock()
    mock.extract = AsyncMock(return_value=sample_auction_profile)
    return mock


@pytest.fixture
def property_service(mock_llm_extractor):
    """Create PropertyExtractionService with mocked dependencies."""
    with patch(
        "ocular.services.property_extraction_service.create_llm_extractor",
        return_value=mock_llm_extractor,
    ):
        service = PropertyExtractionService(llm_provider="mistral-7b")
        return service


class TestPropertyExtractionServiceInitialization:
    """Test service initialization."""

    def test_default_initialization(self):
        """Test service initializes with default LLM provider."""
        with patch(
            "ocular.services.property_extraction_service.create_llm_extractor"
        ) as mock_create:
            service = PropertyExtractionService()
            mock_create.assert_called_once_with(
                provider="mistral-7b", model_name=None
            )

    def test_custom_llm_provider(self):
        """Test service initializes with custom LLM provider."""
        with patch(
            "ocular.services.property_extraction_service.create_llm_extractor"
        ) as mock_create:
            service = PropertyExtractionService(llm_provider="anthropic")
            mock_create.assert_called_with(provider="anthropic", model_name=None)

    def test_custom_model_name(self):
        """Test service initializes with custom model name."""
        with patch(
            "ocular.services.property_extraction_service.create_llm_extractor"
        ) as mock_create:
            service = PropertyExtractionService(
                llm_provider="phi-3-mini", llm_model="custom-model"
            )
            mock_create.assert_called_with(
                provider="phi-3-mini", model_name="custom-model"
            )

    def test_with_settings(self):
        """Test service initializes with custom settings."""
        settings = {
            "MISTRAL_API_KEY": "test_key",
            "OLM_API_ENDPOINT": "http://test.endpoint",
        }
        with patch(
            "ocular.services.property_extraction_service.create_llm_extractor"
        ):
            service = PropertyExtractionService(settings=settings)
            assert service.settings == settings


class TestOCRProcessing:
    """Test OCR processing step."""

    @pytest.mark.asyncio
    async def test_perform_ocr_success(
        self, property_service, sample_pdf_bytes, sample_ocr_result
    ):
        """Test successful OCR processing."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.perform_ocr(
                file_bytes=sample_pdf_bytes, provider="mistral"
            )

            assert isinstance(result, OCRResult)
            assert result.text == sample_ocr_result.text
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_perform_ocr_with_invalid_provider(
        self, property_service, sample_pdf_bytes
    ):
        """Test OCR with invalid provider raises error."""
        with pytest.raises(PropertyExtractionError, match="Invalid OCR provider"):
            await property_service.perform_ocr(
                file_bytes=sample_pdf_bytes, provider="invalid_provider"
            )

    @pytest.mark.asyncio
    async def test_perform_ocr_empty_result(self, property_service, sample_pdf_bytes):
        """Test OCR with empty text result."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(
            return_value=OCRResult(
                text="", confidence=0.0, provider="test", processing_time=0
            )
        )

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(PropertyExtractionError, match="No text extracted"):
                await property_service.perform_ocr(
                    file_bytes=sample_pdf_bytes, provider="mistral"
                )

    @pytest.mark.asyncio
    async def test_perform_ocr_provider_error(
        self, property_service, sample_pdf_bytes
    ):
        """Test handling of OCR provider errors."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(
            side_effect=Exception("OCR processing failed")
        )

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(PropertyExtractionError, match="OCR processing failed"):
                await property_service.perform_ocr(
                    file_bytes=sample_pdf_bytes, provider="mistral"
                )


class TestStructureConversion:
    """Test AI structure conversion step."""

    @pytest.mark.asyncio
    async def test_convert_to_structured_json_success(
        self, property_service, sample_ocr_result, sample_auction_profile
    ):
        """Test successful structure conversion."""
        result = await property_service.convert_to_structured_json(
            extracted_text=sample_ocr_result.text
        )

        assert isinstance(result, AuctionPropertyProfile)
        assert result.property_id == "PROP-TEST-001"
        assert result.appraisal_summary.market_value == 250000.0

    @pytest.mark.asyncio
    async def test_convert_with_additional_context(
        self, property_service, sample_ocr_result, sample_auction_profile
    ):
        """Test conversion with additional context."""
        context = {"portal_id": "PORTAL-123", "tribunal": "Milano"}

        result = await property_service.convert_to_structured_json(
            extracted_text=sample_ocr_result.text, additional_context=context
        )

        assert isinstance(result, AuctionPropertyProfile)
        property_service.llm_extractor.extract.assert_called_once_with(
            ocr_text=sample_ocr_result.text, additional_context=context
        )

    @pytest.mark.asyncio
    async def test_convert_empty_text(self, property_service):
        """Test conversion with empty text."""
        with pytest.raises(PropertyExtractionError, match="cannot be empty"):
            await property_service.convert_to_structured_json(extracted_text="")

    @pytest.mark.asyncio
    async def test_convert_llm_error(self, property_service, sample_ocr_result):
        """Test handling of LLM extraction errors."""
        property_service.llm_extractor.extract = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        with pytest.raises(PropertyExtractionError, match="LLM API error"):
            await property_service.convert_to_structured_json(
                extracted_text=sample_ocr_result.text
            )


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.mark.asyncio
    async def test_process_property_appraisal_success(
        self,
        property_service,
        sample_pdf_bytes,
        sample_ocr_result,
        sample_auction_profile,
    ):
        """Test successful end-to-end processing."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.process_property_appraisal(
                file_bytes=sample_pdf_bytes, provider="mistral"
            )

            assert isinstance(result, dict)
            assert "ocr_result" in result
            assert "structured_data" in result
            assert "metadata" in result

            assert result["structured_data"].property_id == "PROP-TEST-001"
            assert result["metadata"]["provider"] == "mistral"
            assert result["metadata"]["llm_provider"] == "mistral-7b"

    @pytest.mark.asyncio
    async def test_process_with_file_path(
        self,
        property_service,
        sample_ocr_result,
        sample_auction_profile,
        tmp_path,
    ):
        """Test processing with file path instead of bytes."""
        # Create temporary PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.process_property_appraisal(
                file_path=str(pdf_file), provider="mistral"
            )

            assert isinstance(result, dict)
            assert result["structured_data"].property_id == "PROP-TEST-001"

    @pytest.mark.asyncio
    async def test_process_with_additional_context(
        self,
        property_service,
        sample_pdf_bytes,
        sample_ocr_result,
        sample_auction_profile,
    ):
        """Test processing with additional context."""
        context = {
            "portal_id": "PORTAL-123",
            "tribunal": "Tribunale di Milano",
            "procedure_number": "RGE 123/2025",
        }

        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.process_property_appraisal(
                file_bytes=sample_pdf_bytes,
                provider="mistral",
                additional_context=context,
            )

            assert result["metadata"]["portal_id"] == "PORTAL-123"
            assert result["metadata"]["tribunal"] == "Tribunale di Milano"

    @pytest.mark.asyncio
    async def test_process_missing_input(self, property_service):
        """Test processing without file_bytes or file_path raises error."""
        with pytest.raises(PropertyExtractionError, match="Either file_bytes or file_path"):
            await property_service.process_property_appraisal(provider="mistral")

    @pytest.mark.asyncio
    async def test_process_with_validation_error(
        self, property_service, sample_pdf_bytes, sample_ocr_result
    ):
        """Test handling of validation errors in structured output."""
        # Mock LLM to return invalid data
        property_service.llm_extractor.extract = AsyncMock(
            side_effect=ValidationError.from_exception_data(
                "Invalid data", line_errors=[]
            )
        )

        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            with pytest.raises(PropertyExtractionError, match="Validation failed"):
                await property_service.process_property_appraisal(
                    file_bytes=sample_pdf_bytes, provider="mistral"
                )


class TestBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.mark.asyncio
    async def test_process_multiple_files(
        self,
        property_service,
        sample_pdf_bytes,
        sample_ocr_result,
        sample_auction_profile,
    ):
        """Test processing multiple files."""
        files = [
            ("file1.pdf", sample_pdf_bytes),
            ("file2.pdf", sample_pdf_bytes),
            ("file3.pdf", sample_pdf_bytes),
        ]

        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            results = []
            for filename, file_bytes in files:
                result = await property_service.process_property_appraisal(
                    file_bytes=file_bytes, provider="mistral"
                )
                results.append(result)

            assert len(results) == 3
            assert all(isinstance(r["structured_data"], AuctionPropertyProfile) for r in results)


class TestErrorRecovery:
    """Test error recovery and retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(
        self, property_service, sample_pdf_bytes, sample_ocr_result
    ):
        """Test retry logic on transient errors."""
        mock_provider = AsyncMock()
        # First call fails, second succeeds
        mock_provider.process_document = AsyncMock(
            side_effect=[Exception("Temporary error"), sample_ocr_result]
        )

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            # Should retry and succeed
            result = await property_service.perform_ocr(
                file_bytes=sample_pdf_bytes, provider="mistral", max_retries=2
            )

            assert isinstance(result, OCRResult)
            assert mock_provider.process_document.call_count == 2


class TestMetadataGeneration:
    """Test metadata generation."""

    @pytest.mark.asyncio
    async def test_metadata_includes_timing(
        self,
        property_service,
        sample_pdf_bytes,
        sample_ocr_result,
        sample_auction_profile,
    ):
        """Test metadata includes processing times."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.process_property_appraisal(
                file_bytes=sample_pdf_bytes, provider="mistral"
            )

            metadata = result["metadata"]
            assert "ocr_processing_time" in metadata
            assert "total_processing_time" in metadata
            assert metadata["ocr_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_metadata_includes_providers(
        self,
        property_service,
        sample_pdf_bytes,
        sample_ocr_result,
        sample_auction_profile,
    ):
        """Test metadata includes provider information."""
        mock_provider = AsyncMock()
        mock_provider.process_document = AsyncMock(return_value=sample_ocr_result)

        with patch(
            "ocular.services.property_extraction_service.OCRProviderFactory.create_provider",
            return_value=mock_provider,
        ):
            result = await property_service.process_property_appraisal(
                file_bytes=sample_pdf_bytes, provider="olm_ocr"
            )

            metadata = result["metadata"]
            assert metadata["provider"] == "olm_ocr"
            assert metadata["llm_provider"] == "mistral-7b"


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=ocular.services.property_extraction_service",
    ])
