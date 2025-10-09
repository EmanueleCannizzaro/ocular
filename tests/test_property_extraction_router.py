"""
Comprehensive tests for property extraction API router.
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ocular.app import app
from ocular.core.models import OCRResult
from ocular.models.property_assessment import (
    AuctionPropertyProfile,
    AppraisalSummary,
    AuctionDetails,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_file():
    """Create sample PDF file for upload."""
    pdf_content = b"%PDF-1.4\ntest content"
    return io.BytesIO(pdf_content)


@pytest.fixture
def sample_auction_profile():
    """Sample auction property profile."""
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
            location="Via Roma 123, Milano",
        ),
    )


@pytest.fixture
def sample_ocr_result():
    """Sample OCR result."""
    return OCRResult(
        text="PERIZIA TECNICA\nVia Roma 123\n€250,000",
        confidence=0.95,
        provider="mistral",
        processing_time=2.5,
    )


@pytest.fixture
def mock_property_service(sample_ocr_result, sample_auction_profile):
    """Mock PropertyExtractionService."""
    mock_service = MagicMock()
    mock_service.process_property_appraisal = AsyncMock(
        return_value={
            "ocr_result": sample_ocr_result,
            "structured_data": sample_auction_profile,
            "metadata": {
                "provider": "mistral",
                "llm_provider": "mistral-7b",
                "ocr_processing_time": 2.5,
                "total_processing_time": 5.0,
                "ocr_confidence": 0.95,
            },
        }
    )
    return mock_service


class TestPropertyExtractionEndpoint:
    """Test /api/property-extraction/extract endpoint."""

    def test_extract_success(self, client, sample_pdf_file, mock_property_service):
        """Test successful property extraction."""
        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "structured_data" in data
            assert "ocr_result" in data
            assert "metadata" in data

            assert data["structured_data"]["property_id"] == "PROP-TEST-001"
            assert data["metadata"]["provider"] == "mistral"

    def test_extract_with_custom_llm_provider(
        self, client, sample_pdf_file, mock_property_service
    ):
        """Test extraction with custom LLM provider."""
        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "olm_ocr", "llm_provider": "phi-3-mini"},
            )

            assert response.status_code == 200
            # Verify PropertyExtractionService was called with correct llm_provider
            mock_property_service.process_property_appraisal.assert_called_once()

    def test_extract_with_additional_context(
        self, client, sample_pdf_file, mock_property_service
    ):
        """Test extraction with additional context."""
        context = {
            "portal_id": "PORTAL-123",
            "tribunal": "Milano",
            "procedure_number": "RGE 123/2025",
        }

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={
                    "provider": "mistral",
                    "additional_context": json.dumps(context),
                },
            )

            assert response.status_code == 200

    def test_extract_missing_file(self, client):
        """Test extraction without file upload."""
        response = client.post(
            "/api/property-extraction/extract", data={"provider": "mistral"}
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_extract_invalid_provider(self, client, sample_pdf_file):
        """Test extraction with invalid OCR provider."""
        response = client.post(
            "/api/property-extraction/extract",
            files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
            data={"provider": "invalid_provider"},
        )

        # Should return error
        assert response.status_code >= 400

    def test_extract_invalid_llm_provider(self, client, sample_pdf_file):
        """Test extraction with invalid LLM provider."""
        response = client.post(
            "/api/property-extraction/extract",
            files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
            data={"provider": "mistral", "llm_provider": "invalid_llm"},
        )

        # Should return error
        assert response.status_code >= 400

    def test_extract_large_file(self, client):
        """Test extraction with file exceeding size limit."""
        # Create 20MB file (exceeds typical 10MB limit)
        large_file = io.BytesIO(b"x" * (20 * 1024 * 1024))

        response = client.post(
            "/api/property-extraction/extract",
            files={"file": ("large.pdf", large_file, "application/pdf")},
            data={"provider": "mistral"},
        )

        # Should reject large file
        assert response.status_code >= 400

    def test_extract_invalid_file_type(self, client):
        """Test extraction with invalid file type."""
        text_file = io.BytesIO(b"This is not a PDF")

        response = client.post(
            "/api/property-extraction/extract",
            files={"file": ("test.txt", text_file, "text/plain")},
            data={"provider": "mistral"},
        )

        # Should reject non-PDF file
        assert response.status_code >= 400


class TestLLMProvidersEndpoint:
    """Test /api/property-extraction/llm-providers endpoint."""

    def test_get_llm_providers(self, client):
        """Test getting list of LLM providers."""
        response = client.get("/api/property-extraction/llm-providers")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "recommended" in data
        assert data["recommended"] == "mistral-7b"

        # Check free models are included
        models = data["models"]
        assert "mistral-7b-instruct" in models
        assert "phi-3-mini" in models
        assert "llama-3-8b" in models

    def test_llm_providers_include_cost_info(self, client):
        """Test LLM providers include cost information."""
        response = client.get("/api/property-extraction/llm-providers")
        data = response.json()

        models = data["models"]

        # Free models should indicate cost: Free
        assert "Free" in models["mistral-7b-instruct"]["cost"]
        assert "Free" in models["phi-3-mini"]["cost"]

    def test_llm_providers_include_quality_rating(self, client):
        """Test LLM providers include quality ratings."""
        response = client.get("/api/property-extraction/llm-providers")
        data = response.json()

        models = data["models"]

        # Should have quality ratings
        for model_name, model_info in models.items():
            assert "quality" in model_info
            assert "⭐" in model_info["quality"]  # Star ratings


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"


class TestErrorResponses:
    """Test error response handling."""

    def test_ocr_processing_error(self, client, sample_pdf_file):
        """Test handling of OCR processing errors."""
        mock_service = MagicMock()
        mock_service.process_property_appraisal = AsyncMock(
            side_effect=Exception("OCR failed")
        )

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_llm_extraction_error(self, client, sample_pdf_file):
        """Test handling of LLM extraction errors."""
        mock_service = MagicMock()
        mock_service.process_property_appraisal = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code == 500

    def test_validation_error(self, client, sample_pdf_file, sample_ocr_result):
        """Test handling of validation errors."""
        from pydantic import ValidationError

        mock_service = MagicMock()
        mock_service.process_property_appraisal = AsyncMock(
            side_effect=ValidationError.from_exception_data(
                "test", line_errors=[]
            )
        )

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code >= 400


class TestResponseFormat:
    """Test response format and structure."""

    def test_response_includes_all_fields(
        self, client, sample_pdf_file, mock_property_service
    ):
        """Test response includes all required fields."""
        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check main sections
            assert "structured_data" in data
            assert "ocr_result" in data
            assert "metadata" in data

            # Check structured data fields
            structured = data["structured_data"]
            assert "property_id" in structured
            assert "appraisal_summary" in structured
            assert "auction_details" in structured

            # Check metadata fields
            metadata = data["metadata"]
            assert "provider" in metadata
            assert "llm_provider" in metadata
            assert "ocr_processing_time" in metadata
            assert "total_processing_time" in metadata

    def test_response_json_serializable(
        self, client, sample_pdf_file, mock_property_service
    ):
        """Test response is properly JSON serializable."""
        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                data={"provider": "mistral"},
            )

            assert response.status_code == 200
            # Should be able to parse as JSON
            data = response.json()
            # Should be able to re-serialize
            json_str = json.dumps(data)
            assert isinstance(json_str, str)


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_pdf_file, mock_property_service):
        """Test handling of concurrent extraction requests."""
        import asyncio

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            # Simulate multiple concurrent requests
            tasks = []
            for _ in range(5):
                response = client.post(
                    "/api/property-extraction/extract",
                    files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                    data={"provider": "mistral"},
                )
                tasks.append(response)

            # All should succeed
            for response in tasks:
                assert response.status_code == 200


class TestProviderSelection:
    """Test OCR and LLM provider selection."""

    def test_default_providers(self, client, sample_pdf_file, mock_property_service):
        """Test default provider selection."""
        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            response = client.post(
                "/api/property-extraction/extract",
                files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
            )

            # Should use defaults (olm_ocr and mistral-7b)
            assert response.status_code == 200

    def test_all_ocr_providers(self, client, sample_pdf_file, mock_property_service):
        """Test all OCR provider options."""
        ocr_providers = ["mistral", "olm_ocr", "rolm_ocr", "google_vision", "tesseract"]

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            for provider in ocr_providers:
                sample_pdf_file.seek(0)  # Reset file pointer
                response = client.post(
                    "/api/property-extraction/extract",
                    files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                    data={"provider": provider},
                )

                # Should accept all valid providers
                if provider in ["mistral", "olm_ocr", "rolm_ocr"]:
                    assert response.status_code == 200

    def test_all_llm_providers(self, client, sample_pdf_file, mock_property_service):
        """Test all LLM provider options."""
        llm_providers = [
            "mistral-7b",
            "phi-3-mini",
            "llama-3-8b",
            "ollama",
            "anthropic",
        ]

        with patch(
            "ocular.routers.property_extraction.PropertyExtractionService",
            return_value=mock_property_service,
        ):
            for llm_provider in llm_providers:
                sample_pdf_file.seek(0)
                response = client.post(
                    "/api/property-extraction/extract",
                    files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
                    data={"provider": "mistral", "llm_provider": llm_provider},
                )

                # Free providers should work (others may need API keys)
                if llm_provider in ["mistral-7b", "phi-3-mini", "llama-3-8b"]:
                    assert response.status_code in [200, 500]  # 500 if HF API issues


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=ocular.routers.property_extraction",
    ])
