"""
Comprehensive tests for LLM extraction providers.
"""
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ocular.models.property_assessment import AuctionPropertyProfile
from ocular.services.llm_extraction_providers import (
    AnthropicExtractor,
    BaseLLMExtractor,
    HuggingFaceExtractor,
    LLMProvider,
    OllamaExtractor,
    OpenAIExtractor,
    create_llm_extractor,
)


# Sample OCR text for testing
SAMPLE_OCR_TEXT = """
PERIZIA TECNICA DI STIMA

Immobile: Appartamento
Indirizzo: Via Roma 123, Milano
Superficie: 85 mq
Valore di mercato: €250,000
Data asta: 2025-11-15
Prezzo base: €200,000
Stato conservazione: Buono
"""

# Expected structured output
EXPECTED_JSON_OUTPUT = {
    "property_id": "PROP-001",
    "appraisal_summary": {
        "market_value": 250000.0,
        "forced_sale_value": 200000.0,
        "auction_date": "2025-11-15",
        "property_type": "Appartamento",
    },
    "auction_details": {
        "auction_date": "2025-11-15",
        "base_price": 200000.0,
        "location": "Via Roma 123, Milano",
    },
    "valuation_details": {
        "total_area_sqm": 85.0,
        "condition": "Buono",
    },
}


class TestBaseLLMExtractor:
    """Test abstract base class behavior."""

    def test_base_extractor_cannot_be_instantiated(self):
        """Test that BaseLLMExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMExtractor()


class TestHuggingFaceExtractor:
    """Test HuggingFace LLM extractor."""

    @pytest.fixture
    def mock_aiohttp_response(self):
        """Mock aiohttp response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"generated_text": json.dumps(EXPECTED_JSON_OUTPUT)}]
        )
        return mock_response

    @pytest.fixture
    def hf_extractor(self):
        """Create HuggingFace extractor instance."""
        return HuggingFaceExtractor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            api_key="test_hf_key",
        )

    def test_initialization(self, hf_extractor):
        """Test HuggingFace extractor initialization."""
        assert hf_extractor.model_name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert "mistralai/Mistral-7B-Instruct-v0.3" in hf_extractor.api_url

    @pytest.mark.asyncio
    async def test_extract_success(self, hf_extractor, mock_aiohttp_response):
        """Test successful extraction with HuggingFace."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_aiohttp_response

            result = await hf_extractor.extract(SAMPLE_OCR_TEXT)

            assert isinstance(result, AuctionPropertyProfile)
            assert result.appraisal_summary.market_value == 250000.0

    @pytest.mark.asyncio
    async def test_extract_with_retry(self, hf_extractor):
        """Test extraction with retry logic."""
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 503
        mock_response_fail.text = AsyncMock(return_value="Service unavailable")

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(
            return_value=[{"generated_text": json.dumps(EXPECTED_JSON_OUTPUT)}]
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # First call fails, second succeeds
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success,
            ]

            result = await hf_extractor.extract(SAMPLE_OCR_TEXT)
            assert isinstance(result, AuctionPropertyProfile)

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self, hf_extractor):
        """Test handling of invalid JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"generated_text": "invalid json {{{"}]
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception):  # Should raise JSONDecodeError
                await hf_extractor.extract(SAMPLE_OCR_TEXT)

    @pytest.mark.asyncio
    async def test_extract_missing_required_fields(self, hf_extractor):
        """Test handling of response missing required fields."""
        incomplete_output = {"property_id": "PROP-001"}  # Missing required fields

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"generated_text": json.dumps(incomplete_output)}]
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ValidationError):
                await hf_extractor.extract(SAMPLE_OCR_TEXT)


class TestOllamaExtractor:
    """Test Ollama local LLM extractor."""

    @pytest.fixture
    def ollama_extractor(self):
        """Create Ollama extractor instance."""
        return OllamaExtractor(
            model_name="mistral:7b-instruct", base_url="http://localhost:11434"
        )

    def test_initialization(self, ollama_extractor):
        """Test Ollama extractor initialization."""
        assert ollama_extractor.model_name == "mistral:7b-instruct"
        assert ollama_extractor.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_extract_success(self, ollama_extractor):
        """Test successful extraction with Ollama."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"response": json.dumps(EXPECTED_JSON_OUTPUT)}
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await ollama_extractor.extract(SAMPLE_OCR_TEXT)

            assert isinstance(result, AuctionPropertyProfile)
            assert result.property_id == "PROP-001"

    @pytest.mark.asyncio
    async def test_extract_connection_error(self, ollama_extractor):
        """Test handling of Ollama connection errors."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            with pytest.raises(Exception, match="Connection refused"):
                await ollama_extractor.extract(SAMPLE_OCR_TEXT)


class TestAnthropicExtractor:
    """Test Anthropic Claude extractor."""

    @pytest.fixture
    def anthropic_extractor(self):
        """Create Anthropic extractor instance."""
        return AnthropicExtractor(api_key="test_anthropic_key")

    def test_initialization(self, anthropic_extractor):
        """Test Anthropic extractor initialization."""
        assert anthropic_extractor.api_key == "test_anthropic_key"

    @pytest.mark.asyncio
    async def test_extract_success(self, anthropic_extractor):
        """Test successful extraction with Anthropic."""
        # Mock the Anthropic client
        mock_agent = MagicMock()
        mock_agent.run_sync = MagicMock(
            return_value=MagicMock(data=AuctionPropertyProfile(**EXPECTED_JSON_OUTPUT))
        )

        with patch(
            "ocular.services.llm_extraction_providers.Agent",
            return_value=mock_agent,
        ):
            result = await anthropic_extractor.extract(SAMPLE_OCR_TEXT)

            assert isinstance(result, AuctionPropertyProfile)
            assert result.property_id == "PROP-001"

    @pytest.mark.asyncio
    async def test_extract_with_context(self, anthropic_extractor):
        """Test extraction with additional context."""
        additional_context = {"portal_id": "PORTAL-123", "source": "tribunale"}

        mock_agent = MagagicMock()
        mock_agent.run_sync = MagicMock(
            return_value=MagicMock(data=AuctionPropertyProfile(**EXPECTED_JSON_OUTPUT))
        )

        with patch(
            "ocular.services.llm_extraction_providers.Agent",
            return_value=mock_agent,
        ):
            result = await anthropic_extractor.extract(
                SAMPLE_OCR_TEXT, additional_context=additional_context
            )

            assert isinstance(result, AuctionPropertyProfile)


class TestOpenAIExtractor:
    """Test OpenAI GPT extractor."""

    @pytest.fixture
    def openai_extractor(self):
        """Create OpenAI extractor instance."""
        return OpenAIExtractor(api_key="test_openai_key", model_name="gpt-4o-mini")

    def test_initialization(self, openai_extractor):
        """Test OpenAI extractor initialization."""
        assert openai_extractor.api_key == "test_openai_key"
        assert openai_extractor.model_name == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_extract_success(self, openai_extractor):
        """Test successful extraction with OpenAI."""
        mock_agent = MagicMock()
        mock_agent.run_sync = MagicMock(
            return_value=MagicMock(data=AuctionPropertyProfile(**EXPECTED_JSON_OUTPUT))
        )

        with patch(
            "ocular.services.llm_extraction_providers.Agent",
            return_value=mock_agent,
        ):
            result = await openai_extractor.extract(SAMPLE_OCR_TEXT)

            assert isinstance(result, AuctionPropertyProfile)
            assert result.appraisal_summary.market_value == 250000.0


class TestLLMExtractorFactory:
    """Test LLM extractor factory function."""

    def test_create_mistral_7b_extractor(self):
        """Test creating Mistral-7B extractor."""
        extractor = create_llm_extractor(
            provider="mistral-7b", api_key="test_hf_key"
        )

        assert isinstance(extractor, HuggingFaceExtractor)
        assert "Mistral-7B" in extractor.model_name

    def test_create_phi3_extractor(self):
        """Test creating Phi-3-mini extractor."""
        extractor = create_llm_extractor(provider="phi-3-mini", api_key="test_hf_key")

        assert isinstance(extractor, HuggingFaceExtractor)
        assert "Phi-3" in extractor.model_name

    def test_create_llama_extractor(self):
        """Test creating Llama-3 extractor."""
        extractor = create_llm_extractor(
            provider="llama-3-8b", api_key="test_hf_key"
        )

        assert isinstance(extractor, HuggingFaceExtractor)
        assert "Llama-3" in extractor.model_name

    def test_create_ollama_extractor(self):
        """Test creating Ollama extractor."""
        extractor = create_llm_extractor(provider="ollama")

        assert isinstance(extractor, OllamaExtractor)

    def test_create_anthropic_extractor(self):
        """Test creating Anthropic extractor."""
        extractor = create_llm_extractor(
            provider="anthropic", api_key="test_anthropic_key"
        )

        assert isinstance(extractor, AnthropicExtractor)

    def test_create_openai_extractor(self):
        """Test creating OpenAI extractor."""
        extractor = create_llm_extractor(
            provider="openai", api_key="test_openai_key"
        )

        assert isinstance(extractor, OpenAIExtractor)

    def test_create_with_custom_model(self):
        """Test creating extractor with custom model name."""
        extractor = create_llm_extractor(
            provider="mistral-7b",
            model_name="mistralai/Mistral-7B-v0.2",
            api_key="test_hf_key",
        )

        assert isinstance(extractor, HuggingFaceExtractor)
        assert extractor.model_name == "mistralai/Mistral-7B-v0.2"

    def test_create_invalid_provider(self):
        """Test creating extractor with invalid provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_extractor(provider="invalid_provider")

    def test_create_without_required_api_key(self):
        """Test creating extractor without required API key."""
        with pytest.raises(ValueError, match="API key required"):
            create_llm_extractor(provider="anthropic")  # Missing api_key


class TestPromptGeneration:
    """Test prompt generation for different extractors."""

    @pytest.fixture
    def hf_extractor(self):
        return HuggingFaceExtractor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", api_key="test_key"
        )

    def test_prompt_includes_ocr_text(self, hf_extractor):
        """Test that generated prompt includes OCR text."""
        prompt = hf_extractor._build_prompt(SAMPLE_OCR_TEXT)
        assert "Via Roma 123" in prompt
        assert "€250,000" in prompt

    def test_prompt_includes_schema_instructions(self, hf_extractor):
        """Test that prompt includes JSON schema instructions."""
        prompt = hf_extractor._build_prompt(SAMPLE_OCR_TEXT)
        assert "JSON" in prompt.upper()
        assert "property" in prompt.lower()

    def test_prompt_includes_additional_context(self, hf_extractor):
        """Test that prompt includes additional context when provided."""
        context = {"portal_id": "PORTAL-123", "tribunal": "Milano"}
        prompt = hf_extractor._build_prompt(SAMPLE_OCR_TEXT, additional_context=context)

        assert "PORTAL-123" in prompt
        assert "Milano" in prompt


class TestErrorHandling:
    """Test error handling across all extractors."""

    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Test handling of network timeouts."""
        extractor = HuggingFaceExtractor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            api_key="test_key",
            timeout=0.001,  # Very short timeout
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await extractor.extract(SAMPLE_OCR_TEXT)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test handling of rate limit errors."""
        extractor = HuggingFaceExtractor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", api_key="test_key"
        )

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="Rate limit"):
                await extractor.extract(SAMPLE_OCR_TEXT, max_retries=1)

    @pytest.mark.asyncio
    async def test_empty_ocr_text(self):
        """Test handling of empty OCR text."""
        extractor = HuggingFaceExtractor(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", api_key="test_key"
        )

        with pytest.raises(ValueError, match="OCR text cannot be empty"):
            await extractor.extract("")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=ocular.services.llm_extraction_providers"])
