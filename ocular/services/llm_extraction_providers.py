"""
LLM Extraction Providers

This module provides multiple open-source and free LLM options for converting
unstructured OCR text into structured AuctionPropertyProfile JSON.

Supported providers:
- Anthropic Claude (paid, high quality)
- Mistral-7B-Instruct (open-source, free via HF)
- Phi-3-mini (open-source, free via HF or local)
- Llama-3-8B-Instruct (open-source, free via HF)
- OpenAI GPT-4o-mini (paid, low cost)
- Local LLMs via Ollama
"""

import json
import os
from typing import Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

from pydantic import ValidationError

from ..core.logging import get_logger
from ..core.exceptions import ValidationError as OcularValidationError
from ..models.property_assessment import AuctionPropertyProfile


logger = get_logger(__name__)


# LLM Provider Types
LLMProvider = Literal[
    "anthropic",
    "mistral-7b",
    "phi-3-mini",
    "llama-3-8b",
    "openai",
    "ollama"
]


class BaseLLMExtractor(ABC):
    """Base class for LLM-based extraction."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.logger = logger
        self.config = kwargs

    @abstractmethod
    async def extract(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """Extract structured data from OCR text."""
        pass

    def _build_extraction_prompt(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the extraction prompt."""
        schema = AuctionPropertyProfile.model_json_schema()

        prompt = f"""Extract Italian property auction information from this appraisal document and convert it to JSON.

**Document Text:**
{ocr_text}

**Required JSON Schema:**
{json.dumps(schema, indent=2, ensure_ascii=False)}

**Extraction Guidelines:**
1. Extract all dates in format "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
2. Extract prices as numbers (floats) without currency symbols
3. Provide complete addresses with street, number, city, province
4. Extract all cadastral data (Foglio, Particella, Subalterno, Categoria, Rendita)
5. Identify all properties in the lot (apartments, garages, land, etc.)
6. Extract all liens and encumbrances (ipoteche, pignoramenti) with dates and amounts
7. List all compliance issues mentioned
8. Use null for optional fields if not available

**Italian Terms:**
- Tribunale = Court
- R.G. = Registro Generale (procedure number)
- CTU = Expert Appraiser
- Giudice dell'Esecuzione = Execution Judge
- Foglio/Particella = Cadastral identifiers
- Rendita Catastale = Cadastral Revenue
- Ipoteca = Mortgage
- Pignoramento = Seizure
- Base d'asta = Auction Base Price
"""

        if additional_context:
            prompt += f"\n\n**Additional Context from Portal:**\n{json.dumps(additional_context, indent=2, ensure_ascii=False)}"

        prompt += "\n\nProvide ONLY the JSON output, no explanation:"

        return prompt


class AnthropicExtractor(BaseLLMExtractor):
    """Extraction using Anthropic Claude (Pydantic AI)."""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(model_name, **kwargs)

        from pydantic_ai import Agent

        self.agent = Agent(
            f'anthropic:{model_name}',
            result_type=AuctionPropertyProfile,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for Anthropic."""
        return """You are an expert in Italian real estate auction documents and property appraisals.
Extract structured information from Italian property auction appraisal documents (perizie tecniche)
and convert them into structured JSON following the AuctionPropertyProfile schema.
Be precise with numbers, dates, and Italian terminology. Use null for missing optional fields."""

    async def extract(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """Extract using Pydantic AI + Claude."""
        prompt = self._build_extraction_prompt(ocr_text, additional_context)

        try:
            result = await self.agent.run(prompt)
            return result.data
        except Exception as e:
            raise OcularValidationError(f"Anthropic extraction failed: {e}")


class HuggingFaceExtractor(BaseLLMExtractor):
    """Extraction using Hugging Face Inference API (free tier available)."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        self.api_key = os.getenv("HUGGINGFACE_API_KEY") or kwargs.get("api_key")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    async def extract(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """Extract using HuggingFace Inference API."""
        import aiohttp

        prompt = self._build_extraction_prompt(ocr_text, additional_context)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 4000,
                "temperature": 0.1,
                "return_full_text": False
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise OcularValidationError(f"HuggingFace API error: {error}")

                    result = await response.json()

                    # Extract generated text
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = result.get("generated_text", "")

                    # Parse JSON from response
                    json_data = self._extract_json_from_text(generated_text)

                    # Validate with Pydantic
                    return AuctionPropertyProfile(**json_data)

        except ValidationError as e:
            raise OcularValidationError(f"Validation failed: {e}")
        except Exception as e:
            raise OcularValidationError(f"HuggingFace extraction failed: {e}")

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model output."""
        # Try to find JSON in the text
        import re

        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)

        # If no JSON found, try parsing the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise OcularValidationError("Could not extract valid JSON from model output")


class OllamaExtractor(BaseLLMExtractor):
    """Extraction using local Ollama models."""

    def __init__(self, model_name: str = "mistral:7b-instruct", **kwargs):
        super().__init__(model_name, **kwargs)

        self.base_url = kwargs.get("base_url", "http://localhost:11434")

    async def extract(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """Extract using Ollama."""
        import aiohttp

        prompt = self._build_extraction_prompt(ocr_text, additional_context)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4000
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise OcularValidationError(f"Ollama API error: {error}")

                    result = await response.json()
                    generated_text = result.get("response", "")

                    # Extract JSON from response
                    json_data = self._extract_json_from_text(generated_text)

                    # Validate with Pydantic
                    return AuctionPropertyProfile(**json_data)

        except ValidationError as e:
            raise OcularValidationError(f"Validation failed: {e}")
        except Exception as e:
            raise OcularValidationError(f"Ollama extraction failed: {e}")

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model output."""
        import re

        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)

        # Try parsing the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise OcularValidationError("Could not extract valid JSON from model output")


class OpenAIExtractor(BaseLLMExtractor):
    """Extraction using OpenAI GPT models."""

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)

        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY") or kwargs.get("api_key")
        self.client = AsyncOpenAI(api_key=api_key)

    async def extract(
        self,
        ocr_text: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> AuctionPropertyProfile:
        """Extract using OpenAI."""
        prompt = self._build_extraction_prompt(ocr_text, additional_context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in Italian real estate documents. Extract data precisely as JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )

            json_text = response.choices[0].message.content
            json_data = json.loads(json_text)

            # Validate with Pydantic
            return AuctionPropertyProfile(**json_data)

        except ValidationError as e:
            raise OcularValidationError(f"Validation failed: {e}")
        except Exception as e:
            raise OcularValidationError(f"OpenAI extraction failed: {e}")


# Factory function to create extractors
def create_llm_extractor(
    provider: LLMProvider = "anthropic",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLMExtractor:
    """
    Create an LLM extractor based on provider type.

    Args:
        provider: LLM provider type
        model_name: Optional specific model name
        **kwargs: Additional configuration

    Returns:
        Configured LLM extractor

    Examples:
        # Anthropic (paid, best quality)
        extractor = create_llm_extractor("anthropic")

        # Mistral-7B-Instruct (free via HuggingFace)
        extractor = create_llm_extractor("mistral-7b")

        # Phi-3-mini (free via HuggingFace)
        extractor = create_llm_extractor("phi-3-mini")

        # Llama-3-8B (free via HuggingFace)
        extractor = create_llm_extractor("llama-3-8b")

        # Local Ollama
        extractor = create_llm_extractor("ollama", model_name="mistral:7b-instruct")

        # OpenAI (paid, low cost)
        extractor = create_llm_extractor("openai", model_name="gpt-4o-mini")
    """
    extractors = {
        "anthropic": lambda: AnthropicExtractor(
            model_name=model_name or "claude-sonnet-4-20250514",
            **kwargs
        ),
        "mistral-7b": lambda: HuggingFaceExtractor(
            model_name=model_name or "mistralai/Mistral-7B-Instruct-v0.3",
            **kwargs
        ),
        "phi-3-mini": lambda: HuggingFaceExtractor(
            model_name=model_name or "microsoft/Phi-3-mini-4k-instruct",
            **kwargs
        ),
        "llama-3-8b": lambda: HuggingFaceExtractor(
            model_name=model_name or "meta-llama/Meta-Llama-3-8B-Instruct",
            **kwargs
        ),
        "openai": lambda: OpenAIExtractor(
            model_name=model_name or "gpt-4o-mini",
            **kwargs
        ),
        "ollama": lambda: OllamaExtractor(
            model_name=model_name or "mistral:7b-instruct",
            **kwargs
        )
    }

    if provider not in extractors:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Valid options: {list(extractors.keys())}"
        )

    return extractors[provider]()


# Recommended free/open-source configurations
RECOMMENDED_FREE_MODELS = {
    "mistral-7b-instruct": {
        "provider": "mistral-7b",
        "description": "Mistral 7B Instruct v0.3 - High quality, free via HuggingFace",
        "cost": "Free (HF Inference API)",
        "quality": "⭐⭐⭐⭐",
        "speed": "Medium",
        "requires_api_key": False
    },
    "phi-3-mini": {
        "provider": "phi-3-mini",
        "description": "Microsoft Phi-3 Mini - Lightweight, fast, free",
        "cost": "Free (HF Inference API or local)",
        "quality": "⭐⭐⭐",
        "speed": "Fast",
        "requires_api_key": False
    },
    "llama-3-8b": {
        "provider": "llama-3-8b",
        "description": "Meta Llama 3 8B Instruct - High quality, free",
        "cost": "Free (HF Inference API)",
        "quality": "⭐⭐⭐⭐",
        "speed": "Medium",
        "requires_api_key": False
    },
    "ollama-mistral": {
        "provider": "ollama",
        "model_name": "mistral:7b-instruct",
        "description": "Mistral 7B via Ollama - Run locally, 100% free",
        "cost": "Free (local)",
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast (depends on hardware)",
        "requires_api_key": False,
        "requires_setup": "Install Ollama: https://ollama.ai"
    },
    "anthropic-claude": {
        "provider": "anthropic",
        "description": "Anthropic Claude Sonnet - Best quality, paid",
        "cost": "$3 per 1M input tokens",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Fast",
        "requires_api_key": True
    },
    "openai-gpt4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini - Low cost, good quality",
        "cost": "$0.15 per 1M input tokens",
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast",
        "requires_api_key": True
    }
}
