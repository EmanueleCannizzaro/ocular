"""Tests for OLMo OCR model via DeepInfra API using settings configuration."""
import pytest
from openai import OpenAI

from ocular.settings import get_settings


@pytest.fixture(scope="module")
def settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture(scope="module")
def deepinfra_config(settings):
    """Get DeepInfra configuration from settings."""
    olmocr = settings.olmocr

    # Check if DeepInfra is configured
    if not olmocr.deepinfra_api_key:
        pytest.skip("DEEPINFRA_TOKEN environment variable not set")

    return {
        "api_key": olmocr.deepinfra_api_key,
        "base_url": olmocr.deepinfra_base_url,
        "model": olmocr.deepinfra_model,
    }


@pytest.fixture(scope="module")
def openai_client(deepinfra_config):
    """Create OpenAI client configured for DeepInfra endpoint."""
    return OpenAI(
        api_key=deepinfra_config["api_key"],
        base_url=deepinfra_config["base_url"],
    )


@pytest.fixture(scope="module")
def model_name(deepinfra_config):
    """Get the model name from settings."""
    return deepinfra_config["model"]


def test_olmo_simple_completion(openai_client, model_name):
    """Test simple chat completion with OLMo model."""
    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello"}],
    )

    # Verify response structure
    assert chat_completion.choices, "Response should have choices"
    assert chat_completion.choices[0].message.content, "Response should have content"

    # Verify usage information
    assert chat_completion.usage.prompt_tokens > 0, "Should have prompt tokens"
    assert chat_completion.usage.completion_tokens > 0, "Should have completion tokens"

    # Print for manual verification
    print(f"\nResponse: {chat_completion.choices[0].message.content}")
    print(f"Tokens - Prompt: {chat_completion.usage.prompt_tokens}, Completion: {chat_completion.usage.completion_tokens}")


def test_olmo_conversation_with_system_prompt(openai_client, model_name):
    """Test multi-turn conversation with system prompt."""
    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Respond like a michelin starred chef."},
            {"role": "user", "content": "Can you name at least two different techniques to cook lamb?"},
            {
                "role": "assistant",
                "content": "Bonjour! Let me tell you, my friend, cooking lamb is an art form, and I'm more than happy to share with you not two, but three of my favorite techniques to coax out the rich, unctuous flavors and tender textures of this majestic protein. First, we have the classic \"Sous Vide\" method. Next, we have the ancient art of \"Sous le Sable\". And finally, we have the more modern technique of \"Hot Smoking.\""
            },
            {"role": "user", "content": "Tell me more about the second method."},
        ],
    )

    # Verify response structure
    assert chat_completion.choices, "Response should have choices"
    assert chat_completion.choices[0].message.content, "Response should have content"

    # Verify response is contextually relevant
    response_content = chat_completion.choices[0].message.content.lower()
    assert any(term in response_content for term in ["sous le sable", "sand", "technique", "lamb"]), \
        "Response should be contextually relevant to the question"

    # Verify usage information
    assert chat_completion.usage.prompt_tokens > 0, "Should have prompt tokens"
    assert chat_completion.usage.completion_tokens > 0, "Should have completion tokens"

    # Print for manual verification
    print(f"\nResponse: {chat_completion.choices[0].message.content}")
    print(f"Tokens - Prompt: {chat_completion.usage.prompt_tokens}, Completion: {chat_completion.usage.completion_tokens}")


def test_olmo_ocr_task(openai_client, model_name):
    """Test OCR-specific task with OLMo model."""
    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an OCR assistant. Extract text accurately from images."
            },
            {
                "role": "user",
                "content": "Extract the text from this image: [Sample OCR text would go here]"
            },
        ],
    )

    # Verify response
    assert chat_completion.choices, "Response should have choices"
    assert chat_completion.choices[0].message.content, "Response should have content"

    # Verify it's attempting OCR-related response
    response_content = chat_completion.choices[0].message.content.lower()
    assert len(response_content) > 10, "Response should have substantial content"

    print(f"\nOCR Response: {chat_completion.choices[0].message.content}")
    print(f"Tokens - Prompt: {chat_completion.usage.prompt_tokens}, Completion: {chat_completion.usage.completion_tokens}")


@pytest.mark.parametrize("test_message", [
    "What is 2+2?",
    "Tell me a short joke.",
    "Describe a sunset in one sentence.",
])
def test_olmo_various_prompts(openai_client, model_name, test_message):
    """Test various prompts to ensure model responds consistently."""
    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": test_message}],
    )

    # Basic assertions
    assert chat_completion.choices, "Response should have choices"
    assert chat_completion.choices[0].message.content, "Response should have content"
    assert len(chat_completion.choices[0].message.content) > 5, "Response should be meaningful"

    print(f"\nPrompt: {test_message}")
    print(f"Response: {chat_completion.choices[0].message.content[:100]}...")


def test_deepinfra_configuration_loaded(settings):
    """Test that DeepInfra configuration is properly loaded from settings."""
    olmocr = settings.olmocr

    # Check that DeepInfra fields exist
    assert hasattr(olmocr, "deepinfra_api_key"), "Should have deepinfra_api_key field"
    assert hasattr(olmocr, "deepinfra_model"), "Should have deepinfra_model field"
    assert hasattr(olmocr, "deepinfra_base_url"), "Should have deepinfra_base_url field"

    # Verify default values
    assert olmocr.deepinfra_base_url == "https://api.deepinfra.com/v1/openai", "Default base URL should be set"
    assert olmocr.deepinfra_model == "allenai/olmOCR-7B-0825", "Default model should be set"


def test_provider_selection_deepinfra(settings):
    """Test that provider can be set to deepinfra."""
    olmocr = settings.olmocr

    # When provider is set to deepinfra, active endpoints should use deepinfra config
    if olmocr.provider == "deepinfra" and olmocr.deepinfra_api_key:
        assert olmocr.active_api_key == olmocr.deepinfra_api_key, "Active API key should be DeepInfra key"
        assert olmocr.active_endpoint_url == olmocr.deepinfra_base_url, "Active endpoint should be DeepInfra URL"
        assert olmocr.is_configured, "Should be configured when API key is present"
