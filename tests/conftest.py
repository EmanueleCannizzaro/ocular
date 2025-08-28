"""
Pytest configuration and fixtures for Ocular OCR tests.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from ocular.core.models import OCRResult
from ocular.core.enums import ProviderType
from ocular.settings import OcularSettings
from ocular.providers.factory import ProviderFactory


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file for testing."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (100, 50), color='white')
    image_path = temp_dir / "test_image.jpg"
    img.save(image_path, "JPEG")
    
    return image_path


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF file for testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        pdf_path = temp_dir / "test_document.pdf"
        
        # Create a simple PDF with some text
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 700, "This is a sample PDF for OCR testing.")
        c.save()
        
        return pdf_path
    except ImportError:
        # Fallback: create a minimal PDF manually
        pdf_path = temp_dir / "test_document.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
187
%%EOF"""
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)
        
        return pdf_path


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings."""
    settings = OcularSettings(
        environment="testing",
        providers__mistral_api_key="test_key",
        providers__enabled_providers=["mistral"],
        files__max_file_size_mb=10,
        files__temp_dir=temp_dir / "temp",
        files__upload_dir=temp_dir / "uploads",
        processing__enable_caching=False,
        web__debug=True
    )
    
    # Create directories
    settings.files.temp_dir.mkdir(exist_ok=True)
    settings.files.upload_dir.mkdir(exist_ok=True)
    
    return settings


@pytest.fixture
def mock_ocr_result():
    """Create a mock OCR result."""
    return OCRResult(
        text="Sample extracted text",
        confidence=0.95,
        language="en",
        provider="test_provider",
        processing_time=1.5,
        metadata={"test": "data"}
    )


@pytest.fixture
def mock_mistral_provider():
    """Create a mock Mistral provider."""
    provider = Mock()
    provider.provider_type = ProviderType.MISTRAL
    provider.provider_name = "Mock Mistral"
    provider.is_available = AsyncMock(return_value=True)
    provider.initialize = AsyncMock()
    provider.cleanup = AsyncMock()
    provider.extract_text = AsyncMock(return_value=OCRResult(
        text="Mocked text from Mistral",
        confidence=0.95,
        language="en",
        provider="Mock Mistral"
    ))
    provider.extract_structured_data = AsyncMock(return_value={
        "extracted_text": "Mocked structured data",
        "confidence": 0.95
    })
    
    return provider


@pytest.fixture
def provider_configs():
    """Create provider configurations for testing."""
    return {
        ProviderType.MISTRAL: {
            "api_key": "test_key",
            "model": "test_model",
            "timeout": 30,
            "max_retries": 3
        },
        ProviderType.OLM_OCR: {
            "model_name": "test_model",
            "device": "cpu"
        },
        ProviderType.ROLM_OCR: {
            "model_name": "test_model",
            "device": "cpu"
        }
    }


@pytest.fixture
async def mock_cache_service():
    """Create a mock cache service."""
    cache = Mock()
    cache.get_cached_result = AsyncMock(return_value=None)
    cache.cache_result = AsyncMock()
    cache.invalidate_cache = AsyncMock()
    cache.clear_cache = AsyncMock()
    cache.generate_cache_key = Mock(return_value="test_cache_key")
    
    return cache


@pytest.fixture
def large_file_path(temp_dir):
    """Create a large file for size limit testing."""
    large_file = temp_dir / "large_file.txt"
    
    # Create a file larger than typical limits (20MB)
    with open(large_file, 'wb') as f:
        f.write(b'x' * (20 * 1024 * 1024))
    
    return large_file


@pytest.fixture
def invalid_file_path(temp_dir):
    """Create an invalid file for testing."""
    return temp_dir / "non_existent_file.txt"


@pytest.fixture
def unsupported_file_path(temp_dir):
    """Create a file with unsupported extension."""
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("unsupported content")
    
    return unsupported_file


@pytest.fixture
async def mock_provider_factory():
    """Create a mock provider factory."""
    factory = Mock()
    factory.create_provider = Mock()
    factory.create_and_initialize_provider = AsyncMock()
    factory.get_available_providers = AsyncMock(return_value=[ProviderType.MISTRAL])
    factory.validate_provider_configs = AsyncMock(return_value={ProviderType.MISTRAL: True})
    factory.health_check_all = AsyncMock(return_value={
        "mistral": {"available": True, "provider": "Mistral AI"}
    })
    
    return factory


class AsyncContextManagerMock:
    """Mock async context manager for testing."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    session = Mock()
    
    # Mock successful API response
    response = Mock()
    response.status = 200
    response.json = AsyncMock(return_value={
        "choices": [{"message": {"content": "Mocked OCR result"}}]
    })
    
    session.post = Mock(return_value=AsyncContextManagerMock(response))
    session.get = Mock(return_value=AsyncContextManagerMock(response))
    
    return session


@pytest.mark.asyncio
async def pytest_configure(config):
    """Configure pytest for async testing."""
    pytest.test_mode = True


# Utility functions for tests

def create_test_file(path: Path, content: bytes = b"test content", file_type: str = "txt"):
    """Create a test file with specified content."""
    if file_type == "image":
        from PIL import Image
        img = Image.new('RGB', (100, 50), color='white')
        img.save(path, "JPEG")
    elif file_type == "pdf":
        # Simple PDF header
        path.write_bytes(b"%PDF-1.4\n%test pdf content")
    else:
        path.write_bytes(content)
    
    return path


def assert_ocr_result_valid(result: OCRResult):
    """Assert that OCR result has valid structure."""
    assert isinstance(result.text, str)
    assert result.provider is not None
    assert result.confidence is None or (0.0 <= result.confidence <= 1.0)
    assert result.processing_time is None or result.processing_time >= 0.0