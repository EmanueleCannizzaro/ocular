# Developer Guide

Guide for developers who want to contribute to or extend Ocular OCR.

## Table of Contents

- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Contributing](#contributing)
- [Adding New Providers](#adding-new-providers)
- [Testing](#testing)
- [Code Style](#code-style)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ocular.git
cd ocular
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install in development mode**:
```bash
pip install -e ".[dev,test,torch]"
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. **Run tests to verify setup**:
```bash
pytest
```

### Development Dependencies

The `[dev]` extra includes:
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support

## Architecture Overview

Ocular follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Web UI     │  │  REST API   │  │  CLI Tools      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                    Service Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ OCR Service │  │ Doc Service │  │ Validation Svc  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │Cache Service│  │Processing   │                       │
│  └─────────────┘  │Service      │                       │
│                   └─────────────┘                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                    Provider Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Mistral   │  │    OLM      │  │     RoLM        │  │
│  │  Provider   │  │  Provider   │  │   Provider      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                     Core Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Models     │  │ Interfaces  │  │   Exceptions    │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │    Enums    │  │Configuration│                       │
│  └─────────────┘  └─────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### Key Modules

#### Core (`ocular.core`)
- **Models**: Pydantic models for data structures
- **Interfaces**: Abstract base classes defining contracts
- **Enums**: Enumeration types used throughout the system
- **Exceptions**: Custom exception hierarchy

#### Providers (`ocular.providers`)
- **Base**: Abstract base provider class
- **Factory**: Provider creation and management
- **Mistral**: Mistral AI OCR implementation
- **OLM**: Local OCR model implementation  
- **RoLM**: Robust OCR model implementation

#### Services (`ocular.services`)
- **OCRService**: Main orchestration service
- **DocumentService**: File handling and validation
- **ValidationService**: Input/output validation
- **CacheService**: Result caching
- **ProcessingService**: Job management

#### Configuration (`ocular.config`)
- **Settings**: Pydantic settings models
- **Manager**: Configuration management
- **Validators**: Configuration validation

#### Web (`ocular.web`)
- **App**: FastAPI application
- **Templates**: Jinja2 HTML templates
- **Static**: CSS, JS, and other static assets

## Contributing

### Getting Started

1. **Fork the repository** on GitHub
2. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** following the coding standards
4. **Write tests** for your changes
5. **Run the test suite**:
```bash
pytest
black .
flake8
mypy ocular/
```

6. **Commit your changes**:
```bash
git add .
git commit -m "Add feature: your feature description"
```

7. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

8. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Clear description**: Explain what your PR does and why
- **Tests included**: All new code should have tests
- **Documentation updated**: Update docs if you change APIs
- **Backward compatibility**: Don't break existing APIs without discussion
- **Single responsibility**: One feature/fix per PR

## Adding New Providers

To add a new OCR provider, follow these steps:

### 1. Create Provider Class

Create a new file in `ocular/providers/`:

```python
# ocular/providers/your_provider.py
from typing import Optional, Dict, Any
from pathlib import Path

from .base import BaseOCRProvider
from ..core.models import OCRResult
from ..core.enums import ProviderType
from ..core.exceptions import OCRError

class YourProvider(BaseOCRProvider):
    """Your OCR provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your provider-specific settings
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint", "https://api.yourprovider.com")
    
    @property
    def provider_type(self) -> ProviderType:
        # You'll need to add YOUR_PROVIDER to the ProviderType enum
        return ProviderType.YOUR_PROVIDER
    
    @property
    def provider_name(self) -> str:
        return "Your Provider Name"
    
    async def _do_initialize(self) -> None:
        """Initialize the provider."""
        # Perform any setup needed (validate API key, load models, etc.)
        if not self.api_key:
            raise ConfigurationError("API key required for Your Provider")
        
        # Test connection
        await self._test_connection()
    
    async def is_available(self) -> bool:
        """Check if provider is available."""
        try:
            await self._test_connection()
            return True
        except Exception:
            return False
    
    async def _extract_text_impl(
        self, 
        file_path: Path, 
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Implement text extraction."""
        try:
            # Your OCR implementation here
            # This is where you'd call your OCR API or run your model
            
            # Read the file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Process with your OCR service
            extracted_text = await self._process_with_your_service(
                file_content, prompt
            )
            
            # Return standardized result
            return OCRResult(
                text=extracted_text,
                confidence=0.9,  # If your service provides confidence
                language="en",   # If your service detects language
                provider=self.provider_name
            )
            
        except Exception as e:
            raise OCRError(f"Text extraction failed: {e}", provider=self.provider_name)
    
    async def _process_with_your_service(self, content: bytes, prompt: str) -> str:
        """Process content with your OCR service."""
        # Implement your actual OCR logic here
        # This might involve API calls, running local models, etc.
        pass
    
    async def _test_connection(self) -> None:
        """Test connection to your service."""
        # Implement a simple test to verify the service is reachable
        pass
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        return bool(self.api_key)
    
    def get_supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return ['.pdf', '.jpg', '.jpeg', '.png']
```

### 2. Add to ProviderType Enum

```python
# ocular/core/enums.py
class ProviderType(Enum):
    """Available OCR providers."""
    MISTRAL = "mistral"
    OLM_OCR = "olm_ocr" 
    ROLM_OCR = "rolm_ocr"
    YOUR_PROVIDER = "your_provider"  # Add this line
```

### 3. Register in Factory

```python
# ocular/providers/factory.py
from .your_provider import YourProvider

class ProviderFactory:
    _provider_registry: Dict[ProviderType, Type[BaseOCRProvider]] = {
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.OLM_OCR: OLMProvider,
        ProviderType.ROLM_OCR: RoLMProvider,
        ProviderType.YOUR_PROVIDER: YourProvider,  # Add this line
    }
```

### 4. Update Package Exports

```python
# ocular/providers/__init__.py
from .your_provider import YourProvider

__all__ = [
    "BaseOCRProvider",
    "ProviderFactory", 
    "MistralProvider",
    "OLMProvider",
    "RoLMProvider",
    "YourProvider",  # Add this line
]
```

### 5. Add Configuration Support

```python
# ocular/config/settings.py
class ProviderSettings(BaseSettings):
    # ... existing settings ...
    
    # Your provider settings
    your_provider_api_key: Optional[str] = Field(None, env="YOUR_PROVIDER_API_KEY")
    your_provider_endpoint: str = Field("https://api.yourprovider.com", env="YOUR_PROVIDER_ENDPOINT")
```

### 6. Write Tests

```python
# tests/test_providers/test_your_provider.py
import pytest
from unittest.mock import Mock, AsyncMock

from ocular.providers.your_provider import YourProvider
from ocular.core.enums import ProviderType

class TestYourProvider:
    def test_provider_properties(self):
        config = {"api_key": "test_key"}
        provider = YourProvider(config)
        
        assert provider.provider_type == ProviderType.YOUR_PROVIDER
        assert provider.provider_name == "Your Provider Name"
    
    @pytest.mark.asyncio
    async def test_extract_text(self):
        config = {"api_key": "test_key"}
        provider = YourProvider(config)
        
        # Mock your service calls
        with patch.object(provider, '_process_with_your_service') as mock_process:
            mock_process.return_value = "Extracted text"
            
            result = await provider.extract_text(Path("test.jpg"))
            
            assert result.text == "Extracted text"
            assert result.provider == "Your Provider Name"
```

### 7. Update Documentation

Add your provider to the documentation:
- Update `docs/api-reference.md`
- Update `docs/user-guide.md`
- Add configuration examples

## Testing

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── test_core/               # Core functionality tests
│   ├── test_models.py
│   ├── test_enums.py
│   └── test_exceptions.py
├── test_providers/          # Provider tests
│   ├── test_factory.py
│   ├── test_mistral.py
│   └── test_base.py
├── test_services/           # Service layer tests
│   ├── test_ocr_service.py
│   └── test_validation_service.py
└── test_web/               # Web interface tests
    └── test_app.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ocular

# Run specific test file
pytest tests/test_providers/test_factory.py

# Run with verbose output
pytest -v

# Run only async tests
pytest -m asyncio
```

### Writing Tests

#### Unit Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality."""
    service = OCRService(test_settings)
    result = await service.process_document(mock_request)
    assert result.status == ProcessingStatus.COMPLETED
```

#### Integration Tests
```python
def test_end_to_end_processing(sample_image_path, test_settings):
    """Test complete processing pipeline."""
    # Test actual file processing
    pass
```

#### Mock External Services
```python
@patch('ocular.providers.mistral.aiohttp.ClientSession')
async def test_mistral_api_call(mock_session):
    """Test API call without hitting real service."""
    # Configure mock response
    mock_response = Mock()
    mock_response.status = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
    
    mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
    
    # Test your code
    provider = MistralProvider(config)
    result = await provider.extract_text(file_path)
    assert result.text == "test"
```

## Code Style

### Formatting and Linting

We use several tools to maintain code quality:

```bash
# Format code
black .

# Check linting
flake8 ocular/ tests/

# Type checking
mypy ocular/

# All checks
black . && flake8 ocular/ tests/ && mypy ocular/
```

### Pre-commit Hooks

Set up pre-commit hooks to run checks automatically:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

### Style Guidelines

#### Python Code Style
- Follow PEP 8
- Use Black formatting (line length 88)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

#### Naming Conventions
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_CASE`
- Private methods: `_snake_case`

#### Documentation Style
```python
def process_document(file_path: Path, strategy: ProcessingStrategy) -> ProcessingResult:
    """Process a document using OCR.
    
    Args:
        file_path: Path to the document file
        strategy: Processing strategy to use
        
    Returns:
        ProcessingResult containing extracted text and metadata
        
    Raises:
        ValidationError: If file is invalid
        OCRError: If processing fails
    """
```

#### Import Organization
```python
# Standard library
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import pytest
from pydantic import BaseModel

# Local imports  
from ocular.core.models import OCRResult
from ocular.providers.base import BaseOCRProvider
```

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite**:
```bash
pytest --cov=ocular
black . && flake8 && mypy ocular/
```

4. **Create release branch**:
```bash
git checkout -b release/v0.2.0
```

5. **Commit changes**:
```bash
git add .
git commit -m "Bump version to 0.2.0"
```

6. **Create pull request** for release branch
7. **After merge, create tag**:
```bash
git tag v0.2.0
git push origin v0.2.0
```

8. **Create GitHub release** with release notes

### Automated Testing

Our CI/CD pipeline runs:
- Tests on Python 3.9, 3.10, 3.11
- Code formatting checks
- Type checking
- Security scanning
- Documentation builds

### Deployment

Production deployments use:
- Docker containers
- Environment-specific configuration
- Health checks and monitoring
- Gradual rollouts

## Debugging

### Logging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
OCULAR_LOG_LEVEL=DEBUG
```

### Common Debug Patterns

```python
# Service debugging
async def debug_processing():
    service = OCRService.from_env()
    
    # Enable detailed logging
    logging.getLogger('ocular').setLevel(logging.DEBUG)
    
    # Process with debugging
    try:
        result = await service.process_document(request)
        logger.debug(f"Processing successful: {result.to_dict()}")
    except Exception as e:
        logger.exception("Processing failed")
        raise

# Provider debugging
async def debug_provider():
    provider = MistralProvider(config)
    
    # Check availability
    is_available = await provider.is_available()
    logger.debug(f"Provider available: {is_available}")
    
    # Health check
    health = await provider.health_check()
    logger.debug(f"Health check: {health}")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_processing():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    asyncio.run(process_documents())
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support  
- **Code Review**: Submit PRs for feedback
- **Documentation**: Check existing docs first

## License and Copyright

By contributing to Ocular, you agree that your contributions will be licensed under the MIT License. Make sure any third-party code you include is compatible with this license.