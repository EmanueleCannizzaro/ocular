
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from pathlib import Path

from ocular.client import MistralOCRClient
from ocular.exceptions import APIError, OCRError, ConfigurationError
from ocular.models import RootModel
from ocular.config import OcularConfig


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"}):
        yield


@pytest.fixture
def mock_schema_file():
    schema_data = RootModel.model_json_schema()
    with patch("builtins.open", mock_open(read_data=json.dumps(schema_data))) as m:
        yield m


@pytest.fixture
def mock_validate_file_size():
    with patch('ocular.client.MistralOCRClient._validate_file_size') as mock_method:
        yield mock_method


class TestMistralOCRClient:

    def test_init_raises_value_error_if_api_key_is_missing(self, mock_schema_file):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="MISTRAL_API_KEY environment variable is required"):
                MistralOCRClient()

    def test_init_raises_file_not_found_error_if_schema_is_missing(self, mock_env):
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                MistralOCRClient()

    @patch('ocular.client.Mistral')
    def test_init_successful(self, mock_mistral, mock_env, mock_schema_file):
        client = MistralOCRClient()
        assert client.client is not None
        assert client.schema is not None
        mock_mistral.assert_called_with(api_key="test_key")

    @pytest.mark.asyncio
    @patch('ocular.client.Mistral')
    async def test_extract_structured_data_success(self, mock_mistral, mock_env, mock_schema_file, mock_validate_file_size):
        # Arrange
        mock_ocr_response = MagicMock()
        mock_ocr_response.parsed.model_dump_json.return_value = '{"invoices": [{"invoice_number": "123"}]}'
        
        mock_mistral_instance = MagicMock()
        mock_mistral_instance.ocr.process.return_value = mock_ocr_response
        mock_mistral.return_value = mock_mistral_instance

        client = MistralOCRClient()

        file_path = Path("dummy.pdf")
        file_content = b"dummy content"

        m = mock_open(read_data=file_content)
        with patch("builtins.open", m):
            # Act
            result = await client.extract_structured_data(file_path, schema={})

            # Assert
            m.assert_called_once_with(file_path, 'rb')
            mock_mistral_instance.ocr.process.assert_called_once()
            assert result == '{"invoices": [{"invoice_number": "123"}]}'

    @pytest.mark.asyncio
    @patch('ocular.client.Mistral')
    async def test_extract_structured_data_file_not_found(self, mock_mistral, mock_env, mock_schema_file, mock_validate_file_size):
        client = MistralOCRClient()
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                await client.extract_structured_data(Path("non_existent.pdf"), schema={})

    @pytest.mark.asyncio
    @patch('ocular.client.Mistral')
    async def test_extract_structured_data_api_error(self, mock_mistral, mock_env, mock_schema_file, mock_validate_file_size):
        mock_mistral_instance = MagicMock()
        mock_mistral_instance.ocr.process.side_effect = Exception("api error")
        mock_mistral.return_value = mock_mistral_instance

        client = MistralOCRClient()
        
        file_path = Path("dummy.pdf")
        file_content = b"dummy content"
        
        m = mock_open(read_data=file_content)
        with patch("builtins.open", m):
            with pytest.raises(APIError, match="Mistral API error: api error"):
                await client.extract_structured_data(file_path, schema={})
