"""
Tests for provider factory.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from ocular.providers.factory import ProviderFactory, ProviderManager
from ocular.providers.mistral import MistralProvider
from ocular.core.enums import ProviderType
from ocular.core.exceptions import ProviderError, ConfigurationError


class TestProviderFactory:
    """Test ProviderFactory class."""
    
    def test_create_mistral_provider(self):
        """Test creating Mistral provider."""
        config = {
            "api_key": "test_key",
            "model": "test_model",
            "timeout": 30
        }
        
        provider = ProviderFactory.create_provider(ProviderType.MISTRAL, config)
        
        assert isinstance(provider, MistralProvider)
        assert provider.api_key == "test_key"
        assert provider.model == "test_model"
        assert provider.timeout == 30
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        # Mock an unknown provider type
        unknown_type = Mock()
        unknown_type.value = "unknown"
        
        with pytest.raises(ProviderError, match="Unknown provider type"):
            ProviderFactory.create_provider(unknown_type, {})
    
    def test_create_provider_with_invalid_config(self):
        """Test creating provider with config that causes initialization error."""
        config = {"api_key": None}  # Invalid config
        
        # Should create provider but may fail during initialization
        provider = ProviderFactory.create_provider(ProviderType.MISTRAL, config)
        assert isinstance(provider, MistralProvider)
    
    @pytest.mark.asyncio
    async def test_create_and_initialize_provider(self):
        """Test creating and initializing provider."""
        config = {"api_key": "test_key"}
        
        with patch.object(MistralProvider, 'initialize', new_callable=AsyncMock) as mock_init:
            provider = await ProviderFactory.create_and_initialize_provider(
                ProviderType.MISTRAL, config
            )
            
            mock_init.assert_called_once()
            assert isinstance(provider, MistralProvider)
    
    @pytest.mark.asyncio
    async def test_create_and_initialize_provider_failure(self):
        """Test handling initialization failure."""
        config = {"api_key": "test_key"}
        
        with patch.object(MistralProvider, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(MistralProvider, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
                mock_init.side_effect = ConfigurationError("Init failed")
                
                with pytest.raises(ProviderError, match="Failed to initialize provider"):
                    await ProviderFactory.create_and_initialize_provider(
                        ProviderType.MISTRAL, config
                    )
                
                mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_available_providers(self, provider_configs):
        """Test getting available providers."""
        with patch.object(ProviderFactory, 'create_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.is_available = AsyncMock(return_value=True)
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            available = await ProviderFactory.get_available_providers(provider_configs)
            
            assert len(available) > 0
            assert all(isinstance(p, ProviderType) for p in available)
    
    @pytest.mark.asyncio
    async def test_validate_provider_configs(self, provider_configs):
        """Test validating provider configurations."""
        with patch.object(ProviderFactory, 'create_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.validate_config.return_value = True
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            results = await ProviderFactory.validate_provider_configs(provider_configs)
            
            assert isinstance(results, dict)
            assert len(results) == len(provider_configs)
            assert all(isinstance(k, ProviderType) for k in results.keys())
            assert all(isinstance(v, bool) for v in results.values())
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, provider_configs):
        """Test health check for all providers."""
        with patch.object(ProviderFactory, 'create_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.health_check = AsyncMock(return_value={
                "provider": "test",
                "available": True
            })
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            results = await ProviderFactory.health_check_all(provider_configs)
            
            assert isinstance(results, dict)
            assert len(results) == len(provider_configs)
    
    def test_register_custom_provider(self):
        """Test registering custom provider."""
        from ocular.providers.base import BaseOCRProvider
        
        class CustomProvider(BaseOCRProvider):
            @property
            def provider_type(self):
                return ProviderType.MISTRAL  # Reuse existing type for test
            
            @property
            def provider_name(self):
                return "Custom"
            
            async def _do_initialize(self):
                pass
            
            async def is_available(self):
                return True
            
            async def _extract_text_impl(self, file_path, prompt=None, **kwargs):
                pass
        
        # Register custom provider
        ProviderFactory.register_provider(ProviderType.MISTRAL, CustomProvider)
        
        # Create should use custom provider
        provider = ProviderFactory.create_provider(ProviderType.MISTRAL, {})
        assert isinstance(provider, CustomProvider)
        
        # Cleanup - restore original
        ProviderFactory.register_provider(ProviderType.MISTRAL, MistralProvider)
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        info = ProviderFactory.get_provider_info(ProviderType.MISTRAL)
        
        assert isinstance(info, dict)
        assert "type" in info
        assert "name" in info
        assert "class" in info
        assert "supported_formats" in info
        assert info["type"] == ProviderType.MISTRAL.value
    
    def test_create_provider_from_name(self):
        """Test creating provider from string name."""
        config = {"api_key": "test"}
        
        # Test various name formats
        provider1 = ProviderFactory.create_provider_from_name("mistral", config)
        provider2 = ProviderFactory.create_provider_from_name("mistral_ai", config)
        
        assert isinstance(provider1, MistralProvider)
        assert isinstance(provider2, MistralProvider)
        
        # Test unknown name
        with pytest.raises(ProviderError, match="Unknown provider name"):
            ProviderFactory.create_provider_from_name("unknown", config)


class TestProviderManager:
    """Test ProviderManager class."""
    
    @pytest.mark.asyncio
    async def test_get_provider(self):
        """Test getting provider instance."""
        config = {"api_key": "test_key"}
        
        with patch.object(ProviderFactory, 'create_and_initialize_provider') as mock_create:
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            
            async with ProviderManager() as manager:
                provider = await manager.get_provider(ProviderType.MISTRAL, config)
                
                assert provider == mock_provider
                mock_create.assert_called_once_with(ProviderType.MISTRAL, config)
    
    @pytest.mark.asyncio
    async def test_get_provider_cached(self):
        """Test that provider instances are cached."""
        config = {"api_key": "test_key"}
        
        with patch.object(ProviderFactory, 'create_and_initialize_provider') as mock_create:
            mock_provider = Mock()
            mock_create.return_value = mock_provider
            
            async with ProviderManager() as manager:
                provider1 = await manager.get_provider(ProviderType.MISTRAL, config)
                provider2 = await manager.get_provider(ProviderType.MISTRAL, config)
                
                assert provider1 == provider2
                # Should only create once due to caching
                mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_provider(self):
        """Test cleaning up specific provider."""
        config = {"api_key": "test_key"}
        
        with patch.object(ProviderFactory, 'create_and_initialize_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            async with ProviderManager() as manager:
                await manager.get_provider(ProviderType.MISTRAL, config)
                await manager.cleanup_provider(ProviderType.MISTRAL)
                
                mock_provider.cleanup.assert_called_once()
                
                # Provider should be removed from cache
                active = manager.get_active_providers()
                assert ProviderType.MISTRAL not in active
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all providers."""
        config = {"api_key": "test_key"}
        
        with patch.object(ProviderFactory, 'create_and_initialize_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            manager = ProviderManager()
            await manager.get_provider(ProviderType.MISTRAL, config)
            await manager.cleanup_all()
            
            mock_provider.cleanup.assert_called_once()
            
            # All providers should be removed
            active = manager.get_active_providers()
            assert len(active) == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test provider manager as context manager."""
        config = {"api_key": "test_key"}
        
        with patch.object(ProviderFactory, 'create_and_initialize_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.cleanup = AsyncMock()
            mock_create.return_value = mock_provider
            
            async with ProviderManager() as manager:
                await manager.get_provider(ProviderType.MISTRAL, config)
            
            # Cleanup should be called automatically
            mock_provider.cleanup.assert_called_once()