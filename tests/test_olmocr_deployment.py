"""
Test OLMocr model deployment to RunPod.

This test validates the deployment service's ability to deploy the OLMocr model
to RunPod serverless infrastructure.
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ocular.database.models import Deployment, Provider, Model, DeploymentStatus, ProviderStatus
from ocular.database.session import SessionLocal
from ocular.services.deployment_service import DeploymentService, get_deployment_service


@pytest.fixture(scope="function")
def db_session():
    """
    Create a test database session with isolated transaction.

    Uses a nested transaction that rolls back after each test,
    ensuring no test data pollutes the main database.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ocular.database.models import Base

    # Create in-memory SQLite database for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    TestSession = sessionmaker(bind=engine)
    db = TestSession()

    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def runpod_provider(db_session):
    """Create RunPod provider in test database."""
    provider = Provider(
        name="runpod",
        display_name="RunPod Serverless",
        status=ProviderStatus.ACTIVE,
        available_gpus=["RTX 4090", "RTX 4000 Ada", "A100 PCIe"],
        min_cost_per_hour=0.29,
        max_cost_per_hour=4.89,
        extra_metadata={
            "api_url": "https://api.runpod.io/graphql",
            "console_url": "https://console.runpod.io/serverless"
        }
    )
    db_session.add(provider)
    db_session.commit()
    db_session.refresh(provider)
    return provider


@pytest.fixture
def olmocr_model(db_session):
    """Create OLMocr model in test database."""
    model = Model(
        name="olmocr",
        display_name="OLMocr 7B",
        version="allenai/olmOCR-7B-0825",
        base_model="allenai/olmOCR-7B-0825",
        min_vram_gb=16,
        docker_image="ocular/olmocr-runpod:latest",
        supports_vllm=True,
        extra_metadata={
            "huggingface_repo": "allenai/olmOCR-7B-0825",
            "model_type": "vision-language",
            "recommended_gpu": "RTX 4090"
        }
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


@pytest.fixture
def test_deployment(db_session, runpod_provider, olmocr_model):
    """Create test deployment instance."""
    deployment = Deployment(
        deployment_id="test-olmocr-deployment-001",
        deployment_name="OLMocr Test Deployment",
        provider_id=runpod_provider.id,
        model_id=olmocr_model.id,
        status=DeploymentStatus.PENDING,
        gpu_type="RTX 4090",
        region="eur-is-1",
        min_workers=0,
        max_workers=3,
        cost_per_hour=0.39,
        timeout_seconds=600,
        api_key_hash="test_hash_placeholder",
        config={
            "container_disk_gb": 30,
            "volumes_in_gb": 30,
            "idle_timeout": 30,
            "env": {
                "RUNPOD_REGION": "eur-is-1",
                "MODEL_NAME": "allenai/olmOCR-7B-0825"
            }
        }
    )
    db_session.add(deployment)
    db_session.commit()
    db_session.refresh(deployment)
    return deployment


@pytest.fixture
def deployment_service():
    """Get deployment service instance."""
    return get_deployment_service()


@pytest.fixture
def mock_runpod_response():
    """Mock successful RunPod API response."""
    return {
        "data": {
            "createEndpoint": {
                "id": "mock-endpoint-id-12345",
                "name": "OLMocr Test Deployment",
                "status": "INITIALIZING"
            }
        }
    }


class TestOLMocrDeployment:
    """Test suite for OLMocr deployment to RunPod."""

    @pytest.mark.asyncio
    async def test_deploy_olmocr_to_runpod_success(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test successful OLMocr deployment to RunPod."""

        # Mock the HTTP request to RunPod API
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_runpod_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Execute deployment
            result = await deployment_service.deploy_to_runpod(
                deployment=test_deployment,
                api_key="test_api_key_placeholder"
            )

            # Verify result
            assert result["endpoint_id"] == "mock-endpoint-id-12345"
            assert result["endpoint_url"] == "https://api.runpod.ai/v2/mock-endpoint-id-12345/run"
            assert result["status"] == "INITIALIZING"

            # Verify deployment was updated
            assert test_deployment.endpoint_url == result["endpoint_url"]
            assert test_deployment.instance_id == "mock-endpoint-id-12345"
            assert test_deployment.status == DeploymentStatus.DEPLOYING
            assert test_deployment.started_at is not None

    @pytest.mark.asyncio
    async def test_deploy_olmocr_with_custom_docker_image(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test deployment with custom Docker image override."""

        custom_image = "myregistry/custom-olmocr:v2"

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_runpod_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await deployment_service.deploy_to_runpod(
                deployment=test_deployment,
                api_key="test_api_key",
                docker_image=custom_image
            )

            # Verify the custom image was used
            call_args = mock_post.call_args
            payload = call_args.kwargs['json']
            endpoint_config = payload['variables']['input']

            assert endpoint_config['docker_image'] == custom_image
            assert test_deployment.extra_metadata['docker_image'] == custom_image

    @pytest.mark.asyncio
    async def test_deploy_olmocr_with_gpu_override(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test deployment with GPU type override."""

        gpu_type = "A100 PCIe"

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_runpod_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await deployment_service.deploy_to_runpod(
                deployment=test_deployment,
                api_key="test_api_key",
                gpu_type=gpu_type
            )

            # Verify the GPU type was used
            call_args = mock_post.call_args
            payload = call_args.kwargs['json']
            endpoint_config = payload['variables']['input']

            assert endpoint_config['gpu_ids'] == gpu_type

    @pytest.mark.asyncio
    async def test_deploy_olmocr_api_error(
        self,
        deployment_service,
        test_deployment
    ):
        """Test handling of RunPod API errors."""

        error_response = {
            "errors": [
                {
                    "message": "Invalid API key",
                    "extensions": {"code": "UNAUTHENTICATED"}
                }
            ]
        }

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = error_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await deployment_service.deploy_to_runpod(
                    deployment=test_deployment,
                    api_key="invalid_key"
                )

            assert "RunPod API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_deploy_olmocr_configuration(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test that OLMocr-specific configuration is properly set."""

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_runpod_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await deployment_service.deploy_to_runpod(
                deployment=test_deployment,
                api_key="test_api_key"
            )

            # Verify RunPod endpoint configuration
            call_args = mock_post.call_args
            payload = call_args.kwargs['json']
            endpoint_config = payload['variables']['input']

            # Check endpoint name
            assert endpoint_config['name'] == "OLMocr Test Deployment"

            # Check worker configuration
            assert endpoint_config['min_workers'] == 0
            assert endpoint_config['max_workers'] == 3

            # Check timeout
            assert endpoint_config['execution_timeout'] == 600

            # Check GPU type
            assert endpoint_config['gpu_ids'] == "RTX 4090"

            # Check environment variables
            env = endpoint_config['env']
            assert env['MODEL_NAME'] == "allenai/olmOCR-7B-0825"
            assert env['RUNPOD_REGION'] == "eur-is-1"

            # Check disk configuration
            assert endpoint_config['container_disk_in_gb'] == 30
            assert endpoint_config['volumes_in_gb'] == 30

    @pytest.mark.asyncio
    async def test_deploy_olmocr_environment_variables(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test that RunPod S3 credentials are properly passed."""

        # Set environment variables
        with patch.dict(os.environ, {
            'RUNPOD_ACCESS_KEY_ID': 'user_test_key',
            'RUNPOD_SECRET_ACCESS_KEY': 'rps_test_secret',
            'RUNPOD_BUCKET_ENDPOINT_URL': 'https://s3api-eur-is-1.runpod.io/',
            'RUNPOD_NETWORK_VOLUME_ID': 'test-volume-id',
            'RUNPOD_REGION': 'eur-is-1'
        }):
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = mock_runpod_response
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                result = await deployment_service.deploy_to_runpod(
                    deployment=test_deployment,
                    api_key="test_api_key"
                )

                # Verify environment variables were passed
                call_args = mock_post.call_args
                payload = call_args.kwargs['json']
                env = payload['variables']['input']['env']

                assert env['RUNPOD_ACCESS_KEY_ID'] == 'user_test_key'
                assert env['RUNPOD_SECRET_ACCESS_KEY'] == 'rps_test_secret'
                assert env['RUNPOD_BUCKET_ENDPOINT_URL'] == 'https://s3api-eur-is-1.runpod.io/'
                assert env['RUNPOD_NETWORK_VOLUME_ID'] == 'test-volume-id'

    def test_deployment_service_singleton(self):
        """Test that deployment service uses singleton pattern."""
        service1 = get_deployment_service()
        service2 = get_deployment_service()

        assert service1 is service2

    @pytest.mark.asyncio
    async def test_deploy_method_routes_to_runpod(
        self,
        deployment_service,
        test_deployment,
        mock_runpod_response
    ):
        """Test that deploy() method correctly routes to RunPod provider."""

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_runpod_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Use the generic deploy method
            result = await deployment_service.deploy(
                deployment=test_deployment,
                api_key="test_api_key"
            )

            # Verify it called RunPod deployment
            assert result["endpoint_id"] == "mock-endpoint-id-12345"
            assert mock_post.called


@pytest.mark.integration
class TestOLMocrDeploymentIntegration:
    """Integration tests for OLMocr deployment (requires actual RunPod API key)."""

    @pytest.mark.skipif(
        not os.getenv("RUNPOD_API_KEY"),
        reason="RUNPOD_API_KEY not set - skipping integration test"
    )
    @pytest.mark.asyncio
    async def test_real_olmocr_deployment(
        self,
        deployment_service,
        test_deployment
    ):
        """
        Integration test: Deploy OLMocr to real RunPod endpoint.

        WARNING: This test will create a real deployment and may incur costs!
        Only run when RUNPOD_API_KEY is set and you intend to test real deployment.
        """
        api_key = os.getenv("RUNPOD_API_KEY")

        try:
            result = await deployment_service.deploy_to_runpod(
                deployment=test_deployment,
                api_key=api_key
            )

            # Verify real deployment
            assert "endpoint_id" in result
            assert "endpoint_url" in result
            assert result["endpoint_url"].startswith("https://api.runpod.ai/v2/")

            print(f"\n✓ Real deployment successful!")
            print(f"  Endpoint ID: {result['endpoint_id']}")
            print(f"  Endpoint URL: {result['endpoint_url']}")
            print(f"  Status: {result['status']}")
            print(f"\n⚠ Remember to delete this endpoint when done testing!")

        except Exception as e:
            pytest.fail(f"Real deployment failed: {e}")
