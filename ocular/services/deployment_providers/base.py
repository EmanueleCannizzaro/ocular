"""
Base deployment provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class DeploymentConfig:
    """Configuration for a deployment."""

    def __init__(
        self,
        model_id: int,
        provider_id: int,
        deployment_name: str,
        gpu_type: str,
        gpu_count: int = 1,
        max_workers: int = 1,
        min_workers: int = 0,
        timeout_seconds: int = 300,
        autoscaling_enabled: bool = False,
        region: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.provider_id = provider_id
        self.deployment_name = deployment_name
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.timeout_seconds = timeout_seconds
        self.autoscaling_enabled = autoscaling_enabled
        self.region = region
        self.extra_config = extra_config or {}


class DeploymentResult:
    """Result of a deployment operation."""

    def __init__(
        self,
        success: bool,
        instance_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        status: str = "pending",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.instance_id = instance_id
        self.endpoint_url = endpoint_url
        self.status = status
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class BaseDeploymentProvider(ABC):
    """
    Abstract base class for deployment providers.

    Each cloud GPU provider (RunPod, Vast.ai, etc.) should implement this interface.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the deployment provider.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def create_deployment(
        self,
        config: DeploymentConfig,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any]
    ) -> DeploymentResult:
        """
        Create a new deployment instance.

        Args:
            config: Deployment configuration
            model_config: Model-specific configuration (docker_image, etc.)
            provider_config: Provider-specific configuration

        Returns:
            DeploymentResult with instance details
        """
        pass

    @abstractmethod
    async def get_deployment_status(self, instance_id: str) -> DeploymentResult:
        """
        Get the current status of a deployment.

        Args:
            instance_id: Provider's instance/pod ID

        Returns:
            DeploymentResult with current status
        """
        pass

    @abstractmethod
    async def stop_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Stop a running deployment.

        Args:
            instance_id: Provider's instance/pod ID

        Returns:
            DeploymentResult with operation status
        """
        pass

    @abstractmethod
    async def delete_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Delete a deployment instance.

        Args:
            instance_id: Provider's instance/pod ID

        Returns:
            DeploymentResult with operation status
        """
        pass

    @abstractmethod
    async def get_logs(
        self,
        instance_id: str,
        lines: int = 100
    ) -> Dict[str, Any]:
        """
        Retrieve logs from a deployment instance.

        Args:
            instance_id: Provider's instance/pod ID
            lines: Number of log lines to retrieve

        Returns:
            Dict with logs and metadata
        """
        pass

    @abstractmethod
    async def estimate_cost(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        hours: float = 1.0
    ) -> Dict[str, Any]:
        """
        Estimate cost for a deployment configuration.

        Args:
            gpu_type: Type of GPU
            gpu_count: Number of GPUs
            hours: Estimated runtime in hours

        Returns:
            Dict with cost estimate and breakdown
        """
        pass
