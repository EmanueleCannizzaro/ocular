"""
Replicate Deployment Provider.

Provides deployment capabilities for Replicate ML platform.
"""

import httpx
from typing import Dict, Any, Optional, List
from .base import BaseDeploymentProvider, DeploymentConfig, DeploymentResult


class ReplicateProvider(BaseDeploymentProvider):
    """
    Replicate deployment provider implementation.

    Replicate offers hosted ML model deployment with automatic scaling.
    """

    BASE_URL = "https://api.replicate.com/v1"

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize Replicate provider.

        Args:
            api_key: Replicate API token
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

    async def create_deployment(
        self,
        config: DeploymentConfig,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any]
    ) -> DeploymentResult:
        """
        Create a deployment on Replicate.

        Note: Replicate uses a different model - you deploy a model version,
        not a traditional instance.

        Args:
            config: Deployment configuration
            model_config: Model-specific configuration
            provider_config: Provider-specific configuration

        Returns:
            DeploymentResult with deployment details
        """
        try:
            # Get model owner and name from config
            model_ref = model_config.get("replicate_model")
            if not model_ref:
                return DeploymentResult(
                    success=False,
                    error_message="replicate_model not specified in model_config"
                )

            # For Replicate, "deployment" means creating a model if it doesn't exist
            # or using an existing one
            owner, name = model_ref.split("/") if "/" in model_ref else ("", model_ref)

            async with httpx.AsyncClient() as client:
                # Check if model exists
                try:
                    response = await client.get(
                        f"{self.BASE_URL}/models/{owner}/{name}",
                        headers=self.headers,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    model = response.json()
                    model_exists = True
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        model_exists = False
                    else:
                        raise

                # If model doesn't exist, create it
                if not model_exists:
                    create_model_data = {
                        "owner": owner,
                        "name": name,
                        "description": model_config.get("description", ""),
                        "visibility": model_config.get("visibility", "private"),
                        "hardware": self._map_gpu_to_hardware(config.gpu_type)
                    }

                    response = await client.post(
                        f"{self.BASE_URL}/models",
                        json=create_model_data,
                        headers=self.headers,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    model = response.json()

                # Model URL serves as "endpoint"
                model_url = f"https://replicate.com/{owner}/{name}"

                return DeploymentResult(
                    success=True,
                    instance_id=f"{owner}/{name}",
                    endpoint_url=model_url,
                    status="running",
                    metadata={
                        "provider": "replicate",
                        "model_owner": owner,
                        "model_name": name,
                        "visibility": model.get("visibility"),
                        "hardware": model.get("default_example", {}).get("hardware")
                    }
                )

        except httpx.HTTPError as e:
            return DeploymentResult(
                success=False,
                error_message=f"HTTP error: {str(e)}"
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to create deployment: {str(e)}"
            )

    async def get_deployment_status(self, instance_id: str) -> DeploymentResult:
        """
        Get status of a Replicate model.

        Args:
            instance_id: Model identifier (owner/name)

        Returns:
            DeploymentResult with current status
        """
        try:
            owner, name = instance_id.split("/")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/models/{owner}/{name}",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                model = response.json()

                model_url = f"https://replicate.com/{owner}/{name}"

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    endpoint_url=model_url,
                    status="running",  # Replicate models are always "running" if they exist
                    metadata={
                        "provider": "replicate",
                        "visibility": model.get("visibility"),
                        "run_count": model.get("run_count", 0),
                        "latest_version": model.get("latest_version", {}).get("id")
                    }
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return DeploymentResult(
                    success=False,
                    error_message=f"Model {instance_id} not found"
                )
            return DeploymentResult(
                success=False,
                error_message=f"HTTP error: {str(e)}"
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to get status: {str(e)}"
            )

    async def stop_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Stop a Replicate model (not supported - returns success).

        Replicate doesn't have a concept of "stopping" a model.
        Models are always available once deployed.

        Args:
            instance_id: Model identifier

        Returns:
            DeploymentResult indicating operation not supported
        """
        return DeploymentResult(
            success=True,
            instance_id=instance_id,
            status="running",
            metadata={
                "note": "Replicate models cannot be stopped - they scale to zero automatically"
            }
        )

    async def delete_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Delete a Replicate model.

        Args:
            instance_id: Model identifier (owner/name)

        Returns:
            DeploymentResult with operation status
        """
        try:
            owner, name = instance_id.split("/")

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.BASE_URL}/models/{owner}/{name}",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    status="deleted"
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to delete model: {str(e)}"
            )

    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """
        Get list of available hardware on Replicate.

        Returns:
            List of available hardware configurations
        """
        # Replicate's available hardware (as of 2024)
        return [
            {
                "name": "CPU",
                "description": "CPU-only instance",
                "price_per_second": 0.000100
            },
            {
                "name": "Nvidia T4 GPU",
                "description": "Nvidia T4 GPU",
                "price_per_second": 0.000225
            },
            {
                "name": "Nvidia A40 GPU",
                "description": "Nvidia A40 (Large) GPU",
                "price_per_second": 0.000575
            },
            {
                "name": "Nvidia A40 (Large) GPU",
                "description": "Nvidia A40 with more memory",
                "price_per_second": 0.001150
            },
            {
                "name": "Nvidia A100 (40GB) GPU",
                "description": "Nvidia A100 40GB GPU",
                "price_per_second": 0.001150
            },
            {
                "name": "Nvidia A100 (80GB) GPU",
                "description": "Nvidia A100 80GB GPU",
                "price_per_second": 0.001400
            },
        ]

    async def run_prediction(
        self,
        instance_id: str,
        input_data: Dict[str, Any],
        webhook: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a prediction on a Replicate model.

        Args:
            instance_id: Model identifier (owner/name or owner/name:version)
            input_data: Input data for the prediction
            webhook: Optional webhook URL for completion notification

        Returns:
            Prediction result
        """
        try:
            # Parse model reference
            if ":" in instance_id:
                model_ref, version = instance_id.split(":")
            else:
                model_ref = instance_id
                version = None

            prediction_data = {
                "version": version,
                "input": input_data
            }

            if webhook:
                prediction_data["webhook"] = webhook
                prediction_data["webhook_events_filter"] = ["completed"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/predictions",
                    json=prediction_data,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

    def _map_gpu_to_hardware(self, gpu_type: str) -> str:
        """Map our GPU types to Replicate hardware names."""
        hardware_map = {
            "T4": "gpu-t4",
            "A40": "gpu-a40-large",
            "A100": "gpu-a100",
            "A100-80GB": "gpu-a100-80gb"
        }
        return hardware_map.get(gpu_type, "gpu-t4")

    async def get_provider_info(self) -> Dict[str, Any]:
        """
        Get Replicate provider information.

        Returns:
            Provider metadata
        """
        return {
            "name": "Replicate",
            "type": "serverless",
            "supports_autoscaling": True,
            "supports_spot_instances": False,
            "supports_ssh": False,
            "supports_jupyter": False,
            "billing_model": "pay-per-prediction",
            "regions": ["us"],
            "min_deployment_time_minutes": 0,
            "cold_start_seconds": "1-30",
            "scales_to_zero": True,
            "documentation_url": "https://replicate.com/docs"
        }
