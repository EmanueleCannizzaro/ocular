"""
Lambda Labs Deployment Provider.

Provides deployment capabilities for Lambda Labs GPU cloud.
"""

import httpx
from typing import Dict, Any, List
from .base import BaseDeploymentProvider, DeploymentConfig, DeploymentResult


class LambdaLabsProvider(BaseDeploymentProvider):
    """
    Lambda Labs deployment provider implementation.

    Lambda Labs offers on-demand GPU instances with competitive pricing.
    """

    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize Lambda Labs provider.

        Args:
            api_key: Lambda Labs API key
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def create_deployment(
        self,
        config: DeploymentConfig,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any]
    ) -> DeploymentResult:
        """
        Create a deployment on Lambda Labs.

        Args:
            config: Deployment configuration
            model_config: Model-specific configuration
            provider_config: Provider-specific configuration

        Returns:
            DeploymentResult with instance details
        """
        try:
            # Map GPU type to Lambda instance type
            instance_type_map = {
                "A100": "gpu_1x_a100",
                "A100-80GB": "gpu_1x_a100_sxm4",
                "A10": "gpu_1x_a10",
                "RTX 6000 Ada": "gpu_1x_rtx6000",
                "H100": "gpu_1x_h100_pcie"
            }

            instance_type = instance_type_map.get(config.gpu_type)
            if not instance_type:
                return DeploymentResult(
                    success=False,
                    error_message=f"Unsupported GPU type: {config.gpu_type}"
                )

            # Prepare instance launch configuration
            launch_config = {
                "region_name": config.region or "us-west-1",
                "instance_type_name": instance_type,
                "ssh_key_names": provider_config.get("ssh_key_names", []),
                "file_system_names": provider_config.get("file_system_names", []),
                "quantity": config.gpu_count,
                "name": config.deployment_name
            }

            async with httpx.AsyncClient() as client:
                # Launch instance
                response = await client.post(
                    f"{self.BASE_URL}/instance-operations/launch",
                    json=launch_config,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                if not result.get("data"):
                    return DeploymentResult(
                        success=False,
                        error_message="Failed to launch instance"
                    )

                instance_ids = result["data"].get("instance_ids", [])
                if not instance_ids:
                    return DeploymentResult(
                        success=False,
                        error_message="No instances created"
                    )

                instance_id = instance_ids[0]

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    endpoint_url=None,  # Will be populated when instance is running
                    status="pending",
                    metadata={
                        "provider": "lambda_labs",
                        "instance_type": instance_type,
                        "region": config.region,
                        "gpu_count": config.gpu_count
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
        Get status of a Lambda Labs instance.

        Args:
            instance_id: Instance ID

        Returns:
            DeploymentResult with current status
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/instances/{instance_id}",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                instance = result.get("data")
                if not instance:
                    return DeploymentResult(
                        success=False,
                        error_message=f"Instance {instance_id} not found"
                    )

                # Map Lambda status to our status
                lambda_status = instance.get("status", "unknown")
                status_map = {
                    "active": "running",
                    "booting": "deploying",
                    "unhealthy": "error",
                    "terminated": "stopped"
                }
                status = status_map.get(lambda_status, "pending")

                # Get IP address for endpoint
                ip = instance.get("ip")
                endpoint_url = f"ssh://ubuntu@{ip}" if ip else None

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    endpoint_url=endpoint_url,
                    status=status,
                    metadata={
                        "provider": "lambda_labs",
                        "instance_type": instance.get("instance_type", {}).get("name"),
                        "region": instance.get("region", {}).get("name"),
                        "ip_address": ip,
                        "hostname": instance.get("hostname")
                    }
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to get status: {str(e)}"
            )

    async def stop_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Stop a Lambda Labs instance.

        Args:
            instance_id: Instance ID

        Returns:
            DeploymentResult with operation status
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/instance-operations/terminate",
                    json={"instance_ids": [instance_id]},
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                terminated_ids = result.get("data", {}).get("terminated_instances", [])
                success = instance_id in terminated_ids

                return DeploymentResult(
                    success=success,
                    instance_id=instance_id,
                    status="stopped" if success else "error",
                    error_message=None if success else "Failed to terminate instance"
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to stop deployment: {str(e)}"
            )

    async def delete_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Delete a Lambda Labs instance (same as stop).

        Args:
            instance_id: Instance ID

        Returns:
            DeploymentResult with operation status
        """
        return await self.stop_deployment(instance_id)

    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """
        Get list of available GPU types on Lambda Labs.

        Returns:
            List of available GPU configurations
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/instance-types",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                gpus = []
                for instance_type in result.get("data", {}).values():
                    specs = instance_type.get("instance_type", {}).get("specs", {})
                    gpus.append({
                        "name": instance_type.get("instance_type", {}).get("name"),
                        "gpu_type": specs.get("gpu_description"),
                        "gpu_count": specs.get("gpus", 1),
                        "memory_gb": specs.get("memory_gib", 0),
                        "vcpus": specs.get("vcpus", 0),
                        "storage_gb": specs.get("storage_gib", 0),
                        "price_per_hour": instance_type.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
                    })

                return gpus

        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []

    async def get_provider_info(self) -> Dict[str, Any]:
        """
        Get Lambda Labs provider information.

        Returns:
            Provider metadata
        """
        return {
            "name": "Lambda Labs",
            "type": "cloud",
            "supports_autoscaling": False,
            "supports_spot_instances": False,
            "supports_ssh": True,
            "supports_jupyter": True,
            "billing_model": "pay-per-second",
            "regions": ["us-west-1", "us-west-2", "us-east-1", "us-south-1", "europe-central-1", "asia-south-1"],
            "min_deployment_time_minutes": 0,
            "documentation_url": "https://docs.lambdalabs.com/"
        }
