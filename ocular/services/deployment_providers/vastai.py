"""
Vast.ai Deployment Provider.

Provides deployment capabilities for Vast.ai GPU marketplace.
"""

import httpx
from typing import Dict, Any, Optional, List
from .base import BaseDeploymentProvider, DeploymentConfig, DeploymentResult


class VastAIProvider(BaseDeploymentProvider):
    """
    Vast.ai deployment provider implementation.

    Vast.ai is a GPU marketplace offering affordable compute.
    """

    BASE_URL = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize Vast.ai provider.

        Args:
            api_key: Vast.ai API key
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
        Create a deployment on Vast.ai.

        Args:
            config: Deployment configuration
            model_config: Model-specific configuration
            provider_config: Provider-specific configuration

        Returns:
            DeploymentResult with instance details
        """
        try:
            # Search for available instances matching requirements
            search_params = {
                "gpu_name": config.gpu_type,
                "num_gpus": config.gpu_count,
                "verified": True,
                "rentable": True,
                "reliability": 0.95,
                "order": "score-"
            }

            async with httpx.AsyncClient() as client:
                # Find available offers
                search_response = await client.get(
                    f"{self.BASE_URL}/bundles/",
                    params=search_params,
                    headers=self.headers,
                    timeout=30.0
                )
                search_response.raise_for_status()
                offers = search_response.json()

                if not offers or len(offers["offers"]) == 0:
                    return DeploymentResult(
                        success=False,
                        error_message=f"No available instances found for {config.gpu_type}"
                    )

                # Select best offer
                best_offer = offers["offers"][0]
                offer_id = best_offer["id"]

                # Prepare instance configuration
                instance_config = {
                    "client_id": "ocular",
                    "image": model_config.get("docker_image", "pytorch/pytorch:latest"),
                    "env": config.extra_config.get("environment", {}),
                    "disk": provider_config.get("disk_size_gb", 50),
                    "onstart": model_config.get("onstart_script", ""),
                    "runtype": "ssh",
                    "python_utf8": True,
                    "jupyter": False,
                }

                # Create instance
                create_response = await client.put(
                    f"{self.BASE_URL}/asks/{offer_id}/",
                    json=instance_config,
                    headers=self.headers,
                    timeout=30.0
                )
                create_response.raise_for_status()
                result = create_response.json()

                if not result.get("success"):
                    return DeploymentResult(
                        success=False,
                        error_message=result.get("msg", "Unknown error creating instance")
                    )

                instance_id = str(result["new_contract"])

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    endpoint_url=f"ssh://vast.ai:{instance_id}",
                    status="pending",
                    metadata={
                        "provider": "vastai",
                        "gpu_type": config.gpu_type,
                        "gpu_count": config.gpu_count,
                        "offer_id": offer_id,
                        "hourly_cost": best_offer.get("dph_total", 0)
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
        Get status of a Vast.ai instance.

        Args:
            instance_id: Instance/contract ID

        Returns:
            DeploymentResult with current status
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/instances/",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                instances = response.json()

                # Find our instance
                instance = next(
                    (i for i in instances.get("instances", []) if str(i["id"]) == instance_id),
                    None
                )

                if not instance:
                    return DeploymentResult(
                        success=False,
                        error_message=f"Instance {instance_id} not found"
                    )

                # Map Vast.ai status to our status
                vast_status = instance.get("actual_status", "unknown")
                status_map = {
                    "running": "running",
                    "loading": "deploying",
                    "exited": "stopped",
                    "created": "pending"
                }
                status = status_map.get(vast_status, "unknown")

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    endpoint_url=instance.get("ssh_host", ""),
                    status=status,
                    metadata={
                        "provider": "vastai",
                        "gpu_utilization": instance.get("gpu_util", 0),
                        "uptime": instance.get("duration", 0),
                        "cost": instance.get("total_flops_cost", 0)
                    }
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to get status: {str(e)}"
            )

    async def stop_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Stop a Vast.ai instance.

        Args:
            instance_id: Instance/contract ID

        Returns:
            DeploymentResult with operation status
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.BASE_URL}/instances/{instance_id}/",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                return DeploymentResult(
                    success=result.get("success", False),
                    instance_id=instance_id,
                    status="stopped" if result.get("success") else "error",
                    error_message=None if result.get("success") else "Failed to stop instance"
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to stop deployment: {str(e)}"
            )

    async def delete_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Delete a Vast.ai instance (same as stop for Vast.ai).

        Args:
            instance_id: Instance/contract ID

        Returns:
            DeploymentResult with operation status
        """
        return await self.stop_deployment(instance_id)

    async def get_available_gpus(self) -> List[Dict[str, Any]]:
        """
        Get list of available GPU types on Vast.ai.

        Returns:
            List of available GPU configurations
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/bundles/",
                    headers=self.headers,
                    params={"rentable": True},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                # Extract unique GPU types with pricing
                gpus = {}
                for offer in data.get("offers", []):
                    gpu_name = offer.get("gpu_name", "Unknown")
                    if gpu_name not in gpus:
                        gpus[gpu_name] = {
                            "name": gpu_name,
                            "memory_gb": offer.get("gpu_ram", 0),
                            "min_price_per_hour": offer.get("dph_total", 0),
                            "available_count": 1
                        }
                    else:
                        gpus[gpu_name]["available_count"] += 1
                        gpus[gpu_name]["min_price_per_hour"] = min(
                            gpus[gpu_name]["min_price_per_hour"],
                            offer.get("dph_total", 0)
                        )

                return list(gpus.values())

        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []

    async def get_provider_info(self) -> Dict[str, Any]:
        """
        Get Vast.ai provider information.

        Returns:
            Provider metadata
        """
        return {
            "name": "Vast.ai",
            "type": "marketplace",
            "supports_autoscaling": False,
            "supports_spot_instances": True,
            "supports_ssh": True,
            "supports_jupyter": True,
            "billing_model": "pay-per-minute",
            "regions": ["global"],
            "min_deployment_time_minutes": 15,
            "documentation_url": "https://vast.ai/docs/"
        }
