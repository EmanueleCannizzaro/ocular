"""
RunPod deployment provider implementation.

Integrates with RunPod Serverless API to deploy OCR models.
"""

import os
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

from .base import (
    BaseDeploymentProvider,
    DeploymentConfig,
    DeploymentResult
)


class RunPodDeploymentProvider(BaseDeploymentProvider):
    """
    RunPod Serverless deployment provider.

    Docs: https://docs.runpod.io/serverless/endpoints/manage-endpoints
    """

    BASE_URL = "https://api.runpod.ai/v2"
    GRAPHQL_URL = "https://api.runpod.io/graphql"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize RunPod provider.

        Args:
            api_key: RunPod API key (defaults to RUNPOD_API_KEY env var)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not api_key:
            raise ValueError("RunPod API key is required")

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
        Create a RunPod serverless endpoint.

        Creates a new serverless endpoint with the specified configuration.
        """
        try:
            # Extract configuration
            docker_image = model_config.get("docker_image")
            if not docker_image:
                return DeploymentResult(
                    success=False,
                    error_message="Docker image not specified in model configuration"
                )

            # Build GraphQL mutation for creating endpoint
            mutation = """
            mutation {
              saveEndpoint(input: {
                name: "%s"
                template: {
                  imageName: "%s"
                  dockerArgs: ""
                  containerDiskInGb: 20
                  volumeInGb: 0
                  env: [
                    {key: "MODEL_NAME", value: "%s"}
                  ]
                }
                gpuIds: "%s"
                networkVolumeId: ""
                locations: "%s"
                idleTimeout: %d
                scalerType: "QUEUE_DELAY"
                scalerValue: 4
                workersMin: %d
                workersMax: %d
              }) {
                id
                name
                gpuIds
                networkVolumeId
                locations
                workersMin
                workersMax
                idleTimeout
              }
            }
            """ % (
                config.deployment_name,
                docker_image,
                model_config.get("name", "ocr-model"),
                self._map_gpu_to_runpod_id(config.gpu_type),
                config.region or "US",
                config.timeout_seconds,
                config.min_workers,
                config.max_workers
            )

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.GRAPHQL_URL,
                    json={"query": mutation},
                    headers=self.headers
                )

                if response.status_code != 200:
                    return DeploymentResult(
                        success=False,
                        error_message=f"RunPod API error: {response.status_code} - {response.text}"
                    )

                result = response.json()

                # Check for GraphQL errors
                if "errors" in result:
                    error_msg = "; ".join([e.get("message", str(e)) for e in result["errors"]])
                    return DeploymentResult(
                        success=False,
                        error_message=f"RunPod GraphQL error: {error_msg}"
                    )

                # Extract endpoint data
                endpoint_data = result.get("data", {}).get("saveEndpoint", {})
                endpoint_id = endpoint_data.get("id")

                if not endpoint_id:
                    return DeploymentResult(
                        success=False,
                        error_message="No endpoint ID returned from RunPod"
                    )

                # Build endpoint URL
                endpoint_url = f"{self.BASE_URL}/{endpoint_id}"

                return DeploymentResult(
                    success=True,
                    instance_id=endpoint_id,
                    endpoint_url=endpoint_url,
                    status="deploying",
                    metadata={
                        "endpoint_data": endpoint_data,
                        "provider": "runpod",
                        "created_at": datetime.utcnow().isoformat()
                    }
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error_message=f"Failed to create deployment: {str(e)}"
            )

    async def get_deployment_status(self, instance_id: str) -> DeploymentResult:
        """
        Get status of a RunPod endpoint.

        Queries the endpoint to check if it's ready and operational.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check endpoint health
                health_url = f"{self.BASE_URL}/{instance_id}/health"
                response = await client.get(
                    health_url,
                    headers=self.headers
                )

                if response.status_code == 200:
                    health_data = response.json()
                    workers = health_data.get("workers", {})

                    # Determine status based on workers
                    running_workers = workers.get("running", 0)
                    idle_workers = workers.get("idle", 0)

                    if running_workers > 0 or idle_workers > 0:
                        status = "running"
                    else:
                        status = "deploying"

                    return DeploymentResult(
                        success=True,
                        instance_id=instance_id,
                        endpoint_url=f"{self.BASE_URL}/{instance_id}",
                        status=status,
                        metadata={
                            "workers": workers,
                            "checked_at": datetime.utcnow().isoformat()
                        }
                    )
                else:
                    return DeploymentResult(
                        success=False,
                        instance_id=instance_id,
                        status="error",
                        error_message=f"Health check failed: {response.status_code}"
                    )

        except Exception as e:
            return DeploymentResult(
                success=False,
                instance_id=instance_id,
                status="error",
                error_message=f"Failed to check status: {str(e)}"
            )

    async def stop_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Stop a RunPod endpoint by setting workers to 0.

        Note: RunPod serverless endpoints scale to zero automatically,
        so this updates the configuration.
        """
        try:
            # Update endpoint to set workers to 0
            mutation = """
            mutation {
              updateEndpointWorkersMin(input: {
                endpointId: "%s"
                workersMin: 0
              }) {
                id
                workersMin
                workersMax
              }
            }
            """ % instance_id

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.GRAPHQL_URL,
                    json={"query": mutation},
                    headers=self.headers
                )

                if response.status_code != 200:
                    return DeploymentResult(
                        success=False,
                        instance_id=instance_id,
                        error_message=f"Failed to stop endpoint: {response.status_code}"
                    )

                result = response.json()

                if "errors" in result:
                    error_msg = "; ".join([e.get("message", str(e)) for e in result["errors"]])
                    return DeploymentResult(
                        success=False,
                        instance_id=instance_id,
                        error_message=f"GraphQL error: {error_msg}"
                    )

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    status="stopped",
                    metadata={
                        "stopped_at": datetime.utcnow().isoformat()
                    }
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                instance_id=instance_id,
                error_message=f"Failed to stop deployment: {str(e)}"
            )

    async def delete_deployment(self, instance_id: str) -> DeploymentResult:
        """
        Delete a RunPod endpoint permanently.
        """
        try:
            mutation = """
            mutation {
              deleteEndpoint(id: "%s") {
                id
                name
              }
            }
            """ % instance_id

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.GRAPHQL_URL,
                    json={"query": mutation},
                    headers=self.headers
                )

                if response.status_code != 200:
                    return DeploymentResult(
                        success=False,
                        instance_id=instance_id,
                        error_message=f"Failed to delete endpoint: {response.status_code}"
                    )

                result = response.json()

                if "errors" in result:
                    error_msg = "; ".join([e.get("message", str(e)) for e in result["errors"]])
                    return DeploymentResult(
                        success=False,
                        instance_id=instance_id,
                        error_message=f"GraphQL error: {error_msg}"
                    )

                return DeploymentResult(
                    success=True,
                    instance_id=instance_id,
                    status="deleted",
                    metadata={
                        "deleted_at": datetime.utcnow().isoformat()
                    }
                )

        except Exception as e:
            return DeploymentResult(
                success=False,
                instance_id=instance_id,
                error_message=f"Failed to delete deployment: {str(e)}"
            )

    async def get_logs(
        self,
        instance_id: str,
        lines: int = 100
    ) -> Dict[str, Any]:
        """
        Retrieve logs from a RunPod endpoint.

        Note: RunPod serverless logs are available through the dashboard.
        This is a placeholder for future implementation.
        """
        return {
            "success": False,
            "message": "Log retrieval not yet implemented for RunPod serverless endpoints",
            "instance_id": instance_id
        }

    async def estimate_cost(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        hours: float = 1.0
    ) -> Dict[str, Any]:
        """
        Estimate cost for RunPod deployment.

        Uses approximate pricing from RunPod's pricing page.
        """
        # Approximate pricing per GPU type (per hour)
        pricing_map = {
            "RTX 4090": 0.39,
            "RTX A4000": 0.34,
            "RTX 4000 Ada": 0.44,
            "RTX 6000 Ada": 0.89,
            "A40": 0.79,
            "A100 40GB": 1.89,
            "A100 80GB": 2.49,
            "H100": 4.25,
            "L40": 1.19,
        }

        # Find matching GPU price
        price_per_hour = None
        for gpu_key, price in pricing_map.items():
            if gpu_key.lower() in gpu_type.lower():
                price_per_hour = price
                break

        if price_per_hour is None:
            price_per_hour = 0.50  # Default fallback

        total_cost = price_per_hour * gpu_count * hours

        return {
            "success": True,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "hours": hours,
            "price_per_hour": price_per_hour,
            "total_estimated_cost": round(total_cost, 4),
            "currency": "USD",
            "note": "Estimated pricing - actual costs may vary"
        }

    def _map_gpu_to_runpod_id(self, gpu_type: str) -> str:
        """
        Map GPU type string to RunPod GPU ID.

        Args:
            gpu_type: GPU type name (e.g., "RTX 4090")

        Returns:
            RunPod GPU ID string
        """
        # Common RunPod GPU IDs
        gpu_mapping = {
            "RTX 4090": "NVIDIA GeForce RTX 4090",
            "RTX A4000": "NVIDIA RTX A4000",
            "RTX 4000 Ada": "NVIDIA RTX 4000 Ada Generation",
            "RTX 6000 Ada": "NVIDIA RTX 6000 Ada Generation",
            "A40": "NVIDIA A40",
            "A100": "NVIDIA A100-PCIE-40GB",
            "A100 40GB": "NVIDIA A100-PCIE-40GB",
            "A100 80GB": "NVIDIA A100 80GB PCIe",
            "H100": "NVIDIA H100 PCIe",
            "L40": "NVIDIA L40",
        }

        # Find best match
        for gpu_key, runpod_id in gpu_mapping.items():
            if gpu_key.lower() in gpu_type.lower():
                return runpod_id

        # Default fallback
        return gpu_type
