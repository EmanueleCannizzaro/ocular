"""
Deployment service for executing deployment scripts.

Handles deployment of AI models to cloud GPU providers.
"""

import os
import json
import requests
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ocular.database.models import Deployment, DeploymentStatus, Provider, Model
from ocular.database.session import SessionLocal
from ocular.services.deployment_providers import RunPodDeploymentProvider
from ocular.services.deployment_providers.base import DeploymentConfig


class DeploymentService:
    """Service for managing AI model deployments to cloud GPU providers."""

    def __init__(self):
        """Initialize deployment service."""
        self.deployment_templates_dir = Path(__file__).parent.parent.parent / "deployment" / "runpod.io"

    async def deploy_to_runpod(
        self,
        deployment: Deployment,
        api_key: str,
        docker_image: Optional[str] = None,
        gpu_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy model to RunPod serverless.

        Args:
            deployment: Deployment instance from database
            api_key: RunPod API key
            docker_image: Optional Docker image override
            gpu_type: Optional GPU type override

        Returns:
            Dict with deployment result including endpoint_id and endpoint_url
        """
        # Get model and provider info
        db = SessionLocal()
        try:
            model = db.query(Model).filter(Model.id == deployment.model_id).first()
            provider = db.query(Provider).filter(Provider.id == deployment.provider_id).first()

            if not model or not provider:
                raise ValueError("Model or provider not found")

            # Determine Docker image
            if not docker_image:
                if model.docker_image:
                    docker_image = model.docker_image
                else:
                    # Default image based on model name
                    docker_username = os.getenv("DOCKER_USERNAME", "ocular")
                    docker_image = f"{docker_username}/{model.name}-runpod:latest"

            # Determine GPU type
            if not gpu_type:
                gpu_type = deployment.gpu_type

            # Build endpoint configuration
            endpoint_config = {
                "name": deployment.deployment_name,
                "template_id": None,  # Custom template
                "gpu_ids": gpu_type,
                "network_volume_id": os.getenv("RUNPOD_NETWORK_VOLUME_ID"),
                "container_disk_in_gb": deployment.config.get("container_disk_gb", 30) if deployment.config else 30,
                "volumes_in_gb": deployment.config.get("volumes_in_gb", 30) if deployment.config else 30,
                "min_workers": deployment.min_workers,
                "max_workers": deployment.max_workers,
                "idle_timeout": deployment.config.get("idle_timeout", 30) if deployment.config else 30,
                "execution_timeout": deployment.timeout_seconds,
                "docker_image": docker_image,
                "env": {
                    "RUNPOD_ACCESS_KEY_ID": os.getenv("RUNPOD_ACCESS_KEY_ID", ""),
                    "RUNPOD_SECRET_ACCESS_KEY": os.getenv("RUNPOD_SECRET_ACCESS_KEY", ""),
                    "RUNPOD_BUCKET_ENDPOINT_URL": os.getenv("RUNPOD_BUCKET_ENDPOINT_URL", ""),
                    "RUNPOD_NETWORK_VOLUME_ID": os.getenv("RUNPOD_NETWORK_VOLUME_ID", ""),
                    "RUNPOD_REGION": deployment.region or os.getenv("RUNPOD_REGION", "eur-is-1"),
                    "MODEL_NAME": model.base_model or model.name
                }
            }

            # Add custom environment variables from deployment config
            if deployment.config and "env" in deployment.config:
                endpoint_config["env"].update(deployment.config["env"])

            # Call RunPod API
            result = await self._create_runpod_endpoint(api_key, endpoint_config)

            # Update deployment with endpoint information
            deployment.endpoint_url = result.get("endpoint_url")
            deployment.instance_id = result.get("endpoint_id")
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.started_at = datetime.utcnow()

            # Save additional metadata
            deployment.extra_metadata = deployment.extra_metadata or {}
            deployment.extra_metadata.update({
                "docker_image": docker_image,
                "gpu_type": gpu_type,
                "deployment_result": result
            })

            db.commit()

            return result

        finally:
            db.close()

    async def _create_runpod_endpoint(
        self,
        api_key: str,
        endpoint_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create RunPod endpoint via GraphQL API.

        Args:
            api_key: RunPod API key
            endpoint_config: Endpoint configuration

        Returns:
            Dict with endpoint_id, endpoint_url, status

        Raises:
            Exception: If API call fails
        """
        api_url = "https://api.runpod.io/graphql"

        # GraphQL mutation
        mutation = """
        mutation CreateEndpoint($input: EndpointInput!) {
            createEndpoint(input: $input) {
                id
                name
                status
            }
        }
        """

        payload = {
            "query": mutation,
            "variables": {
                "input": endpoint_config
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Make async HTTP request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(api_url, json=payload, headers=headers, timeout=30)
        )

        response.raise_for_status()
        result = response.json()

        # Check for errors
        if "errors" in result:
            error_msg = json.dumps(result["errors"], indent=2)
            raise Exception(f"RunPod API error: {error_msg}")

        # Extract endpoint data
        endpoint_data = result.get("data", {}).get("createEndpoint", {})
        endpoint_id = endpoint_data.get("id")

        if not endpoint_id:
            raise Exception("No endpoint ID returned from RunPod API")

        return {
            "endpoint_id": endpoint_id,
            "endpoint_url": f"https://api.runpod.ai/v2/{endpoint_id}/run",
            "status": endpoint_data.get("status"),
            "name": endpoint_data.get("name")
        }

    async def deploy_to_vastai(
        self,
        deployment: Deployment,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Deploy model to Vast.ai.

        Args:
            deployment: Deployment instance from database
            api_key: Vast.ai API key

        Returns:
            Dict with deployment result

        Raises:
            NotImplementedError: Vast.ai deployment not yet implemented
        """
        raise NotImplementedError("Vast.ai deployment not yet implemented")

    async def deploy_to_lambda(
        self,
        deployment: Deployment,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Deploy model to Lambda Labs.

        Args:
            deployment: Deployment instance from database
            api_key: Lambda Labs API key

        Returns:
            Dict with deployment result

        Raises:
            NotImplementedError: Lambda Labs deployment not yet implemented
        """
        raise NotImplementedError("Lambda Labs deployment not yet implemented")

    async def deploy(
        self,
        deployment: Deployment,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Deploy model to the appropriate cloud provider using deployment providers.

        Args:
            deployment: Deployment instance from database
            api_key: Provider API key

        Returns:
            Dict with deployment result

        Raises:
            ValueError: If provider is not supported
        """
        db = SessionLocal()
        try:
            provider = db.query(Provider).filter(Provider.id == deployment.provider_id).first()
            model = db.query(Model).filter(Model.id == deployment.model_id).first()

            if not provider:
                raise ValueError("Provider not found")
            if not model:
                raise ValueError("Model not found")

            provider_name = provider.name.lower()

            # Create deployment configuration
            config = DeploymentConfig(
                model_id=deployment.model_id,
                provider_id=deployment.provider_id,
                deployment_name=deployment.deployment_name,
                gpu_type=deployment.gpu_type,
                gpu_count=deployment.gpu_count,
                max_workers=deployment.max_workers,
                min_workers=deployment.min_workers,
                timeout_seconds=deployment.timeout_seconds,
                autoscaling_enabled=deployment.autoscaling_enabled,
                region=deployment.region,
                extra_config=deployment.config
            )

            # Model configuration
            model_config = {
                "name": model.name,
                "docker_image": model.docker_image,
                "base_model": model.base_model,
                "config_template": model.config_template,
                "supports_vllm": model.supports_vllm
            }

            # Provider configuration
            provider_config = provider.extra_metadata or {}

            # Deploy using appropriate provider
            if provider_name == "runpod":
                provider_instance = RunPodDeploymentProvider(api_key=api_key)
                result = await provider_instance.create_deployment(
                    config=config,
                    model_config=model_config,
                    provider_config=provider_config
                )

                # Update deployment with result
                if result.success:
                    deployment.status = DeploymentStatus.DEPLOYING
                    deployment.instance_id = result.instance_id
                    deployment.endpoint_url = result.endpoint_url
                    deployment.started_at = result.timestamp

                    # Store metadata
                    deployment.extra_metadata = deployment.extra_metadata or {}
                    deployment.extra_metadata.update(result.metadata)
                else:
                    deployment.status = DeploymentStatus.ERROR
                    deployment.error_message = result.error_message
                    deployment.last_error_at = result.timestamp

                db.commit()

                return {
                    "success": result.success,
                    "endpoint_id": result.instance_id,
                    "endpoint_url": result.endpoint_url,
                    "status": result.status,
                    "error_message": result.error_message
                }

            elif provider_name == "vastai" or provider_name == "vast.ai":
                return await self.deploy_to_vastai(deployment, api_key)
            elif provider_name == "lambda" or provider_name == "lambda_labs":
                return await self.deploy_to_lambda(deployment, api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

        finally:
            db.close()

    async def stop_deployment(self, deployment: Deployment, api_key: str) -> bool:
        """
        Stop a running deployment.

        Args:
            deployment: Deployment instance from database
            api_key: Provider API key

        Returns:
            True if successful

        Raises:
            ValueError: If provider is not supported
        """
        db = SessionLocal()
        try:
            provider = db.query(Provider).filter(Provider.id == deployment.provider_id).first()

            if not provider:
                raise ValueError("Provider not found")

            provider_name = provider.name.lower()

            if provider_name == "runpod":
                provider_instance = RunPodDeploymentProvider(api_key=api_key)
                result = await provider_instance.stop_deployment(deployment.instance_id)

                if result.success:
                    deployment.status = DeploymentStatus.STOPPED
                    deployment.stopped_at = result.timestamp
                else:
                    deployment.error_message = result.error_message
                    deployment.last_error_at = result.timestamp

                db.commit()
                return result.success

            elif provider_name == "vastai" or provider_name == "vast.ai":
                raise NotImplementedError("Vast.ai stop not yet implemented")
            elif provider_name == "lambda" or provider_name == "lambda_labs":
                raise NotImplementedError("Lambda Labs stop not yet implemented")
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

        finally:
            db.close()

    async def delete_deployment(self, deployment: Deployment, api_key: str) -> bool:
        """
        Delete a deployment from the cloud provider.

        Args:
            deployment: Deployment instance from database
            api_key: Provider API key

        Returns:
            True if successful

        Raises:
            ValueError: If provider is not supported
        """
        db = SessionLocal()
        try:
            provider = db.query(Provider).filter(Provider.id == deployment.provider_id).first()

            if not provider:
                raise ValueError("Provider not found")

            provider_name = provider.name.lower()

            if provider_name == "runpod":
                provider_instance = RunPodDeploymentProvider(api_key=api_key)
                result = await provider_instance.delete_deployment(deployment.instance_id)

                if result.success:
                    deployment.status = DeploymentStatus.DELETED
                    deployment.stopped_at = result.timestamp
                else:
                    deployment.error_message = result.error_message
                    deployment.last_error_at = result.timestamp

                db.commit()
                return result.success

            elif provider_name == "vastai" or provider_name == "vast.ai":
                raise NotImplementedError("Vast.ai delete not yet implemented")
            elif provider_name == "lambda" or provider_name == "lambda_labs":
                raise NotImplementedError("Lambda Labs delete not yet implemented")
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

        finally:
            db.close()


# Singleton instance
_deployment_service = None


def get_deployment_service() -> DeploymentService:
    """Get or create deployment service instance."""
    global _deployment_service
    if _deployment_service is None:
        _deployment_service = DeploymentService()
    return _deployment_service
