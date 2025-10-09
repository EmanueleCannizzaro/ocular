"""
API Router for Advanced Deployment Configuration.

Provides endpoints for managing deployment settings including environment variables,
scaling policies, health checks, and resource limits.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from sqlalchemy.orm import Session

from ocular.database.session import get_db
from ocular.models.deployments import (
    EnvironmentVariablesRequest, ScalingPolicyRequest, HealthCheckRequest, 
    ResourceLimitsRequest, DockerConfigRequest, NetworkConfigRequest, 
    GPUConfigRequest
)
from ocular.routers import templates
from ocular.services.deployment_config_service import (
    DeploymentConfigService,
    ScalingPolicy,
    HealthCheck
)


router = APIRouter()


# API Endpoints

@router.post("/deployments/{deployment_id}/config/environment")
async def set_environment_variables(
    deployment_id: str,
    request: EnvironmentVariablesRequest,
    db: Session = Depends(get_db)
):
    """
    Set environment variables for a deployment.

    Environment variables are passed to the container at runtime.
    """
    try:
        deployment = DeploymentConfigService.set_environment_variables(
            db=db,
            deployment_id=deployment_id,
            env_vars=request.env_vars,
            merge=request.merge
        )
        return {
            "success": True,
            "message": "Environment variables updated",
            "deployment_id": deployment.deployment_id,
            "environment": deployment.config.get("environment", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update environment: {str(e)}")


@router.post("/deployments/{deployment_id}/config/scaling-policy")
async def set_scaling_policy(
    deployment_id: str,
    request: ScalingPolicyRequest,
    db: Session = Depends(get_db)
):
    """
    Configure autoscaling policy for a deployment.

    Defines how the deployment scales based on load and resource utilization.
    """
    try:
        policy = ScalingPolicy.create_policy(
            min_instances=request.min_instances,
            max_instances=request.max_instances,
            target_utilization=request.target_utilization,
            scale_up_threshold=request.scale_up_threshold,
            scale_down_threshold=request.scale_down_threshold,
            cooldown_period_seconds=request.cooldown_period_seconds,
            metrics=request.metrics
        )

        deployment = DeploymentConfigService.set_scaling_policy(
            db=db,
            deployment_id=deployment_id,
            policy=policy
        )

        return {
            "success": True,
            "message": "Scaling policy updated",
            "deployment_id": deployment.deployment_id,
            "policy": deployment.config.get("scaling_policy", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scaling policy: {str(e)}")


@router.post("/deployments/{deployment_id}/config/health-check")
async def set_health_check(
    deployment_id: str,
    request: HealthCheckRequest,
    db: Session = Depends(get_db)
):
    """
    Configure health check for a deployment.

    Health checks monitor the deployment and can trigger automatic recovery.
    """
    try:
        health_check = HealthCheck.create_config(
            enabled=request.enabled,
            endpoint=request.endpoint,
            interval_seconds=request.interval_seconds,
            timeout_seconds=request.timeout_seconds,
            unhealthy_threshold=request.unhealthy_threshold,
            healthy_threshold=request.healthy_threshold,
            http_method=request.http_method,
            expected_status_codes=request.expected_status_codes,
            headers=request.headers
        )

        deployment = DeploymentConfigService.set_health_check(
            db=db,
            deployment_id=deployment_id,
            health_check=health_check
        )

        return {
            "success": True,
            "message": "Health check configured",
            "deployment_id": deployment.deployment_id,
            "health_check": deployment.config.get("health_check", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure health check: {str(e)}")


@router.post("/deployments/{deployment_id}/config/resources")
async def set_resource_limits(
    deployment_id: str,
    request: ResourceLimitsRequest,
    db: Session = Depends(get_db)
):
    """
    Set resource limits for a deployment.

    Defines maximum CPU, memory, GPU memory, and disk usage.
    """
    try:
        deployment = DeploymentConfigService.set_resource_limits(
            db=db,
            deployment_id=deployment_id,
            cpu_limit=request.cpu_limit,
            memory_limit=request.memory_limit,
            gpu_memory_limit=request.gpu_memory_limit,
            disk_limit=request.disk_limit
        )

        return {
            "success": True,
            "message": "Resource limits updated",
            "deployment_id": deployment.deployment_id,
            "resources": deployment.config.get("resources", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update resource limits: {str(e)}")


@router.post("/deployments/{deployment_id}/config/docker")
async def set_docker_config(
    deployment_id: str,
    request: DockerConfigRequest,
    db: Session = Depends(get_db)
):
    """
    Configure Docker-specific settings.

    Customize Dockerfile, build args, runtime args, and container settings.
    """
    try:
        deployment = DeploymentConfigService.set_docker_config(
            db=db,
            deployment_id=deployment_id,
            dockerfile_path=request.dockerfile_path,
            build_args=request.build_args,
            docker_args=request.docker_args,
            entrypoint=request.entrypoint,
            command=request.command,
            working_dir=request.working_dir
        )

        return {
            "success": True,
            "message": "Docker configuration updated",
            "deployment_id": deployment.deployment_id,
            "docker": deployment.config.get("docker", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update Docker config: {str(e)}")


@router.post("/deployments/{deployment_id}/config/network")
async def set_network_config(
    deployment_id: str,
    request: NetworkConfigRequest,
    db: Session = Depends(get_db)
):
    """
    Configure networking settings.

    Set up port mappings, public exposure, IP whitelist, and custom domains.
    """
    try:
        ports = [p.dict() for p in request.ports] if request.ports else None

        deployment = DeploymentConfigService.set_network_config(
            db=db,
            deployment_id=deployment_id,
            ports=ports,
            expose_public=request.expose_public,
            allowed_ips=request.allowed_ips,
            custom_domain=request.custom_domain,
            ssl_enabled=request.ssl_enabled
        )

        return {
            "success": True,
            "message": "Network configuration updated",
            "deployment_id": deployment.deployment_id,
            "network": deployment.config.get("network", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update network config: {str(e)}")


@router.post("/deployments/{deployment_id}/config/gpu")
async def set_gpu_config(
    deployment_id: str,
    request: GPUConfigRequest,
    db: Session = Depends(get_db)
):
    """
    Configure advanced GPU settings.

    Set GPU preferences, multi-GPU strategy, CUDA version, and utilization targets.
    """
    try:
        deployment = DeploymentConfigService.set_advanced_gpu_config(
            db=db,
            deployment_id=deployment_id,
            gpu_types=request.gpu_types,
            multi_gpu_strategy=request.multi_gpu_strategy,
            cuda_version=request.cuda_version,
            tensor_cores=request.tensor_cores,
            gpu_utilization_target=request.gpu_utilization_target
        )

        return {
            "success": True,
            "message": "GPU configuration updated",
            "deployment_id": deployment.deployment_id,
            "gpu": deployment.config.get("gpu", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update GPU config: {str(e)}")


@router.get("/deployments/{deployment_id}/config")
async def get_deployment_config(
    deployment_id: str,
    db: Session = Depends(get_db)
):
    """
    Get complete configuration for a deployment.

    Returns all configuration settings including basic, scaling, environment,
    health checks, resources, Docker, network, and GPU settings.
    """
    try:
        config = DeploymentConfigService.get_full_config(db, deployment_id)
        return config
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/deployments/{deployment_id}/config/validate")
async def validate_config(
    deployment_id: str,
    config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Validate deployment configuration without saving.

    Checks if the provided configuration is valid.
    """
    try:
        is_valid, error_message = DeploymentConfigService.validate_config(config)

        return {
            "valid": is_valid,
            "error": error_message,
            "deployment_id": deployment_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
