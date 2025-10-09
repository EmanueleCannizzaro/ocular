"""
Advanced Deployment Configuration Service.

Handles complex deployment settings including environment variables,
scaling policies, health checks, and advanced GPU configurations.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from ocular.database.models import Deployment


class ScalingPolicy:
    """Scaling policy configuration."""

    @staticmethod
    def create_policy(
        min_instances: int = 0,
        max_instances: int = 10,
        target_utilization: float = 0.8,
        scale_up_threshold: float = 0.9,
        scale_down_threshold: float = 0.3,
        cooldown_period_seconds: int = 300,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create autoscaling policy configuration.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_utilization: Target utilization percentage (0-1)
            scale_up_threshold: Threshold to trigger scale up
            scale_down_threshold: Threshold to trigger scale down
            cooldown_period_seconds: Cooldown between scaling actions
            metrics: Metrics to monitor for scaling

        Returns:
            Scaling policy dictionary
        """
        return {
            "enabled": True,
            "min_instances": min_instances,
            "max_instances": max_instances,
            "target_utilization": target_utilization,
            "scale_up_threshold": scale_up_threshold,
            "scale_down_threshold": scale_down_threshold,
            "cooldown_period_seconds": cooldown_period_seconds,
            "metrics": metrics or ["cpu", "gpu", "memory", "requests_per_second"],
            "scale_up_increment": 1,
            "scale_down_increment": 1,
        }


class HealthCheck:
    """Health check configuration."""

    @staticmethod
    def create_config(
        enabled: bool = True,
        endpoint: str = "/health",
        interval_seconds: int = 30,
        timeout_seconds: int = 10,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
        http_method: str = "GET",
        expected_status_codes: List[int] = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Create health check configuration.

        Args:
            enabled: Whether health checks are enabled
            endpoint: Health check endpoint path
            interval_seconds: Check interval
            timeout_seconds: Request timeout
            unhealthy_threshold: Failures before marking unhealthy
            healthy_threshold: Successes before marking healthy
            http_method: HTTP method for health check
            expected_status_codes: Expected HTTP status codes
            headers: Additional headers for health check request

        Returns:
            Health check configuration dictionary
        """
        return {
            "enabled": enabled,
            "endpoint": endpoint,
            "interval_seconds": interval_seconds,
            "timeout_seconds": timeout_seconds,
            "unhealthy_threshold": unhealthy_threshold,
            "healthy_threshold": healthy_threshold,
            "http_method": http_method,
            "expected_status_codes": expected_status_codes or [200, 204],
            "headers": headers or {},
            "retry_attempts": 2,
        }


class DeploymentConfigService:
    """Service for managing advanced deployment configurations."""

    @staticmethod
    def set_environment_variables(
        db: Session,
        deployment_id: str,
        env_vars: Dict[str, str],
        merge: bool = True
    ) -> Deployment:
        """
        Set environment variables for a deployment.

        Args:
            db: Database session
            deployment_id: Deployment ID
            env_vars: Environment variables dictionary
            merge: If True, merge with existing vars; if False, replace

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}

        if merge:
            existing_env = config.get("environment", {})
            existing_env.update(env_vars)
            config["environment"] = existing_env
        else:
            config["environment"] = env_vars

        deployment.config = config
        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_scaling_policy(
        db: Session,
        deployment_id: str,
        policy: Dict[str, Any]
    ) -> Deployment:
        """
        Configure autoscaling policy for a deployment.

        Args:
            db: Database session
            deployment_id: Deployment ID
            policy: Scaling policy configuration

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        # Validate policy
        required_fields = ["min_instances", "max_instances"]
        for field in required_fields:
            if field not in policy:
                raise ValueError(f"Missing required field: {field}")

        if policy["min_instances"] > policy["max_instances"]:
            raise ValueError("min_instances cannot be greater than max_instances")

        # Update deployment
        deployment.min_workers = policy["min_instances"]
        deployment.max_workers = policy["max_instances"]
        deployment.autoscaling_enabled = policy.get("enabled", True)

        # Store full policy in config
        config = deployment.config or {}
        config["scaling_policy"] = policy
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_health_check(
        db: Session,
        deployment_id: str,
        health_check: Dict[str, Any]
    ) -> Deployment:
        """
        Configure health check for a deployment.

        Args:
            db: Database session
            deployment_id: Deployment ID
            health_check: Health check configuration

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}
        config["health_check"] = health_check
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_resource_limits(
        db: Session,
        deployment_id: str,
        cpu_limit: Optional[str] = None,
        memory_limit: Optional[str] = None,
        gpu_memory_limit: Optional[str] = None,
        disk_limit: Optional[str] = None
    ) -> Deployment:
        """
        Set resource limits for a deployment.

        Args:
            db: Database session
            deployment_id: Deployment ID
            cpu_limit: CPU limit (e.g., "2", "2000m")
            memory_limit: Memory limit (e.g., "4Gi", "4096Mi")
            gpu_memory_limit: GPU memory limit (e.g., "8Gi")
            disk_limit: Disk limit (e.g., "20Gi")

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}
        resources = config.get("resources", {})

        limits = resources.get("limits", {})
        if cpu_limit:
            limits["cpu"] = cpu_limit
        if memory_limit:
            limits["memory"] = memory_limit
        if gpu_memory_limit:
            limits["gpu_memory"] = gpu_memory_limit
        if disk_limit:
            limits["disk"] = disk_limit

        resources["limits"] = limits
        config["resources"] = resources
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_docker_config(
        db: Session,
        deployment_id: str,
        dockerfile_path: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
        docker_args: Optional[List[str]] = None,
        entrypoint: Optional[List[str]] = None,
        command: Optional[List[str]] = None,
        working_dir: Optional[str] = None
    ) -> Deployment:
        """
        Configure Docker-specific settings.

        Args:
            db: Database session
            deployment_id: Deployment ID
            dockerfile_path: Path to Dockerfile
            build_args: Build-time arguments
            docker_args: Runtime Docker arguments
            entrypoint: Container entrypoint
            command: Container command
            working_dir: Working directory

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}
        docker_config = config.get("docker", {})

        if dockerfile_path:
            docker_config["dockerfile_path"] = dockerfile_path
        if build_args:
            docker_config["build_args"] = build_args
        if docker_args:
            docker_config["docker_args"] = docker_args
        if entrypoint:
            docker_config["entrypoint"] = entrypoint
        if command:
            docker_config["command"] = command
        if working_dir:
            docker_config["working_dir"] = working_dir

        config["docker"] = docker_config
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_network_config(
        db: Session,
        deployment_id: str,
        ports: Optional[List[Dict[str, Any]]] = None,
        expose_public: bool = False,
        allowed_ips: Optional[List[str]] = None,
        custom_domain: Optional[str] = None,
        ssl_enabled: bool = False
    ) -> Deployment:
        """
        Configure networking settings.

        Args:
            db: Database session
            deployment_id: Deployment ID
            ports: Port mappings [{"container": 8080, "host": 8080, "protocol": "tcp"}]
            expose_public: Whether to expose publicly
            allowed_ips: IP whitelist
            custom_domain: Custom domain name
            ssl_enabled: Enable SSL/TLS

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}
        network = config.get("network", {})

        if ports:
            network["ports"] = ports
        network["expose_public"] = expose_public
        if allowed_ips:
            network["allowed_ips"] = allowed_ips
        if custom_domain:
            network["custom_domain"] = custom_domain
        network["ssl_enabled"] = ssl_enabled

        config["network"] = network
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def set_advanced_gpu_config(
        db: Session,
        deployment_id: str,
        gpu_types: Optional[List[str]] = None,
        multi_gpu_strategy: Optional[str] = None,
        cuda_version: Optional[str] = None,
        tensor_cores: bool = False,
        gpu_utilization_target: Optional[float] = None
    ) -> Deployment:
        """
        Configure advanced GPU settings.

        Args:
            db: Database session
            deployment_id: Deployment ID
            gpu_types: Preferred GPU types in order
            multi_gpu_strategy: Strategy for multi-GPU ("data_parallel", "model_parallel")
            cuda_version: Required CUDA version
            tensor_cores: Require tensor cores
            gpu_utilization_target: Target GPU utilization (0-1)

        Returns:
            Updated Deployment instance
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}
        gpu_config = config.get("gpu", {})

        if gpu_types:
            gpu_config["preferred_types"] = gpu_types
        if multi_gpu_strategy:
            gpu_config["multi_gpu_strategy"] = multi_gpu_strategy
        if cuda_version:
            gpu_config["cuda_version"] = cuda_version
        gpu_config["tensor_cores_required"] = tensor_cores
        if gpu_utilization_target:
            gpu_config["utilization_target"] = gpu_utilization_target

        config["gpu"] = gpu_config
        deployment.config = config

        db.commit()
        db.refresh(deployment)

        return deployment

    @staticmethod
    def get_full_config(
        db: Session,
        deployment_id: str
    ) -> Dict[str, Any]:
        """
        Get complete configuration for a deployment.

        Args:
            db: Database session
            deployment_id: Deployment ID

        Returns:
            Full configuration dictionary
        """
        deployment = db.query(Deployment).filter(
            Deployment.deployment_id == deployment_id
        ).first()

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        config = deployment.config or {}

        return {
            "deployment_id": deployment.deployment_id,
            "deployment_name": deployment.deployment_name,
            "basic": {
                "gpu_type": deployment.gpu_type,
                "gpu_count": deployment.gpu_count,
                "region": deployment.region,
                "deployment_type": deployment.deployment_type.value,
            },
            "scaling": {
                "min_workers": deployment.min_workers,
                "max_workers": deployment.max_workers,
                "timeout_seconds": deployment.timeout_seconds,
                "autoscaling_enabled": deployment.autoscaling_enabled,
                "policy": config.get("scaling_policy", {}),
            },
            "environment": config.get("environment", {}),
            "health_check": config.get("health_check", {}),
            "resources": config.get("resources", {}),
            "docker": config.get("docker", {}),
            "network": config.get("network", {}),
            "gpu": config.get("gpu", {}),
            "metadata": deployment.extra_metadata or {},
        }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate deployment configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate scaling policy
        if "scaling_policy" in config:
            policy = config["scaling_policy"]
            if policy.get("min_instances", 0) > policy.get("max_instances", 0):
                return False, "min_instances cannot be greater than max_instances"

        # Validate health check
        if "health_check" in config:
            hc = config["health_check"]
            if hc.get("interval_seconds", 30) < 10:
                return False, "Health check interval must be at least 10 seconds"
            if hc.get("timeout_seconds", 10) > hc.get("interval_seconds", 30):
                return False, "Health check timeout cannot exceed interval"

        # Validate network
        if "network" in config:
            network = config["network"]
            if "ports" in network:
                for port in network["ports"]:
                    if not isinstance(port.get("container"), int):
                        return False, "Container port must be an integer"

        return True, None
