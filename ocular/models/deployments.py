from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# Pydantic Models

class EnvironmentVariablesRequest(BaseModel):
    """Request model for setting environment variables."""
    env_vars: Dict[str, str] = Field(..., description="Environment variables")
    merge: bool = Field(True, description="Merge with existing vars or replace")


class ScalingPolicyRequest(BaseModel):
    """Request model for scaling policy."""
    min_instances: int = Field(0, ge=0, description="Minimum instances")
    max_instances: int = Field(10, ge=1, le=100, description="Maximum instances")
    target_utilization: float = Field(0.8, ge=0.1, le=1.0, description="Target utilization")
    scale_up_threshold: float = Field(0.9, ge=0.1, le=1.0)
    scale_down_threshold: float = Field(0.3, ge=0.1, le=1.0)
    cooldown_period_seconds: int = Field(300, ge=60, description="Cooldown period")
    metrics: List[str] = Field(default=["cpu", "gpu", "memory", "requests_per_second"])


class HealthCheckRequest(BaseModel):
    """Request model for health check configuration."""
    enabled: bool = Field(True)
    endpoint: str = Field("/health", description="Health check endpoint")
    interval_seconds: int = Field(30, ge=10, le=300)
    timeout_seconds: int = Field(10, ge=5, le=60)
    unhealthy_threshold: int = Field(3, ge=1, le=10)
    healthy_threshold: int = Field(2, ge=1, le=10)
    http_method: str = Field("GET", description="HTTP method")
    expected_status_codes: List[int] = Field(default=[200, 204])
    headers: Dict[str, str] = Field(default={})


class ResourceLimitsRequest(BaseModel):
    """Request model for resource limits."""
    cpu_limit: Optional[str] = Field(None, description="CPU limit (e.g., '2', '2000m')")
    memory_limit: Optional[str] = Field(None, description="Memory limit (e.g., '4Gi')")
    gpu_memory_limit: Optional[str] = Field(None, description="GPU memory limit")
    disk_limit: Optional[str] = Field(None, description="Disk limit")


class DockerConfigRequest(BaseModel):
    """Request model for Docker configuration."""
    dockerfile_path: Optional[str] = None
    build_args: Optional[Dict[str, str]] = None
    docker_args: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    command: Optional[List[str]] = None
    working_dir: Optional[str] = None


class PortMapping(BaseModel):
    """Port mapping configuration."""
    container: int = Field(..., ge=1, le=65535)
    host: int = Field(..., ge=1, le=65535)
    protocol: str = Field("tcp", description="Protocol (tcp/udp)")


class NetworkConfigRequest(BaseModel):
    """Request model for network configuration."""
    ports: Optional[List[PortMapping]] = None
    expose_public: bool = Field(False)
    allowed_ips: Optional[List[str]] = None
    custom_domain: Optional[str] = None
    ssl_enabled: bool = Field(False)


class GPUConfigRequest(BaseModel):
    """Request model for advanced GPU configuration."""
    gpu_types: Optional[List[str]] = Field(None, description="Preferred GPU types")
    multi_gpu_strategy: Optional[str] = Field(None, description="Multi-GPU strategy")
    cuda_version: Optional[str] = Field(None, description="CUDA version")
    tensor_cores: bool = Field(False, description="Require tensor cores")
    gpu_utilization_target: Optional[float] = Field(None, ge=0.0, le=1.0)