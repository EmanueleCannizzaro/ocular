"""Pydantic models for deployment selector options."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class DeploymentModelOption(BaseModel):
    """Lightweight model option for deployment selectors."""

    id: int
    name: str
    display_name: str
    version: str
    status: str
    min_vram_gb: int
    supports_vllm: bool
    description: Optional[str]
    recommended_gpu: Optional[str]
    license: Optional[str]
    pipeline_tag: Optional[str]
    metadata: dict


class DeploymentProviderOption(BaseModel):
    """Lightweight provider option for deployment selectors."""

    id: int
    name: str
    display_name: str
    status: str
    available_gpus: List[str]
    supports_autoscaling: bool
    supports_spot_instances: bool
    min_cost_per_hour: Optional[float]
    max_cost_per_hour: Optional[float]
    description: Optional[str]
    metadata: dict


class DeploymentOptionsResponse(BaseModel):
    """Aggregated options returned to the deployment wizard."""

    models: List[DeploymentModelOption]
    providers: List[DeploymentProviderOption]
    generated_at: str

    @classmethod
    def from_payload(
        cls,
        models: List[DeploymentModelOption],
        providers: List[DeploymentProviderOption],
    ) -> "DeploymentOptionsResponse":
        """Build a response with a generated timestamp."""

        return cls(
            models=models,
            providers=providers,
            generated_at=datetime.utcnow().isoformat(),
        )
