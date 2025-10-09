"""
Provider Comparison Service.

Helps users compare different cloud GPU providers and select the best option.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ProviderComparison:
    """Comparison data for cloud GPU providers."""
    provider_name: str
    pricing_model: str
    min_cost_per_hour: float
    max_cost_per_hour: float
    supports_autoscaling: bool
    supports_spot_instances: bool
    supports_ssh: bool
    deployment_speed: str  # "instant", "fast" (<5min), "moderate" (<15min)
    reliability_score: float  # 0-100
    gpu_availability: str  # "high", "medium", "low"
    regions: List[str]
    best_for: List[str]
    limitations: List[str]
    documentation_quality: str  # "excellent", "good", "fair"


class ProviderComparisonService:
    """Service for comparing deployment providers."""

    @staticmethod
    def get_all_providers() -> List[ProviderComparison]:
        """
        Get comparison data for all supported providers.

        Returns:
            List of ProviderComparison objects
        """
        return [
            # RunPod
            ProviderComparison(
                provider_name="RunPod",
                pricing_model="Pay-per-second",
                min_cost_per_hour=0.30,
                max_cost_per_hour=5.00,
                supports_autoscaling=True,
                supports_spot_instances=True,
                supports_ssh=True,
                deployment_speed="fast",
                reliability_score=90.0,
                gpu_availability="high",
                regions=["US-East", "US-West", "EU-Central", "Asia-Pacific"],
                best_for=["ML inference", "Training", "Cost-sensitive workloads", "Serverless"],
                limitations=["Community pods may have lower reliability", "Network storage costs extra"],
                documentation_quality="excellent"
            ),

            # Vast.ai
            ProviderComparison(
                provider_name="Vast.ai",
                pricing_model="Pay-per-minute",
                min_cost_per_hour=0.10,
                max_cost_per_hour=3.00,
                supports_autoscaling=False,
                supports_spot_instances=True,
                supports_ssh=True,
                deployment_speed="moderate",
                reliability_score=75.0,
                gpu_availability="high",
                regions=["Global (marketplace)"],
                best_for=["Budget-conscious projects", "Development/testing", "Flexible GPU needs"],
                limitations=["Marketplace model - variable reliability", "No managed services", "Manual instance selection"],
                documentation_quality="good"
            ),

            # Lambda Labs
            ProviderComparison(
                provider_name="Lambda Labs",
                pricing_model="Pay-per-second",
                min_cost_per_hour=0.60,
                max_cost_per_hour=4.00,
                supports_autoscaling=False,
                supports_spot_instances=False,
                supports_ssh=True,
                deployment_speed="instant",
                reliability_score=95.0,
                gpu_availability="medium",
                regions=["US-West", "US-East", "US-South", "EU-Central", "Asia-South"],
                best_for=["Enterprise workloads", "High reliability requirements", "Long-running jobs"],
                limitations=["Limited GPU availability", "No spot instances", "Higher minimum cost"],
                documentation_quality="excellent"
            ),

            # Replicate
            ProviderComparison(
                provider_name="Replicate",
                pricing_model="Pay-per-prediction",
                min_cost_per_hour=0.36,  # Estimated based on average use
                max_cost_per_hour=5.04,  # Estimated based on average use
                supports_autoscaling=True,
                supports_spot_instances=False,
                supports_ssh=False,
                deployment_speed="instant",
                reliability_score=98.0,
                gpu_availability="high",
                regions=["US (managed)"],
                best_for=["API-based inference", "Serverless deployments", "Auto-scaling workloads", "Public models"],
                limitations=["No SSH access", "Prediction-based billing", "Less control over infrastructure"],
                documentation_quality="excellent"
            ),

            # Together AI
            ProviderComparison(
                provider_name="Together AI",
                pricing_model="Pay-per-token",
                min_cost_per_hour=0.20,  # Estimated
                max_cost_per_hour=3.00,  # Estimated
                supports_autoscaling=True,
                supports_spot_instances=False,
                supports_ssh=False,
                deployment_speed="instant",
                reliability_score=95.0,
                gpu_availability="high",
                regions=["US (managed)"],
                best_for=["LLM inference", "Chat applications", "Token-based pricing", "Managed deployments"],
                limitations=["Focused on LLMs", "No custom containers", "Limited to supported models"],
                documentation_quality="excellent"
            ),

            # Modal
            ProviderComparison(
                provider_name="Modal",
                pricing_model="Pay-per-second",
                min_cost_per_hour=0.00,  # Free tier available
                max_cost_per_hour=4.00,
                supports_autoscaling=True,
                supports_spot_instances=False,
                supports_ssh=False,
                deployment_speed="instant",
                reliability_score=92.0,
                gpu_availability="high",
                regions=["US (managed)"],
                best_for=["Python applications", "Serverless functions", "Auto-scaling", "Developer-friendly"],
                limitations=["Python-focused", "Less control over infrastructure", "Newer platform"],
                documentation_quality="excellent"
            ),
        ]

    @staticmethod
    def compare_for_use_case(
        use_case: str,
        budget_per_hour: Optional[float] = None,
        required_features: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare providers for a specific use case.

        Args:
            use_case: Use case category (e.g., "training", "inference", "development")
            budget_per_hour: Maximum budget per hour
            required_features: List of required features (e.g., "ssh", "autoscaling")

        Returns:
            List of providers ranked by suitability
        """
        providers = ProviderComparisonService.get_all_providers()
        scored_providers = []

        for provider in providers:
            score = 0
            reasons = []

            # Check budget constraint
            if budget_per_hour and provider.min_cost_per_hour > budget_per_hour:
                continue  # Skip providers outside budget

            # Score based on use case
            if use_case in [item.lower() for item in provider.best_for]:
                score += 30
                reasons.append(f"Optimized for {use_case}")

            # Score based on required features
            if required_features:
                if "ssh" in required_features and provider.supports_ssh:
                    score += 10
                    reasons.append("SSH access supported")
                if "autoscaling" in required_features and provider.supports_autoscaling:
                    score += 10
                    reasons.append("Autoscaling supported")
                if "spot" in required_features and provider.supports_spot_instances:
                    score += 10
                    reasons.append("Spot instances available")

            # Score based on reliability
            score += provider.reliability_score / 5  # Max 20 points

            # Score based on deployment speed
            speed_scores = {"instant": 15, "fast": 10, "moderate": 5}
            score += speed_scores.get(provider.deployment_speed, 0)

            # Score based on GPU availability
            availability_scores = {"high": 15, "medium": 10, "low": 5}
            score += availability_scores.get(provider.gpu_availability, 0)

            scored_providers.append({
                "provider": provider.provider_name,
                "score": round(score, 1),
                "reasons": reasons,
                "min_cost_per_hour": provider.min_cost_per_hour,
                "max_cost_per_hour": provider.max_cost_per_hour,
                "reliability": provider.reliability_score,
                "deployment_speed": provider.deployment_speed,
                "supports_autoscaling": provider.supports_autoscaling,
                "supports_ssh": provider.supports_ssh,
                "regions": provider.regions,
                "best_for": provider.best_for,
                "limitations": provider.limitations
            })

        # Sort by score (descending)
        scored_providers.sort(key=lambda x: x["score"], reverse=True)

        return scored_providers

    @staticmethod
    def get_cheapest_option(
        gpu_type: str,
        min_gpu_count: int = 1
    ) -> Dict[str, Any]:
        """
        Find the cheapest provider for a given GPU type.

        Args:
            gpu_type: GPU type (e.g., "A100", "H100")
            min_gpu_count: Minimum number of GPUs needed

        Returns:
            Dictionary with cheapest provider information
        """
        providers = ProviderComparisonService.get_all_providers()

        # Filter providers that support the GPU type (simplified)
        # In reality, you'd query actual pricing for the specific GPU
        cheapest = min(providers, key=lambda p: p.min_cost_per_hour)

        return {
            "provider": cheapest.provider_name,
            "estimated_cost_per_hour": cheapest.min_cost_per_hour * min_gpu_count,
            "pricing_model": cheapest.pricing_model,
            "deployment_speed": cheapest.deployment_speed,
            "note": f"Actual cost may vary based on specific {gpu_type} pricing"
        }

    @staticmethod
    def get_provider_matrix() -> Dict[str, Any]:
        """
        Get a complete comparison matrix of all providers.

        Returns:
            Dictionary with comprehensive comparison data
        """
        providers = ProviderComparisonService.get_all_providers()

        matrix = {
            "providers": [p.provider_name for p in providers],
            "features": {
                "autoscaling": {p.provider_name: p.supports_autoscaling for p in providers},
                "spot_instances": {p.provider_name: p.supports_spot_instances for p in providers},
                "ssh_access": {p.provider_name: p.supports_ssh for p in providers},
            },
            "pricing": {
                p.provider_name: {
                    "model": p.pricing_model,
                    "min_per_hour": p.min_cost_per_hour,
                    "max_per_hour": p.max_cost_per_hour
                }
                for p in providers
            },
            "reliability": {p.provider_name: p.reliability_score for p in providers},
            "deployment_speed": {p.provider_name: p.deployment_speed for p in providers},
            "gpu_availability": {p.provider_name: p.gpu_availability for p in providers},
            "best_for": {p.provider_name: p.best_for for p in providers},
            "limitations": {p.provider_name: p.limitations for p in providers}
        }

        return matrix

    @staticmethod
    def recommend_provider(
        workload_type: str,
        duration_hours: Optional[float] = None,
        budget: Optional[float] = None,
        priority: str = "cost"  # "cost", "reliability", "speed"
    ) -> Dict[str, Any]:
        """
        Get provider recommendation based on requirements.

        Args:
            workload_type: Type of workload (training, inference, development)
            duration_hours: Expected duration in hours
            budget: Total budget
            priority: What to prioritize (cost, reliability, speed)

        Returns:
            Recommendation with reasoning
        """
        providers = ProviderComparisonService.get_all_providers()

        # Calculate budget per hour if total budget given
        budget_per_hour = budget / duration_hours if budget and duration_hours else None

        # Score providers based on priority
        scored = []
        for provider in providers:
            score = 0

            if priority == "cost":
                # Prefer lower cost
                score = 100 - (provider.min_cost_per_hour * 10)
            elif priority == "reliability":
                score = provider.reliability_score
            elif priority == "speed":
                speed_scores = {"instant": 100, "fast": 80, "moderate": 60}
                score = speed_scores.get(provider.deployment_speed, 40)

            # Apply budget constraint
            if budget_per_hour and provider.min_cost_per_hour > budget_per_hour:
                score = 0  # Eliminate from consideration

            scored.append((provider, score))

        # Get best match
        scored.sort(key=lambda x: x[1], reverse=True)
        best_provider, best_score = scored[0]

        recommendation = {
            "recommended_provider": best_provider.provider_name,
            "confidence_score": round(best_score, 1),
            "reasoning": [],
            "estimated_cost": None,
            "alternatives": []
        }

        # Add reasoning
        if priority == "cost":
            recommendation["reasoning"].append(f"Lowest cost option at ${best_provider.min_cost_per_hour}/hour")
        if priority == "reliability":
            recommendation["reasoning"].append(f"High reliability score of {best_provider.reliability_score}%")
        if priority == "speed":
            recommendation["reasoning"].append(f"{best_provider.deployment_speed.capitalize()} deployment speed")

        if workload_type.lower() in [item.lower() for item in best_provider.best_for]:
            recommendation["reasoning"].append(f"Optimized for {workload_type} workloads")

        # Calculate estimated cost
        if duration_hours:
            estimated_cost = best_provider.min_cost_per_hour * duration_hours
            recommendation["estimated_cost"] = {
                "amount": round(estimated_cost, 2),
                "duration_hours": duration_hours,
                "hourly_rate": best_provider.min_cost_per_hour
            }

        # Add alternatives
        for provider, score in scored[1:3]:  # Top 2 alternatives
            recommendation["alternatives"].append({
                "provider": provider.provider_name,
                "score": round(score, 1),
                "cost_per_hour": provider.min_cost_per_hour
            })

        return recommendation
