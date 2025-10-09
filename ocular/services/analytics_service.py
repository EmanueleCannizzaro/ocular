"""
Analytics Service for Ocular OCR Platform.

Provides metrics calculation, aggregation, and reporting for deployments,
models, and system performance.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import func, and_, case
from sqlalchemy.orm import Session

from ocular.database.models import Deployment, DeploymentStatus, Model, Provider


class AnalyticsService:
    """Service for analytics and metrics calculation."""

    def __init__(self, db: Session):
        """Initialize analytics service with database session."""
        self.db = db

    def get_deployment_timeline(
        self,
        days: int = 30,
        provider_id: Optional[int] = None,
        model_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get deployment timeline data for the specified period.

        Args:
            days: Number of days to analyze
            provider_id: Optional filter by provider
            model_id: Optional filter by model

        Returns:
            Timeline data with daily deployment counts and statuses
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        # Build query
        query = self.db.query(
            func.date(Deployment.created_at).label('date'),
            Deployment.status,
            func.count(Deployment.id).label('count')
        ).filter(
            Deployment.created_at >= start_date
        )

        if provider_id:
            query = query.filter(Deployment.provider_id == provider_id)
        if model_id:
            query = query.filter(Deployment.model_id == model_id)

        query = query.group_by(
            func.date(Deployment.created_at),
            Deployment.status
        ).order_by(func.date(Deployment.created_at))

        results = query.all()

        # Organize data by date and status
        timeline = {}
        for result in results:
            date_str = result.date.isoformat()
            if date_str not in timeline:
                timeline[date_str] = {
                    'date': date_str,
                    'total': 0,
                    'running': 0,
                    'stopped': 0,
                    'failed': 0,
                    'deploying': 0
                }

            timeline[date_str]['total'] += result.count

            status_key = result.status.value.lower() if hasattr(result.status, 'value') else str(result.status).lower()
            if status_key in timeline[date_str]:
                timeline[date_str][status_key] += result.count

        return {
            'period_days': days,
            'start_date': start_date.isoformat(),
            'end_date': datetime.utcnow().isoformat(),
            'timeline': list(timeline.values())
        }

    def get_cost_analytics(
        self,
        days: int = 30,
        provider_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate cost analytics for deployments.

        Args:
            days: Number of days to analyze
            provider_id: Optional filter by provider

        Returns:
            Cost breakdown and trends
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get active deployments
        query = self.db.query(Deployment).filter(
            Deployment.created_at >= start_date
        )

        if provider_id:
            query = query.filter(Deployment.provider_id == provider_id)

        deployments = query.all()

        # Calculate costs (simplified - in production, use actual pricing)
        daily_costs = {}
        total_cost = 0
        cost_by_gpu = {}
        cost_by_model = {}

        for deployment in deployments:
            # Estimate cost based on GPU type and uptime
            # This is a simplified calculation
            cost_class = deployment.cost_class
            gpu_type = deployment.gpu_type

            # Hourly rates (example rates in USD)
            gpu_hourly_rates = {
                'A40': 0.79,
                'A100': 1.69,
                'RTX 4090': 0.69,
                'L40': 0.99,
                'H100': 3.99
            }

            hourly_rate = gpu_hourly_rates.get(gpu_type, 1.0)

            # Calculate uptime hours
            if deployment.status == DeploymentStatus.RUNNING:
                uptime = (datetime.utcnow() - deployment.created_at).total_seconds() / 3600
            elif deployment.ended_at:
                uptime = (deployment.ended_at - deployment.created_at).total_seconds() / 3600
            else:
                uptime = 0

            deployment_cost = uptime * hourly_rate * deployment.gpu_count
            total_cost += deployment_cost

            # Group by date
            deployment_date = deployment.created_at.date().isoformat()
            if deployment_date not in daily_costs:
                daily_costs[deployment_date] = 0
            daily_costs[deployment_date] += deployment_cost

            # Group by GPU type
            if gpu_type not in cost_by_gpu:
                cost_by_gpu[gpu_type] = 0
            cost_by_gpu[gpu_type] += deployment_cost

            # Group by model
            if deployment.model_id:
                model_name = deployment.model.name if deployment.model else 'Unknown'
                if model_name not in cost_by_model:
                    cost_by_model[model_name] = 0
                cost_by_model[model_name] += deployment_cost

        # Calculate daily breakdown
        daily_breakdown = [
            {'date': date, 'cost': round(cost, 2)}
            for date, cost in sorted(daily_costs.items())
        ]

        return {
            'period_days': days,
            'total_cost': round(total_cost, 2),
            'average_daily_cost': round(total_cost / max(days, 1), 2),
            'daily_breakdown': daily_breakdown,
            'cost_by_gpu': {gpu: round(cost, 2) for gpu, cost in cost_by_gpu.items()},
            'cost_by_model': {model: round(cost, 2) for model, cost in cost_by_model.items()},
            'currency': 'USD'
        }

    def get_gpu_utilization(
        self,
        provider_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate GPU utilization metrics.

        Args:
            provider_id: Optional filter by provider

        Returns:
            GPU utilization statistics
        """
        # Get all deployments
        query = self.db.query(Deployment)

        if provider_id:
            query = query.filter(Deployment.provider_id == provider_id)

        deployments = query.all()

        # Calculate utilization by GPU type
        gpu_stats = {}
        total_gpus = 0
        active_gpus = 0

        for deployment in deployments:
            gpu_type = deployment.gpu_type
            gpu_count = deployment.gpu_count

            if gpu_type not in gpu_stats:
                gpu_stats[gpu_type] = {
                    'total': 0,
                    'active': 0,
                    'idle': 0,
                    'utilization_percent': 0
                }

            gpu_stats[gpu_type]['total'] += gpu_count
            total_gpus += gpu_count

            if deployment.status == DeploymentStatus.RUNNING:
                gpu_stats[gpu_type]['active'] += gpu_count
                active_gpus += gpu_count
            else:
                gpu_stats[gpu_type]['idle'] += gpu_count

        # Calculate utilization percentages
        for gpu_type, stats in gpu_stats.items():
            if stats['total'] > 0:
                stats['utilization_percent'] = round(
                    (stats['active'] / stats['total']) * 100,
                    2
                )

        overall_utilization = round(
            (active_gpus / total_gpus * 100) if total_gpus > 0 else 0,
            2
        )

        return {
            'overall_utilization_percent': overall_utilization,
            'total_gpus': total_gpus,
            'active_gpus': active_gpus,
            'idle_gpus': total_gpus - active_gpus,
            'by_gpu_type': gpu_stats
        }

    def get_model_usage_stats(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get usage statistics for models.

        Args:
            days: Number of days to analyze

        Returns:
            Model usage statistics
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get deployment counts by model
        query = self.db.query(
            Model.id,
            Model.name,
            Model.display_name,
            func.count(Deployment.id).label('deployment_count')
        ).join(
            Deployment,
            Deployment.model_id == Model.id
        ).filter(
            Deployment.created_at >= start_date
        ).group_by(
            Model.id,
            Model.name,
            Model.display_name
        ).order_by(
            func.count(Deployment.id).desc()
        )

        results = query.all()

        model_stats = []
        for result in results:
            model_stats.append({
                'model_id': result.id,
                'model_name': result.name,
                'display_name': result.display_name or result.name,
                'deployment_count': result.deployment_count
            })

        return {
            'period_days': days,
            'total_models_used': len(model_stats),
            'models': model_stats
        }

    def get_performance_trends(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate performance trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics and trends
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get deployments over time
        query = self.db.query(
            func.date(Deployment.created_at).label('date'),
            func.count(Deployment.id).label('total_deployments'),
            func.sum(
                case(
                    (Deployment.status == DeploymentStatus.RUNNING, 1),
                    else_=0
                )
            ).label('successful_deployments'),
            func.sum(
                case(
                    (Deployment.status == DeploymentStatus.ERROR, 1),
                    else_=0
                )
            ).label('failed_deployments')
        ).filter(
            Deployment.created_at >= start_date
        ).group_by(
            func.date(Deployment.created_at)
        ).order_by(
            func.date(Deployment.created_at)
        )

        results = query.all()

        trends = []
        total_deployments = 0
        total_successful = 0
        total_failed = 0

        for result in results:
            total_deployments += result.total_deployments
            total_successful += result.successful_deployments or 0
            total_failed += result.failed_deployments or 0

            success_rate = (
                (result.successful_deployments / result.total_deployments * 100)
                if result.total_deployments > 0 else 0
            )

            trends.append({
                'date': result.date.isoformat(),
                'total_deployments': result.total_deployments,
                'successful_deployments': result.successful_deployments or 0,
                'failed_deployments': result.failed_deployments or 0,
                'success_rate_percent': round(success_rate, 2)
            })

        overall_success_rate = (
            (total_successful / total_deployments * 100)
            if total_deployments > 0 else 0
        )

        return {
            'period_days': days,
            'total_deployments': total_deployments,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'overall_success_rate_percent': round(overall_success_rate, 2),
            'daily_trends': trends
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard summary with key metrics.

        Returns:
            Dashboard summary with all key metrics
        """
        # Active deployments
        active_deployments = self.db.query(Deployment).filter(
            Deployment.status == DeploymentStatus.RUNNING
        ).count()

        # Total deployments
        total_deployments = self.db.query(Deployment).count()

        # Total models
        total_models = self.db.query(Model).count()

        # Total providers
        total_providers = self.db.query(Provider).count()

        # Recent deployments (last 24 hours)
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_deployments = self.db.query(Deployment).filter(
            Deployment.created_at >= last_24h
        ).count()

        # Failed deployments (last 24 hours)
        recent_failures = self.db.query(Deployment).filter(
            and_(
                Deployment.created_at >= last_24h,
                Deployment.status == DeploymentStatus.ERROR
            )
        ).count()

        # Get cost analytics for last 7 days
        cost_data = self.get_cost_analytics(days=7)

        # Get GPU utilization
        gpu_data = self.get_gpu_utilization()

        return {
            'active_deployments': active_deployments,
            'total_deployments': total_deployments,
            'total_models': total_models,
            'total_providers': total_providers,
            'recent_deployments_24h': recent_deployments,
            'recent_failures_24h': recent_failures,
            'estimated_daily_cost': cost_data['average_daily_cost'],
            'gpu_utilization_percent': gpu_data['overall_utilization_percent'],
            'total_gpus': gpu_data['total_gpus'],
            'active_gpus': gpu_data['active_gpus'],
            'timestamp': datetime.utcnow().isoformat()
        }


def get_analytics_service(db: Session) -> AnalyticsService:
    """
    Factory function to create analytics service instance.

    Args:
        db: Database session

    Returns:
        AnalyticsService instance
    """
    return AnalyticsService(db)
