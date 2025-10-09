"""
Version Management Service.

Handles model version tracking, rollback, comparison, and A/B testing.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ocular.database.models import (
    Model, ModelVersion, VersionDeployment, VersionStatus
)


class VersionService:
    """Service for managing model versions."""

    @staticmethod
    def create_version(
        db: Session,
        model_id: int,
        version_number: str,
        changelog: Optional[str] = None,
        version_name: Optional[str] = None,
        version_tag: Optional[str] = None,
        docker_image: Optional[str] = None,
        huggingface_repo: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create a new model version.

        Args:
            db: Database session
            model_id: Model ID
            version_number: Version identifier (e.g., 'v1.0.0')
            changelog: Description of changes
            version_name: Human-readable version name
            version_tag: Tag like 'latest', 'stable', 'beta'
            docker_image: Docker image for this version
            huggingface_repo: HuggingFace repository
            config: Version-specific configuration
            created_by: User who created this version

        Returns:
            Created ModelVersion instance
        """
        # Check if model exists
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        # Create new version
        version = ModelVersion(
            model_id=model_id,
            version_number=version_number,
            version_name=version_name,
            version_tag=version_tag,
            changelog=changelog,
            docker_image=docker_image,
            huggingface_repo=huggingface_repo,
            config=config or {},
            created_by=created_by,
            status=VersionStatus.DRAFT,
        )

        db.add(version)
        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def get_version(db: Session, version_id: int) -> Optional[ModelVersion]:
        """Get a specific model version by ID."""
        return db.query(ModelVersion).filter(ModelVersion.id == version_id).first()

    @staticmethod
    def get_model_versions(
        db: Session,
        model_id: int,
        status: Optional[VersionStatus] = None,
        limit: int = 50
    ) -> List[ModelVersion]:
        """
        Get all versions for a model.

        Args:
            db: Database session
            model_id: Model ID
            status: Filter by version status
            limit: Maximum number of versions to return

        Returns:
            List of ModelVersion instances
        """
        query = db.query(ModelVersion).filter(ModelVersion.model_id == model_id)

        if status:
            query = query.filter(ModelVersion.status == status)

        return query.order_by(desc(ModelVersion.created_at)).limit(limit).all()

    @staticmethod
    def get_active_version(db: Session, model_id: int) -> Optional[ModelVersion]:
        """Get the currently active version for a model."""
        return db.query(ModelVersion).filter(
            and_(
                ModelVersion.model_id == model_id,
                ModelVersion.is_active == True,
                ModelVersion.status == VersionStatus.ACTIVE
            )
        ).first()

    @staticmethod
    def activate_version(
        db: Session,
        version_id: int,
        deactivate_others: bool = True
    ) -> ModelVersion:
        """
        Activate a specific version, optionally deactivating others.

        Args:
            db: Database session
            version_id: Version ID to activate
            deactivate_others: If True, deactivate all other versions

        Returns:
            Activated ModelVersion instance
        """
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not version:
            raise ValueError(f"Version with ID {version_id} not found")

        # Deactivate other versions if requested
        if deactivate_others:
            db.query(ModelVersion).filter(
                and_(
                    ModelVersion.model_id == version.model_id,
                    ModelVersion.id != version_id
                )
            ).update({"is_active": False})

        # Activate this version
        version.is_active = True
        version.status = VersionStatus.ACTIVE
        version.deployed_at = datetime.utcnow()

        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def rollback_to_version(
        db: Session,
        model_id: int,
        version_id: int
    ) -> ModelVersion:
        """
        Rollback to a previous version by activating it.

        Args:
            db: Database session
            model_id: Model ID
            version_id: Version ID to rollback to

        Returns:
            Activated ModelVersion instance
        """
        # Verify version belongs to model
        version = db.query(ModelVersion).filter(
            and_(
                ModelVersion.id == version_id,
                ModelVersion.model_id == model_id
            )
        ).first()

        if not version:
            raise ValueError(f"Version {version_id} not found for model {model_id}")

        # Activate this version (will deactivate others)
        return VersionService.activate_version(db, version_id, deactivate_others=True)

    @staticmethod
    def compare_versions(
        db: Session,
        version_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple versions side-by-side.

        Args:
            db: Database session
            version_ids: List of version IDs to compare

        Returns:
            List of version data dictionaries with comparison metrics
        """
        if len(version_ids) < 2:
            raise ValueError("At least 2 versions required for comparison")

        if len(version_ids) > 5:
            raise ValueError("Maximum 5 versions can be compared at once")

        versions = db.query(ModelVersion).filter(
            ModelVersion.id.in_(version_ids)
        ).all()

        if len(versions) != len(version_ids):
            raise ValueError("Some versions not found")

        # Convert to dictionaries with additional comparison metrics
        comparison_data = []
        for version in versions:
            data = version.to_dict()

            # Add deployment count
            deployment_count = db.query(VersionDeployment).filter(
                VersionDeployment.version_id == version.id
            ).count()
            data["deployment_count"] = deployment_count

            # Calculate success rate from deployments
            deployments = db.query(VersionDeployment).filter(
                VersionDeployment.version_id == version.id
            ).all()

            total_requests = sum(d.total_requests for d in deployments)
            successful_requests = sum(d.successful_requests for d in deployments)

            data["total_requests"] = total_requests
            data["success_rate"] = round(
                (successful_requests / total_requests * 100), 2
            ) if total_requests > 0 else 0

            comparison_data.append(data)

        return comparison_data

    @staticmethod
    def update_version_metrics(
        db: Session,
        version_id: int,
        benchmark_scores: Optional[Dict[str, Any]] = None,
        avg_inference_time_ms: Optional[float] = None,
        avg_tokens_per_second: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> ModelVersion:
        """
        Update performance metrics for a version.

        Args:
            db: Database session
            version_id: Version ID
            benchmark_scores: Benchmark test results
            avg_inference_time_ms: Average inference time
            avg_tokens_per_second: Average token generation speed
            quality_score: Overall quality metric

        Returns:
            Updated ModelVersion instance
        """
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not version:
            raise ValueError(f"Version with ID {version_id} not found")

        if benchmark_scores is not None:
            version.benchmark_scores = benchmark_scores

        if avg_inference_time_ms is not None:
            version.avg_inference_time_ms = avg_inference_time_ms

        if avg_tokens_per_second is not None:
            version.avg_tokens_per_second = avg_tokens_per_second

        if quality_score is not None:
            version.quality_score = quality_score

        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def setup_ab_test(
        db: Session,
        version_id_a: int,
        version_id_b: int,
        traffic_split: float = 50.0,
        duration_days: int = 7,
    ) -> tuple[ModelVersion, ModelVersion]:
        """
        Setup A/B testing between two versions.

        Args:
            db: Database session
            version_id_a: First version ID (baseline)
            version_id_b: Second version ID (test)
            traffic_split: Percentage of traffic for version B (0-100)
            duration_days: Duration of A/B test in days

        Returns:
            Tuple of (version_a, version_b)
        """
        version_a = db.query(ModelVersion).filter(ModelVersion.id == version_id_a).first()
        version_b = db.query(ModelVersion).filter(ModelVersion.id == version_id_b).first()

        if not version_a or not version_b:
            raise ValueError("One or both versions not found")

        if version_a.model_id != version_b.model_id:
            raise ValueError("Versions must belong to the same model")

        # Setup A/B test for both versions
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=duration_days)

        version_a.ab_test_enabled = True
        version_a.ab_test_traffic_percentage = 100.0 - traffic_split
        version_a.ab_test_start_date = start_date
        version_a.ab_test_end_date = end_date

        version_b.ab_test_enabled = True
        version_b.ab_test_traffic_percentage = traffic_split
        version_b.ab_test_start_date = start_date
        version_b.ab_test_end_date = end_date

        db.commit()
        db.refresh(version_a)
        db.refresh(version_b)

        return version_a, version_b

    @staticmethod
    def end_ab_test(
        db: Session,
        version_id: int,
        promote_to_active: bool = False
    ) -> ModelVersion:
        """
        End an A/B test for a version.

        Args:
            db: Database session
            version_id: Version ID
            promote_to_active: If True, promote this version to active

        Returns:
            Updated ModelVersion instance
        """
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not version:
            raise ValueError(f"Version with ID {version_id} not found")

        version.ab_test_enabled = False
        version.ab_test_end_date = datetime.utcnow()

        if promote_to_active:
            version.is_active = True
            version.status = VersionStatus.ACTIVE

            # Deactivate other versions
            db.query(ModelVersion).filter(
                and_(
                    ModelVersion.model_id == version.model_id,
                    ModelVersion.id != version_id
                )
            ).update({"is_active": False})

        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def deprecate_version(
        db: Session,
        version_id: int,
        reason: Optional[str] = None
    ) -> ModelVersion:
        """
        Deprecate a version.

        Args:
            db: Database session
            version_id: Version ID
            reason: Reason for deprecation

        Returns:
            Deprecated ModelVersion instance
        """
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not version:
            raise ValueError(f"Version with ID {version_id} not found")

        version.status = VersionStatus.DEPRECATED
        version.is_active = False
        version.deprecated_at = datetime.utcnow()

        if reason:
            metadata = version.extra_metadata or {}
            metadata["deprecation_reason"] = reason
            version.extra_metadata = metadata

        db.commit()
        db.refresh(version)

        return version

    @staticmethod
    def get_version_timeline(
        db: Session,
        model_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get version history timeline for a model.

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            List of version events in chronological order
        """
        versions = db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id
        ).order_by(ModelVersion.created_at).all()

        timeline = []
        for version in versions:
            # Add creation event
            timeline.append({
                "version_id": version.id,
                "version_number": version.version_number,
                "event_type": "created",
                "timestamp": version.created_at,
                "details": {
                    "status": version.status.value,
                    "changelog": version.changelog,
                }
            })

            # Add deployment event
            if version.deployed_at:
                timeline.append({
                    "version_id": version.id,
                    "version_number": version.version_number,
                    "event_type": "deployed",
                    "timestamp": version.deployed_at,
                    "details": {
                        "status": version.status.value,
                    }
                })

            # Add deprecation event
            if version.deprecated_at:
                timeline.append({
                    "version_id": version.id,
                    "version_number": version.version_number,
                    "event_type": "deprecated",
                    "timestamp": version.deprecated_at,
                    "details": {
                        "reason": version.extra_metadata.get("deprecation_reason") if version.extra_metadata else None,
                    }
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline
