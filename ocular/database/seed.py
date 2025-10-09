"""
Database seeding script for initial providers and models.
Reads from ocular.json configuration file.
"""

import json
from pathlib import Path
from sqlalchemy.orm import Session

from ocular.database.models import (
    Provider,
    ProviderStatus,
    Model,
    ModelStatus,
    DeploymentType
)
from ocular.database.session import SessionLocal, init_db


def load_ocular_config() -> dict:
    """Load configuration from ocular.json."""
    config_file = Path(__file__).parent.parent.parent / "ocular.json"
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def seed_providers(db: Session, config: dict, update_existing: bool = False) -> int:
    """
    Seed providers table from ocular.json configuration.

    Args:
        db: Database session
        config: Configuration dict from ocular.json
        update_existing: If True, update existing providers with new data

    Returns:
        Number of providers processed
    """
    providers_data = config.get("cloud_providers", [])
    count = 0

    for provider_data in providers_data:
        # Check if provider already exists
        existing = db.query(Provider).filter(Provider.name == provider_data["name"]).first()

        if existing:
            if not update_existing:
                print(f"  ⊙ Provider '{provider_data['name']}' already exists, skipping")
                continue

            # Update existing provider
            existing.display_name = provider_data.get("display_name", existing.display_name)
            existing.description = provider_data.get("description")
            existing.website_url = provider_data.get("website")
            existing.documentation_url = provider_data.get("documentation_url")

            # Map status from config to enum
            status_str = provider_data.get("status", "INACTIVE").upper()
            if status_str == "ACTIVE":
                existing.status = ProviderStatus.ACTIVE
            elif status_str == "MAINTENANCE":
                existing.status = ProviderStatus.MAINTENANCE
            else:
                existing.status = ProviderStatus.INACTIVE

            existing.available_gpus = provider_data.get("available_gpus", [])

            # Extract features
            features = provider_data.get("features", {})
            existing.supports_autoscaling = features.get("autoscaling", False)
            existing.supports_spot_instances = features.get("spot_instances", False)

            # Store comprehensive metadata
            existing.extra_metadata = {
                "long_description": provider_data.get("long_description"),
                "pricing": provider_data.get("pricing", {}),
                "regions": provider_data.get("regions", []),
                "global_regions": provider_data.get("global_regions"),
                "features": features,
                "best_for": provider_data.get("best_for", []),
                "pros": provider_data.get("pros", []),
                "cons": provider_data.get("cons", []),
                "use_cases": provider_data.get("use_cases", [])
            }

            print(f"  ✓ Updated provider: {provider_data['name']}")
        else:
            # Create new provider
            status_str = provider_data.get("status", "INACTIVE").upper()
            if status_str == "ACTIVE":
                status = ProviderStatus.ACTIVE
            elif status_str == "MAINTENANCE":
                status = ProviderStatus.MAINTENANCE
            else:
                status = ProviderStatus.INACTIVE

            features = provider_data.get("features", {})

            # Extract pricing if available
            pricing = provider_data.get("pricing", {})
            min_cost = pricing.get("min_price_per_hour")
            max_cost = pricing.get("max_price_per_hour")

            provider = Provider(
                name=provider_data["name"],
                display_name=provider_data.get("display_name", provider_data["name"]),
                description=provider_data.get("description"),
                website_url=provider_data.get("website"),
                documentation_url=provider_data.get("documentation_url"),
                status=status,
                available_gpus=provider_data.get("available_gpus", []),
                min_cost_per_hour=min_cost,
                max_cost_per_hour=max_cost,
                supports_autoscaling=features.get("supports_autoscaling", False),
                supports_spot_instances=features.get("supports_spot_instances", False),
                extra_metadata={
                    "long_description": provider_data.get("long_description"),
                    "pricing": pricing,
                    "regions": provider_data.get("regions", []),
                    "global_regions": provider_data.get("global_regions"),
                    "features": features,
                    "best_for": provider_data.get("best_for", []),
                    "pros": provider_data.get("pros", []),
                    "cons": provider_data.get("cons", []),
                    "use_cases": provider_data.get("use_cases", [])
                }
            )

            db.add(provider)
            print(f"  + Added provider: {provider_data['name']}")

        count += 1

    db.commit()
    return count


def seed_models(db: Session, config: dict, update_existing: bool = False) -> int:
    """
    Seed models table from ocular.json configuration.

    Args:
        db: Database session
        config: Configuration dict from ocular.json
        update_existing: If True, update existing models with new data

    Returns:
        Number of models processed
    """
    models_data = config.get("ai_models", [])
    count = 0

    for model_data in models_data:
        # Check if model already exists
        existing = db.query(Model).filter(Model.name == model_data["name"]).first()

        if existing:
            if not update_existing:
                print(f"  ⊙ Model '{model_data['name']}' already exists, skipping")
                continue

            # Update existing model
            existing.display_name = model_data.get("display_name", existing.display_name)
            existing.version = model_data.get("version", existing.version)
            existing.description = model_data.get("description")

            # Map status from config to enum
            status_str = model_data.get("status", "INACTIVE").upper()
            if status_str == "ACTIVE":
                existing.status = ModelStatus.AVAILABLE
            elif status_str == "BETA":
                existing.status = ModelStatus.BETA
            else:
                existing.status = ModelStatus.DEPRECATED

            existing.min_vram_gb = model_data.get("min_vram_gb", existing.min_vram_gb)
            existing.recommended_gpu = model_data.get("recommended_gpu")
            existing.docker_image = model_data.get("docker_image")
            existing.supports_vllm = model_data.get("supports_vllm", False)

            # Determine deployment type
            if model_data.get("supports_vllm"):
                existing.default_deployment_type = DeploymentType.VLLM
            else:
                existing.default_deployment_type = DeploymentType.DOCKER

            existing.config_template = model_data.get("configuration", {})
            existing.extra_metadata = model_data.get("configuration", {})

            print(f"  ✓ Updated model: {model_data['name']}")
        else:
            # Create new model
            status_str = model_data.get("status", "INACTIVE").upper()
            if status_str == "ACTIVE":
                status = ModelStatus.AVAILABLE
            elif status_str == "BETA":
                status = ModelStatus.BETA
            else:
                status = ModelStatus.DEPRECATED

            # Determine deployment type
            if model_data.get("supports_vllm"):
                deployment_type = DeploymentType.VLLM
            else:
                deployment_type = DeploymentType.DOCKER

            model = Model(
                name=model_data["name"],
                display_name=model_data.get("display_name", model_data["name"]),
                version=model_data.get("version", "latest"),
                description=model_data.get("description"),
                status=status,
                min_vram_gb=model_data.get("min_vram_gb", 16),
                recommended_gpu=model_data.get("recommended_gpu"),
                docker_image=model_data.get("docker_image"),
                supports_vllm=model_data.get("supports_vllm", False),
                default_deployment_type=deployment_type,
                config_template=model_data.get("configuration", {}),
                extra_metadata=model_data.get("configuration", {})
            )

            db.add(model)
            print(f"  + Added model: {model_data['name']}")

        count += 1

    db.commit()
    return count


def seed_database(update_existing: bool = False) -> None:
    """
    Seed the database with providers and models from ocular.json.

    Args:
        update_existing: If True, update existing records with new data from config
    """
    print("=" * 60)
    print("Database Seeding Script")
    print("=" * 60)

    print("\n📊 Initializing database...")
    init_db()
    print("✓ Database initialized")

    print("\n📁 Loading configuration from ocular.json...")
    config = load_ocular_config()
    print("✓ Configuration loaded")
    print(f"   Cloud Providers: {len(config.get('cloud_providers', []))}")
    print(f"   AI Models: {len(config.get('ai_models', []))}")

    db = SessionLocal()

    try:
        # Seed providers
        print("\n🌐 Seeding cloud GPU providers...")
        provider_count = seed_providers(db, config, update_existing)
        print(f"✓ Processed {provider_count} providers")

        # Seed models
        print("\n🤖 Seeding AI models...")
        model_count = seed_models(db, config, update_existing)
        print(f"✓ Processed {model_count} models")

        print("\n" + "=" * 60)
        print("✅ Database seeding completed successfully!")
        print(f"   Providers: {provider_count}")
        print(f"   Models: {model_count}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error seeding database: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    # Check for --update flag
    update_existing = "--update" in sys.argv

    if update_existing:
        print("\n⚠️  Running in UPDATE mode - existing records will be updated")

    seed_database(update_existing=update_existing)
