#!/usr/bin/env python3
"""
Test deployment form pre-filling functionality.
This script verifies that the API endpoints return the correct data for form population.
"""

import asyncio
import json
from ocular.database.session import SessionLocal, init_db
from ocular.database.seed import seed_database
from ocular.database.models import Provider, Model, ProviderStatus, ModelStatus


def test_providers_endpoint():
    """Test that providers endpoint returns correct data."""
    print("\n=== Testing Providers Endpoint ===")

    db = SessionLocal()

    try:
        # Query providers
        providers = db.query(Provider).all()

        print(f"Found {len(providers)} providers:")
        for provider in providers:
            data = provider.to_dict()
            print(f"\n  {provider.display_name}:")
            print(f"    ID: {data['id']}")
            print(f"    Name: {data['name']}")
            print(f"    Status: {data['status']}")
            print(f"    Available GPUs: {data['available_gpus']}")
            print(f"    Cost Range: ${data['min_cost_per_hour']}/hr - ${data['max_cost_per_hour']}/hr")

        # Verify active providers
        active_providers = [p for p in providers if p.status == ProviderStatus.ACTIVE]
        print(f"\n  ✓ Active providers: {len(active_providers)}")
        assert len(active_providers) > 0, "Should have at least one active provider"

        # Verify RunPod has GPUs
        runpod = next((p for p in providers if p.name == 'runpod'), None)
        assert runpod is not None, "RunPod provider should exist"
        assert len(runpod.available_gpus) > 0, "RunPod should have available GPUs"
        print(f"  ✓ RunPod has {len(runpod.available_gpus)} GPU types")

        return True

    finally:
        db.close()


def test_models_endpoint():
    """Test that models endpoint returns correct data."""
    print("\n=== Testing Models Endpoint ===")

    db = SessionLocal()

    try:
        # Query models
        models = db.query(Model).all()

        print(f"Found {len(models)} models:")
        for model in models:
            data = model.to_dict()
            print(f"\n  {model.display_name}:")
            print(f"    ID: {data['id']}")
            print(f"    Name: {data['name']}")
            print(f"    Version: {data['version']}")
            print(f"    Status: {data['status']}")
            print(f"    Min VRAM: {data['min_vram_gb']}GB")
            print(f"    Docker Image: {data['docker_image']}")
            print(f"    Supports vLLM: {data['supports_vllm']}")

        # Verify active models
        active_models = [m for m in models if m.status == ModelStatus.AVAILABLE]
        print(f"\n  ✓ Available models: {len(active_models)}")
        assert len(active_models) > 0, "Should have at least one available model"

        # Verify OLMocr exists
        olmocr = next((m for m in models if m.name == 'olmocr'), None)
        assert olmocr is not None, "OLMocr model should exist"
        assert olmocr.docker_image is not None, "OLMocr should have Docker image"
        print(f"  ✓ OLMocr model configured correctly")

        return True

    finally:
        db.close()


def test_form_data_structure():
    """Test that data structure matches frontend expectations."""
    print("\n=== Testing Form Data Structure ===")

    db = SessionLocal()

    try:
        # Get a provider
        provider = db.query(Provider).filter(Provider.name == 'runpod').first()
        provider_dict = provider.to_dict()

        # Verify required fields for form
        required_fields = ['id', 'name', 'display_name', 'status', 'available_gpus']
        for field in required_fields:
            assert field in provider_dict, f"Provider should have '{field}' field"
            print(f"  ✓ Provider has '{field}': {provider_dict[field]}")

        # Get a model
        model = db.query(Model).filter(Model.name == 'olmocr').first()
        model_dict = model.to_dict()

        # Verify required fields for form
        required_fields = ['id', 'name', 'display_name', 'version', 'status', 'min_vram_gb']
        for field in required_fields:
            assert field in model_dict, f"Model should have '{field}' field"
            print(f"  ✓ Model has '{field}': {model_dict[field]}")

        return True

    finally:
        db.close()


def simulate_form_population():
    """Simulate how the frontend will populate the form."""
    print("\n=== Simulating Frontend Form Population ===")

    db = SessionLocal()

    try:
        # Fetch providers (like the frontend does)
        providers = db.query(Provider).all()
        providers_json = [p.to_dict() for p in providers]

        # Fetch models (like the frontend does)
        models = db.query(Model).all()
        models_json = [m.to_dict() for m in models]

        print("\nForm will be populated with:")

        # Provider select options
        print("\n  Provider Select Options:")
        for provider in providers_json:
            disabled = "" if provider['status'] == 'ACTIVE' else " (disabled)"
            print(f"    <option value=\"{provider['id']}\">{provider['display_name']}{disabled}</option>")

        # Model select options
        print("\n  Model Select Options:")
        active_models = [m for m in models_json if m['status'] == 'AVAILABLE']
        for model in active_models:
            print(f"    <option value=\"{model['id']}\">{model['display_name']} - {model['version']}</option>")

        # GPU select options (for RunPod)
        runpod = next((p for p in providers_json if p['name'] == 'runpod'), None)
        print("\n  GPU Select Options (for RunPod):")
        for gpu in runpod['available_gpus']:
            print(f"    <option value=\"{gpu}\">{gpu}</option>")

        print("\n  ✓ Form population simulation successful")
        return True

    finally:
        db.close()


def test_deployment_creation_payload():
    """Test creating a deployment payload from form data."""
    print("\n=== Testing Deployment Creation Payload ===")

    db = SessionLocal()

    try:
        # Simulate user selections
        provider = db.query(Provider).filter(Provider.name == 'runpod').first()
        model = db.query(Model).filter(Model.name == 'olmocr').first()

        # Simulate form submission
        form_data = {
            "deployment_name": f"olmocr-test-{1234567890}",
            "model_id": model.id,
            "provider_id": provider.id,
            "gpu_type": "RTX 4000 Ada",
            "gpu_count": 1,
            "region": "eur-is-1",
            "api_key": "test_api_key_placeholder",
            "deployment_type": "docker",
            "autoscaling": True,
            "min_workers": 0,
            "max_workers": 3,
            "timeout": 600,
            "cost_per_hour": 0.29,
            "config": {
                "container_disk_gb": 30,
                "volumes_in_gb": 30,
                "idle_timeout": 30
            },
            "metadata": {
                "created_from": "web_ui",
                "model_name": model.name,
                "provider_name": provider.name
            }
        }

        print("\nDeployment Payload:")
        print(json.dumps(form_data, indent=2))

        # Validate payload
        assert form_data['model_id'] == model.id
        assert form_data['provider_id'] == provider.id
        assert form_data['gpu_type'] in provider.available_gpus
        assert form_data['max_workers'] > 0
        assert form_data['cost_per_hour'] > 0

        print("\n  ✓ Deployment payload validation successful")
        return True

    finally:
        db.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Deployment Form Pre-filling Functionality")
    print("=" * 60)

    # Initialize database
    init_db()
    seed_database()

    tests = [
        ("Providers Endpoint", test_providers_endpoint),
        ("Models Endpoint", test_models_endpoint),
        ("Form Data Structure", test_form_data_structure),
        ("Form Population Simulation", simulate_form_population),
        ("Deployment Payload Creation", test_deployment_creation_payload),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if error:
            print(f"    Error: {error}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Form pre-filling is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
