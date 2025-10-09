"""
RunPod Deployment Example - Complete Workflow

This example demonstrates how to:
1. Create a deployment configuration from template
2. Deploy a model to RunPod
3. Monitor deployment status
4. Test the deployed model
5. Manage deployment lifecycle

Requirements:
    pip install requests pydantic
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class DeploymentConfig(BaseModel):
    """Deployment configuration model."""
    name: str
    model_id: int
    provider_id: int
    gpu_type: str
    gpu_count: int
    deployment_type: str = "vllm"
    environment_vars: Dict[str, str] = {}
    scaling_policy: Optional[Dict[str, Any]] = None
    health_check: Optional[Dict[str, Any]] = None
    resource_limits: Optional[Dict[str, Any]] = None


class OcularRunPodDeployer:
    """Complete RunPod deployment manager for Ocular platform."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def create_deployment_from_template(
        self,
        template_name: str,
        model_id: int,
        gpu_type: str = "A40",
        gpu_count: int = 1,
        custom_env: Optional[Dict[str, str]] = None
    ) -> DeploymentConfig:
        """
        Create deployment configuration from a predefined template.

        Args:
            template_name: Name of the deployment template
            model_id: ID of the model to deploy
            gpu_type: GPU type (A40, A100, H100, etc.)
            gpu_count: Number of GPUs
            custom_env: Custom environment variables

        Returns:
            DeploymentConfig object
        """
        # Default templates
        templates = {
            "vllm_optimized": {
                "deployment_type": "vllm",
                "environment_vars": {
                    "MODEL_PATH": "/models",
                    "MAX_MODEL_LEN": "4096",
                    "TENSOR_PARALLEL_SIZE": str(gpu_count),
                    "GPU_MEMORY_UTILIZATION": "0.9",
                    "ENABLE_PREFIX_CACHING": "true",
                    "ENABLE_CHUNKED_PREFILL": "true"
                },
                "scaling_policy": {
                    "min_instances": 1,
                    "max_instances": 5,
                    "target_utilization": 0.75,
                    "scale_up_threshold": 0.85,
                    "scale_down_threshold": 0.30,
                    "cooldown_period_seconds": 180,
                    "metrics": ["gpu", "requests_per_second"]
                },
                "health_check": {
                    "enabled": True,
                    "endpoint": "/health",
                    "interval_seconds": 30,
                    "timeout_seconds": 5,
                    "unhealthy_threshold": 3,
                    "healthy_threshold": 2
                },
                "resource_limits": {
                    "cpu_limit": "8",
                    "memory_limit": "32Gi",
                    "gpu_memory_limit": "48Gi" if gpu_type == "A100" else "24Gi"
                }
            },
            "docker_standard": {
                "deployment_type": "docker",
                "environment_vars": {
                    "MODEL_PATH": "/models",
                    "PORT": "8000"
                },
                "scaling_policy": {
                    "min_instances": 1,
                    "max_instances": 3,
                    "target_utilization": 0.70
                },
                "health_check": {
                    "enabled": True,
                    "endpoint": "/health",
                    "interval_seconds": 60
                }
            },
            "production_ha": {  # High Availability
                "deployment_type": "vllm",
                "environment_vars": {
                    "MODEL_PATH": "/models",
                    "MAX_MODEL_LEN": "8192",
                    "TENSOR_PARALLEL_SIZE": str(gpu_count),
                    "GPU_MEMORY_UTILIZATION": "0.95",
                    "ENABLE_PREFIX_CACHING": "true",
                    "ENABLE_CHUNKED_PREFILL": "true",
                    "MAX_NUM_SEQS": "512"
                },
                "scaling_policy": {
                    "min_instances": 2,  # Always at least 2 for HA
                    "max_instances": 10,
                    "target_utilization": 0.70,
                    "scale_up_threshold": 0.80,
                    "scale_down_threshold": 0.40,
                    "cooldown_period_seconds": 300
                },
                "health_check": {
                    "enabled": True,
                    "endpoint": "/health",
                    "interval_seconds": 15,
                    "timeout_seconds": 3,
                    "unhealthy_threshold": 2,
                    "healthy_threshold": 3
                },
                "resource_limits": {
                    "cpu_limit": "16",
                    "memory_limit": "64Gi",
                    "gpu_memory_limit": "80Gi" if gpu_type == "H100" else "48Gi"
                }
            }
        }

        template = templates.get(template_name, templates["vllm_optimized"])

        # Merge custom environment variables
        env_vars = template["environment_vars"].copy()
        if custom_env:
            env_vars.update(custom_env)

        config = DeploymentConfig(
            name=f"{template_name}_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_id=model_id,
            provider_id=1,  # RunPod provider ID
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            deployment_type=template["deployment_type"],
            environment_vars=env_vars,
            scaling_policy=template.get("scaling_policy"),
            health_check=template.get("health_check"),
            resource_limits=template.get("resource_limits")
        )

        return config

    def deploy_to_runpod(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy model to RunPod using configuration.

        Args:
            config: DeploymentConfig object

        Returns:
            Deployment response with ID, status, endpoint
        """
        url = f"{self.base_url}/api/deployments"

        # Prepare deployment payload
        payload = {
            "model_id": config.model_id,
            "provider_id": config.provider_id,
            "gpu_type": config.gpu_type,
            "gpu_count": config.gpu_count,
            "deployment_type": config.deployment_type,
            "config": {
                "environment_vars": config.environment_vars,
                "vllm_config": {
                    "max_model_len": int(config.environment_vars.get("MAX_MODEL_LEN", "4096")),
                    "tensor_parallel_size": config.gpu_count,
                    "gpu_memory_utilization": float(config.environment_vars.get("GPU_MEMORY_UTILIZATION", "0.9")),
                    "enable_prefix_caching": config.environment_vars.get("ENABLE_PREFIX_CACHING", "true").lower() == "true",
                    "enable_chunked_prefill": config.environment_vars.get("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
                }
            }
        }

        print(f"🚀 Deploying to RunPod...")
        print(f"   Model ID: {config.model_id}")
        print(f"   GPU: {config.gpu_count}x {config.gpu_type}")
        print(f"   Type: {config.deployment_type}")

        response = requests.post(url, json=payload, headers=self.headers)

        if response.status_code == 200:
            deployment = response.json()
            print(f"✅ Deployment created successfully!")
            print(f"   Deployment ID: {deployment['id']}")
            print(f"   Status: {deployment['status']}")
            return deployment
        else:
            raise Exception(f"Deployment failed: {response.status_code} - {response.text}")

    def configure_deployment(self, deployment_id: int, config: DeploymentConfig):
        """
        Apply advanced configuration to deployment.

        Args:
            deployment_id: Deployment ID
            config: DeploymentConfig with settings
        """
        print(f"\n🔧 Configuring deployment {deployment_id}...")

        # Set environment variables
        if config.environment_vars:
            env_url = f"{self.base_url}/api/deployments/{deployment_id}/config/environment"
            response = requests.post(
                env_url,
                json={"env_vars": config.environment_vars, "merge": True},
                headers=self.headers
            )
            if response.ok:
                print(f"   ✅ Environment variables configured")

        # Set scaling policy
        if config.scaling_policy:
            scaling_url = f"{self.base_url}/api/deployments/{deployment_id}/config/scaling-policy"
            response = requests.post(
                scaling_url,
                json=config.scaling_policy,
                headers=self.headers
            )
            if response.ok:
                print(f"   ✅ Scaling policy configured")

        # Set health check
        if config.health_check:
            health_url = f"{self.base_url}/api/deployments/{deployment_id}/config/health-check"
            response = requests.post(
                health_url,
                json=config.health_check,
                headers=self.headers
            )
            if response.ok:
                print(f"   ✅ Health check configured")

        # Set resource limits
        if config.resource_limits:
            resource_url = f"{self.base_url}/api/deployments/{deployment_id}/config/resources"
            response = requests.post(
                resource_url,
                json=config.resource_limits,
                headers=self.headers
            )
            if response.ok:
                print(f"   ✅ Resource limits configured")

    def monitor_deployment(self, deployment_id: int, timeout: int = 600):
        """
        Monitor deployment until it's running or timeout.

        Args:
            deployment_id: Deployment ID
            timeout: Timeout in seconds

        Returns:
            Final deployment status
        """
        url = f"{self.base_url}/api/deployments/{deployment_id}"

        print(f"\n⏳ Monitoring deployment {deployment_id}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(url, headers=self.headers)

            if response.ok:
                deployment = response.json()
                status = deployment["status"]

                print(f"   Status: {status}", end="\r")

                if status == "running":
                    print(f"\n✅ Deployment is running!")
                    print(f"   Endpoint: {deployment.get('endpoint_url')}")
                    return deployment
                elif status == "error":
                    print(f"\n❌ Deployment failed!")
                    print(f"   Error: {deployment.get('error_message')}")
                    return deployment

            time.sleep(10)

        print(f"\n⏰ Timeout reached after {timeout}s")
        return None

    def test_deployment(self, deployment_id: int, test_text: str = "Hello, test!"):
        """
        Test deployed model with a sample request.

        Args:
            deployment_id: Deployment ID
            test_text: Test input text

        Returns:
            Test result
        """
        # Get deployment info
        url = f"{self.base_url}/api/deployments/{deployment_id}"
        response = requests.get(url, headers=self.headers)

        if not response.ok:
            raise Exception("Deployment not found")

        deployment = response.json()
        endpoint_url = deployment.get("endpoint_url")

        if not endpoint_url:
            raise Exception("No endpoint URL available")

        print(f"\n🧪 Testing deployment...")
        print(f"   Endpoint: {endpoint_url}")
        print(f"   Test input: {test_text}")

        # Send test request to deployed model
        test_payload = {
            "prompt": test_text,
            "max_tokens": 100
        }

        response = requests.post(
            f"{endpoint_url}/v1/completions",
            json=test_payload
        )

        if response.ok:
            result = response.json()
            print(f"   ✅ Test successful!")
            print(f"   Response: {result.get('choices', [{}])[0].get('text', '')[:100]}...")
            return result
        else:
            print(f"   ❌ Test failed: {response.status_code}")
            return None

    def get_deployment_metrics(self, deployment_id: int):
        """
        Get deployment metrics and statistics.

        Args:
            deployment_id: Deployment ID

        Returns:
            Metrics data
        """
        url = f"{self.base_url}/api/analytics/deployment-metrics/{deployment_id}"
        response = requests.get(url, headers=self.headers)

        if response.ok:
            metrics = response.json()
            print(f"\n📊 Deployment Metrics:")
            print(f"   Requests: {metrics.get('total_requests', 0)}")
            print(f"   Avg Latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
            print(f"   GPU Utilization: {metrics.get('gpu_utilization', 0):.1f}%")
            print(f"   Uptime: {metrics.get('uptime_hours', 0):.1f}h")
            print(f"   Cost: ${metrics.get('total_cost', 0):.2f}")
            return metrics

        return None

    def stop_deployment(self, deployment_id: int):
        """
        Stop a running deployment.

        Args:
            deployment_id: Deployment ID
        """
        url = f"{self.base_url}/api/deployments/{deployment_id}/stop"
        response = requests.post(url, headers=self.headers)

        if response.ok:
            print(f"🛑 Deployment {deployment_id} stopped")
        else:
            print(f"❌ Failed to stop deployment: {response.text}")

    def delete_deployment(self, deployment_id: int):
        """
        Delete a deployment.

        Args:
            deployment_id: Deployment ID
        """
        url = f"{self.base_url}/api/deployments/{deployment_id}/delete"
        response = requests.post(url, headers=self.headers)

        if response.ok:
            print(f"🗑️  Deployment {deployment_id} deleted")
        else:
            print(f"❌ Failed to delete deployment: {response.text}")


def main():
    """
    Complete deployment workflow example.
    """
    # Initialize deployer
    deployer = OcularRunPodDeployer(
        base_url="http://localhost:8000",
        api_key="your_api_key_here"  # Optional
    )

    # Step 1: Create deployment configuration from template
    config = deployer.create_deployment_from_template(
        template_name="vllm_optimized",  # or "docker_standard", "production_ha"
        model_id=1,  # OLMocr model ID
        gpu_type="A40",
        gpu_count=1,
        custom_env={
            "CUSTOM_VAR": "custom_value",
            "LOG_LEVEL": "INFO"
        }
    )

    print("\n📋 Deployment Configuration:")
    print(json.dumps(config.dict(), indent=2))

    # Step 2: Deploy to RunPod
    deployment = deployer.deploy_to_runpod(config)
    deployment_id = deployment["id"]

    # Step 3: Apply advanced configuration
    deployer.configure_deployment(deployment_id, config)

    # Step 4: Monitor deployment
    final_deployment = deployer.monitor_deployment(deployment_id, timeout=600)

    if final_deployment and final_deployment["status"] == "running":
        # Step 5: Test deployment
        test_result = deployer.test_deployment(
            deployment_id,
            test_text="Extract text from this document"
        )

        # Step 6: Get metrics
        time.sleep(30)  # Wait for some metrics
        metrics = deployer.get_deployment_metrics(deployment_id)

        # Step 7: Cleanup (optional)
        # deployer.stop_deployment(deployment_id)
        # deployer.delete_deployment(deployment_id)

        print(f"\n✅ Deployment workflow completed successfully!")
    else:
        print(f"\n❌ Deployment workflow failed")


if __name__ == "__main__":
    main()
