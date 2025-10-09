#!/usr/bin/env python3
"""
Multi-Platform AI Model Deployment Script
Supports: RunPod, Modal, Replicate, Google Cloud Run, DigitalOcean, Lambda Labs

This script reads the multi-platform-config.yaml file and deploys AI models
to the specified cloud GPU platforms.
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Platform(Enum):
    """Supported cloud GPU platforms"""
    RUNPOD = "runpod"
    MODAL = "modal"
    REPLICATE = "replicate"
    GOOGLE_CLOUD_RUN = "google_cloud_run"
    DIGITALOCEAN = "digitalocean"
    LAMBDA_LABS = "lambda_labs"

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    platform: Platform
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    endpoint_id: Optional[str] = None
    error_message: Optional[str] = None
    deployment_time: Optional[float] = None
    cost_estimate: Optional[float] = None

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class DeploymentError(Exception):
    """Raised when deployment fails"""
    pass

class MultiPlatformDeployer:
    """Multi-platform AI model deployment orchestrator"""

    def __init__(self, config_path: str):
        """Initialize the deployer with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results: List[DeploymentResult] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file"""
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self._validate_config(config)
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_sections = ['global', 'platforms']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")

        # Validate platform configurations
        platforms = config.get('platforms', {})
        for platform_name, platform_config in platforms.items():
            if platform_config.get('enabled', False):
                if 'config' not in platform_config:
                    raise ConfigurationError(f"Missing config for platform: {platform_name}")

    def _substitute_environment_variables(self, text: str) -> str:
        """Substitute environment variables in configuration values"""
        import re

        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)

        # Handle ${VAR} and ${VAR:default} patterns
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, text)

    def _process_config_values(self, obj: Any) -> Any:
        """Recursively process configuration values to substitute environment variables"""
        if isinstance(obj, dict):
            return {k: self._process_config_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._process_config_values(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_environment_variables(obj)
        else:
            return obj

    async def deploy_to_runpod(self) -> DeploymentResult:
        """Deploy to RunPod serverless"""
        logger.info("Deploying to RunPod...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['runpod']['config']
            global_config = self.config['global']

            # Process environment variables
            platform_config = self._process_config_values(platform_config)

            # Check if runpodctl is available
            if not self._check_command_available('runpodctl'):
                raise DeploymentError("runpodctl CLI not found. Please install it first.")

            # Prepare deployment command
            cmd = [
                'runpodctl', 'deploy',
                '--name', platform_config['endpoint_name'],
                '--image', platform_config['base_image'],
                '--gpu', platform_config['gpu_types'][0],
                '--region', platform_config['regions'][0]
            ]

            # Add environment variables
            env_vars = {**global_config.get('environment', {}),
                       **platform_config.get('environment', {})}

            for key, value in env_vars.items():
                cmd.extend(['--env', f'{key}={value}'])

            # Execute deployment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Parse endpoint ID from output (this would need platform-specific parsing)
                endpoint_id = self._parse_runpod_endpoint_id(result.stdout)
                endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

                return DeploymentResult(
                    platform=Platform.RUNPOD,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=endpoint_url,
                    endpoint_id=endpoint_id,
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"RunPod deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"RunPod deployment error: {e}")
            return DeploymentResult(
                platform=Platform.RUNPOD,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    async def deploy_to_modal(self) -> DeploymentResult:
        """Deploy to Modal"""
        logger.info("Deploying to Modal...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['modal']['config']

            # Generate Modal application code
            modal_app_code = self._generate_modal_app(platform_config)

            # Write Modal app to temporary file
            modal_app_path = Path("/tmp/modal_app.py")
            with open(modal_app_path, 'w') as f:
                f.write(modal_app_code)

            # Deploy using Modal CLI
            cmd = ['modal', 'deploy', str(modal_app_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                endpoint_url = self._parse_modal_endpoint_url(result.stdout)

                return DeploymentResult(
                    platform=Platform.MODAL,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=endpoint_url,
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"Modal deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Modal deployment error: {e}")
            return DeploymentResult(
                platform=Platform.MODAL,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    async def deploy_to_replicate(self) -> DeploymentResult:
        """Deploy to Replicate"""
        logger.info("Deploying to Replicate...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['replicate']['config']

            # Generate Replicate cog.yaml and predict.py
            self._generate_replicate_files(platform_config)

            # Deploy using Replicate CLI
            cmd = ['cog', 'push', f"r8.im/{os.getenv('REPLICATE_USERNAME')}/{platform_config['model_name']}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

            if result.returncode == 0:
                model_url = f"https://replicate.com/{os.getenv('REPLICATE_USERNAME')}/{platform_config['model_name']}"

                return DeploymentResult(
                    platform=Platform.REPLICATE,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=model_url,
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"Replicate deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Replicate deployment error: {e}")
            return DeploymentResult(
                platform=Platform.REPLICATE,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    async def deploy_to_google_cloud_run(self) -> DeploymentResult:
        """Deploy to Google Cloud Run"""
        logger.info("Deploying to Google Cloud Run...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['google_cloud_run']['config']

            # Build and push container image
            await self._build_and_push_container(platform_config)

            # Deploy to Cloud Run
            cmd = [
                'gcloud', 'run', 'deploy', platform_config['service_name'],
                '--image', platform_config['container']['image'],
                '--platform', 'managed',
                '--region', platform_config['region'],
                '--gpu', '1',
                '--gpu-type', platform_config['resources']['limits']['gpu']['type'],
                '--memory', platform_config['resources']['limits']['memory'],
                '--cpu', platform_config['resources']['limits']['cpu'],
                '--max-instances', str(platform_config['scaling']['max_instances']),
                '--allow-unauthenticated'
            ]

            # Add environment variables
            for key, value in platform_config.get('environment', {}).items():
                cmd.extend(['--set-env-vars', f'{key}={value}'])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                service_url = self._parse_cloud_run_url(result.stdout)

                return DeploymentResult(
                    platform=Platform.GOOGLE_CLOUD_RUN,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=service_url,
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"Cloud Run deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Google Cloud Run deployment error: {e}")
            return DeploymentResult(
                platform=Platform.GOOGLE_CLOUD_RUN,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    async def deploy_to_digitalocean(self) -> DeploymentResult:
        """Deploy to DigitalOcean"""
        logger.info("Deploying to DigitalOcean...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['digitalocean']['config']

            # Generate DigitalOcean App Platform spec
            app_spec = self._generate_do_app_spec(platform_config)

            # Deploy using doctl
            cmd = ['doctl', 'apps', 'create', '--spec', '-']
            result = subprocess.run(
                cmd,
                input=json.dumps(app_spec),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                app_data = json.loads(result.stdout)
                app_url = app_data.get('live_url', '')

                return DeploymentResult(
                    platform=Platform.DIGITALOCEAN,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=app_url,
                    endpoint_id=app_data.get('id'),
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"DigitalOcean deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"DigitalOcean deployment error: {e}")
            return DeploymentResult(
                platform=Platform.DIGITALOCEAN,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    async def deploy_to_lambda_labs(self) -> DeploymentResult:
        """Deploy to Lambda Labs"""
        logger.info("Deploying to Lambda Labs...")
        start_time = time.time()

        try:
            platform_config = self.config['platforms']['lambda_labs']['config']

            # Launch instance
            cmd = [
                'lambda', 'cloud', 'instance', 'launch',
                '--instance-type', platform_config['instance_type'],
                '--ssh-key-name', platform_config['ssh_key_names'][0],
                '--name', platform_config['instance_name']
            ]

            if 'file_system_names' in platform_config:
                cmd.extend(['--file-system-name', platform_config['file_system_names'][0]])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                instance_data = self._parse_lambda_instance_data(result.stdout)

                # Wait for instance to be ready and execute cloud-init
                await self._setup_lambda_instance(instance_data, platform_config)

                return DeploymentResult(
                    platform=Platform.LAMBDA_LABS,
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=f"http://{instance_data['ip']}:8080",
                    endpoint_id=instance_data['id'],
                    deployment_time=time.time() - start_time
                )
            else:
                raise DeploymentError(f"Lambda Labs deployment failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Lambda Labs deployment error: {e}")
            return DeploymentResult(
                platform=Platform.LAMBDA_LABS,
                status=DeploymentStatus.FAILED,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH"""
        try:
            subprocess.run(['which', command], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _parse_runpod_endpoint_id(self, output: str) -> Optional[str]:
        """Parse endpoint ID from RunPod CLI output"""
        # This would need to be implemented based on actual RunPod CLI output format
        import re
        match = re.search(r'Endpoint ID: ([a-zA-Z0-9-]+)', output)
        return match.group(1) if match else None

    def _parse_modal_endpoint_url(self, output: str) -> Optional[str]:
        """Parse endpoint URL from Modal CLI output"""
        import re
        match = re.search(r'✓ Created web endpoint: (https://[^\s]+)', output)
        return match.group(1) if match else None

    def _parse_cloud_run_url(self, output: str) -> Optional[str]:
        """Parse service URL from Cloud Run CLI output"""
        import re
        match = re.search(r'Service URL: (https://[^\s]+)', output)
        return match.group(1) if match else None

    def _parse_lambda_instance_data(self, output: str) -> Dict[str, str]:
        """Parse instance data from Lambda Labs CLI output"""
        # This would need to be implemented based on actual Lambda CLI output
        return {
            'id': 'instance-123',
            'ip': '192.168.1.100'
        }

    def _generate_modal_app(self, config: Dict[str, Any]) -> str:
        """Generate Modal application code"""
        return f'''
import modal

# Create Modal app
app = modal.App("{config['app_name']}")

# Define image with requirements
image = modal.Image.debian_slim().pip_install({config['requirements']})

# Define GPU configuration
gpu_config = modal.gpu.{config['gpu_config']['type']}(count={config['gpu_config']['count']})

@app.function(
    image=image,
    gpu=gpu_config,
    cpu={config['cpu']},
    memory={config['memory']},
    timeout={config['timeout']}
)
@modal.web_endpoint(method="POST")
def inference(request_data: dict):
    import vllm
    from transformers import AutoTokenizer

    # Initialize model (this would be cached across invocations)
    model = vllm.LLM(
        model="{self.config['global']['model_name']}",
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )

    # Process request
    messages = request_data.get("messages", [])
    max_tokens = request_data.get("max_tokens", 4096)

    # Generate response
    outputs = model.generate(messages, max_tokens=max_tokens)

    return {{"choices": outputs}}

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {{"status": "healthy"}}
'''

    def _generate_replicate_files(self, config: Dict[str, Any]) -> None:
        """Generate Replicate cog.yaml and predict.py files"""

        # Generate cog.yaml
        cog_config = {
            'build': {
                'gpu': True,
                'cuda': config['build']['cuda_version'],
                'python_version': config['build']['python_version'],
                'python_packages': config['build']['python_packages'],
                'system_packages': config['build']['system_packages']
            },
            'predict': 'predict.py:Predictor'
        }

        with open('cog.yaml', 'w') as f:
            yaml.dump(cog_config, f)

        # Generate predict.py
        predict_code = f'''
import os
from typing import List, Dict, Any
import torch
import vllm
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        self.model = vllm.LLM(
            model="{self.config['global']['model_name']}",
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            trust_remote_code=True
        )

    def predict(
        self,
        messages: List[Dict[str, Any]] = Input(description="Chat messages with image support"),
        max_tokens: int = Input(default=4096, ge=1, le=8192, description="Maximum tokens to generate"),
        temperature: float = Input(default=0.1, ge=0.0, le=2.0, description="Sampling temperature"),
        top_p: float = Input(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter"),
    ) -> Dict[str, Any]:
        """Generate response from model"""

        # Process messages and generate response
        outputs = self.model.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        return {{
            "choices": outputs,
            "model": "{self.config['global']['model_name']}"
        }}
'''

        with open('predict.py', 'w') as f:
            f.write(predict_code)

    def _generate_do_app_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DigitalOcean App Platform specification"""
        return {
            'name': config['app_name'],
            'services': config['services'],
            'databases': config.get('databases', []),
            'domains': config.get('domains', [])
        }

    async def _build_and_push_container(self, config: Dict[str, Any]) -> None:
        """Build and push container image for Cloud Run"""
        # This would implement Docker build and push logic
        logger.info("Building and pushing container image...")
        pass

    async def _setup_lambda_instance(self, instance_data: Dict[str, str], config: Dict[str, Any]) -> None:
        """Set up Lambda Labs instance with cloud-init script"""
        # This would implement SSH connection and script execution
        logger.info("Setting up Lambda Labs instance...")
        pass

    async def deploy_to_platform(self, platform: Platform) -> DeploymentResult:
        """Deploy to a specific platform"""
        deployment_methods = {
            Platform.RUNPOD: self.deploy_to_runpod,
            Platform.MODAL: self.deploy_to_modal,
            Platform.REPLICATE: self.deploy_to_replicate,
            Platform.GOOGLE_CLOUD_RUN: self.deploy_to_google_cloud_run,
            Platform.DIGITALOCEAN: self.deploy_to_digitalocean,
            Platform.LAMBDA_LABS: self.deploy_to_lambda_labs
        }

        method = deployment_methods.get(platform)
        if not method:
            raise DeploymentError(f"Unsupported platform: {platform}")

        return await method()

    async def deploy_multi_platform(self, platforms: List[Platform]) -> List[DeploymentResult]:
        """Deploy to multiple platforms concurrently"""
        logger.info(f"Deploying to platforms: {[p.value for p in platforms]}")

        tasks = []
        for platform in platforms:
            if self.config['platforms'][platform.value].get('enabled', False):
                tasks.append(self.deploy_to_platform(platform))
            else:
                logger.warning(f"Platform {platform.value} is disabled in configuration")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(DeploymentResult(
                    platform=platforms[i],
                    status=DeploymentStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        self.results.extend(processed_results)
        return processed_results

    async def test_deployments(self) -> Dict[Platform, bool]:
        """Test all deployed endpoints"""
        logger.info("Testing deployed endpoints...")

        test_results = {}
        for result in self.results:
            if result.status == DeploymentStatus.RUNNING and result.endpoint_url:
                try:
                    # Perform health check
                    response = requests.get(f"{result.endpoint_url}/health", timeout=30)
                    test_results[result.platform] = response.status_code == 200
                except Exception as e:
                    logger.error(f"Test failed for {result.platform.value}: {e}")
                    test_results[result.platform] = False
            else:
                test_results[result.platform] = False

        return test_results

    def generate_report(self) -> str:
        """Generate deployment report"""
        report = ["=== Multi-Platform Deployment Report ===\n"]

        for result in self.results:
            report.append(f"Platform: {result.platform.value}")
            report.append(f"Status: {result.status.value}")

            if result.endpoint_url:
                report.append(f"Endpoint: {result.endpoint_url}")

            if result.deployment_time:
                report.append(f"Deployment Time: {result.deployment_time:.2f}s")

            if result.error_message:
                report.append(f"Error: {result.error_message}")

            report.append("")

        return "\n".join(report)

    def save_deployment_info(self, output_file: str) -> None:
        """Save deployment information to file"""
        deployment_info = {
            'timestamp': time.time(),
            'deployments': [
                {
                    'platform': result.platform.value,
                    'status': result.status.value,
                    'endpoint_url': result.endpoint_url,
                    'endpoint_id': result.endpoint_id,
                    'deployment_time': result.deployment_time,
                    'error_message': result.error_message
                }
                for result in self.results
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        logger.info(f"Deployment information saved to {output_file}")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Multi-Platform AI Model Deployment")
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--platforms',
        nargs='+',
        choices=[p.value for p in Platform],
        help='Platforms to deploy to (default: all enabled)'
    )
    parser.add_argument(
        '--strategy',
        choices=['multi_platform', 'cost_optimized', 'performance_optimized'],
        default='multi_platform',
        help='Deployment strategy'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test deployments after completion'
    )
    parser.add_argument(
        '--output',
        help='Output file for deployment information'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deployed without deploying'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize deployer
        deployer = MultiPlatformDeployer(args.config)

        # Determine platforms to deploy to
        if args.platforms:
            platforms = [Platform(p) for p in args.platforms]
        else:
            # Deploy to all enabled platforms
            platforms = [
                Platform(name) for name, config in deployer.config['platforms'].items()
                if config.get('enabled', False)
            ]

        if not platforms:
            logger.error("No platforms specified or enabled")
            sys.exit(1)

        if args.dry_run:
            logger.info("DRY RUN: Would deploy to platforms:")
            for platform in platforms:
                logger.info(f"  - {platform.value}")
            return

        # Deploy to platforms
        results = asyncio.run(deployer.deploy_multi_platform(platforms))

        # Test deployments if requested
        if args.test:
            test_results = asyncio.run(deployer.test_deployments())
            logger.info("Test Results:")
            for platform, success in test_results.items():
                status = "✓ PASS" if success else "✗ FAIL"
                logger.info(f"  {platform.value}: {status}")

        # Generate and display report
        report = deployer.generate_report()
        print(report)

        # Save deployment information
        if args.output:
            deployer.save_deployment_info(args.output)

        # Exit with error code if any deployments failed
        failed_deployments = [r for r in results if r.status == DeploymentStatus.FAILED]
        if failed_deployments:
            logger.error(f"{len(failed_deployments)} deployment(s) failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()