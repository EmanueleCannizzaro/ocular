/**
 * Advanced Deployment Configuration JavaScript
 *
 * Handles configuration UI for environment variables, scaling policies,
 * health checks, resource limits, Docker settings, network, and GPU config.
 */

let currentDeploymentId = null;
let currentConfig = {};

/**
 * Show advanced configuration modal for a deployment
 */
async function showDeploymentConfig(deploymentId) {
    currentDeploymentId = deploymentId;

    try {
        // Fetch current configuration
        const response = await fetch(`/api/deployments/${deploymentId}/config`);
        if (!response.ok) throw new Error('Failed to fetch configuration');

        currentConfig = await response.json();

        // Populate all configuration sections
        populateBasicInfo(currentConfig.basic);
        populateEnvironmentVariables(currentConfig.environment);
        populateScalingPolicy(currentConfig.scaling);
        populateHealthCheck(currentConfig.health_check);
        populateResourceLimits(currentConfig.resources);
        populateDockerConfig(currentConfig.docker);
        populateNetworkConfig(currentConfig.network);
        populateGPUConfig(currentConfig.gpu);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('deploymentConfigModal'));
        modal.show();

    } catch (error) {
        console.error('Error loading configuration:', error);
        alert('Failed to load deployment configuration: ' + error.message);
    }
}

/**
 * Populate basic deployment information
 */
function populateBasicInfo(basic) {
    document.getElementById('configDeploymentName').textContent = currentConfig.deployment_name || 'Unknown';
    document.getElementById('configGPUType').textContent = basic.gpu_type || 'N/A';
    document.getElementById('configGPUCount').textContent = basic.gpu_count || 1;
    document.getElementById('configRegion').textContent = basic.region || 'N/A';
}

/**
 * Populate environment variables section
 */
function populateEnvironmentVariables(env) {
    const container = document.getElementById('envVarsList');
    let html = '';

    const envVars = env || {};
    const keys = Object.keys(envVars);

    if (keys.length === 0) {
        html = '<p class="text-muted">No environment variables configured.</p>';
    } else {
        html = '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Key</th><th>Value</th><th>Action</th></tr></thead><tbody>';

        keys.forEach(key => {
            const value = envVars[key];
            const maskedValue = key.toLowerCase().includes('key') || key.toLowerCase().includes('secret') || key.toLowerCase().includes('password')
                ? '*'.repeat(Math.min(value.length, 20))
                : value;

            html += `
                <tr>
                    <td><code>${key}</code></td>
                    <td><code>${maskedValue}</code></td>
                    <td>
                        <button class="btn btn-sm btn-danger" onclick="removeEnvVar('${key}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
    }

    container.innerHTML = html;
}

/**
 * Add new environment variable
 */
async function addEnvVar() {
    const key = document.getElementById('newEnvKey').value.trim();
    const value = document.getElementById('newEnvValue').value;

    if (!key) {
        alert('Environment variable key is required');
        return;
    }

    try {
        const newEnvVars = { ...currentConfig.environment, [key]: value };

        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/environment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                env_vars: { [key]: value },
                merge: true
            })
        });

        if (!response.ok) throw new Error('Failed to add environment variable');

        // Clear inputs
        document.getElementById('newEnvKey').value = '';
        document.getElementById('newEnvValue').value = '';

        // Refresh configuration
        await showDeploymentConfig(currentDeploymentId);

        alert('Environment variable added successfully');

    } catch (error) {
        console.error('Error adding environment variable:', error);
        alert('Failed to add environment variable: ' + error.message);
    }
}

/**
 * Remove environment variable
 */
async function removeEnvVar(key) {
    if (!confirm(`Remove environment variable "${key}"?`)) return;

    try {
        const newEnvVars = { ...currentConfig.environment };
        delete newEnvVars[key];

        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/environment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                env_vars: newEnvVars,
                merge: false
            })
        });

        if (!response.ok) throw new Error('Failed to remove environment variable');

        await showDeploymentConfig(currentDeploymentId);
        alert('Environment variable removed successfully');

    } catch (error) {
        console.error('Error removing environment variable:', error);
        alert('Failed to remove environment variable: ' + error.message);
    }
}

/**
 * Populate scaling policy section
 */
function populateScalingPolicy(scaling) {
    const policy = scaling.policy || {};

    document.getElementById('scalingEnabled').checked = scaling.autoscaling_enabled || false;
    document.getElementById('minInstances').value = scaling.min_workers || 0;
    document.getElementById('maxInstances').value = scaling.max_workers || 10;
    document.getElementById('targetUtilization').value = (policy.target_utilization || 0.8) * 100;
    document.getElementById('scaleUpThreshold').value = (policy.scale_up_threshold || 0.9) * 100;
    document.getElementById('scaleDownThreshold').value = (policy.scale_down_threshold || 0.3) * 100;
    document.getElementById('cooldownPeriod').value = policy.cooldown_period_seconds || 300;
}

/**
 * Save scaling policy
 */
async function saveScalingPolicy() {
    const data = {
        min_instances: parseInt(document.getElementById('minInstances').value),
        max_instances: parseInt(document.getElementById('maxInstances').value),
        target_utilization: parseFloat(document.getElementById('targetUtilization').value) / 100,
        scale_up_threshold: parseFloat(document.getElementById('scaleUpThreshold').value) / 100,
        scale_down_threshold: parseFloat(document.getElementById('scaleDownThreshold').value) / 100,
        cooldown_period_seconds: parseInt(document.getElementById('cooldownPeriod').value),
        metrics: ['cpu', 'gpu', 'memory', 'requests_per_second']
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/scaling-policy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save scaling policy');

        alert('Scaling policy saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving scaling policy:', error);
        alert('Failed to save scaling policy: ' + error.message);
    }
}

/**
 * Populate health check section
 */
function populateHealthCheck(healthCheck) {
    const hc = healthCheck || {};

    document.getElementById('healthCheckEnabled').checked = hc.enabled !== false;
    document.getElementById('healthCheckEndpoint').value = hc.endpoint || '/health';
    document.getElementById('healthCheckInterval').value = hc.interval_seconds || 30;
    document.getElementById('healthCheckTimeout').value = hc.timeout_seconds || 10;
    document.getElementById('unhealthyThreshold').value = hc.unhealthy_threshold || 3;
    document.getElementById('healthyThreshold').value = hc.healthy_threshold || 2;
}

/**
 * Save health check configuration
 */
async function saveHealthCheck() {
    const data = {
        enabled: document.getElementById('healthCheckEnabled').checked,
        endpoint: document.getElementById('healthCheckEndpoint').value,
        interval_seconds: parseInt(document.getElementById('healthCheckInterval').value),
        timeout_seconds: parseInt(document.getElementById('healthCheckTimeout').value),
        unhealthy_threshold: parseInt(document.getElementById('unhealthyThreshold').value),
        healthy_threshold: parseInt(document.getElementById('healthyThreshold').value),
        http_method: 'GET',
        expected_status_codes: [200, 204],
        headers: {}
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/health-check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save health check');

        alert('Health check configuration saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving health check:', error);
        alert('Failed to save health check: ' + error.message);
    }
}

/**
 * Populate resource limits section
 */
function populateResourceLimits(resources) {
    const limits = resources.limits || {};

    document.getElementById('cpuLimit').value = limits.cpu || '';
    document.getElementById('memoryLimit').value = limits.memory || '';
    document.getElementById('gpuMemoryLimit').value = limits.gpu_memory || '';
    document.getElementById('diskLimit').value = limits.disk || '';
}

/**
 * Save resource limits
 */
async function saveResourceLimits() {
    const data = {
        cpu_limit: document.getElementById('cpuLimit').value || null,
        memory_limit: document.getElementById('memoryLimit').value || null,
        gpu_memory_limit: document.getElementById('gpuMemoryLimit').value || null,
        disk_limit: document.getElementById('diskLimit').value || null
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/resources`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save resource limits');

        alert('Resource limits saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving resource limits:', error);
        alert('Failed to save resource limits: ' + error.message);
    }
}

/**
 * Populate Docker configuration section
 */
function populateDockerConfig(docker) {
    const dc = docker || {};

    document.getElementById('dockerfilePath').value = dc.dockerfile_path || '';
    document.getElementById('dockerEntrypoint').value = (dc.entrypoint || []).join(' ');
    document.getElementById('dockerCommand').value = (dc.command || []).join(' ');
    document.getElementById('dockerWorkingDir').value = dc.working_dir || '';
}

/**
 * Save Docker configuration
 */
async function saveDockerConfig() {
    const entrypointStr = document.getElementById('dockerEntrypoint').value.trim();
    const commandStr = document.getElementById('dockerCommand').value.trim();

    const data = {
        dockerfile_path: document.getElementById('dockerfilePath').value || null,
        entrypoint: entrypointStr ? entrypointStr.split(' ') : null,
        command: commandStr ? commandStr.split(' ') : null,
        working_dir: document.getElementById('dockerWorkingDir').value || null,
        build_args: {},
        docker_args: []
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/docker`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save Docker configuration');

        alert('Docker configuration saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving Docker config:', error);
        alert('Failed to save Docker configuration: ' + error.message);
    }
}

/**
 * Populate network configuration section
 */
function populateNetworkConfig(network) {
    const nc = network || {};

    document.getElementById('exposePublic').checked = nc.expose_public || false;
    document.getElementById('customDomain').value = nc.custom_domain || '';
    document.getElementById('sslEnabled').checked = nc.ssl_enabled || false;
    document.getElementById('allowedIPs').value = (nc.allowed_ips || []).join(', ');
}

/**
 * Save network configuration
 */
async function saveNetworkConfig() {
    const allowedIPsStr = document.getElementById('allowedIPs').value.trim();

    const data = {
        expose_public: document.getElementById('exposePublic').checked,
        custom_domain: document.getElementById('customDomain').value || null,
        ssl_enabled: document.getElementById('sslEnabled').checked,
        allowed_ips: allowedIPsStr ? allowedIPsStr.split(',').map(ip => ip.trim()) : null,
        ports: []
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/network`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save network configuration');

        alert('Network configuration saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving network config:', error);
        alert('Failed to save network configuration: ' + error.message);
    }
}

/**
 * Populate GPU configuration section
 */
function populateGPUConfig(gpu) {
    const gc = gpu || {};

    document.getElementById('preferredGPUs').value = (gc.preferred_types || []).join(', ');
    document.getElementById('multiGPUStrategy').value = gc.multi_gpu_strategy || '';
    document.getElementById('cudaVersion').value = gc.cuda_version || '';
    document.getElementById('tensorCoresRequired').checked = gc.tensor_cores_required || false;
    document.getElementById('gpuUtilizationTarget').value = (gc.utilization_target || 0.8) * 100;
}

/**
 * Save GPU configuration
 */
async function saveGPUConfig() {
    const gpuTypesStr = document.getElementById('preferredGPUs').value.trim();

    const data = {
        gpu_types: gpuTypesStr ? gpuTypesStr.split(',').map(t => t.trim()) : null,
        multi_gpu_strategy: document.getElementById('multiGPUStrategy').value || null,
        cuda_version: document.getElementById('cudaVersion').value || null,
        tensor_cores: document.getElementById('tensorCoresRequired').checked,
        gpu_utilization_target: parseFloat(document.getElementById('gpuUtilizationTarget').value) / 100
    };

    try {
        const response = await fetch(`/api/deployments/${currentDeploymentId}/config/gpu`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to save GPU configuration');

        alert('GPU configuration saved successfully');
        await showDeploymentConfig(currentDeploymentId);

    } catch (error) {
        console.error('Error saving GPU config:', error);
        alert('Failed to save GPU configuration: ' + error.message);
    }
}

/**
 * Export complete configuration as JSON
 */
function exportConfiguration() {
    const blob = new Blob([JSON.stringify(currentConfig, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deployment-${currentDeploymentId}-config.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
