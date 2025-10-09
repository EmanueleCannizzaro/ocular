// AI Model Deployment Page JavaScript

// Configuration
const API_BASE_URL = window.location.origin;
const REFRESH_INTERVAL = 30000; // 30 seconds

// State
let activeDeployments = [];
let selectedProvider = null;
let selectedModel = null;
let autoRefreshTimer = null;
let availableProviders = [];
let availableModels = [];
let availableGPUs = [];
let currentSettings = null;
let deploymentOptionsPromise = null;

function isActiveStatus(status) {
    const normalized = (status || '').toString().toUpperCase();
    return ['ACTIVE', 'AVAILABLE', 'READY', 'ENABLED'].includes(normalized);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Deployment page initialized');

    // Load initial data
    loadSettings();
    loadDeploymentOptions();
    loadDeployments();
    loadMetrics();

    // Set up auto-refresh
    startAutoRefresh();

    // Event listeners
    setupEventListeners();
});

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Provider selection
    document.querySelectorAll('.provider-card').forEach(card => {
        card.addEventListener('click', function() {
            const provider = this.getAttribute('data-provider');
            if (!this.classList.contains('disabled')) {
                selectProvider(provider);
            }
        });
    });
}

/**
 * Select a cloud provider
 */
function selectProvider(provider) {
    selectedProvider = provider;

    // Update UI
    document.querySelectorAll('.provider-card').forEach(card => {
        card.classList.remove('selected');
    });

    const selectedCard = document.querySelector(`[data-provider="${provider}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }

    console.log(`Selected provider: ${provider}`);
    showToast(`Selected ${provider} as cloud provider`, 'info');
}

/**
 * Load active deployments
 */
async function loadDeployments() {
    const tbody = document.getElementById('deployments-tbody');

    try {
        // Simulated deployment data - replace with actual API call
        const deployments = await fetchDeployments();

        if (deployments.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <h5>No Active Deployments</h5>
                        <p>Deploy your first AI model to get started</p>
                        <button class="btn btn-primary" onclick="showNewDeployment()">
                            <i class="fas fa-plus me-1"></i> New Deployment
                        </button>
                    </td>
                </tr>
            `;
            return;
        }

        // Render deployments
        tbody.innerHTML = deployments.map(deployment => `
            <tr>
                <td>
                    <strong>${deployment.model}</strong>
                    <br>
                    <small class="text-muted">${deployment.version}</small>
                </td>
                <td>
                    <span class="badge bg-secondary">${deployment.provider}</span>
                </td>
                <td>
                    <span class="gpu-badge">${deployment.gpu_type}</span>
                </td>
                <td>
                    <span class="status-indicator ${deployment.status}"></span>
                    <span class="deployment-status-badge ${deployment.status}">
                        ${deployment.status}
                    </span>
                </td>
                <td>
                    <code class="endpoint-url" onclick="copyToClipboard('${deployment.endpoint}')">
                        ${deployment.endpoint}
                    </code>
                </td>
                <td>
                    <span class="cost-display ${deployment.cost_class}">
                        $${deployment.cost_per_hour}/hr
                    </span>
                </td>
                <td>
                    <button class="action-btn action-btn-view" onclick="viewDeployment('${deployment.id}')" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${deployment.status === 'running' ? `
                        <button class="action-btn action-btn-stop" onclick="stopDeployment('${deployment.id}')" title="Stop">
                            <i class="fas fa-stop"></i>
                        </button>
                    ` : `
                        <button class="action-btn action-btn-restart" onclick="restartDeployment('${deployment.id}')" title="Start">
                            <i class="fas fa-play"></i>
                        </button>
                    `}
                    <button class="action-btn action-btn-delete" onclick="deleteDeployment('${deployment.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        activeDeployments = deployments;

    } catch (error) {
        console.error('Error loading deployments:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to load deployments
                </td>
            </tr>
        `;
    }
}

/**
 * Load metrics
 */
async function loadMetrics() {
    try {
        const metrics = await fetchMetrics();

        document.getElementById('active-deployments').textContent = metrics.active_deployments || '0';
        document.getElementById('gpu-count').textContent = metrics.gpu_count || '0';
        document.getElementById('monthly-cost').textContent = `$${metrics.monthly_cost || '0'}`;
        document.getElementById('avg-response').textContent = metrics.avg_response || '--';

    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

/**
 * Fetch deployments from API
 */
async function fetchDeployments() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/deployments`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const deployments = await response.json();

        // Transform API response to match UI expectations
        return deployments.map(d => ({
            id: d.id,
            model: d.model_display_name || d.model_name,
            version: d.model_version,
            provider: d.provider_name,
            gpu_type: d.gpu_type,
            status: d.status,
            endpoint: d.endpoint || 'Not assigned',
            cost_per_hour: d.cost_per_hour.toFixed(2),
            cost_class: d.cost_class
        }));

    } catch (error) {
        console.error('Error fetching deployments:', error);
        return [];
    }
}

/**
 * Fetch metrics from API
 */
async function fetchMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/deployments/metrics`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();

    } catch (error) {
        console.error('Error fetching metrics:', error);
        return {
            active_deployments: 0,
            gpu_count: 0,
            monthly_cost: 0,
            avg_response: 'N/A',
            total_requests: 0,
            total_cost: 0
        };
    }
}

/**
 * Load providers from API (uses ocular.json config)
 */
async function loadProviders(force = false) {
    await loadDeploymentOptions(force);
}

async function loadModels(force = false) {
    await loadDeploymentOptions(force);
}

async function loadDeploymentOptions(force = false) {
    if (deploymentOptionsPromise && !force) {
        return deploymentOptionsPromise;
    }

    deploymentOptionsPromise = (async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/deployments/options`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            availableModels = data.models || [];
            availableProviders = data.providers || [];

            console.log('Loaded deployment options:', data);

            updateModelCards();
            updateProviderCards();
            populateFormOptions();

            return data;
        } catch (error) {
            console.error('Error loading deployment options:', error);
            availableModels = [];
            availableProviders = [];
            populateFormOptions();
            return null;
        }
    })();

    return deploymentOptionsPromise;
}

/**
 * Update provider cards with database data
 */
function updateProviderCards() {
    const providerCardsContainer = document.getElementById('provider-cards');
    if (!providerCardsContainer || availableProviders.length === 0) return;

    providerCardsContainer.innerHTML = availableProviders.map(provider => {
        const isActive = isActiveStatus(provider.status);
        const iconMap = {
            'runpod': 'fa-server',
            'vastai': 'fa-hdd',
            'lambda': 'fa-bolt'
        };
        const colorMap = {
            'runpod': 'primary',
            'vastai': 'info',
            'lambda': 'warning'
        };

        return `
            <div class="col-md-4 mb-3">
                <div class="provider-card ${isActive ? '' : 'disabled'}" data-provider="${provider.name}">
                    <div class="provider-logo">
                        <i class="fas ${iconMap[provider.name] || 'fa-cloud'} fa-3x text-${colorMap[provider.name] || 'secondary'}"></i>
                    </div>
                    <h5 class="mt-3">${provider.display_name}</h5>
                    <p class="text-muted small">${provider.description || 'Cloud GPU deployment'}</p>
                    <div class="provider-status">
                        <span class="badge bg-${isActive ? 'success' : 'secondary'}">
                            ${isActive ? 'Available' : 'Coming Soon'}
                        </span>
                    </div>
                    <button class="btn btn-outline-${isActive ? colorMap[provider.name] : 'secondary'} btn-sm mt-2"
                            onclick="selectProvider('${provider.name}')"
                            ${!isActive ? 'disabled' : ''}>
                        <i class="fas fa-cog me-1"></i> Configure
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Update model cards with database data
 */
function updateModelCards() {
    const modelsGrid = document.getElementById('models-grid');
    if (!modelsGrid || availableModels.length === 0) return;

    modelsGrid.innerHTML = availableModels.map(model => {
        const isReady = model.status === 'ACTIVE';

        return `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="model-card">
                    <div class="model-header">
                        <h6>${model.display_name}</h6>
                        <span class="badge bg-${isReady ? 'success' : 'warning'}">
                            ${isReady ? 'Ready' : model.status}
                        </span>
                    </div>
                    <div class="model-details">
                        <p class="text-muted small mb-2">${model.description || model.version}</p>
                        <div class="model-specs">
                            <span class="spec-item">
                                <i class="fas fa-memory me-1"></i> ${model.min_vram_gb}GB VRAM
                            </span>
                            ${model.recommended_gpu ? `
                                <span class="spec-item">
                                    <i class="fas fa-microchip me-1"></i> GPU: ${model.recommended_gpu}
                                </span>
                            ` : ''}
                        </div>
                    </div>
                    <button class="btn btn-sm btn-primary w-100 mt-2"
                            onclick="deployModel('${model.name}', ${model.id})"
                            ${!isReady ? 'disabled' : ''}>
                        <i class="fas fa-rocket me-1"></i> Deploy
                    </button>
                </div>
            </div>
        `;
    }).join('');

    // Add custom model card
    modelsGrid.innerHTML += `
        <div class="col-md-6 col-lg-4 mb-3">
            <div class="model-card">
                <div class="model-header">
                    <h6>Custom Model</h6>
                    <span class="badge bg-info">Configure</span>
                </div>
                <div class="model-details">
                    <p class="text-muted small mb-2">Deploy your own model</p>
                    <div class="model-specs">
                        <span class="spec-item"><i class="fas fa-question me-1"></i> Custom specs</span>
                    </div>
                </div>
                <button class="btn btn-sm btn-outline-primary w-100 mt-2" onclick="deployCustomModel()">
                    <i class="fas fa-cog me-1"></i> Configure
                </button>
            </div>
        </div>
    `;
}

/**
 * Show new deployment modal
 */
async function showNewDeployment() {
    // Ensure models and providers are loaded
    if (availableModels.length === 0) {
        await loadModels();
    }
    if (availableProviders.length === 0) {
        await loadProviders();
    }
    if (!currentSettings) {
        await loadSettings();
    }

    // Populate form options from database
    populateFormOptions();

    const modal = new bootstrap.Modal(document.getElementById('newDeploymentModal'));
    modal.show();
}

/**
 * Populate form select options from database
 */
function populateFormOptions() {
    console.log('Populating form options...');
    console.log('Available models:', availableModels.length);
    console.log('Available providers:', availableProviders.length);

    // Populate model select
    const modelSelect = document.getElementById('model-select');
    const modelAlert = document.getElementById('model-select-alert');
    if (modelSelect) {
        if (availableModels.length > 0) {
            const activeModels = availableModels.filter(model => isActiveStatus(model.status));
            const modelsToRender = activeModels.length ? activeModels : availableModels;
            console.log('Renderable models:', modelsToRender.length);

            modelSelect.innerHTML = '<option value="">Choose a model...</option>' +
                modelsToRender.map(model => `
                    <option value="${model.id}" data-name="${model.name}">
                        ${model.display_name} - ${model.version}
                    </option>
                `).join('') +
                '<option value="custom">Custom Model</option>';

            modelSelect.disabled = modelsToRender.length === 0;
            modelSelect.selectedIndex = 0;
            if (modelAlert) {
                modelAlert.classList.toggle('d-none', modelsToRender.length > 0);
            }
            console.log('Model select populated with', modelSelect.options.length, 'options');
        } else {
            console.log('No models from database.');
            modelSelect.innerHTML = '<option value="">No deployable models found</option>';
            modelSelect.disabled = true;
            if (modelAlert) {
                modelAlert.classList.remove('d-none');
            }
        }
    } else {
        console.error('Model select element not found!');
    }

    // Populate provider select
    const providerSelect = document.getElementById('provider-select');
    const providerAlert = document.getElementById('provider-select-alert');
    if (providerSelect) {
        if (availableProviders.length > 0) {
            const activeProviders = availableProviders.filter(provider => isActiveStatus(provider.status));
            const providersToRender = activeProviders.length ? activeProviders : availableProviders;
            console.log('Renderable providers:', providersToRender.length);

            providerSelect.innerHTML = '<option value="">Choose a provider...</option>' +
                providersToRender.map(provider => {
                    const label = isActiveStatus(provider.status)
                        ? provider.display_name
                        : `${provider.display_name} (${provider.status || 'Unavailable'})`;
                    return `
                        <option value="${provider.id}" data-name="${provider.name}">
                            ${label}
                        </option>
                    `;
                }).join('');

            providerSelect.disabled = providersToRender.length === 0;
            providerSelect.selectedIndex = 0;
            if (providerAlert) {
                providerAlert.classList.toggle('d-none', providersToRender.length > 0);
            }
            console.log('Provider select populated with', providerSelect.options.length, 'options');
        } else {
            console.log('No providers from database.');
            providerSelect.innerHTML = '<option value="">No providers configured</option>';
            providerSelect.disabled = true;
            if (providerAlert) {
                providerAlert.classList.remove('d-none');
            }
        }
    } else {
        console.error('Provider select element not found!');
    }

    console.log('Form options population complete');
}

/**
 * Update GPU options based on selected provider
 */
function updateGPUOptions(providerId) {
    const provider = availableProviders.find(p => p.id == providerId);
    const gpuSelect = document.getElementById('gpu-select');

    if (!provider || !gpuSelect) return;

    if (provider.available_gpus && provider.available_gpus.length > 0) {
        gpuSelect.innerHTML = '<option value="">Select GPU...</option>' +
            provider.available_gpus.map(gpu => `
                <option value="${gpu}">${gpu}</option>
            `).join('');
    } else {
        gpuSelect.innerHTML = '<option value="">No GPUs available</option>';
    }
}

/**
 * Deploy a specific model
 */
function deployModel(modelName, modelId) {
    showNewDeployment();

    // Pre-fill the model selection
    setTimeout(() => {
        const modelSelect = document.getElementById('model-select');
        if (modelSelect && modelId) {
            modelSelect.value = modelId;

            // Also pre-fill deployment name
            const deploymentNameInput = document.getElementById('deployment-name');
            if (deploymentNameInput) {
                deploymentNameInput.value = `${modelName}-${Date.now()}`;
            }
        }
    }, 100);
}

/**
 * Deploy custom model
 */
function deployCustomModel() {
    showNewDeployment();

    setTimeout(() => {
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            modelSelect.value = 'custom';
        }
    }, 100);
}

/**
 * Submit deployment
 */
async function submitDeployment() {
    const form = document.getElementById('deployment-form');

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    // Get selected provider and model for additional data
    const modelId = parseInt(document.getElementById('model-select').value);
    const providerId = parseInt(document.getElementById('provider-select').value);
    const selectedModel = availableModels.find(m => m.id === modelId);
    const selectedProvider = availableProviders.find(p => p.id === providerId);

    // Calculate cost per hour (example - should come from pricing API)
    const gpuType = document.getElementById('gpu-select').value;
    const costPerHour = calculateCostPerHour(gpuType, selectedProvider);

    const deploymentData = {
        deployment_name: document.getElementById('deployment-name').value,
        model_id: modelId,
        provider_id: providerId,
        gpu_type: gpuType,
        gpu_count: 1,
        region: selectedProvider?.extra_metadata?.default_region || 'eur-is-1',
        api_key: document.getElementById('api-key').value,
        deployment_type: document.getElementById('deployment-type').value,
        autoscaling: document.getElementById('autoscaling').value === 'enabled',
        min_workers: 0,
        max_workers: parseInt(document.getElementById('max-workers').value),
        timeout: parseInt(document.getElementById('timeout').value),
        cost_per_hour: costPerHour,
        config: {
            container_disk_gb: 30,
            volumes_in_gb: 30,
            idle_timeout: 30
        },
        metadata: {
            created_from: 'web_ui',
            model_name: selectedModel?.name,
            provider_name: selectedProvider?.name
        }
    };

    console.log('Submitting deployment:', deploymentData);

    // Show loading state
    showLoading('Deploying model to cloud provider...');

    try {
        const response = await fetch(`${API_BASE_URL}/api/deployments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(deploymentData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        hideLoading();

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('newDeploymentModal'));
        modal.hide();

        // Show success message with endpoint info
        showToast(
            `Deployment started successfully! Endpoint ID: ${result.instance_id || 'Pending'}`,
            'success'
        );

        // Reload deployments and metrics
        setTimeout(() => {
            loadDeployments();
            loadMetrics();
        }, 1000);

    } catch (error) {
        console.error('Deployment error:', error);
        hideLoading();
        showToast('Deployment failed: ' + error.message, 'error');
    }
}

/**
 * Calculate cost per hour for GPU type
 */
function calculateCostPerHour(gpuType, provider) {
    // Pricing map (should ideally come from API or provider metadata)
    const pricingMap = {
        'RTX 4000 Ada': 0.29,
        'RTX A4000': 0.34,
        'RTX A5000': 0.59,
        'RTX A6000': 0.79,
        'A100 PCIe': 1.89,
        'A100 SXM': 2.49
    };

    return pricingMap[gpuType] || provider?.min_cost_per_hour || 0.50;
}

/**
 * View deployment details
 */
function viewDeployment(deploymentId) {
    console.log('Viewing deployment:', deploymentId);
    showToast('Deployment details coming soon', 'info');
    // TODO: Implement deployment details modal
}

/**
 * Stop deployment
 */
async function stopDeployment(deploymentId) {
    if (!confirm('Are you sure you want to stop this deployment?')) {
        return;
    }

    console.log('Stopping deployment:', deploymentId);
    showLoading('Stopping deployment...');

    try {
        // Simulated API call
        await new Promise(resolve => setTimeout(resolve, 1500));

        hideLoading();
        showToast('Deployment stopped successfully', 'success');
        loadDeployments();

    } catch (error) {
        console.error('Error stopping deployment:', error);
        hideLoading();
        showToast('Failed to stop deployment', 'error');
    }
}

/**
 * Restart deployment
 */
async function restartDeployment(deploymentId) {
    console.log('Restarting deployment:', deploymentId);
    showLoading('Starting deployment...');

    try {
        // Simulated API call
        await new Promise(resolve => setTimeout(resolve, 1500));

        hideLoading();
        showToast('Deployment started successfully', 'success');
        loadDeployments();

    } catch (error) {
        console.error('Error restarting deployment:', error);
        hideLoading();
        showToast('Failed to start deployment', 'error');
    }
}

/**
 * Delete deployment
 */
async function deleteDeployment(deploymentId) {
    if (!confirm('Are you sure you want to delete this deployment? This action cannot be undone.')) {
        return;
    }

    console.log('Deleting deployment:', deploymentId);
    showLoading('Deleting deployment...');

    try {
        // Simulated API call
        await new Promise(resolve => setTimeout(resolve, 1500));

        hideLoading();
        showToast('Deployment deleted successfully', 'success');
        loadDeployments();
        loadMetrics();

    } catch (error) {
        console.error('Error deleting deployment:', error);
        hideLoading();
        showToast('Failed to delete deployment', 'error');
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Endpoint copied to clipboard', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to copy endpoint', 'error');
    });
}

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
    }

    autoRefreshTimer = setInterval(() => {
        loadDeployments();
        loadMetrics();
    }, REFRESH_INTERVAL);

    console.log('Auto-refresh started');
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
        autoRefreshTimer = null;
        console.log('Auto-refresh stopped');
    }
}

/**
 * Show loading overlay
 */
function showLoading(message = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const container = document.querySelector('.toast-container') || createToastContainer();

    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas ${getToastIcon(type)} me-2"></i>
            <div>${message}</div>
        </div>
    `;

    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Create toast container
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
}

/**
 * Get icon for toast type
 */
function getToastIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return icons[type] || icons.info;
}

/**
 * Load settings from API
 */
async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/settings`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        currentSettings = data.settings;
        console.log('Loaded settings:', currentSettings);
    } catch (error) {
        console.error('Error loading settings:', error);
        currentSettings = {};
    }
}

/**
 * Handle model selection change
 */
function handleModelChange() {
    const modelSelect = document.getElementById('model-select');
    const modelId = parseInt(modelSelect.value);
    selectedModel = availableModels.find(m => m.id === modelId);

    if (selectedModel) {
        // Show model configuration section
        document.getElementById('step-model-config').style.display = 'block';
        generateModelConfigFields(selectedModel);

        // Check if provider is also selected
        if (selectedProvider) {
            document.getElementById('step-model-upload').style.display = 'block';
            document.getElementById('step-options').style.display = 'block';

            // Pre-populate S3 bucket name and HF model ID
            const bucketInput = document.getElementById('s3-bucket-name');
            const hfModelInput = document.getElementById('hf-model-id');

            if (bucketInput && currentSettings?.RUNPOD_NETWORK_VOLUME_ID) {
                bucketInput.value = currentSettings.RUNPOD_NETWORK_VOLUME_ID;
            }

            if (hfModelInput && selectedModel.version) {
                hfModelInput.value = selectedModel.version;
            }

            // Pre-populate HF token if available
            const hfTokenInput = document.getElementById('hf-token');
            if (hfTokenInput && currentSettings?.HF_TOKEN) {
                hfTokenInput.value = currentSettings.HF_TOKEN;
            }
        }
    }
}

/**
 * Handle provider selection change
 */
function handleProviderChange() {
    const providerSelect = document.getElementById('provider-select');
    const providerId = parseInt(providerSelect.value);
    selectedProvider = availableProviders.find(p => p.id === providerId);

    if (selectedProvider) {
        // Show provider configuration section
        document.getElementById('step-provider-config').style.display = 'block';
        generateProviderConfigFields(selectedProvider);

        // Update GPU options
        updateGPUOptions(providerId);

        // Check if model is also selected
        if (selectedModel) {
            document.getElementById('step-model-upload').style.display = 'block';
            document.getElementById('step-options').style.display = 'block';
        }
    }
}

/**
 * Handle model source change (HuggingFace/S3/Upload)
 */
function handleModelSourceChange() {
    const source = document.getElementById('model-source').value;

    // Hide all source-specific sections
    document.getElementById('huggingface-source').style.display = 'none';
    document.getElementById('s3-source').style.display = 'none';
    document.getElementById('upload-source').style.display = 'none';

    // Show selected source section
    if (source === 'huggingface') {
        document.getElementById('huggingface-source').style.display = 'block';
    } else if (source === 's3') {
        document.getElementById('s3-source').style.display = 'block';
    } else if (source === 'upload') {
        document.getElementById('upload-source').style.display = 'block';
    }
}

/**
 * Toggle S3 upload section visibility
 */
function toggleS3Upload() {
    const skipUpload = document.getElementById('skip-s3-upload').checked;
    const uploadSection = document.getElementById('step-model-upload');

    if (skipUpload) {
        // Collapse all S3-related fields except the checkbox
        uploadSection.querySelectorAll('.row').forEach(row => {
            if (!row.querySelector('#skip-s3-upload')) {
                row.style.opacity = '0.5';
                row.style.pointerEvents = 'none';
            }
        });
    } else {
        // Restore all fields
        uploadSection.querySelectorAll('.row').forEach(row => {
            row.style.opacity = '1';
            row.style.pointerEvents = 'auto';
        });
    }
}

/**
 * Upload model files to S3
 */
async function uploadModelToS3() {
    const files = document.getElementById('model-files').files;
    const bucketName = document.getElementById('s3-bucket-name').value;
    const modelSource = document.getElementById('model-source').value;

    if (!bucketName) {
        showToast('Please enter S3 bucket name', 'error');
        return null;
    }

    if (modelSource === 'upload' && files.length === 0) {
        showToast('Please select model files to upload', 'error');
        return null;
    }

    try {
        const formData = new FormData();

        // Add all files to form data
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        formData.append('bucket_name', bucketName);
        formData.append('model_name', selectedModel.name);

        // Show upload progress
        const progressBar = document.getElementById('upload-progress-bar');
        const progressContainer = document.getElementById('upload-progress');
        const statusDiv = document.getElementById('upload-status');

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        statusDiv.innerHTML = '<small class="text-muted">Uploading files to S3...</small>';

        const response = await fetch(`${API_BASE_URL}/api/deployments/upload-model`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();

        // Update progress to 100%
        progressBar.style.width = '100%';
        statusDiv.innerHTML = `<small class="text-success"><i class="fas fa-check-circle me-1"></i>Upload complete! ${result.file_count} files uploaded</small>`;

        return result.s3_path;

    } catch (error) {
        console.error('S3 upload error:', error);

        const statusDiv = document.getElementById('upload-status');
        statusDiv.innerHTML = `<small class="text-danger"><i class="fas fa-exclamation-circle me-1"></i>Upload failed: ${error.message}</small>`;

        showToast('Failed to upload model to S3: ' + error.message, 'error');
        return null;
    }
}

/**
 * Generate provider-specific configuration fields
 */
function generateProviderConfigFields(provider) {
    const container = document.getElementById('provider-config-fields');
    if (!container) return;

    let html = '';

    if (provider.name === 'runpod') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">RunPod API Key</label>
                    <div class="input-group">
                        <input type="password" class="form-control" id="runpod-api-key-input"
                               value="${currentSettings?.RUNPOD_API_KEY || ''}"
                               placeholder="Enter RunPod API key">
                        <button class="btn btn-outline-secondary" type="button" onclick="togglePasswordField('runpod-api-key-input')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                    <small class="text-muted">From settings: ${currentSettings?.RUNPOD_API_KEY ? '✓ Configured' : '✗ Not configured'}</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Region</label>
                    <select class="form-select" id="runpod-region-input">
                        <option value="eur-is-1" ${currentSettings?.RUNPOD_REGION === 'eur-is-1' ? 'selected' : ''}>EUR-IS-1 (Iceland)</option>
                        <option value="us-ca-1" ${currentSettings?.RUNPOD_REGION === 'us-ca-1' ? 'selected' : ''}>US-CA-1 (California)</option>
                        <option value="us-tx-1" ${currentSettings?.RUNPOD_REGION === 'us-tx-1' ? 'selected' : ''}>US-TX-1 (Texas)</option>
                    </select>
                    <small class="text-muted">From settings: ${currentSettings?.RUNPOD_REGION || 'Default'}</small>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Network Volume ID <span class="text-muted">(optional)</span></label>
                    <input type="text" class="form-control" id="runpod-volume-id-input"
                           value="${currentSettings?.RUNPOD_NETWORK_VOLUME_ID || ''}"
                           placeholder="volume-id">
                    <small class="text-muted">For persistent model storage</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">S3 Bucket Endpoint <span class="text-muted">(optional)</span></label>
                    <input type="url" class="form-control" id="runpod-bucket-endpoint-input"
                           value="${currentSettings?.RUNPOD_BUCKET_ENDPOINT_URL || ''}"
                           placeholder="https://s3api-eur-is-1.runpod.io/">
                    <small class="text-muted">For file upload/download</small>
                </div>
            </div>
        `;
    } else if (provider.name === 'vastai') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Vast.ai API Key</label>
                    <div class="input-group">
                        <input type="password" class="form-control" id="vastai-api-key-input"
                               value="${currentSettings?.VASTAI_API_KEY || ''}"
                               placeholder="Enter Vast.ai API key">
                        <button class="btn btn-outline-secondary" type="button" onclick="togglePasswordField('vastai-api-key-input')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                    <small class="text-muted">From settings: ${currentSettings?.VASTAI_API_KEY ? '✓ Configured' : '✗ Not configured'}</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Instance ID</label>
                    <input type="text" class="form-control" id="vastai-instance-id-input"
                           value="${currentSettings?.VASTAI_INSTANCE_ID || ''}"
                           placeholder="instance-12345">
                    <small class="text-muted">Your Vast.ai instance identifier</small>
                </div>
            </div>
        `;
    } else if (provider.name === 'deepinfra') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">DeepInfra API Token</label>
                    <div class="input-group">
                        <input type="password" class="form-control" id="deepinfra-token-input"
                               value="${currentSettings?.DEEPINFRA_TOKEN || ''}"
                               placeholder="Enter DeepInfra token">
                        <button class="btn btn-outline-secondary" type="button" onclick="togglePasswordField('deepinfra-token-input')">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                    <small class="text-muted">From settings: ${currentSettings?.DEEPINFRA_TOKEN ? '✓ Configured' : '✗ Not configured'}</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Base URL</label>
                    <input type="url" class="form-control" id="deepinfra-base-url-input"
                           value="${currentSettings?.DEEPINFRA_BASE_URL || 'https://api.deepinfra.com/v1/openai'}"
                           placeholder="https://api.deepinfra.com/v1/openai">
                    <small class="text-muted">API endpoint URL</small>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Generate model-specific configuration fields
 */
function generateModelConfigFields(model) {
    const container = document.getElementById('model-config-fields');
    if (!container) return;

    const modelName = model.name.toLowerCase();
    let html = '';

    if (modelName === 'olmocr') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Model Name</label>
                    <input type="text" class="form-control" id="olmocr-model-name-input"
                           value="${currentSettings?.OLMOCR_MODEL_NAME || model.version}"
                           placeholder="allenai/olmOCR-7B-0825">
                    <small class="text-muted">From settings: ${currentSettings?.OLMOCR_MODEL_NAME || 'Default'}</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Docker Image</label>
                    <input type="text" class="form-control" id="olmocr-docker-image-input"
                           value="${model.docker_image || ''}"
                           placeholder="ocular/olmocr-runpod:latest">
                    <small class="text-muted">Container image for deployment</small>
                </div>
            </div>
            <div class="row">
                <div class="col-md-4 mb-3">
                    <label class="form-label">Max File Size (MB)</label>
                    <input type="number" class="form-control" id="olmocr-max-file-size-input"
                           value="${currentSettings?.OLMOCR_MAX_FILE_SIZE_MB || 50}" min="1" max="500">
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Timeout (seconds)</label>
                    <input type="number" class="form-control" id="olmocr-timeout-input"
                           value="${currentSettings?.OLMOCR_TIMEOUT_SECONDS || 300}" min="30" max="600">
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Allowed Extensions</label>
                    <input type="text" class="form-control" id="olmocr-extensions-input"
                           value="${currentSettings?.OLMOCR_ALLOWED_EXTENSIONS || '.pdf,.png,.jpg,.jpeg'}"
                           placeholder=".pdf,.png,.jpg">
                </div>
            </div>
        `;
    } else if (modelName === 'rolmocr') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Model Version</label>
                    <input type="text" class="form-control" id="rolmocr-version-input"
                           value="${model.version}"
                           placeholder="Model version">
                    <small class="text-muted">RoLMocr model version</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Docker Image</label>
                    <input type="text" class="form-control" id="rolmocr-docker-image-input"
                           value="${model.docker_image || ''}"
                           placeholder="ocular/rolmocr-runpod:latest">
                    <small class="text-muted">Container image for deployment</small>
                </div>
            </div>
        `;
    } else if (modelName === 'pixtral') {
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Model Version</label>
                    <input type="text" class="form-control" id="pixtral-version-input"
                           value="${model.version}"
                           placeholder="mistralai/pixtral-12b">
                    <small class="text-muted">Pixtral model version</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Docker Image</label>
                    <input type="text" class="form-control" id="pixtral-docker-image-input"
                           value="${model.docker_image || ''}"
                           placeholder="ocular/pixtral-runpod:latest">
                    <small class="text-muted">Container image for deployment</small>
                </div>
            </div>
        `;
    } else {
        // Generic model configuration
        html = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Model Identifier</label>
                    <input type="text" class="form-control" id="custom-model-name-input"
                           value="${model.version}"
                           placeholder="huggingface/model-name">
                    <small class="text-muted">HuggingFace or custom model identifier</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Docker Image</label>
                    <input type="text" class="form-control" id="custom-docker-image-input"
                           value="${model.docker_image || ''}"
                           placeholder="your-registry/image:tag">
                    <small class="text-muted">Container image for deployment</small>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Toggle password field visibility
 */
function togglePasswordField(fieldId) {
    const field = document.getElementById(fieldId);
    const button = field.nextElementSibling;
    const icon = button.querySelector('i');

    if (field.type === 'password') {
        field.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        field.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
});
