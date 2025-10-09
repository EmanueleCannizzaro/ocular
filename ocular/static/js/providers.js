/**
 * Providers Management JavaScript
 * Handles management of OCR providers and cloud GPU providers
 */

const API_BASE_URL = window.location.origin;
let ocrProviders = [];
let cloudProviders = [];
let currentEditingProvider = null;

/**
 * Initialize the page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Providers management page loaded');
    loadOCRProviders();
    loadCloudProviders();
});

/**
 * Load OCR providers status
 */
async function loadOCRProviders() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/ocr-providers/status`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        ocrProviders = data.providers || [];

        console.log('Loaded OCR providers:', ocrProviders);
        renderOCRProviders();

    } catch (error) {
        console.error('Error loading OCR providers:', error);
        showNotification('Error loading OCR providers: ' + error.message, 'danger');

        // Show error state
        document.getElementById('ocr-providers-grid').innerHTML = `
            <div class="col-12 text-center text-danger py-5">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <p>Error loading OCR providers</p>
                <small>${error.message}</small>
            </div>
        `;
    }
}

/**
 * Render OCR providers grid with status cards
 */
function renderOCRProviders() {
    const grid = document.getElementById('ocr-providers-grid');

    if (!grid) {
        console.error('OCR providers grid not found');
        return;
    }

    // Calculate stats
    const available = ocrProviders.filter(p => p.available).length;
    const unavailable = ocrProviders.filter(p => !p.available).length;
    const total = ocrProviders.length;

    // Update stats
    document.getElementById('ocr-available-count').textContent = available;
    document.getElementById('ocr-unavailable-count').textContent = unavailable;
    document.getElementById('ocr-total-count').textContent = total;

    if (ocrProviders.length === 0) {
        grid.innerHTML = `
            <div class="col-12 text-center text-muted py-5">
                <i class="fas fa-info-circle fa-2x mb-3"></i>
                <p>No OCR providers configured</p>
            </div>
        `;
        return;
    }

    // Render provider cards
    grid.innerHTML = ocrProviders.map(provider => `
        <div class="col-md-4 mb-4">
            <div class="card ${provider.available ? 'border-success' : 'border-warning'} h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-3">
                        <div class="provider-icon me-3">
                            ${getProviderIcon(provider.name)}
                        </div>
                        <div>
                            <h5 class="card-title mb-0">${provider.display_name}</h5>
                            <small class="text-muted">${provider.name}</small>
                        </div>
                    </div>

                    <div class="mb-3">
                        ${provider.available
                            ? '<span class="badge bg-success"><i class="fas fa-check-circle"></i> Available</span>'
                            : '<span class="badge bg-warning"><i class="fas fa-exclamation-triangle"></i> Unavailable</span>'}
                    </div>

                    ${provider.description ? `<p class="card-text small text-muted">${provider.description}</p>` : ''}

                    ${provider.configuration_status ? `
                        <div class="small">
                            <strong>Status:</strong> ${provider.configuration_status}
                        </div>
                    ` : ''}

                    ${!provider.available && provider.missing_config ? `
                        <div class="alert alert-warning small mt-2 mb-0">
                            <strong>Missing:</strong> ${provider.missing_config.join(', ')}
                        </div>
                    ` : ''}
                </div>
            </div>
        </div>
    `).join('');
}

/**
 * Get icon for provider
 */
function getProviderIcon(providerName) {
    const icons = {
        'mistral': '<i class="fas fa-robot fa-2x text-primary"></i>',
        'google_vision': '<i class="fab fa-google fa-2x text-danger"></i>',
        'aws_textract': '<i class="fab fa-aws fa-2x text-warning"></i>',
        'azure_document_intelligence': '<i class="fab fa-microsoft fa-2x text-info"></i>',
        'tesseract': '<i class="fas fa-font fa-2x text-secondary"></i>',
        'olm_ocr': '<i class="fas fa-brain fa-2x text-success"></i>',
        'rolm_ocr': '<i class="fas fa-brain fa-2x text-success"></i>'
    };

    return icons[providerName] || '<i class="fas fa-cube fa-2x text-secondary"></i>';
}

/**
 * Check/refresh OCR providers status
 */
async function checkOCRProviders() {
    // Show loading state
    const btn = event.target.closest('button');
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Checking...';
    btn.disabled = true;

    try {
        await loadOCRProviders();
        showNotification('OCR providers status refreshed', 'success');
    } finally {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
    }
}

/**
 * Load cloud GPU providers
 */
async function loadCloudProviders() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/providers`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        cloudProviders = data.providers || [];

        console.log('Loaded cloud providers:', cloudProviders);
        renderCloudProviders();

    } catch (error) {
        console.error('Error loading cloud providers:', error);
        showNotification('Error loading cloud providers: ' + error.message, 'danger');

        // Show error state
        document.getElementById('cloud-providers-grid').innerHTML = `
            <div class="col-12 text-center text-danger py-5">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <p>Error loading cloud providers</p>
                <small>${error.message}</small>
            </div>
        `;
    }
}

/**
 * Render cloud providers grid
 */
function renderCloudProviders() {
    const grid = document.getElementById('cloud-providers-grid');

    if (!grid) {
        console.error('Cloud providers grid not found');
        return;
    }

    if (cloudProviders.length === 0) {
        grid.innerHTML = `
            <div class="col-12 text-center text-muted py-5">
                <i class="fas fa-cloud fa-3x mb-3"></i>
                <p>No cloud providers configured</p>
                <button class="btn btn-primary" onclick="showAddProviderModal()">
                    <i class="fas fa-plus me-2"></i>
                    Add Your First Provider
                </button>
            </div>
        `;
        return;
    }

    grid.innerHTML = cloudProviders.map(provider => {
        const metadata = provider.metadata || {};
        const pricing = metadata.pricing || {};
        const features = metadata.features || {};

        return `
        <div class="col-md-6 col-lg-4 mb-4">
            <div class="card ${provider.status === 'active' ? 'border-success' : 'border-secondary'} h-100">
                <div class="card-body d-flex flex-column">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <div>
                            <h5 class="card-title mb-1">${provider.display_name}</h5>
                            ${provider.status === 'active'
                                ? '<span class="badge bg-success">Active</span>'
                                : '<span class="badge bg-secondary">Coming Soon</span>'}
                        </div>
                    </div>

                    <p class="card-text text-muted small mb-3">${provider.description || ''}</p>

                    ${pricing.min_price_per_hour ? `
                        <div class="mb-3">
                            <div class="small">
                                <i class="fas fa-dollar-sign text-primary me-1"></i>
                                <strong>Pricing:</strong> $${pricing.min_price_per_hour}/hr - $${pricing.max_price_per_hour}/hr
                            </div>
                            ${pricing.pricing_details ? `
                                <div class="text-muted small mt-1">${pricing.pricing_details}</div>
                            ` : ''}
                        </div>
                    ` : ''}

                    <div class="mb-3">
                        <strong class="small d-block mb-2">Available GPUs:</strong>
                        <div class="d-flex flex-wrap gap-1">
                            ${provider.available_gpus && provider.available_gpus.length > 0
                                ? provider.available_gpus.slice(0, 4).map(gpu =>
                                    `<span class="badge bg-secondary small">${gpu}</span>`
                                ).join('')
                                : '<span class="text-muted small">No GPUs listed</span>'}
                            ${provider.available_gpus && provider.available_gpus.length > 4
                                ? `<span class="badge bg-light text-dark small">+${provider.available_gpus.length - 4} more</span>`
                                : ''}
                        </div>
                    </div>

                    ${Object.keys(features).length > 0 ? `
                        <div class="mb-3">
                            <strong class="small d-block mb-2">Key Features:</strong>
                            <div class="small">
                                ${features.per_second_billing ? '<i class="fas fa-check text-success me-1"></i> Per-second billing<br>' : ''}
                                ${features.serverless_inference ? '<i class="fas fa-check text-success me-1"></i> Serverless inference<br>' : ''}
                                ${features.kubernetes_native ? '<i class="fas fa-check text-success me-1"></i> Kubernetes-native<br>' : ''}
                                ${features.supports_autoscaling ? '<i class="fas fa-check text-success me-1"></i> Auto-scaling<br>' : ''}
                                ${features.marketplace_model ? '<i class="fas fa-check text-success me-1"></i> Marketplace pricing<br>' : ''}
                            </div>
                        </div>
                    ` : ''}

                    <div class="mt-auto pt-3 border-top">
                        <div class="d-flex gap-2">
                            <button class="btn btn-sm btn-primary flex-fill" onclick="showProviderDetails(${provider.id})">
                                <i class="fas fa-info-circle me-1"></i> View Details
                            </button>
                            ${provider.website_url ? `
                                <a href="${provider.website_url}" target="_blank" class="btn btn-sm btn-outline-secondary">
                                    <i class="fas fa-external-link-alt"></i>
                                </a>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `}).join('');
}

/**
 * Get status badge HTML
 */
function getProviderStatusBadge(status) {
    const statusMap = {
        'ACTIVE': 'success',
        'INACTIVE': 'secondary',
        'TESTING': 'warning'
    };

    const color = statusMap[status] || 'secondary';
    return `<span class="badge bg-${color}">${status || 'UNKNOWN'}</span>`;
}

/**
 * Show add provider modal
 */
function showAddProviderModal() {
    currentEditingProvider = null;

    // Reset form
    document.getElementById('provider-form').reset();
    document.getElementById('provider-id').value = '';
    document.getElementById('provider-modal-title-text').textContent = 'Add Cloud Provider';

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('providerModal'));
    modal.show();
}

/**
 * Edit an existing provider
 */
async function editProvider(providerId) {
    const provider = cloudProviders.find(p => p.id === providerId);

    if (!provider) {
        showNotification('Provider not found', 'danger');
        return;
    }

    currentEditingProvider = provider;

    // Populate form
    document.getElementById('provider-id').value = provider.id;
    document.getElementById('provider-name').value = provider.name || '';
    document.getElementById('provider-display-name').value = provider.display_name || '';
    document.getElementById('provider-description').value = provider.description || '';
    document.getElementById('provider-website').value = provider.website || '';
    document.getElementById('provider-status').value = provider.status || 'INACTIVE';

    // Populate GPUs
    if (provider.available_gpus && provider.available_gpus.length > 0) {
        document.getElementById('available-gpus').value = provider.available_gpus.join('\n');
    } else {
        document.getElementById('available-gpus').value = '';
    }

    // Populate features
    if (provider.features) {
        document.getElementById('supports-autoscaling').checked = provider.features.supports_autoscaling || false;
        document.getElementById('supports-spot-instances').checked = provider.features.supports_spot_instances || false;
        document.getElementById('supports-custom-images').checked = provider.features.supports_custom_images || false;
        document.getElementById('supports-persistent-storage').checked = provider.features.supports_persistent_storage || false;
    }

    // Update modal title
    document.getElementById('provider-modal-title-text').textContent = 'Edit Cloud Provider';

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('providerModal'));
    modal.show();
}

/**
 * Save provider (create or update)
 */
async function saveProvider() {
    const form = document.getElementById('provider-form');

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const providerId = document.getElementById('provider-id').value;
    const isEdit = !!providerId;

    // Parse available GPUs (one per line)
    const gpusText = document.getElementById('available-gpus').value;
    const availableGpus = gpusText
        .split('\n')
        .map(gpu => gpu.trim())
        .filter(gpu => gpu.length > 0);

    const providerData = {
        name: document.getElementById('provider-name').value.trim(),
        display_name: document.getElementById('provider-display-name').value.trim(),
        description: document.getElementById('provider-description').value.trim(),
        website: document.getElementById('provider-website').value.trim(),
        status: document.getElementById('provider-status').value,
        available_gpus: availableGpus,
        features: {
            supports_autoscaling: document.getElementById('supports-autoscaling').checked,
            supports_spot_instances: document.getElementById('supports-spot-instances').checked,
            supports_custom_images: document.getElementById('supports-custom-images').checked,
            supports_persistent_storage: document.getElementById('supports-persistent-storage').checked
        }
    };

    try {
        const url = isEdit
            ? `${API_BASE_URL}/api/providers/${providerId}`
            : `${API_BASE_URL}/api/providers`;

        const method = isEdit ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(providerData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        showNotification(
            isEdit ? 'Provider updated successfully' : 'Provider created successfully',
            'success'
        );

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('providerModal'));
        modal.hide();

        // Reload providers
        await loadCloudProviders();

    } catch (error) {
        console.error('Error saving provider:', error);
        showNotification('Error saving provider: ' + error.message, 'danger');
    }
}

/**
 * Delete a provider
 */
async function deleteProvider(providerId) {
    const provider = cloudProviders.find(p => p.id === providerId);

    if (!provider) {
        showNotification('Provider not found', 'danger');
        return;
    }

    if (!confirm(`Are you sure you want to delete "${provider.display_name}"?\n\nThis action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/providers/${providerId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        showNotification('Provider deleted successfully', 'success');

        // Reload providers
        await loadCloudProviders();

    } catch (error) {
        console.error('Error deleting provider:', error);
        showNotification('Error deleting provider: ' + error.message, 'danger');
    }
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
    toast.show();

    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

/**
 * Show detailed provider information modal
 */
async function showProviderDetails(providerId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/providers/${providerId}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const provider = await response.json();
        const metadata = provider.metadata || {};
        const pricing = metadata.pricing || {};
        const features = metadata.features || {};
        const best_for = metadata.best_for || [];
        const pros = metadata.pros || [];
        const cons = metadata.cons || [];
        const use_cases = metadata.use_cases || [];
        const regions = metadata.regions || [];

        const modalHtml = `
            <div class="modal fade" id="providerDetailsModal" tabindex="-1">
                <div class="modal-dialog modal-lg modal-dialog-scrollable">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <div>
                                <h5 class="modal-title">${provider.display_name}</h5>
                                <small>${provider.status === 'active' ?
                                    '<span class="badge bg-success">Active & Ready</span>' :
                                    '<span class="badge bg-secondary">Coming Soon</span>'}</small>
                            </div>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-4">
                                <p class="lead">${provider.description || ''}</p>
                                ${metadata.long_description ? `<p class="text-muted">${metadata.long_description}</p>` : ''}
                            </div>

                            ${pricing.min_price_per_hour ? `
                                <div class="card mb-3 border-primary">
                                    <div class="card-body">
                                        <h6 class="card-title">
                                            <i class="fas fa-dollar-sign text-primary me-2"></i>Pricing
                                        </h6>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="text-center p-3 bg-light rounded">
                                                    <div class="h4 text-primary mb-0">$${pricing.min_price_per_hour}/hr</div>
                                                    <small class="text-muted">Starting from</small>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <div class="text-center p-3 bg-light rounded">
                                                    <div class="h4 text-primary mb-0">$${pricing.max_price_per_hour}/hr</div>
                                                    <small class="text-muted">Up to</small>
                                                </div>
                                            </div>
                                        </div>
                                        ${pricing.pricing_details ? `<p class="mt-3 mb-0 small text-muted">${pricing.pricing_details}</p>` : ''}
                                    </div>
                                </div>
                            ` : ''}

                            <div class="card mb-3">
                                <div class="card-body">
                                    <h6 class="card-title">
                                        <i class="fas fa-microchip text-info me-2"></i>Available GPUs
                                    </h6>
                                    <div class="d-flex flex-wrap gap-2">
                                        ${provider.available_gpus && provider.available_gpus.length > 0
                                            ? provider.available_gpus.map(gpu =>
                                                `<span class="badge bg-secondary">${gpu}</span>`
                                            ).join('')
                                            : '<span class="text-muted">No GPUs listed</span>'}
                                    </div>
                                </div>
                            </div>

                            ${regions.length > 0 ? `
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h6 class="card-title">
                                            <i class="fas fa-globe text-success me-2"></i>Regions
                                        </h6>
                                        <div class="d-flex flex-wrap gap-2">
                                            ${regions.map(region => `<span class="badge bg-success-subtle text-success">${region}</span>`).join('')}
                                        </div>
                                        ${metadata.global_regions ? `<small class="text-muted mt-2 d-block">${metadata.global_regions}</small>` : ''}
                                    </div>
                                </div>
                            ` : ''}

                            ${Object.keys(features).length > 0 ? `
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h6 class="card-title">
                                            <i class="fas fa-star text-warning me-2"></i>Features
                                        </h6>
                                        <div class="row">
                                            <div class="col-md-6">
                                                ${features.per_second_billing ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Per-second billing</div>' : ''}
                                                ${features.serverless_inference ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Serverless inference</div>' : ''}
                                                ${features.kubernetes_native ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Kubernetes-native</div>' : ''}
                                                ${features.supports_autoscaling ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Auto-scaling</div>' : ''}
                                                ${features.marketplace_model ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Marketplace pricing</div>' : ''}
                                                ${features.bare_metal_performance ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Bare metal performance</div>' : ''}
                                            </div>
                                            <div class="col-md-6">
                                                ${features.supports_spot_instances ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Spot instances</div>' : ''}
                                                ${features.supports_custom_images ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Custom images</div>' : ''}
                                                ${features.supports_persistent_storage ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Persistent storage</div>' : ''}
                                                ${features.openai_compatible_api ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>OpenAI-compatible API</div>' : ''}
                                                ${features.lambda_stack ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>Lambda Stack</div>' : ''}
                                                ${features.one_click_clusters ? '<div class="mb-2"><i class="fas fa-check text-success me-2"></i>1-Click Clusters</div>' : ''}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ` : ''}

                            ${best_for.length > 0 ? `
                                <div class="card mb-3 bg-success-subtle">
                                    <div class="card-body">
                                        <h6 class="card-title text-success">
                                            <i class="fas fa-bullseye me-2"></i>Best For
                                        </h6>
                                        <ul class="mb-0">
                                            ${best_for.map(item => `<li>${item}</li>`).join('')}
                                        </ul>
                                    </div>
                                </div>
                            ` : ''}

                            <div class="row">
                                ${pros.length > 0 ? `
                                    <div class="col-md-6">
                                        <div class="card mb-3 border-success">
                                            <div class="card-body">
                                                <h6 class="card-title text-success">
                                                    <i class="fas fa-thumbs-up me-2"></i>Pros
                                                </h6>
                                                <ul class="small mb-0">
                                                    ${pros.map(pro => `<li>${pro}</li>`).join('')}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                ` : ''}

                                ${cons.length > 0 ? `
                                    <div class="col-md-6">
                                        <div class="card mb-3 border-warning">
                                            <div class="card-body">
                                                <h6 class="card-title text-warning">
                                                    <i class="fas fa-exclamation-triangle me-2"></i>Cons
                                                </h6>
                                                <ul class="small mb-0">
                                                    ${cons.map(con => `<li>${con}</li>`).join('')}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                ` : ''}
                            </div>

                            ${use_cases.length > 0 ? `
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h6 class="card-title">
                                            <i class="fas fa-briefcase me-2"></i>Common Use Cases
                                        </h6>
                                        <div class="d-flex flex-wrap gap-2">
                                            ${use_cases.map(useCase => `<span class="badge bg-info">${useCase}</span>`).join('')}
                                        </div>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                        <div class="modal-footer">
                            ${provider.documentation_url ? `
                                <a href="${provider.documentation_url}" target="_blank" class="btn btn-outline-secondary">
                                    <i class="fas fa-book me-1"></i> Documentation
                                </a>
                            ` : ''}
                            ${provider.website_url ? `
                                <a href="${provider.website_url}" target="_blank" class="btn btn-primary">
                                    <i class="fas fa-external-link-alt me-1"></i> Visit Website
                                </a>
                            ` : ''}
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove any existing modal
        const existingModal = document.getElementById('providerDetailsModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add new modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('providerDetailsModal'));
        modal.show();

        // Clean up modal after it's hidden
        document.getElementById('providerDetailsModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });

    } catch (error) {
        console.error('Error loading provider details:', error);
        showNotification('Failed to load provider details', 'danger');
    }
}
