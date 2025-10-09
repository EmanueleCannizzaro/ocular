/**
 * AI Models Management JavaScript
 * Handles CRUD operations for AI models with HuggingFace integration
 */

const API_BASE_URL = window.location.origin;
let availableModels = [];
let filteredModels = [];
let currentEditingModel = null;
let selectedModelIds = new Set();

/**
 * Initialize the page
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Models management page loaded');
    setupHuggingFacePreviewModal();
    loadModels();
});

/**
 * Load all models from the API
 */
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        availableModels = data.models || [];
        filteredModels = [...availableModels];

        console.log('Loaded models:', availableModels);

        // Fetch HuggingFace stats for models that have versions
        fetchHuggingFaceStats();

        applyFilters();

    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Error loading models: ' + error.message, 'danger');

        // Show empty state
        document.getElementById('models-tbody').innerHTML = `
            <tr>
                <td colspan="9" class="text-center text-muted py-5">
                    <i class="fas fa-info-circle fa-2x mb-3"></i>
                    <p>No models available. Click "Add Model" to get started.</p>
                </td>
            </tr>
        `;
    }
}

/**
 * Fetch HuggingFace stats for all models with versions
 */
async function fetchHuggingFaceStats() {
    for (const model of availableModels) {
        if (model.version && !model.metadata?.huggingface_stats) {
            try {
                const hfResponse = await fetch(`https://huggingface.co/api/models/${model.version}`);
                if (hfResponse.ok) {
                    const hfData = await hfResponse.json();

                    // Store stats in metadata
                    if (!model.metadata) model.metadata = {};
                    model.metadata.huggingface_stats = {
                        downloads: hfData.downloads,
                        likes: hfData.likes,
                        created_at: hfData.created_at,
                        last_modified: hfData.last_modified
                    };

                    // Re-render table to show updated stats
                    renderModelsTable();
                }
            } catch (error) {
                console.warn(`Could not fetch HF stats for ${model.name}:`, error);
            }
        }
    }
}

/**
 * Apply filters to models table
 */
function applyFilters() {
    const statusFilter = document.getElementById('filter-status').value;
    const opensourceFilter = document.getElementById('filter-opensource').value;
    const cloudStorageFilter = document.getElementById('filter-cloud-storage').value;
    const vllmFilter = document.getElementById('filter-vllm').value;
    const vramFilter = document.getElementById('filter-vram').value;
    const providerFilter = document.getElementById('filter-provider').value;
    const searchFilter = document.getElementById('filter-search').value.toLowerCase();
    const minDownloads = parseInt(document.getElementById('filter-min-downloads').value) || 0;
    const minLikes = parseInt(document.getElementById('filter-min-likes').value) || 0;
    const tagsFilter = document.getElementById('filter-tags').value.toLowerCase();

    filteredModels = availableModels.filter(model => {
        // Status filter
        if (statusFilter) {
            const modelStatus = (model.status || '').toLowerCase();
            if (modelStatus !== statusFilter.toLowerCase()) {
                return false;
            }
        }

        // Open source filter
        if (opensourceFilter) {
            const licenseValue = (model.license || model.metadata?.huggingface_stats?.license || '').toString().toLowerCase();
            const isOpenSource = typeof model.metadata?.is_opensource === 'boolean'
                ? model.metadata.is_opensource
                : (licenseValue && !['', 'unknown', 'proprietary', 'llama2', 'facebooklicense', 'bigscience-bloom-rail-1.0', 'cc-by-nc-4.0'].includes(licenseValue));
            if (opensourceFilter === 'true' && !isOpenSource) return false;
            if (opensourceFilter === 'false' && isOpenSource) return false;
        }

        // Cloud storage filter
        if (cloudStorageFilter) {
            const hasCloudStorage = model.metadata?.s3_uploads && model.metadata.s3_uploads.length > 0;
            if (cloudStorageFilter === 'true' && !hasCloudStorage) return false;
            if (cloudStorageFilter === 'false' && hasCloudStorage) return false;
        }

        // vLLM support filter
        if (vllmFilter) {
            const supportsVLLM = (model.supports_vllm !== undefined
                ? model.supports_vllm
                : model.metadata?.supports_vllm) || false;
            if (vllmFilter === 'true' && !supportsVLLM) return false;
            if (vllmFilter === 'false' && supportsVLLM) return false;
        }

        // VRAM requirement filter
        if (vramFilter) {
            const vramGB = Number(model.min_vram_gb ?? model.metadata?.min_vram_gb ?? model.metadata?.vram_gb ?? 0);

            if (vramFilter === '0-10' && vramGB > 10) return false;
            if (vramFilter === '10-20' && (vramGB < 10 || vramGB > 20)) return false;
            if (vramFilter === '20-40' && (vramGB < 20 || vramGB > 40)) return false;
            if (vramFilter === '40-80' && (vramGB < 40 || vramGB > 80)) return false;
            if (vramFilter === '80+' && vramGB < 80) return false;
        }

        // Provider type filter
        if (providerFilter) {
            const providerType = (
                model.metadata?.provider_type ||
                model.metadata?.model_category ||
                model.metadata?.category ||
                model.metadata?.provider ||
                model.model_type ||
                model.default_deployment_type ||
                ''
            ).toString().toLowerCase();

            if (!providerType.includes(providerFilter.toLowerCase())) {
                return false;
            }
        }

        // HuggingFace downloads filter
        if (minDownloads > 0) {
            const downloads = model.downloads ?? model.metadata?.huggingface_stats?.downloads ?? 0;
            if (downloads < minDownloads) return false;
        }

        // HuggingFace likes filter
        if (minLikes > 0) {
            const likes = model.likes ?? model.metadata?.huggingface_stats?.likes ?? 0;
            if (likes < minLikes) return false;
        }

        // Tags filter (searches in HuggingFace tags)
        if (tagsFilter) {
            const tags = model.metadata?.huggingface_stats?.tags || [];
            const hasMatchingTag = tags.some(tag => tag.toLowerCase().includes(tagsFilter));
            if (!hasMatchingTag) return false;
        }

        // Search filter
        if (searchFilter) {
            const searchableText = [
                model.name,
                model.display_name,
                model.version,
                model.description
            ].join(' ').toLowerCase();

            if (!searchableText.includes(searchFilter)) {
                return false;
            }
        }

        return true;
    });

    renderModelsTable();
}

/**
 * Clear all filters
 */
function clearFilters() {
    document.getElementById('filter-status').value = '';
    document.getElementById('filter-opensource').value = '';
    document.getElementById('filter-cloud-storage').value = '';
    document.getElementById('filter-vllm').value = '';
    document.getElementById('filter-vram').value = '';
    document.getElementById('filter-provider').value = '';
    document.getElementById('filter-search').value = '';
    document.getElementById('filter-min-downloads').value = '';
    document.getElementById('filter-min-likes').value = '';
    document.getElementById('filter-tags').value = '';

    applyFilters();
}

/**
 * Render the models table
 */
function renderModelsTable() {
    const tbody = document.getElementById('models-tbody');
    const countBadge = document.getElementById('model-count');

    if (!tbody) {
        console.error('Models table body not found');
        return;
    }

    // Update count badge
    if (countBadge) {
        countBadge.textContent = filteredModels.length;
    }

    if (filteredModels.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="10" class="text-center text-muted py-5">
                    <i class="fas fa-info-circle fa-2x mb-3"></i>
                    <p>${availableModels.length === 0 ? 'No models available. Click "Add Model" to get started.' : 'No models match the current filters.'}</p>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = filteredModels.map(model => {
        // Check if model has cloud storage uploads
        const hasCloudStorage = model.metadata?.s3_uploads && model.metadata.s3_uploads.length > 0;
        const isSelected = selectedModelIds.has(model.id);

        // Generate HuggingFace URL
        const hfUrl = model.repo_url || (model.version ? `https://huggingface.co/${model.version}` : null);

        // Get HuggingFace stats from metadata
        const hfStats = model.metadata?.huggingface_stats || {};
        const downloadsRaw = model.downloads ?? hfStats.downloads;
        const likesRaw = model.likes ?? hfStats.likes;
        const lastModifiedRaw = model.last_modified || hfStats.last_modified || hfStats.created_at;
        const license = (model.license || hfStats.license || 'unknown').toString();
        const licenseLower = license.toLowerCase();
        const isOpenSource = typeof model.metadata?.is_opensource === 'boolean'
            ? model.metadata.is_opensource
            : (licenseLower && !['unknown', 'proprietary', 'llama2', 'facebooklicense', 'bigscience-bloom-rail-1.0', 'cc-by-nc-4.0'].includes(licenseLower));

        const downloads = Number.isFinite(downloadsRaw) ? Number(downloadsRaw).toLocaleString() : '—';
        const likes = Number.isFinite(likesRaw) ? Number(likesRaw).toLocaleString() : '—';
        const lastModified = lastModifiedRaw ? new Date(lastModifiedRaw).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }) : '—';

        const safeModelName = (model.display_name || model.name || '').replace(/"/g, '&quot;');

        return `
        <tr>
            <td>
                <input type="checkbox" class="form-check-input model-checkbox"
                       data-model-id="${model.id}"
                       ${isSelected ? 'checked' : ''}
                       onchange="toggleModelSelection(${model.id})">
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <i class="fas fa-cube me-2 text-primary"></i>
                    <div>
                        ${hfUrl ?
                            `<strong><a href="#" role="button" class="text-decoration-none model-preview-link" data-hf-url="${hfUrl}" data-model-name="${safeModelName}">
                                ${model.display_name || model.name}
                                <i class="fas fa-external-link-alt ms-1" style="font-size: 0.8em;"></i>
                            </a></strong>` :
                            `<strong>${model.display_name || model.name}</strong>`
                        }
                        <br>
                        <small class="text-muted">${model.name}</small>
                    </div>
                </div>
            </td>
            <td>
                <code class="small">${model.version || 'N/A'}</code>
            </td>
            <td>
                <span class="badge bg-info">${model.min_vram_gb || 0} GB</span>
                <br>
                <small class="text-muted">${model.recommended_gpu || 'Any'}</small>
            </td>
            <td class="small">
                <div class="mb-1">
                    <i class="fas fa-download text-primary"></i> ${downloads}
                </div>
                <div class="mb-1">
                    <i class="fas fa-heart text-danger"></i> ${likes}
                </div>
                <div class="text-muted" style="font-size: 0.75rem;">
                    <i class="fas fa-calendar"></i> ${lastModified}
                </div>
            </td>
            <td class="text-center">
                ${hasCloudStorage ?
                    `<span class="badge bg-success" title="${model.metadata.s3_uploads.length} file(s) uploaded">
                        <i class="fas fa-cloud-check"></i> Yes
                    </span>` :
                    '<span class="badge bg-secondary"><i class="fas fa-cloud"></i> No</span>'}
            </td>
            <td class="text-center">
                ${isOpenSource ?
                    '<span class="badge bg-success"><i class="fab fa-osi"></i> Open</span>' :
                    '<span class="badge bg-warning"><i class="fas fa-lock"></i> Restricted</span>'}
                <div class="small text-muted mt-1">${license.toUpperCase()}</div>
            </td>
            <td class="text-center">
                ${model.supports_vllm ?
                    '<span class="badge bg-success"><i class="fas fa-check"></i> Yes</span>' :
                    '<span class="badge bg-secondary"><i class="fas fa-times"></i> No</span>'}
            </td>
            <td>
                ${getStatusBadge(model.status)}
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-warning" onclick="showTestModal(${model.id})" title="Test Model">
                        <i class="fas fa-vial"></i>
                    </button>
                    <button class="btn btn-outline-secondary" onclick="showVersionHistory(${model.id})" title="Version History">
                        <i class="fas fa-code-branch"></i>
                    </button>
                    <button class="btn btn-outline-success" onclick="showUploadModal(${model.id})" title="Upload to Cloud">
                        <i class="fas fa-cloud-upload-alt"></i>
                    </button>
                    <button class="btn btn-outline-primary" onclick="editModel(${model.id})" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-outline-info" onclick="viewModelDetails(${model.id})" title="View Details & Graph">
                        <i class="fas fa-chart-line"></i>
                    </button>
                    <button class="btn btn-outline-danger" onclick="deleteModel(${model.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `}).join('');

    initializeModelPreviewLinks();
}

/**
 * Get status badge HTML
 */
function getStatusBadge(status) {
    const statusMap = {
        'available': { color: 'success', label: 'Available' },
        'beta': { color: 'warning', label: 'Beta' },
        'deprecated': { color: 'danger', label: 'Deprecated' }
    };

    const statusInfo = statusMap[status] || { color: 'secondary', label: status || 'Unknown' };
    return `<span class="badge bg-${statusInfo.color}">${statusInfo.label}</span>`;
}

/**
 * Show add model modal
 */
function showAddModelModal() {
    currentEditingModel = null;

    // Reset form
    document.getElementById('model-form').reset();
    document.getElementById('model-id').value = '';
    document.getElementById('model-modal-title-text').textContent = 'Add AI Model';
    document.getElementById('supports-vllm').checked = false;

    // Clear search results
    const resultsDiv = document.getElementById('hf-search-results');
    if (resultsDiv) {
        resultsDiv.innerHTML = '';
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('modelModal'));
    modal.show();
}

function setupHuggingFacePreviewModal() {
    const modalElement = document.getElementById('huggingfacePreviewModal');
    if (!modalElement) return;

    const iframe = document.getElementById('huggingface-preview-frame');
    const spinner = document.getElementById('huggingface-preview-spinner');

    modalElement.addEventListener('hidden.bs.modal', () => {
        if (iframe) {
            iframe.src = '';
        }
        if (spinner) {
            spinner.classList.add('d-none');
            if (spinner.dataset.timeoutId) {
                clearTimeout(Number(spinner.dataset.timeoutId));
                delete spinner.dataset.timeoutId;
            }
        }
    });

    if (iframe && spinner) {
        iframe.addEventListener('load', () => {
            spinner.classList.add('d-none');
        });
    }
}

function initializeModelPreviewLinks() {
    const links = document.querySelectorAll('.model-preview-link');
    links.forEach(link => {
        if (link.dataset.previewInit === 'true') return;
        link.dataset.previewInit = 'true';
        link.addEventListener('click', event => {
            event.preventDefault();
            const url = link.getAttribute('data-hf-url');
            const name = link.getAttribute('data-model-name');
            openModelPreviewModal(url, name);
        });
    });
}

function openModelPreviewModal(url, modelName) {
    if (!url) {
        showNotification('No Hugging Face URL available for this model.', 'warning');
        return;
    }

    const modalElement = document.getElementById('huggingfacePreviewModal');
    if (!modalElement) {
        window.open(url, '_blank', 'noopener');
        return;
    }

    const iframe = document.getElementById('huggingface-preview-frame');
    const fallbackLink = document.getElementById('huggingface-preview-fallback');
    const title = document.getElementById('huggingface-preview-title');
    const spinner = document.getElementById('huggingface-preview-spinner');

    const resolvedName = (modelName || '').replace(/&quot;/g, '"');
    if (title) {
        title.textContent = resolvedName || 'Model Preview';
    }
    if (fallbackLink) {
        fallbackLink.href = url;
    }
    if (spinner) {
        spinner.classList.remove('d-none');
        if (spinner.dataset.timeoutId) {
            clearTimeout(Number(spinner.dataset.timeoutId));
        }
        spinner.dataset.timeoutId = String(setTimeout(() => {
            spinner.classList.add('d-none');
        }, 5000));
    }
    if (iframe) {
        iframe.src = '';
        iframe.src = url;
    }

    const modal = bootstrap.Modal.getOrCreateInstance(modalElement);
    modal.show();
}

/**
 * Search HuggingFace Hub for models
 */
async function searchHuggingFace() {
    const query = document.getElementById('hf-search').value.trim();
    const resultsDiv = document.getElementById('hf-search-results');

    if (!query) {
        showNotification('Please enter a search term', 'warning');
        return;
    }

    // Show loading state
    resultsDiv.innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border spinner-border-sm text-primary"></div>
            <p class="mt-2 mb-0 text-muted">Searching HuggingFace Hub...</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/api/models/search-huggingface?q=${encodeURIComponent(query)}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const models = data.models || [];

        if (models.length === 0) {
            resultsDiv.innerHTML = `
                <div class="alert alert-info mb-0">
                    <i class="fas fa-info-circle me-2"></i>
                    No models found for "${query}"
                </div>
            `;
            return;
        }

        // Render search results
        resultsDiv.innerHTML = `
            <div class="list-group">
                ${models.map(model => `
                    <button type="button" class="list-group-item list-group-item-action"
                            onclick="selectHuggingFaceModel('${model.id}', '${model.name}')">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">${model.name}</h6>
                            <small class="text-muted">
                                <i class="fas fa-download me-1"></i>${model.downloads || 0}
                            </small>
                        </div>
                        <p class="mb-1 small">${model.description || 'No description available'}</p>
                        <small class="text-muted">
                            ${model.tags ? model.tags.slice(0, 3).map(tag =>
                                `<span class="badge bg-secondary me-1">${tag}</span>`
                            ).join('') : ''}
                        </small>
                    </button>
                `).join('')}
            </div>
        `;

    } catch (error) {
        console.error('Error searching HuggingFace:', error);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger mb-0">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Error searching HuggingFace Hub: ${error.message}
            </div>
        `;
    }
}

/**
 * Select a model from HuggingFace search results
 */
function selectHuggingFaceModel(modelId, modelName) {
    // Pre-fill form fields
    document.getElementById('model-version').value = modelId;
    document.getElementById('model-display-name').value = modelName;

    // Generate a sanitized name for internal use
    const sanitizedName = modelName.toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');

    document.getElementById('model-name').value = sanitizedName;

    // Clear search results
    document.getElementById('hf-search-results').innerHTML = '';
    document.getElementById('hf-search').value = '';

    showNotification('Model selected from HuggingFace Hub', 'success');
}

/**
 * Edit an existing model
 */
async function editModel(modelId) {
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        showNotification('Model not found', 'danger');
        return;
    }

    currentEditingModel = model;

    // Populate form
    document.getElementById('model-id').value = model.id;
    document.getElementById('model-name').value = model.name || '';
    document.getElementById('model-display-name').value = model.display_name || '';
    document.getElementById('model-version').value = model.version || '';
    document.getElementById('model-description').value = model.description || '';
    document.getElementById('docker-image').value = model.docker_image || '';
    document.getElementById('min-vram').value = model.min_vram_gb || '';
    document.getElementById('recommended-gpu').value = model.recommended_gpu || '';
    document.getElementById('model-status').value = model.status || 'INACTIVE';
    document.getElementById('supports-vllm').checked = model.supports_vllm || false;

    // Update modal title
    document.getElementById('model-modal-title-text').textContent = 'Edit AI Model';

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('modelModal'));
    modal.show();
}

/**
 * View model details with evolution graph and HuggingFace info
 */
async function viewModelDetails(modelId) {
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        showNotification('Model not found', 'danger');
        return;
    }

    // Generate chart data
    const { chartHtml, data: chartData } = generateModelEvolutionGraph(model);

    // Check cloud storage status
    const hasCloudStorage = model.metadata?.s3_uploads && model.metadata.s3_uploads.length > 0;
    const isOpenSource = model.metadata?.is_opensource || false;
    const hfUrl = model.repo_url || (model.version ? `https://huggingface.co/${model.version}` : null);

    // Fetch HuggingFace model information if model has a version
    let hfInfo = null;
    if (model.version) {
        try {
            const hfResponse = await fetch(`https://huggingface.co/api/models/${model.version}`);
            if (hfResponse.ok) {
                hfInfo = await hfResponse.json();
            }
        } catch (error) {
            console.warn('Could not fetch HuggingFace info:', error);
        }
    }

    // Create details modal content
    const detailsHtml = `
        <div class="modal fade" id="modelDetailsModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-cube me-2"></i>
                            ${model.display_name || model.name}
                            ${hfUrl ? `<a href="${hfUrl}" target="_blank" class="ms-2"><i class="fas fa-external-link-alt small"></i></a>` : ''}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Model Evolution Graph -->
                        ${chartHtml}

                        <!-- Model Details -->
                        <div class="card">
                            <div class="card-header">
                                <h6 class="mb-0">
                                    <i class="fas fa-info-circle me-2"></i>
                                    Model Information
                                </h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label text-muted small">Internal Name</label>
                                        <p class="mb-0"><code>${model.name}</code></p>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label text-muted small">Version/ID</label>
                                        <p class="mb-0"><code class="small">${model.version || 'N/A'}</code></p>
                                    </div>
                                    <div class="col-12 mb-3">
                                        <label class="form-label text-muted small">Description</label>
                                        <p class="mb-0">${model.description || 'No description available'}</p>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label text-muted small">Docker Image</label>
                                        <p class="mb-0"><code class="small">${model.docker_image || 'N/A'}</code></p>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label text-muted small">Status</label>
                                        <p class="mb-0">${getStatusBadge(model.status)}</p>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Min VRAM</label>
                                        <p class="mb-0"><span class="badge bg-info">${model.min_vram_gb || 0} GB</span></p>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Recommended GPU</label>
                                        <p class="mb-0">${model.recommended_gpu || 'Any'}</p>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">vLLM Support</label>
                                        <p class="mb-0">${model.supports_vllm ?
                                            '<span class="badge bg-success"><i class="fas fa-check"></i> Yes</span>' :
                                            '<span class="badge bg-secondary"><i class="fas fa-times"></i> No</span>'}</p>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Open Source</label>
                                        <p class="mb-0">${isOpenSource ?
                                            '<span class="badge bg-success"><i class="fab fa-osi"></i> Yes</span>' :
                                            '<span class="badge bg-warning"><i class="fas fa-lock"></i> Proprietary</span>'}</p>
                                    </div>
                                </div>

                                ${hasCloudStorage ? `
                                    <hr>
                                    <h6 class="mb-3">
                                        <i class="fas fa-cloud me-2"></i>
                                        Cloud Storage Uploads (${model.metadata.s3_uploads.length})
                                    </h6>
                                    <div class="list-group">
                                        ${model.metadata.s3_uploads.map(upload => `
                                            <div class="list-group-item">
                                                <div class="d-flex w-100 justify-content-between">
                                                    <h6 class="mb-1">${upload.filename}</h6>
                                                    <small class="text-muted">${(upload.size_bytes / 1024 / 1024).toFixed(2)} MB</small>
                                                </div>
                                                <p class="mb-1 small">
                                                    <a href="${upload.url}" target="_blank" class="text-decoration-none">
                                                        ${upload.url}
                                                        <i class="fas fa-external-link-alt ms-1"></i>
                                                    </a>
                                                </p>
                                                <small class="text-muted">
                                                    Uploaded: ${new Date(upload.uploaded_at).toLocaleString()} •
                                                    Provider: ${upload.provider} •
                                                    ${upload.public ? '<span class="badge bg-success">Public</span>' : '<span class="badge bg-secondary">Private</span>'}
                                                </small>
                                            </div>
                                        `).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        </div>

                        <!-- HuggingFace Information -->
                        ${hfInfo ? `
                        <div class="card mt-3">
                            <div class="card-header bg-warning bg-opacity-10">
                                <h6 class="mb-0">
                                    <i class="fas fa-robot me-2"></i>
                                    HuggingFace Model Information
                                </h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    ${hfInfo.downloads ? `
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Downloads</label>
                                        <p class="mb-0">
                                            <i class="fas fa-download text-primary me-1"></i>
                                            <strong>${hfInfo.downloads.toLocaleString()}</strong>
                                        </p>
                                    </div>
                                    ` : ''}
                                    ${hfInfo.likes ? `
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Likes</label>
                                        <p class="mb-0">
                                            <i class="fas fa-heart text-danger me-1"></i>
                                            <strong>${hfInfo.likes.toLocaleString()}</strong>
                                        </p>
                                    </div>
                                    ` : ''}
                                    ${hfInfo.created_at ? `
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Created</label>
                                        <p class="mb-0">${new Date(hfInfo.created_at).toLocaleDateString()}</p>
                                    </div>
                                    ` : ''}
                                    ${hfInfo.last_modified ? `
                                    <div class="col-md-3 mb-3">
                                        <label class="form-label text-muted small">Last Modified</label>
                                        <p class="mb-0">${new Date(hfInfo.last_modified).toLocaleDateString()}</p>
                                    </div>
                                    ` : ''}
                                </div>

                                ${hfInfo.tags && hfInfo.tags.length > 0 ? `
                                    <div class="mb-3">
                                        <label class="form-label text-muted small">Tags</label>
                                        <div>
                                            ${hfInfo.tags.slice(0, 10).map(tag =>
                                                `<span class="badge bg-secondary me-1 mb-1">${tag}</span>`
                                            ).join('')}
                                            ${hfInfo.tags.length > 10 ? `<span class="badge bg-light text-dark">+${hfInfo.tags.length - 10} more</span>` : ''}
                                        </div>
                                    </div>
                                ` : ''}

                                ${hfInfo.pipeline_tag ? `
                                    <div class="mb-3">
                                        <label class="form-label text-muted small">Pipeline Tag</label>
                                        <p class="mb-0">
                                            <span class="badge bg-primary">${hfInfo.pipeline_tag}</span>
                                        </p>
                                    </div>
                                ` : ''}

                                ${hfInfo.library_name ? `
                                    <div class="mb-3">
                                        <label class="form-label text-muted small">Library</label>
                                        <p class="mb-0">
                                            <span class="badge bg-info">${hfInfo.library_name}</span>
                                        </p>
                                    </div>
                                ` : ''}

                                ${hfInfo.author ? `
                                    <div class="mb-3">
                                        <label class="form-label text-muted small">Author</label>
                                        <p class="mb-0">
                                            <a href="https://huggingface.co/${hfInfo.author}" target="_blank" class="text-decoration-none">
                                                ${hfInfo.author}
                                                <i class="fas fa-external-link-alt ms-1 small"></i>
                                            </a>
                                        </p>
                                    </div>
                                ` : ''}

                                ${hfInfo.card_data?.license ? `
                                    <div class="mb-3">
                                        <label class="form-label text-muted small">License</label>
                                        <p class="mb-0">
                                            <span class="badge bg-success">${hfInfo.card_data.license}</span>
                                        </p>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-success" onclick="showUploadModal(${model.id}); bootstrap.Modal.getInstance(document.getElementById('modelDetailsModal')).hide();">
                            <i class="fas fa-cloud-upload-alt me-1"></i> Upload to Cloud
                        </button>
                        <button type="button" class="btn btn-primary" onclick="editModel(${model.id}); bootstrap.Modal.getInstance(document.getElementById('modelDetailsModal')).hide();">
                            <i class="fas fa-edit me-1"></i> Edit Model
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing details modal if present
    const existingModal = document.getElementById('modelDetailsModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add to body and show
    document.body.insertAdjacentHTML('beforeend', detailsHtml);
    const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
    modal.show();

    // Initialize chart after modal is shown
    document.getElementById('modelDetailsModal').addEventListener('shown.bs.modal', function() {
        initModelEvolutionChart(chartData);
    });

    // Clean up after modal is hidden
    document.getElementById('modelDetailsModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

/**
 * Save model (create or update)
 */
async function saveModel() {
    const form = document.getElementById('model-form');

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const modelId = document.getElementById('model-id').value;
    const isEdit = !!modelId;

    const modelData = {
        name: document.getElementById('model-name').value.trim(),
        display_name: document.getElementById('model-display-name').value.trim(),
        version: document.getElementById('model-version').value.trim(),
        description: document.getElementById('model-description').value.trim(),
        docker_image: document.getElementById('docker-image').value.trim(),
        min_vram_gb: parseFloat(document.getElementById('min-vram').value) || 0,
        recommended_gpu: document.getElementById('recommended-gpu').value.trim(),
        status: document.getElementById('model-status').value,
        supports_vllm: document.getElementById('supports-vllm').checked
    };

    try {
        const url = isEdit
            ? `${API_BASE_URL}/api/models/${modelId}`
            : `${API_BASE_URL}/api/models`;

        const method = isEdit ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(modelData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        showNotification(
            isEdit ? 'Model updated successfully' : 'Model created successfully',
            'success'
        );

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('modelModal'));
        modal.hide();

        // Reload models
        await loadModels();

    } catch (error) {
        console.error('Error saving model:', error);
        showNotification('Error saving model: ' + error.message, 'danger');
    }
}

/**
 * Delete a model
 */
async function deleteModel(modelId) {
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        showNotification('Model not found', 'danger');
        return;
    }

    if (!confirm(`Are you sure you want to delete "${model.display_name || model.name}"?\n\nThis action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        showNotification('Model deleted successfully', 'success');

        // Reload models
        await loadModels();

    } catch (error) {
        console.error('Error deleting model:', error);
        showNotification('Error deleting model: ' + error.message, 'danger');
    }
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    // Create toast element if it doesn't exist
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
 * Show upload to cloud storage modal
 */
async function showUploadModal(modelId) {
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        showNotification('Model not found', 'danger');
        return;
    }

    // Reset form
    document.getElementById('upload-form').reset();
    document.getElementById('upload-model-id').value = modelId;
    document.getElementById('upload-progress').style.display = 'none';

    // Load available buckets from configuration
    try {
        const response = await fetch(`${API_BASE_URL}/api/deployments/storage/buckets`);
        if (response.ok) {
            const data = await response.json();
            const bucketSelect = document.getElementById('upload-bucket');

            // Clear existing options
            bucketSelect.innerHTML = '';

            // Add bucket options
            data.buckets.forEach(bucket => {
                const option = document.createElement('option');
                option.value = bucket.name;
                option.textContent = bucket.display_name;
                option.dataset.provider = bucket.provider;
                if (bucket.default) {
                    option.selected = true;
                }
                bucketSelect.appendChild(option);
            });

            // Update provider based on selected bucket
            bucketSelect.addEventListener('change', function() {
                const selectedOption = this.options[this.selectedIndex];
                if (selectedOption.dataset.provider) {
                    document.getElementById('upload-provider').value = selectedOption.dataset.provider;
                }
            });

            // Set initial provider
            if (bucketSelect.options.length > 0) {
                const firstOption = bucketSelect.options[0];
                if (firstOption.dataset.provider) {
                    document.getElementById('upload-provider').value = firstOption.dataset.provider;
                }
            }
        }
    } catch (error) {
        console.warn('Could not load bucket configuration:', error);
        // Use default bucket input
    }

    // Show/hide file upload based on upload source
    const uploadSource = document.getElementById('upload-source');
    const fileUploadGroup = document.getElementById('file-upload-group');

    if (uploadSource) {
        uploadSource.addEventListener('change', function() {
            if (this.value === 'huggingface') {
                fileUploadGroup.style.display = 'none';
                document.getElementById('upload-file').removeAttribute('required');
            } else {
                fileUploadGroup.style.display = 'block';
                document.getElementById('upload-file').setAttribute('required', 'required');
            }
        });

        // Trigger initial update
        uploadSource.dispatchEvent(new Event('change'));
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
    modal.show();
}

/**
 * Upload model file to cloud storage
 */
async function uploadToCloudStorage() {
    const form = document.getElementById('upload-form');

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const modelId = document.getElementById('upload-model-id').value;
    const bucket = document.getElementById('upload-bucket').value || document.getElementById('upload-bucket-custom')?.value;
    const provider = document.getElementById('upload-provider').value;
    const makePublic = document.getElementById('upload-public').checked;
    const uploadSource = document.getElementById('upload-source')?.value || 'file';

    if (!bucket) {
        showNotification('Please select or enter a bucket name', 'warning');
        return;
    }

    // Show progress
    document.getElementById('upload-progress').style.display = 'block';
    const progressText = document.getElementById('upload-progress-text');

    try {
        let response;

        if (uploadSource === 'huggingface') {
            // Download from HuggingFace and upload
            progressText.textContent = 'Downloading model from HuggingFace...';

            const formData = new FormData();
            formData.append('bucket', bucket);
            formData.append('provider', provider);
            formData.append('make_public', makePublic);

            response = await fetch(`${API_BASE_URL}/api/deployments/models/${modelId}/upload-from-hf`, {
                method: 'POST',
                body: formData
            });
        } else {
            // Upload file directly
            const file = document.getElementById('upload-file').files[0];

            if (!file) {
                showNotification('Please select a file to upload', 'warning');
                document.getElementById('upload-progress').style.display = 'none';
                return;
            }

            progressText.textContent = `Uploading ${file.name}...`;

            const formData = new FormData();
            formData.append('file', file);
            formData.append('bucket', bucket);
            formData.append('provider', provider);
            formData.append('make_public', makePublic);

            response = await fetch(`${API_BASE_URL}/api/deployments/models/${modelId}/upload`, {
                method: 'POST',
                body: formData
            });
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (uploadSource === 'huggingface') {
            showNotification(
                `Model downloaded from HuggingFace and uploaded! ${result.upload.file_count} files, ${formatBytes(result.upload.total_size_bytes)}`,
                'success'
            );
        } else {
            showNotification('Model uploaded to cloud storage successfully!', 'success');
        }

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('uploadModal'));
        modal.hide();

        // Reload models to show updated cloud storage status
        await loadModels();

    } catch (error) {
        console.error('Error uploading model:', error);
        showNotification('Upload failed: ' + error.message, 'danger');
    } finally {
        document.getElementById('upload-progress').style.display = 'none';
    }
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Generate model evolution graph using Chart.js
 */
function generateModelEvolutionGraph(model) {
    // Generate sample data for model evolution
    // In a real implementation, this would come from deployment history
    const deploymentHistory = model.metadata?.deployment_history || [];

    // If no history, generate sample data
    const data = deploymentHistory.length > 0 ? deploymentHistory : [
        { date: '2024-01', downloads: 100, deployments: 5 },
        { date: '2024-02', downloads: 250, deployments: 12 },
        { date: '2024-03', downloads: 450, deployments: 18 },
        { date: '2024-04', downloads: 680, deployments: 25 },
        { date: '2024-05', downloads: 920, deployments: 32 },
        { date: '2024-06', downloads: 1200, deployments: 40 }
    ];

    const chartHtml = `
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="mb-0">
                    <i class="fas fa-chart-line me-2"></i>
                    Model Evolution & Usage
                </h6>
            </div>
            <div class="card-body">
                <canvas id="model-evolution-chart" height="100"></canvas>
            </div>
        </div>
    `;

    return { chartHtml, data };
}

/**
 * Initialize Chart.js for model evolution
 */
function initModelEvolutionChart(data) {
    const ctx = document.getElementById('model-evolution-chart');

    if (!ctx) {
        console.error('Chart canvas not found');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [
                {
                    label: 'Downloads',
                    data: data.map(d => d.downloads),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    yAxisID: 'y',
                },
                {
                    label: 'Deployments',
                    data: data.map(d => d.deployments),
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Downloads'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Deployments'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            }
        },
    });
}
