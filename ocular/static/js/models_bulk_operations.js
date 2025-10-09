/**
 * Bulk Operations for Models Management
 * Handles selection and bulk actions on multiple models
 */

/**
 * Toggle individual model selection
 */
function toggleModelSelection(modelId) {
    if (selectedModelIds.has(modelId)) {
        selectedModelIds.delete(modelId);
    } else {
        selectedModelIds.add(modelId);
    }
    updateBulkActionsBar();
    updateSelectAllCheckbox();
}

/**
 * Toggle select all checkbox
 */
function toggleSelectAll() {
    const selectAllCheckbox = document.getElementById('select-all-checkbox');

    if (selectAllCheckbox.checked) {
        // Select all visible models
        filteredModels.forEach(model => selectedModelIds.add(model.id));
    } else {
        // Deselect all
        selectedModelIds.clear();
    }

    updateBulkActionsBar();
    renderModelsTable();
}

/**
 * Select all models (button action)
 */
function selectAllModels() {
    filteredModels.forEach(model => selectedModelIds.add(model.id));
    updateBulkActionsBar();
    renderModelsTable();
    updateSelectAllCheckbox();
}

/**
 * Clear all selections
 */
function clearSelection() {
    selectedModelIds.clear();
    updateBulkActionsBar();
    renderModelsTable();
    updateSelectAllCheckbox();
}

/**
 * Update bulk actions bar visibility and count
 */
function updateBulkActionsBar() {
    const bulkActionsBar = document.getElementById('bulk-actions-bar');
    const selectedCount = document.getElementById('selected-count');

    if (selectedModelIds.size > 0) {
        bulkActionsBar.style.display = 'block';
        selectedCount.textContent = selectedModelIds.size;
    } else {
        bulkActionsBar.style.display = 'none';
    }
}

/**
 * Update select all checkbox state
 */
function updateSelectAllCheckbox() {
    const selectAllCheckbox = document.getElementById('select-all-checkbox');

    if (!selectAllCheckbox) return;

    if (selectedModelIds.size === 0) {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = false;
    } else if (selectedModelIds.size === filteredModels.length) {
        selectAllCheckbox.checked = true;
        selectAllCheckbox.indeterminate = false;
    } else {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = true;
    }
}

/**
 * Export selected models to JSON
 */
function exportSelectedModels() {
    if (selectedModelIds.size === 0) {
        showNotification('Please select at least one model to export', 'warning');
        return;
    }

    const selectedModels = availableModels.filter(m => selectedModelIds.has(m.id));

    // Create JSON blob
    const dataStr = JSON.stringify(selectedModels, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    // Create download link
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocular-models-export-${new Date().toISOString().split('T')[0]}.json`;

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification(`Exported ${selectedModelIds.size} model(s) successfully`, 'success');
}

/**
 * Bulk update status of selected models
 */
async function bulkUpdateStatus() {
    if (selectedModelIds.size === 0) {
        showNotification('Please select at least one model to update', 'warning');
        return;
    }

    // Show status selection modal
    const statusOptions = [
        { value: 'available', label: 'Available', color: 'success' },
        { value: 'beta', label: 'Beta', color: 'warning' },
        { value: 'deprecated', label: 'Deprecated', color: 'danger' }
    ];

    const modalHtml = `
        <div class="modal fade" id="bulkStatusModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-edit me-2"></i>
                            Update Status for ${selectedModelIds.size} Model(s)
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">New Status</label>
                            <select class="form-select" id="bulk-status-select">
                                ${statusOptions.map(opt => `
                                    <option value="${opt.value}">
                                        ${opt.label}
                                    </option>
                                `).join('')}
                            </select>
                        </div>
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            This will update the status for all ${selectedModelIds.size} selected model(s).
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="executeBulkStatusUpdate()">
                            <i class="fas fa-check me-1"></i> Update Status
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existingModal = document.getElementById('bulkStatusModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('bulkStatusModal'));
    modal.show();
}

/**
 * Execute bulk status update
 */
async function executeBulkStatusUpdate() {
    const newStatus = document.getElementById('bulk-status-select').value;

    const modal = bootstrap.Modal.getInstance(document.getElementById('bulkStatusModal'));
    modal.hide();

    let successCount = 0;
    let errorCount = 0;

    for (const modelId of selectedModelIds) {
        try {
            const model = availableModels.find(m => m.id === modelId);

            const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...model,
                    status: newStatus
                })
            });

            if (response.ok) {
                successCount++;
                // Update local model
                model.status = newStatus;
            } else {
                errorCount++;
            }
        } catch (error) {
            console.error(`Error updating model ${modelId}:`, error);
            errorCount++;
        }
    }

    // Clear selection and refresh
    clearSelection();
    renderModelsTable();

    if (errorCount === 0) {
        showNotification(`Successfully updated ${successCount} model(s)`, 'success');
    } else {
        showNotification(`Updated ${successCount} model(s), ${errorCount} failed`, 'warning');
    }
}

/**
 * Bulk upload selected models to cloud
 */
async function bulkUploadToCloud() {
    if (selectedModelIds.size === 0) {
        showNotification('Please select at least one model to upload', 'warning');
        return;
    }

    const modalHtml = `
        <div class="modal fade" id="bulkUploadModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-cloud-upload-alt me-2"></i>
                            Bulk Upload ${selectedModelIds.size} Model(s) from HuggingFace
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">S3 Bucket</label>
                            <select class="form-select" id="bulk-upload-bucket">
                                <option value="">Loading buckets...</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Storage Provider</label>
                            <select class="form-select" id="bulk-upload-provider" disabled>
                                <option value="runpod">RunPod Storage</option>
                                <option value="aws">AWS S3</option>
                                <option value="custom">Custom S3-Compatible</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="bulk-upload-public">
                                <label class="form-check-label" for="bulk-upload-public">
                                    Make publicly accessible
                                </label>
                            </div>
                        </div>
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <strong>Note:</strong> This will download models from HuggingFace and upload to cloud storage.
                            Only models with a HuggingFace version will be processed. This may take several minutes.
                        </div>
                        <div id="bulk-upload-progress" style="display: none;">
                            <div class="progress mb-2">
                                <div class="progress-bar progress-bar-striped progress-bar-animated"
                                     id="bulk-upload-progress-bar" role="progressbar" style="width: 0%"></div>
                            </div>
                            <small class="text-muted" id="bulk-upload-status">Preparing...</small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="executeBulkUpload()">
                            <i class="fas fa-cloud-upload-alt me-1"></i> Upload All
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existingModal = document.getElementById('bulkUploadModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Load buckets
    try {
        const response = await fetch(`${API_BASE_URL}/api/deployments/storage/buckets`);
        if (response.ok) {
            const data = await response.json();
            const bucketSelect = document.getElementById('bulk-upload-bucket');
            bucketSelect.innerHTML = '';

            data.buckets.forEach(bucket => {
                const option = document.createElement('option');
                option.value = bucket.name;
                option.textContent = bucket.display_name;
                option.dataset.provider = bucket.provider;
                if (bucket.default) option.selected = true;
                bucketSelect.appendChild(option);
            });

            // Update provider based on bucket
            bucketSelect.addEventListener('change', function() {
                const selectedOption = this.options[this.selectedIndex];
                if (selectedOption.dataset.provider) {
                    document.getElementById('bulk-upload-provider').value = selectedOption.dataset.provider;
                }
            });

            // Set initial provider
            if (bucketSelect.options.length > 0) {
                const firstOption = bucketSelect.options[0];
                if (firstOption.dataset.provider) {
                    document.getElementById('bulk-upload-provider').value = firstOption.dataset.provider;
                }
            }
        }
    } catch (error) {
        console.warn('Could not load buckets:', error);
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('bulkUploadModal'));
    modal.show();
}

/**
 * Execute bulk upload
 */
async function executeBulkUpload() {
    const bucket = document.getElementById('bulk-upload-bucket').value;
    const provider = document.getElementById('bulk-upload-provider').value;
    const makePublic = document.getElementById('bulk-upload-public').checked;

    if (!bucket) {
        showNotification('Please select a bucket', 'warning');
        return;
    }

    // Show progress
    document.getElementById('bulk-upload-progress').style.display = 'block';

    const selectedModels = availableModels.filter(m =>
        selectedModelIds.has(m.id) && m.version  // Only models with HF version
    );

    if (selectedModels.length === 0) {
        showNotification('No models with HuggingFace versions selected', 'warning');
        document.getElementById('bulk-upload-progress').style.display = 'none';
        return;
    }

    let successCount = 0;
    let errorCount = 0;
    const total = selectedModels.length;

    for (let i = 0; i < selectedModels.length; i++) {
        const model = selectedModels[i];

        // Update progress
        const progress = ((i + 1) / total) * 100;
        document.getElementById('bulk-upload-progress-bar').style.width = `${progress}%`;
        document.getElementById('bulk-upload-status').textContent =
            `Uploading ${i + 1} of ${total}: ${model.name}...`;

        try {
            const formData = new FormData();
            formData.append('bucket', bucket);
            formData.append('provider', provider);
            formData.append('make_public', makePublic);

            const response = await fetch(`${API_BASE_URL}/api/deployments/models/${model.id}/upload-from-hf`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                successCount++;
            } else {
                errorCount++;
            }
        } catch (error) {
            console.error(`Error uploading model ${model.id}:`, error);
            errorCount++;
        }
    }

    // Hide modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('bulkUploadModal'));
    modal.hide();

    // Clear selection and refresh
    clearSelection();
    await loadModels();

    if (errorCount === 0) {
        showNotification(`Successfully uploaded ${successCount} model(s) to cloud storage`, 'success');
    } else {
        showNotification(`Uploaded ${successCount} model(s), ${errorCount} failed`, 'warning');
    }
}

/**
 * Delete selected models
 */
async function deleteSelectedModels() {
    if (selectedModelIds.size === 0) {
        showNotification('Please select at least one model to delete', 'warning');
        return;
    }

    if (!confirm(`Are you sure you want to delete ${selectedModelIds.size} model(s)? This action cannot be undone.`)) {
        return;
    }

    let successCount = 0;
    let errorCount = 0;

    for (const modelId of selectedModelIds) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                successCount++;
            } else {
                errorCount++;
            }
        } catch (error) {
            console.error(`Error deleting model ${modelId}:`, error);
            errorCount++;
        }
    }

    // Clear selection and refresh
    clearSelection();
    await loadModels();

    if (errorCount === 0) {
        showNotification(`Successfully deleted ${successCount} model(s)`, 'success');
    } else {
        showNotification(`Deleted ${successCount} model(s), ${errorCount} failed`, 'warning');
    }
}

/**
 * Compare selected models
 */
function compareSelectedModels() {
    if (selectedModelIds.size < 2) {
        showNotification('Please select at least 2 models to compare', 'warning');
        return;
    }

    if (selectedModelIds.size > 5) {
        showNotification('Please select no more than 5 models to compare', 'warning');
        return;
    }

    const selectedModels = availableModels.filter(m => selectedModelIds.has(m.id));

    // Build comparison table
    const comparisonRows = [
        {
            label: 'Model Name',
            getValue: (m) => m.display_name || m.name
        },
        {
            label: 'Version',
            getValue: (m) => `<code class="small">${m.version || 'N/A'}</code>`
        },
        {
            label: 'Status',
            getValue: (m) => getStatusBadge(m.status)
        },
        {
            label: 'VRAM Requirement',
            getValue: (m) => `<span class="badge bg-info">${m.min_vram_gb || 0} GB</span>`
        },
        {
            label: 'Recommended GPU',
            getValue: (m) => m.recommended_gpu || 'Any'
        },
        {
            label: 'Docker Image',
            getValue: (m) => m.docker_image ? `<code class="small">${m.docker_image}</code>` : '—'
        },
        {
            label: 'vLLM Support',
            getValue: (m) => m.supports_vllm ?
                '<span class="badge bg-success"><i class="fas fa-check"></i> Yes</span>' :
                '<span class="badge bg-secondary"><i class="fas fa-times"></i> No</span>'
        },
        {
            label: 'Open Source',
            getValue: (m) => (m.metadata?.is_opensource || false) ?
                '<span class="badge bg-success"><i class="fab fa-osi"></i> Yes</span>' :
                '<span class="badge bg-warning"><i class="fas fa-lock"></i> Proprietary</span>'
        },
        {
            label: 'Cloud Storage',
            getValue: (m) => {
                const hasCloudStorage = m.metadata?.s3_uploads && m.metadata.s3_uploads.length > 0;
                return hasCloudStorage ?
                    `<span class="badge bg-success" title="${m.metadata.s3_uploads.length} file(s)"><i class="fas fa-cloud-check"></i> ${m.metadata.s3_uploads.length}</span>` :
                    '<span class="badge bg-secondary"><i class="fas fa-cloud"></i> No</span>';
            }
        },
        {
            label: 'HF Downloads',
            getValue: (m) => {
                const downloads = m.metadata?.huggingface_stats?.downloads;
                return downloads ? downloads.toLocaleString() : '—';
            }
        },
        {
            label: 'HF Likes',
            getValue: (m) => {
                const likes = m.metadata?.huggingface_stats?.likes;
                return likes ? likes.toLocaleString() : '—';
            }
        },
        {
            label: 'Created',
            getValue: (m) => {
                const createdAt = m.metadata?.huggingface_stats?.created_at;
                return createdAt ? new Date(createdAt).toLocaleDateString() : '—';
            }
        },
        {
            label: 'Description',
            getValue: (m) => `<small>${m.description || '—'}</small>`
        }
    ];

    const tableHtml = `
        <table class="table table-bordered table-sm">
            <thead class="table-light">
                <tr>
                    <th style="width: 150px;">Attribute</th>
                    ${selectedModels.map(m => `<th class="text-center">${m.display_name || m.name}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${comparisonRows.map(row => `
                    <tr>
                        <td class="fw-bold">${row.label}</td>
                        ${selectedModels.map(m => `<td class="text-center">${row.getValue(m)}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    const modalHtml = `
        <div class="modal fade" id="compareModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-balance-scale me-2"></i>
                            Model Comparison (${selectedModels.length} models)
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="table-responsive">
                            ${tableHtml}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="exportComparison()">
                            <i class="fas fa-download me-1"></i> Export Comparison
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existingModal = document.getElementById('compareModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('compareModal'));
    modal.show();
}

/**
 * Export model comparison
 */
function exportComparison() {
    const selectedModels = availableModels.filter(m => selectedModelIds.has(m.id));

    const comparisonData = {
        exported_at: new Date().toISOString(),
        models_count: selectedModels.length,
        models: selectedModels.map(m => ({
            name: m.name,
            display_name: m.display_name,
            version: m.version,
            status: m.status,
            vram_gb: m.min_vram_gb,
            recommended_gpu: m.recommended_gpu,
            docker_image: m.docker_image,
            supports_vllm: m.supports_vllm,
            is_opensource: m.metadata?.is_opensource || false,
            cloud_storage_uploads: m.metadata?.s3_uploads?.length || 0,
            huggingface_stats: m.metadata?.huggingface_stats || {},
            description: m.description
        }))
    };

    // Create JSON blob
    const dataStr = JSON.stringify(comparisonData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    // Create download link
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocular-model-comparison-${new Date().toISOString().split('T')[0]}.json`;

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('Comparison exported successfully', 'success');
}
