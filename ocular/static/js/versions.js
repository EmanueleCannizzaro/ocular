/**
 * Model Version Management JavaScript
 *
 * Handles version history, comparison, rollback, and A/B testing UI.
 */

// Global state
let currentModelId = null;
let versions = [];
let selectedVersionsForComparison = [];

/**
 * Show version history modal for a model
 */
async function showVersionHistory(modelId) {
    currentModelId = modelId;
    selectedVersionsForComparison = [];

    try {
        // Fetch versions for this model
        const response = await fetch(`/api/models/${modelId}/versions`);
        if (!response.ok) throw new Error('Failed to fetch versions');

        versions = await response.json();

        // Fetch model details
        const modelResponse = await fetch(`/api/models/${modelId}`);
        const model = await modelResponse.json();

        // Update modal content
        document.getElementById('versionHistoryModelName').textContent = model.display_name || model.name;

        // Render version list
        renderVersionList(versions);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('versionHistoryModal'));
        modal.show();

    } catch (error) {
        console.error('Error loading version history:', error);
        alert('Failed to load version history: ' + error.message);
    }
}

/**
 * Render version list in the modal
 */
function renderVersionList(versions) {
    const container = document.getElementById('versionsList');

    if (!versions || versions.length === 0) {
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No versions found for this model. Create your first version to get started.
            </div>
        `;
        return;
    }

    let html = '';

    versions.forEach(version => {
        const isActive = version.is_active;
        const statusBadge = getStatusBadge(version.status);
        const activeBadge = isActive ? '<span class="badge bg-success ms-2">Active</span>' : '';

        html += `
            <div class="card mb-3 ${isActive ? 'border-success' : ''}">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <div class="d-flex align-items-center">
                                <input type="checkbox" class="form-check-input me-3"
                                       onchange="toggleVersionSelection(${version.id})"
                                       ${selectedVersionsForComparison.includes(version.id) ? 'checked' : ''}>
                                <div>
                                    <h5 class="mb-1">
                                        ${version.version_number}
                                        ${version.version_tag ? `<span class="badge bg-secondary">${version.version_tag}</span>` : ''}
                                        ${activeBadge}
                                    </h5>
                                    <p class="text-muted mb-0 small">
                                        ${version.version_name || 'No name'}
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            ${statusBadge}
                            ${version.ab_test_enabled ? '<span class="badge bg-info ms-2">A/B Test</span>' : ''}
                            <div class="small text-muted mt-1">
                                <i class="far fa-clock"></i> ${formatDate(version.created_at)}
                            </div>
                        </div>
                        <div class="col-md-3 text-end">
                            <div class="btn-group">
                                ${!isActive ? `
                                    <button class="btn btn-sm btn-success" onclick="activateVersion(${version.id})" title="Activate">
                                        <i class="fas fa-check"></i>
                                    </button>
                                ` : ''}
                                <button class="btn btn-sm btn-info" onclick="viewVersionDetails(${version.id})" title="View Details">
                                    <i class="fas fa-eye"></i>
                                </button>
                                <button class="btn btn-sm btn-warning" onclick="editVersion(${version.id})" title="Edit">
                                    <i class="fas fa-edit"></i>
                                </button>
                                ${version.status !== 'deprecated' ? `
                                    <button class="btn btn-sm btn-danger" onclick="deprecateVersion(${version.id})" title="Deprecate">
                                        <i class="fas fa-ban"></i>
                                    </button>
                                ` : ''}
                            </div>
                        </div>
                    </div>

                    ${version.changelog ? `
                        <div class="mt-2">
                            <small class="text-muted">
                                <strong>Changelog:</strong> ${version.changelog}
                            </small>
                        </div>
                    ` : ''}

                    ${version.avg_tokens_per_second || version.quality_score ? `
                        <div class="mt-2">
                            <small class="text-muted">
                                ${version.avg_tokens_per_second ? `
                                    <i class="fas fa-tachometer-alt"></i> ${version.avg_tokens_per_second.toFixed(2)} tok/s
                                ` : ''}
                                ${version.quality_score ? `
                                    <i class="fas fa-star ms-3"></i> Quality: ${version.quality_score.toFixed(1)}/100
                                ` : ''}
                            </small>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    // Update comparison button state
    updateComparisonButton();
}

/**
 * Toggle version selection for comparison
 */
function toggleVersionSelection(versionId) {
    const index = selectedVersionsForComparison.indexOf(versionId);
    if (index > -1) {
        selectedVersionsForComparison.splice(index, 1);
    } else {
        if (selectedVersionsForComparison.length >= 5) {
            alert('You can compare maximum 5 versions at once');
            return;
        }
        selectedVersionsForComparison.push(versionId);
    }
    updateComparisonButton();
}

/**
 * Update comparison button state
 */
function updateComparisonButton() {
    const button = document.getElementById('compareVersionsBtn');
    if (button) {
        button.disabled = selectedVersionsForComparison.length < 2;
        button.textContent = `Compare Selected (${selectedVersionsForComparison.length})`;
    }
}

/**
 * Compare selected versions
 */
async function compareSelectedVersions() {
    if (selectedVersionsForComparison.length < 2) {
        alert('Please select at least 2 versions to compare');
        return;
    }

    try {
        const response = await fetch('/api/versions/compare', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(selectedVersionsForComparison)
        });

        if (!response.ok) throw new Error('Failed to compare versions');

        const comparisonData = await response.json();
        showComparisonModal(comparisonData);

    } catch (error) {
        console.error('Error comparing versions:', error);
        alert('Failed to compare versions: ' + error.message);
    }
}

/**
 * Show version comparison modal
 */
function showComparisonModal(comparisonData) {
    const tbody = document.getElementById('versionComparisonBody');

    const attributes = [
        { key: 'version_number', label: 'Version' },
        { key: 'version_tag', label: 'Tag' },
        { key: 'status', label: 'Status' },
        { key: 'is_active', label: 'Active' },
        { key: 'avg_tokens_per_second', label: 'Tokens/Sec', format: (v) => v ? v.toFixed(2) : 'N/A' },
        { key: 'avg_inference_time_ms', label: 'Inference Time (ms)', format: (v) => v ? v.toFixed(1) : 'N/A' },
        { key: 'quality_score', label: 'Quality Score', format: (v) => v ? v.toFixed(1) + '/100' : 'N/A' },
        { key: 'deployment_count', label: 'Deployments' },
        { key: 'total_requests', label: 'Total Requests' },
        { key: 'success_rate', label: 'Success Rate', format: (v) => v + '%' },
        { key: 'created_at', label: 'Created', format: (v) => formatDate(v) },
    ];

    let html = '';
    attributes.forEach(attr => {
        html += '<tr><td class="fw-bold">' + attr.label + '</td>';
        comparisonData.forEach(version => {
            let value = version[attr.key];
            if (attr.format) {
                value = attr.format(value);
            } else if (typeof value === 'boolean') {
                value = value ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-danger"></i>';
            } else if (!value) {
                value = '-';
            }
            html += '<td>' + value + '</td>';
        });
        html += '</tr>';
    });

    tbody.innerHTML = html;

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('versionComparisonModal'));
    modal.show();
}

/**
 * Activate a version
 */
async function activateVersion(versionId) {
    if (!confirm('Are you sure you want to activate this version? This will deactivate all other versions.')) {
        return;
    }

    try {
        const response = await fetch(`/api/versions/${versionId}/activate`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to activate version');

        alert('Version activated successfully');
        // Reload version list
        showVersionHistory(currentModelId);

    } catch (error) {
        console.error('Error activating version:', error);
        alert('Failed to activate version: ' + error.message);
    }
}

/**
 * Rollback to a version
 */
async function rollbackToVersion(modelId, versionId) {
    if (!confirm('Are you sure you want to rollback to this version? This will activate this version and deactivate all others.')) {
        return;
    }

    try {
        const response = await fetch(`/api/models/${modelId}/rollback/${versionId}`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Failed to rollback');

        alert('Rollback successful');
        showVersionHistory(modelId);

    } catch (error) {
        console.error('Error during rollback:', error);
        alert('Failed to rollback: ' + error.message);
    }
}

/**
 * Deprecate a version
 */
async function deprecateVersion(versionId) {
    const reason = prompt('Enter reason for deprecation (optional):');
    if (reason === null) return; // User cancelled

    try {
        const url = `/api/versions/${versionId}/deprecate${reason ? '?reason=' + encodeURIComponent(reason) : ''}`;
        const response = await fetch(url, {method: 'POST'});

        if (!response.ok) throw new Error('Failed to deprecate version');

        alert('Version deprecated successfully');
        showVersionHistory(currentModelId);

    } catch (error) {
        console.error('Error deprecating version:', error);
        alert('Failed to deprecate version: ' + error.message);
    }
}

/**
 * View version details
 */
async function viewVersionDetails(versionId) {
    try {
        const response = await fetch(`/api/versions/${versionId}`);
        if (!response.ok) throw new Error('Failed to fetch version details');

        const version = await response.json();

        // Build details HTML
        let html = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Basic Information</h6>
                    <dl class="row">
                        <dt class="col-sm-4">Version:</dt>
                        <dd class="col-sm-8">${version.version_number}</dd>

                        <dt class="col-sm-4">Name:</dt>
                        <dd class="col-sm-8">${version.version_name || 'N/A'}</dd>

                        <dt class="col-sm-4">Tag:</dt>
                        <dd class="col-sm-8">${version.version_tag || 'N/A'}</dd>

                        <dt class="col-sm-4">Status:</dt>
                        <dd class="col-sm-8">${getStatusBadge(version.status)}</dd>

                        <dt class="col-sm-4">Active:</dt>
                        <dd class="col-sm-8">${version.is_active ? '<i class="fas fa-check text-success"></i>' : '<i class="fas fa-times text-muted"></i>'}</dd>
                    </dl>
                </div>

                <div class="col-md-6">
                    <h6>Performance Metrics</h6>
                    <dl class="row">
                        <dt class="col-sm-6">Tokens/Second:</dt>
                        <dd class="col-sm-6">${version.avg_tokens_per_second ? version.avg_tokens_per_second.toFixed(2) : 'N/A'}</dd>

                        <dt class="col-sm-6">Inference Time:</dt>
                        <dd class="col-sm-6">${version.avg_inference_time_ms ? version.avg_inference_time_ms.toFixed(1) + ' ms' : 'N/A'}</dd>

                        <dt class="col-sm-6">Quality Score:</dt>
                        <dd class="col-sm-6">${version.quality_score ? version.quality_score.toFixed(1) + '/100' : 'N/A'}</dd>
                    </dl>
                </div>
            </div>

            ${version.changelog ? `
                <div class="mt-3">
                    <h6>Changelog</h6>
                    <p>${version.changelog}</p>
                </div>
            ` : ''}

            ${version.docker_image ? `
                <div class="mt-3">
                    <h6>Docker Image</h6>
                    <code>${version.docker_image}</code>
                </div>
            ` : ''}

            <div class="mt-3">
                <h6>Timestamps</h6>
                <small class="text-muted">
                    Created: ${formatDate(version.created_at)}<br>
                    ${version.deployed_at ? `Deployed: ${formatDate(version.deployed_at)}<br>` : ''}
                    ${version.deprecated_at ? `Deprecated: ${formatDate(version.deprecated_at)}` : ''}
                </small>
            </div>
        `;

        document.getElementById('versionDetailsContent').innerHTML = html;

        const modal = new bootstrap.Modal(document.getElementById('versionDetailsModal'));
        modal.show();

    } catch (error) {
        console.error('Error loading version details:', error);
        alert('Failed to load version details: ' + error.message);
    }
}

/**
 * Create new version
 */
async function createNewVersion() {
    const form = document.getElementById('newVersionForm');
    const formData = new FormData(form);

    const data = {
        model_id: currentModelId,
        version_number: formData.get('version_number'),
        version_name: formData.get('version_name'),
        version_tag: formData.get('version_tag'),
        changelog: formData.get('changelog'),
        docker_image: formData.get('docker_image'),
    };

    try {
        const response = await fetch('/api/versions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to create version');

        alert('Version created successfully');

        // Close modal and refresh list
        bootstrap.Modal.getInstance(document.getElementById('newVersionModal')).hide();
        form.reset();
        showVersionHistory(currentModelId);

    } catch (error) {
        console.error('Error creating version:', error);
        alert('Failed to create version: ' + error.message);
    }
}

/**
 * Show new version modal
 */
function showNewVersionModal() {
    const modal = new bootstrap.Modal(document.getElementById('newVersionModal'));
    modal.show();
}

/**
 * Setup A/B test
 */
async function setupABTest() {
    if (selectedVersionsForComparison.length !== 2) {
        alert('Please select exactly 2 versions for A/B testing');
        return;
    }

    const trafficSplit = prompt('Enter traffic percentage for version B (0-100):', '50');
    if (trafficSplit === null) return;

    const durationDays = prompt('Enter test duration in days:', '7');
    if (durationDays === null) return;

    try {
        const response = await fetch('/api/versions/ab-test/setup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                version_id_a: selectedVersionsForComparison[0],
                version_id_b: selectedVersionsForComparison[1],
                traffic_split: parseFloat(trafficSplit),
                duration_days: parseInt(durationDays)
            })
        });

        if (!response.ok) throw new Error('Failed to setup A/B test');

        alert('A/B test configured successfully');
        showVersionHistory(currentModelId);

    } catch (error) {
        console.error('Error setting up A/B test:', error);
        alert('Failed to setup A/B test: ' + error.message);
    }
}

/**
 * Utility functions
 */
function getStatusBadge(status) {
    const badges = {
        'draft': '<span class="badge bg-secondary">Draft</span>',
        'active': '<span class="badge bg-success">Active</span>',
        'deprecated': '<span class="badge bg-warning">Deprecated</span>',
        'archived': '<span class="badge bg-dark">Archived</span>',
    };
    return badges[status] || '<span class="badge bg-secondary">' + status + '</span>';
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function editVersion(versionId) {
    alert('Edit functionality coming soon');
}
