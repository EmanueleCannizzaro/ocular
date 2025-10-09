/**
 * User Settings page functionality for Ocular OCR.
 *
 * Handles user profile, API keys, and quota management.
 */

// ============================================================================
// State Management
// ============================================================================

let currentUser = null;
let apiKeys = [];

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadCurrentUser();
    await loadAPIKeys();
    await loadQuotaInfo();
    setupEventListeners();
});

function setupEventListeners() {
    // Profile update form
    const profileForm = document.getElementById('profileUpdateForm');
    if (profileForm) {
        profileForm.addEventListener('submit', handleProfileUpdate);
    }

    // API key creation form
    const apiKeyForm = document.getElementById('createAPIKeyForm');
    if (apiKeyForm) {
        apiKeyForm.addEventListener('submit', handleCreateAPIKey);
    }
}

// ============================================================================
// User Profile Management
// ============================================================================

async function loadCurrentUser() {
    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Redirect to login
                window.location.href = '/login';
                return;
            }
            throw new Error('Failed to load user profile');
        }

        currentUser = await response.json();
        displayUserProfile(currentUser);

    } catch (error) {
        console.error('Error loading user:', error);
        showAlert('error', 'Failed to load user profile');
    }
}

function displayUserProfile(user) {
    // Update profile form
    document.getElementById('email').value = user.email || '';
    document.getElementById('username').value = user.username || '';
    document.getElementById('firstName').value = user.first_name || '';
    document.getElementById('lastName').value = user.last_name || '';

    // Update profile display
    document.getElementById('userEmail').textContent = user.email;
    document.getElementById('userRole').textContent = user.role;
    document.getElementById('subscriptionTier').textContent = user.subscription_tier;

    // Update avatar
    const avatarImg = document.getElementById('userAvatar');
    if (avatarImg && user.avatar_url) {
        avatarImg.src = user.avatar_url;
    }
}

async function handleProfileUpdate(event) {
    event.preventDefault();

    const formData = {
        first_name: document.getElementById('firstName').value,
        last_name: document.getElementById('lastName').value,
        username: document.getElementById('username').value
    };

    try {
        const response = await fetch('/api/auth/me', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Failed to update profile');
        }

        const updatedUser = await response.json();
        currentUser = updatedUser;
        displayUserProfile(updatedUser);

        showAlert('success', 'Profile updated successfully');

    } catch (error) {
        console.error('Error updating profile:', error);
        showAlert('error', error.message || 'Failed to update profile');
    }
}

// ============================================================================
// Quota Management
// ============================================================================

async function loadQuotaInfo() {
    try {
        const response = await fetch('/api/auth/quota/check', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load quota information');
        }

        const quotaData = await response.json();
        displayQuotaInfo(quotaData);

    } catch (error) {
        console.error('Error loading quota:', error);
        showAlert('error', 'Failed to load quota information');
    }
}

function displayQuotaInfo(quotaData) {
    // Update quota display
    document.getElementById('quotaUsed').textContent = quotaData.used.toLocaleString();
    document.getElementById('quotaLimit').textContent = quotaData.quota.toLocaleString();
    document.getElementById('quotaRemaining').textContent = quotaData.remaining.toLocaleString();

    // Update progress bar
    const progressBar = document.getElementById('quotaProgressBar');
    if (progressBar) {
        const percentage = quotaData.percentage;
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);

        // Change color based on usage
        progressBar.className = 'progress-bar';
        if (percentage >= 90) {
            progressBar.classList.add('bg-danger');
        } else if (percentage >= 70) {
            progressBar.classList.add('bg-warning');
        } else {
            progressBar.classList.add('bg-success');
        }
    }
}

// ============================================================================
// API Key Management
// ============================================================================

async function loadAPIKeys() {
    try {
        const response = await fetch('/api/auth/api-keys?include_revoked=true', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load API keys');
        }

        apiKeys = await response.json();
        displayAPIKeys(apiKeys);

    } catch (error) {
        console.error('Error loading API keys:', error);
        showAlert('error', 'Failed to load API keys');
    }
}

function displayAPIKeys(keys) {
    const container = document.getElementById('apiKeysList');
    if (!container) return;

    if (keys.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-key fa-3x mb-3"></i>
                <p>No API keys created yet</p>
                <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createAPIKeyModal">
                    <i class="fas fa-plus"></i> Create API Key
                </button>
            </div>
        `;
        return;
    }

    const html = keys.map(key => `
        <div class="card mb-3">
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h5 class="card-title mb-1">
                            ${escapeHtml(key.key_name)}
                            ${key.status === 'revoked' ? '<span class="badge bg-danger ms-2">Revoked</span>' : ''}
                            ${key.status === 'expired' ? '<span class="badge bg-warning ms-2">Expired</span>' : ''}
                        </h5>
                        <p class="card-text text-muted mb-2">
                            <small>
                                <strong>Prefix:</strong> ${escapeHtml(key.key_prefix)}...
                                <br>
                                <strong>Created:</strong> ${formatDate(key.created_at)}
                                ${key.expires_at ? `<br><strong>Expires:</strong> ${formatDate(key.expires_at)}` : ''}
                                ${key.last_used_at ? `<br><strong>Last used:</strong> ${formatDate(key.last_used_at)}` : ''}
                            </small>
                        </p>
                        <div class="d-flex flex-wrap gap-2">
                            ${key.scopes.map(scope => `
                                <span class="badge bg-secondary">${escapeHtml(scope)}</span>
                            `).join('')}
                        </div>
                    </div>
                    <div class="col-md-4 text-end">
                        <div class="mb-2">
                            <small class="text-muted">Used ${key.usage_count} times</small>
                        </div>
                        ${key.status === 'active' ? `
                            <button class="btn btn-sm btn-warning" onclick="revokeAPIKey(${key.id}, '${escapeHtml(key.key_name)}')">
                                <i class="fas fa-ban"></i> Revoke
                            </button>
                        ` : ''}
                        <button class="btn btn-sm btn-danger" onclick="deleteAPIKey(${key.id}, '${escapeHtml(key.key_name)}')">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

function showCreateAPIKeyModal() {
    const modal = new bootstrap.Modal(document.getElementById('createAPIKeyModal'));
    modal.show();
}

async function handleCreateAPIKey(event) {
    event.preventDefault();

    const formData = {
        key_name: document.getElementById('keyName').value,
        scopes: getSelectedScopes(),
        expires_in_days: document.getElementById('expiresInDays').value ?
            parseInt(document.getElementById('expiresInDays').value) : null,
        rate_limit_per_minute: document.getElementById('rateLimitPerMinute').value ?
            parseInt(document.getElementById('rateLimitPerMinute').value) : null,
        rate_limit_per_hour: document.getElementById('rateLimitPerHour').value ?
            parseInt(document.getElementById('rateLimitPerHour').value) : null
    };

    try {
        const response = await fetch('/api/auth/api-keys', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Failed to create API key');
        }

        const result = await response.json();

        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('createAPIKeyModal'));
        modal.hide();

        // Show API key in a special modal (only time it will be shown)
        showAPIKeyResult(result.full_key, result.warning);

        // Reload API keys
        await loadAPIKeys();

        // Reset form
        event.target.reset();

    } catch (error) {
        console.error('Error creating API key:', error);
        showAlert('error', error.message || 'Failed to create API key');
    }
}

function getSelectedScopes() {
    const checkboxes = document.querySelectorAll('input[name="scopes"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function showAPIKeyResult(apiKey, warning) {
    const modalHtml = `
        <div class="modal fade" id="apiKeyResultModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header bg-success text-white">
                        <h5 class="modal-title">API Key Created Successfully</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle"></i> ${escapeHtml(warning)}
                        </div>
                        <p><strong>Your API Key:</strong></p>
                        <div class="input-group mb-3">
                            <input type="text" class="form-control font-monospace" id="generatedAPIKey" value="${escapeHtml(apiKey)}" readonly>
                            <button class="btn btn-outline-secondary" type="button" onclick="copyAPIKey()">
                                <i class="fas fa-copy"></i> Copy
                            </button>
                        </div>
                        <p class="text-muted small">
                            Store this key securely. You will not be able to see it again.
                        </p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-primary" data-bs-dismiss="modal">I've Saved It</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if any
    const existingModal = document.getElementById('apiKeyResultModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('apiKeyResultModal'));
    modal.show();

    // Remove modal from DOM when hidden
    document.getElementById('apiKeyResultModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

function copyAPIKey() {
    const input = document.getElementById('generatedAPIKey');
    input.select();
    input.setSelectionRange(0, 99999); // For mobile devices

    navigator.clipboard.writeText(input.value).then(() => {
        showAlert('success', 'API key copied to clipboard');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showAlert('error', 'Failed to copy to clipboard');
    });
}

async function revokeAPIKey(keyId, keyName) {
    if (!confirm(`Are you sure you want to revoke the API key "${keyName}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/auth/api-keys/${keyId}/revoke`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to revoke API key');
        }

        showAlert('success', `API key "${keyName}" has been revoked`);
        await loadAPIKeys();

    } catch (error) {
        console.error('Error revoking API key:', error);
        showAlert('error', error.message || 'Failed to revoke API key');
    }
}

async function deleteAPIKey(keyId, keyName) {
    if (!confirm(`Are you sure you want to delete the API key "${keyName}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/auth/api-keys/${keyId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to delete API key');
        }

        showAlert('success', `API key "${keyName}" has been deleted`);
        await loadAPIKeys();

    } catch (error) {
        console.error('Error deleting API key:', error);
        showAlert('error', error.message || 'Failed to delete API key');
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function getAuthToken() {
    // Try to get Clerk session token
    if (window.Clerk && window.Clerk.session) {
        return window.Clerk.session.getToken();
    }

    // Fall back to localStorage
    return localStorage.getItem('auth_token') || '';
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showAlert(type, message) {
    const alertHtml = `
        <div class="alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show" role="alert">
            ${escapeHtml(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const container = document.getElementById('alertContainer') || document.querySelector('.container');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                const bsAlert = bootstrap.Alert.getInstance(alert) || new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}
