/**
 * Admin Dashboard for Ocular OCR.
 *
 * Handles user management, role updates, quota management, and audit logs.
 */

// ============================================================================
// State Management
// ============================================================================

let users = [];
let auditLogs = [];
let selectedUserId = null;

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadUsers();
    await loadAuditLogs();
    setupEventListeners();
});

function setupEventListeners() {
    // Filter controls
    document.getElementById('roleFilter').addEventListener('change', filterUsers);
    document.getElementById('statusFilter').addEventListener('change', filterUsers);
    document.getElementById('searchUser').addEventListener('input', filterUsers);

    // Role update form
    const roleForm = document.getElementById('updateRoleForm');
    if (roleForm) {
        roleForm.addEventListener('submit', handleRoleUpdate);
    }

    // Quota update form
    const quotaForm = document.getElementById('updateQuotaForm');
    if (quotaForm) {
        quotaForm.addEventListener('submit', handleQuotaUpdate);
    }
}

// ============================================================================
// User Management
// ============================================================================

async function loadUsers() {
    try {
        const response = await fetch('/api/auth/users', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load users');
        }

        users = await response.json();
        displayUsers(users);

    } catch (error) {
        console.error('Error loading users:', error);
        showAlert('error', 'Failed to load users');
    }
}

function displayUsers(userList) {
    const container = document.getElementById('usersList');
    if (!container) return;

    if (userList.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-users fa-3x mb-3"></i>
                <p>No users found</p>
            </div>
        `;
        return;
    }

    const html = userList.map(user => `
        <tr>
            <td>
                ${user.avatar_url ? `<img src="${user.avatar_url}" class="rounded-circle me-2" width="32" height="32">` : ''}
                <strong>${escapeHtml(user.email)}</strong>
                ${user.username ? `<br><small class="text-muted">@${escapeHtml(user.username)}</small>` : ''}
            </td>
            <td>
                ${user.first_name || ''} ${user.last_name || ''}
            </td>
            <td>
                <span class="badge bg-${getRoleBadgeClass(user.role)}">${user.role}</span>
            </td>
            <td>
                <span class="badge bg-${user.is_active ? 'success' : 'danger'}">
                    ${user.is_active ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>
                <span class="badge bg-info">${user.subscription_tier}</span>
            </td>
            <td>
                <small>
                    ${user.quotas.api_calls.used.toLocaleString()} / ${user.quotas.api_calls.limit.toLocaleString()}
                    <div class="progress mt-1" style="height: 5px;">
                        <div class="progress-bar ${getQuotaBarClass(user.quotas.api_calls.used, user.quotas.api_calls.limit)}"
                             style="width: ${(user.quotas.api_calls.used / user.quotas.api_calls.limit * 100)}%"></div>
                    </div>
                </small>
            </td>
            <td>
                <small class="text-muted">${formatDate(user.created_at)}</small>
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="showUserDetails(${user.id})" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-secondary" onclick="showRoleModal(${user.id})" title="Update Role">
                        <i class="fas fa-user-tag"></i>
                    </button>
                    <button class="btn btn-outline-info" onclick="showQuotaModal(${user.id})" title="Update Quotas">
                        <i class="fas fa-chart-pie"></i>
                    </button>
                    ${user.is_active ? `
                        <button class="btn btn-outline-warning" onclick="deactivateUser(${user.id})" title="Deactivate">
                            <i class="fas fa-ban"></i>
                        </button>
                    ` : `
                        <button class="btn btn-outline-success" onclick="activateUser(${user.id})" title="Activate">
                            <i class="fas fa-check"></i>
                        </button>
                    `}
                </div>
            </td>
        </tr>
    `).join('');

    container.innerHTML = html;

    // Update stats
    updateUserStats(userList);
}

function updateUserStats(userList) {
    document.getElementById('totalUsers').textContent = userList.length;
    document.getElementById('activeUsers').textContent = userList.filter(u => u.is_active).length;
    document.getElementById('adminUsers').textContent = userList.filter(u => u.role === 'admin').length;
}

function filterUsers() {
    const roleFilter = document.getElementById('roleFilter').value;
    const statusFilter = document.getElementById('statusFilter').value;
    const searchTerm = document.getElementById('searchUser').value.toLowerCase();

    let filtered = users;

    // Filter by role
    if (roleFilter) {
        filtered = filtered.filter(u => u.role === roleFilter);
    }

    // Filter by status
    if (statusFilter) {
        const isActive = statusFilter === 'active';
        filtered = filtered.filter(u => u.is_active === isActive);
    }

    // Filter by search term
    if (searchTerm) {
        filtered = filtered.filter(u =>
            u.email.toLowerCase().includes(searchTerm) ||
            (u.username && u.username.toLowerCase().includes(searchTerm)) ||
            (u.first_name && u.first_name.toLowerCase().includes(searchTerm)) ||
            (u.last_name && u.last_name.toLowerCase().includes(searchTerm))
        );
    }

    displayUsers(filtered);
}

function showRoleModal(userId) {
    selectedUserId = userId;
    const user = users.find(u => u.id === userId);

    if (user) {
        document.getElementById('roleModalUsername').textContent = user.email;
        document.getElementById('newRole').value = user.role;

        const modal = new bootstrap.Modal(document.getElementById('updateRoleModal'));
        modal.show();
    }
}

async function handleRoleUpdate(event) {
    event.preventDefault();

    const newRole = document.getElementById('newRole').value;

    try {
        const response = await fetch(`/api/auth/users/${selectedUserId}/role`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify({ role: newRole })
        });

        if (!response.ok) {
            throw new Error('Failed to update role');
        }

        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('updateRoleModal'));
        modal.hide();

        showAlert('success', 'User role updated successfully');
        await loadUsers();

    } catch (error) {
        console.error('Error updating role:', error);
        showAlert('error', error.message || 'Failed to update role');
    }
}

function showQuotaModal(userId) {
    selectedUserId = userId;
    const user = users.find(u => u.id === userId);

    if (user) {
        document.getElementById('quotaModalUsername').textContent = user.email;
        document.getElementById('apiCallsQuota').value = user.quotas.api_calls.limit;
        document.getElementById('deploymentQuota').value = user.quotas.deployments;
        document.getElementById('storageQuota').value = user.quotas.storage_gb;

        const modal = new bootstrap.Modal(document.getElementById('updateQuotaModal'));
        modal.show();
    }
}

async function handleQuotaUpdate(event) {
    event.preventDefault();

    const quotaData = {
        api_calls_quota: parseInt(document.getElementById('apiCallsQuota').value),
        deployment_quota: parseInt(document.getElementById('deploymentQuota').value),
        storage_quota_gb: parseFloat(document.getElementById('storageQuota').value)
    };

    try {
        const response = await fetch(`/api/auth/users/${selectedUserId}/quotas`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(quotaData)
        });

        if (!response.ok) {
            throw new Error('Failed to update quotas');
        }

        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('updateQuotaModal'));
        modal.hide();

        showAlert('success', 'User quotas updated successfully');
        await loadUsers();

    } catch (error) {
        console.error('Error updating quotas:', error);
        showAlert('error', error.message || 'Failed to update quotas');
    }
}

async function deactivateUser(userId) {
    const user = users.find(u => u.id === userId);
    if (!user) return;

    if (!confirm(`Are you sure you want to deactivate user "${user.email}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/auth/users/${userId}/deactivate`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to deactivate user');
        }

        showAlert('success', `User "${user.email}" has been deactivated`);
        await loadUsers();

    } catch (error) {
        console.error('Error deactivating user:', error);
        showAlert('error', error.message || 'Failed to deactivate user');
    }
}

async function activateUser(userId) {
    const user = users.find(u => u.id === userId);
    if (!user) return;

    try {
        const response = await fetch(`/api/auth/users/${userId}/activate`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to activate user');
        }

        showAlert('success', `User "${user.email}" has been activated`);
        await loadUsers();

    } catch (error) {
        console.error('Error activating user:', error);
        showAlert('error', error.message || 'Failed to activate user');
    }
}

// ============================================================================
// Audit Logs
// ============================================================================

async function loadAuditLogs() {
    try {
        const response = await fetch('/api/auth/audit-logs?limit=50', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load audit logs');
        }

        auditLogs = await response.json();
        displayAuditLogs(auditLogs);

    } catch (error) {
        console.error('Error loading audit logs:', error);
        showAlert('error', 'Failed to load audit logs');
    }
}

function displayAuditLogs(logs) {
    const container = document.getElementById('auditLogsList');
    if (!container) return;

    if (logs.length === 0) {
        container.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted py-4">No audit logs found</td>
            </tr>
        `;
        return;
    }

    const html = logs.map(log => `
        <tr>
            <td><small class="text-muted">${formatDate(log.created_at)}</small></td>
            <td>
                ${log.user_id ? `<span class="badge bg-secondary">User ${log.user_id}</span>` : '<span class="badge bg-dark">System</span>'}
            </td>
            <td><span class="badge bg-${getActionBadgeClass(log.action)}">${log.action}</span></td>
            <td><span class="badge bg-info">${log.resource_type}</span></td>
            <td>
                <small>${escapeHtml(log.description || 'N/A')}</small>
            </td>
            <td>
                <span class="badge bg-${log.status === 'success' ? 'success' : 'danger'}">${log.status}</span>
            </td>
        </tr>
    `).join('');

    container.innerHTML = html;
}

// ============================================================================
// Utility Functions
// ============================================================================

function getRoleBadgeClass(role) {
    const classes = {
        'admin': 'danger',
        'user': 'primary',
        'viewer': 'secondary'
    };
    return classes[role] || 'secondary';
}

function getActionBadgeClass(action) {
    const classes = {
        'create': 'success',
        'update': 'info',
        'delete': 'danger',
        'deploy': 'primary',
        'stop': 'warning',
        'login': 'secondary',
        'logout': 'secondary',
        'api_call': 'light'
    };
    return classes[action] || 'secondary';
}

function getQuotaBarClass(used, limit) {
    const percentage = (used / limit) * 100;
    if (percentage >= 90) return 'bg-danger';
    if (percentage >= 70) return 'bg-warning';
    return 'bg-success';
}

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
