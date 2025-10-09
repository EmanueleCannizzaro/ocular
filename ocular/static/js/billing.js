/**
 * Billing Dashboard for Ocular OCR.
 *
 * Handles invoices, cost tracking, and budget alerts.
 */

// ============================================================================
// State Management
// ============================================================================

let invoices = [];
let costs = [];
let budgetAlerts = [];
let costChart = null;

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await Promise.all([
        loadInvoices(),
        loadCurrentMonthCost(),
        loadCostSummary(),
        loadBudgetAlerts()
    ]);
    setupEventListeners();
});

function setupEventListeners() {
    // Budget alert form
    const alertForm = document.getElementById('createBudgetAlertForm');
    if (alertForm) {
        alertForm.addEventListener('submit', handleCreateBudgetAlert);
    }

    // Invoice filters
    const statusFilter = document.getElementById('invoiceStatusFilter');
    if (statusFilter) {
        statusFilter.addEventListener('change', filterInvoices);
    }
}

// ============================================================================
// Invoice Management
// ============================================================================

async function loadInvoices() {
    try {
        const response = await fetch('/api/billing/invoices', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load invoices');
        }

        invoices = await response.json();
        displayInvoices(invoices);

    } catch (error) {
        console.error('Error loading invoices:', error);
        showAlert('error', 'Failed to load invoices');
    }
}

function displayInvoices(invoiceList) {
    const container = document.getElementById('invoicesList');
    if (!container) return;

    if (invoiceList.length === 0) {
        container.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted py-4">No invoices found</td>
            </tr>
        `;
        return;
    }

    const html = invoiceList.map(invoice => `
        <tr>
            <td>
                <strong>${escapeHtml(invoice.invoice_number)}</strong>
                <br>
                <small class="text-muted">${formatDate(invoice.invoice_date)}</small>
            </td>
            <td>
                <span class="badge bg-${getInvoiceStatusBadge(invoice.status)}">
                    ${invoice.status.toUpperCase()}
                </span>
            </td>
            <td>$${invoice.total.toFixed(2)}</td>
            <td>$${invoice.amount_paid.toFixed(2)}</td>
            <td>$${invoice.amount_due.toFixed(2)}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="viewInvoice(${invoice.id})">
                    <i class="fas fa-eye"></i> View
                </button>
                ${invoice.status === 'pending' || invoice.status === 'draft' ? `
                    <button class="btn btn-sm btn-primary" onclick="payInvoice(${invoice.id})">
                        <i class="fas fa-credit-card"></i> Pay
                    </button>
                ` : ''}
            </td>
        </tr>
    `).join('');

    container.innerHTML = html;
}

function filterInvoices() {
    const statusFilter = document.getElementById('invoiceStatusFilter').value;

    let filtered = invoices;
    if (statusFilter) {
        filtered = invoices.filter(inv => inv.status === statusFilter);
    }

    displayInvoices(filtered);
}

function getInvoiceStatusBadge(status) {
    const badges = {
        'paid': 'success',
        'pending': 'warning',
        'draft': 'secondary',
        'failed': 'danger',
        'cancelled': 'dark',
        'refunded': 'info'
    };
    return badges[status] || 'secondary';
}

async function viewInvoice(invoiceId) {
    try {
        const response = await fetch(`/api/billing/invoices/${invoiceId}`, {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load invoice details');
        }

        const invoice = await response.json();
        showInvoiceModal(invoice);

    } catch (error) {
        console.error('Error loading invoice:', error);
        showAlert('error', 'Failed to load invoice details');
    }
}

function showInvoiceModal(invoice) {
    const lineItemsHtml = invoice.line_items.map(item => `
        <tr>
            <td>${escapeHtml(item.description)}</td>
            <td class="text-end">${item.quantity || 1}</td>
            <td class="text-end">$${item.amount.toFixed(2)}</td>
        </tr>
    `).join('');

    const modalHtml = `
        <div class="modal fade" id="invoiceDetailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Invoice ${escapeHtml(invoice.invoice_number)}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <p><strong>Invoice Date:</strong> ${formatDate(invoice.invoice_date)}</p>
                                <p><strong>Due Date:</strong> ${formatDate(invoice.due_date)}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Status:</strong> <span class="badge bg-${getInvoiceStatusBadge(invoice.status)}">${invoice.status.toUpperCase()}</span></p>
                                <p><strong>Period:</strong> ${formatDate(invoice.period_start)} - ${formatDate(invoice.period_end)}</p>
                            </div>
                        </div>

                        <h6>Line Items</h6>
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Description</th>
                                    <th class="text-end">Quantity</th>
                                    <th class="text-end">Amount</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${lineItemsHtml}
                            </tbody>
                            <tfoot>
                                <tr>
                                    <td colspan="2" class="text-end"><strong>Subtotal:</strong></td>
                                    <td class="text-end">$${invoice.subtotal.toFixed(2)}</td>
                                </tr>
                                <tr>
                                    <td colspan="2" class="text-end"><strong>Tax:</strong></td>
                                    <td class="text-end">$${invoice.tax.toFixed(2)}</td>
                                </tr>
                                <tr>
                                    <td colspan="2" class="text-end"><strong>Total:</strong></td>
                                    <td class="text-end"><strong>$${invoice.total.toFixed(2)}</strong></td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        ${invoice.status === 'pending' || invoice.status === 'draft' ? `
                            <button type="button" class="btn btn-primary" onclick="payInvoice(${invoice.id})">
                                <i class="fas fa-credit-card"></i> Pay Now
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal
    const existingModal = document.getElementById('invoiceDetailModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add and show modal
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('invoiceDetailModal'));
    modal.show();
}

async function payInvoice(invoiceId) {
    // In a real implementation, this would integrate with Stripe Elements
    showAlert('info', 'Payment integration coming soon. This would open Stripe payment form.');
}

async function generateMonthlyInvoice() {
    if (!confirm('Generate invoice for last month? This will create an invoice based on your usage.')) {
        return;
    }

    try {
        const response = await fetch('/api/billing/invoices/generate-monthly', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to generate invoice');
        }

        const invoice = await response.json();
        showAlert('success', `Invoice ${invoice.invoice_number} generated successfully`);
        await loadInvoices();

    } catch (error) {
        console.error('Error generating invoice:', error);
        showAlert('error', error.message || 'Failed to generate invoice');
    }
}

// ============================================================================
// Cost Tracking
// ============================================================================

async function loadCurrentMonthCost() {
    try {
        const response = await fetch('/api/billing/costs/current-month', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load current month cost');
        }

        const data = await response.json();
        displayCurrentMonthCost(data);

    } catch (error) {
        console.error('Error loading current month cost:', error);
        showAlert('error', 'Failed to load cost information');
    }
}

function displayCurrentMonthCost(data) {
    // Update total cost
    const totalCostEl = document.getElementById('currentMonthCost');
    if (totalCostEl) {
        totalCostEl.textContent = `$${data.total_cost.toFixed(2)}`;
    }

    // Update breakdown
    const breakdownEl = document.getElementById('costBreakdown');
    if (breakdownEl && data.breakdown) {
        const html = Object.entries(data.breakdown).map(([type, cost]) => `
            <div class="d-flex justify-content-between mb-2">
                <span>${type.replace('_', ' ').toUpperCase()}:</span>
                <strong>$${cost.toFixed(2)}</strong>
            </div>
        `).join('');
        breakdownEl.innerHTML = html;
    }
}

async function loadCostSummary(period = 'month') {
    try {
        const response = await fetch(`/api/billing/costs/summary?period=${period}`, {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load cost summary');
        }

        const data = await response.json();
        displayCostChart(data);

    } catch (error) {
        console.error('Error loading cost summary:', error);
    }
}

function displayCostChart(data) {
    const ctx = document.getElementById('costChart');
    if (!ctx) return;

    if (costChart) {
        costChart.destroy();
    }

    const labels = Object.keys(data.breakdown);
    const values = Object.values(data.breakdown);

    costChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels.map(l => l.replace('_', ' ').toUpperCase()),
            datasets: [{
                data: values,
                backgroundColor: [
                    '#0d6efd',
                    '#198754',
                    '#ffc107',
                    '#dc3545',
                    '#6c757d',
                    '#0dcaf0'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: $${context.parsed.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================================================
// Budget Alerts
// ============================================================================

async function loadBudgetAlerts() {
    try {
        const response = await fetch('/api/billing/budget-alerts', {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load budget alerts');
        }

        budgetAlerts = await response.json();
        displayBudgetAlerts(budgetAlerts);

    } catch (error) {
        console.error('Error loading budget alerts:', error);
        showAlert('error', 'Failed to load budget alerts');
    }
}

function displayBudgetAlerts(alerts) {
    const container = document.getElementById('budgetAlertsList');
    if (!container) return;

    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-bell-slash fa-3x mb-3"></i>
                <p>No budget alerts configured</p>
                <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createBudgetAlertModal">
                    <i class="fas fa-plus"></i> Create Alert
                </button>
            </div>
        `;
        return;
    }

    const html = alerts.map(alert => `
        <div class="card mb-3 ${alert.is_triggered ? 'border-danger' : ''}">
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h5 class="card-title mb-1">
                            ${escapeHtml(alert.alert_name)}
                            ${alert.is_triggered ? '<span class="badge bg-danger ms-2">TRIGGERED</span>' : ''}
                            ${!alert.is_active ? '<span class="badge bg-secondary ms-2">INACTIVE</span>' : ''}
                        </h5>
                        <p class="card-text mb-2">
                            <strong>Threshold:</strong> $${alert.threshold_amount.toFixed(2)} (${alert.period_type})
                            <br>
                            <strong>Current:</strong> $${alert.current_amount.toFixed(2)} (${alert.percentage.toFixed(1)}%)
                        </p>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar ${alert.percentage >= 100 ? 'bg-danger' : alert.percentage >= 80 ? 'bg-warning' : 'bg-success'}"
                                 role="progressbar" style="width: ${Math.min(alert.percentage, 100)}%">
                                ${alert.percentage.toFixed(0)}%
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-end">
                        ${alert.is_active ? `
                            <button class="btn btn-sm btn-warning" onclick="deactivateBudgetAlert(${alert.id})">
                                <i class="fas fa-pause"></i> Deactivate
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

async function handleCreateBudgetAlert(event) {
    event.preventDefault();

    const formData = {
        alert_name: document.getElementById('alertName').value,
        threshold_amount: parseFloat(document.getElementById('thresholdAmount').value),
        alert_type: document.getElementById('alertType').value,
        period_type: document.getElementById('periodType').value,
        notify_email: document.getElementById('notifyEmail').checked,
        notify_slack: document.getElementById('notifySlack').checked,
        slack_webhook_url: document.getElementById('slackWebhook').value || null
    };

    try {
        const response = await fetch('/api/billing/budget-alerts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Failed to create budget alert');
        }

        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('createBudgetAlertModal'));
        modal.hide();

        showAlert('success', 'Budget alert created successfully');
        await loadBudgetAlerts();

        // Reset form
        event.target.reset();

    } catch (error) {
        console.error('Error creating budget alert:', error);
        showAlert('error', error.message || 'Failed to create budget alert');
    }
}

async function deactivateBudgetAlert(alertId) {
    if (!confirm('Deactivate this budget alert?')) {
        return;
    }

    try {
        const response = await fetch(`/api/billing/budget-alerts/${alertId}/deactivate`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to deactivate alert');
        }

        showAlert('success', 'Budget alert deactivated');
        await loadBudgetAlerts();

    } catch (error) {
        console.error('Error deactivating alert:', error);
        showAlert('error', error.message || 'Failed to deactivate alert');
    }
}

async function checkBudgetAlerts() {
    try {
        const response = await fetch('/api/billing/budget-alerts/check', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to check alerts');
        }

        const result = await response.json();

        if (result.triggered_alerts > 0) {
            showAlert('warning', `${result.triggered_alerts} budget alert(s) have been triggered!`);
        } else {
            showAlert('success', 'All budget alerts are within limits');
        }

        await loadBudgetAlerts();

    } catch (error) {
        console.error('Error checking alerts:', error);
        showAlert('error', 'Failed to check budget alerts');
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
    return date.toLocaleDateString();
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
