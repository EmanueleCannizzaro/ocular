/**
 * Enhanced Observability Dashboard
 *
 * Provides real-time logs, traces, performance metrics, alerts, and error tracking.
 */

const API_BASE = window.location.origin + '/api/observability';

class ObservabilityDashboard {
    constructor() {
        this.refreshInterval = 30000; // 30 seconds
        this.autoRefresh = true;
        this.refreshTimer = null;

        // Data stores
        this.logs = [];
        this.traces = {};
        this.metrics = {};
        this.alerts = [];
        this.errors = {};

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadDashboard();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('data-bs-target');
                this.onTabChange(target);
            });
        });

        // Refresh buttons
        document.getElementById('refresh-dashboard')?.addEventListener('click', () => {
            this.loadDashboard();
        });

        document.getElementById('refresh-logs')?.addEventListener('click', () => {
            this.loadLogs();
        });

        document.getElementById('refresh-metrics')?.addEventListener('click', () => {
            this.loadMetrics();
        });

        document.getElementById('refresh-alerts')?.addEventListener('click', () => {
            this.loadAlerts();
        });

        // Auto-refresh toggle
        document.getElementById('auto-refresh-toggle')?.addEventListener('change', (e) => {
            this.autoRefresh = e.target.checked;
            if (this.autoRefresh) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });

        // Log level filter
        document.getElementById('log-level-filter')?.addEventListener('change', () => {
            this.loadLogs();
        });

        // Time range filter
        document.getElementById('time-range-filter')?.addEventListener('change', () => {
            this.loadLogs();
            this.loadMetrics();
        });
    }

    async loadDashboard() {
        try {
            showLoading('Loading dashboard...');

            const response = await fetch(`${API_BASE}/dashboard`);
            if (!response.ok) throw new Error('Failed to load dashboard');

            const data = await response.json();

            // Update summary cards
            this.updateSummaryCards(data);

            // Load individual sections
            await Promise.all([
                this.loadLogs(),
                this.loadMetrics(),
                this.loadAlerts(),
                this.loadErrors(),
            ]);

            hideLoading();
        } catch (error) {
            console.error('Error loading dashboard:', error);
            hideLoading();
            showToast('Failed to load dashboard', 'error');
        }
    }

    updateSummaryCards(data) {
        // Update log statistics
        document.getElementById('total-logs').textContent = data.log_stats.total_logs.toLocaleString();
        document.getElementById('error-rate').textContent = (data.log_stats.error_rate * 100).toFixed(2) + '%';

        // Update active alerts
        document.getElementById('active-alerts-count').textContent = data.active_alerts;

        // Update metrics count
        document.getElementById('total-metrics').textContent = data.total_metrics.toLocaleString();

        // Update active traces
        document.getElementById('active-traces').textContent = data.active_traces;

        // Update log level breakdown
        const logLevels = data.log_stats.by_level;
        document.getElementById('debug-logs').textContent = logLevels.debug || 0;
        document.getElementById('info-logs').textContent = logLevels.info || 0;
        document.getElementById('warning-logs').textContent = logLevels.warning || 0;
        document.getElementById('error-logs').textContent = logLevels.error || 0;
    }

    async loadLogs() {
        try {
            const levelFilter = document.getElementById('log-level-filter')?.value || '';
            const timeRange = parseInt(document.getElementById('time-range-filter')?.value || '24');

            const params = new URLSearchParams({
                limit: '100',
                since_hours: timeRange,
            });

            if (levelFilter) {
                params.append('level', levelFilter);
            }

            const response = await fetch(`${API_BASE}/logs?${params}`);
            if (!response.ok) throw new Error('Failed to load logs');

            const data = await response.json();
            this.logs = data.logs;

            this.renderLogs();
        } catch (error) {
            console.error('Error loading logs:', error);
            showToast('Failed to load logs', 'error');
        }
    }

    renderLogs() {
        const container = document.getElementById('logs-container');
        if (!container) return;

        if (this.logs.length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No logs found</div>';
            return;
        }

        const logsHtml = this.logs.map(log => {
            const levelClass = this.getLogLevelClass(log.level);
            const timestamp = new Date(log.timestamp).toLocaleString();

            return `
                <div class="log-entry ${levelClass}" data-trace-id="${log.trace_id || ''}">
                    <div class="log-header">
                        <span class="log-timestamp">${timestamp}</span>
                        <span class="badge bg-${levelClass}">${log.level.toUpperCase()}</span>
                        <span class="log-source">${log.source}</span>
                        ${log.trace_id ? `<span class="log-trace-id" title="Trace ID">${log.trace_id.substring(0, 8)}</span>` : ''}
                    </div>
                    <div class="log-message">${this.escapeHtml(log.message)}</div>
                    ${log.error_details ? `
                        <div class="log-error-details">
                            <details>
                                <summary>Error Details</summary>
                                <pre>${this.escapeHtml(JSON.stringify(log.error_details, null, 2))}</pre>
                            </details>
                        </div>
                    ` : ''}
                    ${Object.keys(log.metadata || {}).length > 0 ? `
                        <div class="log-metadata">
                            <details>
                                <summary>Metadata</summary>
                                <pre>${this.escapeHtml(JSON.stringify(log.metadata, null, 2))}</pre>
                            </details>
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        container.innerHTML = logsHtml;

        // Add click handlers for trace IDs
        container.querySelectorAll('.log-trace-id').forEach(el => {
            el.addEventListener('click', (e) => {
                const traceId = e.target.closest('.log-entry').getAttribute('data-trace-id');
                this.viewTrace(traceId);
            });
        });
    }

    getLogLevelClass(level) {
        const map = {
            debug: 'secondary',
            info: 'info',
            warning: 'warning',
            error: 'danger',
            critical: 'danger',
        };
        return map[level.toLowerCase()] || 'secondary';
    }

    async viewTrace(traceId) {
        try {
            const response = await fetch(`${API_BASE}/traces/${traceId}`);
            if (!response.ok) throw new Error('Failed to load trace');

            const data = await response.json();
            this.showTraceModal(traceId, data.spans);
        } catch (error) {
            console.error('Error loading trace:', error);
            showToast('Failed to load trace details', 'error');
        }
    }

    showTraceModal(traceId, spans) {
        const modal = new bootstrap.Modal(document.getElementById('trace-modal'));
        const container = document.getElementById('trace-spans-container');

        // Build trace tree
        const spansHtml = spans.map(span => {
            const duration = span.duration_ms ? `${span.duration_ms.toFixed(2)}ms` : 'N/A';
            const statusClass = span.status === 'success' ? 'success' : 'danger';

            return `
                <div class="trace-span">
                    <div class="span-header">
                        <strong>${span.operation_name}</strong>
                        <span class="badge bg-${statusClass}">${span.status}</span>
                        <span class="span-duration">${duration}</span>
                    </div>
                    <div class="span-details">
                        <div>Span ID: ${span.span_id}</div>
                        <div>Start: ${new Date(span.start_time).toLocaleString()}</div>
                        ${span.end_time ? `<div>End: ${new Date(span.end_time).toLocaleString()}</div>` : ''}
                        ${Object.keys(span.tags || {}).length > 0 ? `
                            <details>
                                <summary>Tags</summary>
                                <pre>${this.escapeHtml(JSON.stringify(span.tags, null, 2))}</pre>
                            </details>
                        ` : ''}
                        ${span.logs && span.logs.length > 0 ? `
                            <details>
                                <summary>Logs (${span.logs.length})</summary>
                                ${span.logs.map(log => `
                                    <div class="span-log">
                                        <small>${log.timestamp}: ${log.message}</small>
                                    </div>
                                `).join('')}
                            </details>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');

        document.getElementById('trace-id-display').textContent = traceId;
        container.innerHTML = spansHtml;
        modal.show();
    }

    async loadMetrics() {
        try {
            const timeRange = parseInt(document.getElementById('time-range-filter')?.value || '24');

            const params = new URLSearchParams({
                since_hours: timeRange,
                limit: '1000',
            });

            const response = await fetch(`${API_BASE}/metrics?${params}`);
            if (!response.ok) throw new Error('Failed to load metrics');

            const data = await response.json();
            this.metrics = data.metrics;

            this.renderMetrics();
        } catch (error) {
            console.error('Error loading metrics:', error);
            showToast('Failed to load metrics', 'error');
        }
    }

    renderMetrics() {
        const container = document.getElementById('metrics-container');
        if (!container) return;

        // Group metrics by name
        const metricsByName = {};
        this.metrics.forEach(metric => {
            if (!metricsByName[metric.name]) {
                metricsByName[metric.name] = [];
            }
            metricsByName[metric.name].push(metric);
        });

        if (Object.keys(metricsByName).length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No metrics found</div>';
            return;
        }

        const metricsHtml = Object.entries(metricsByName).map(([name, metrics]) => {
            const values = metrics.map(m => m.value);
            const avg = values.reduce((a, b) => a + b, 0) / values.length;
            const min = Math.min(...values);
            const max = Math.max(...values);

            return `
                <div class="metric-card card mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">${name}</h6>
                        <small class="text-muted">${metrics[0].metric_type}</small>
                    </div>
                    <div class="card-body">
                        <div class="metric-stats">
                            <div class="metric-stat">
                                <div class="stat-label">Min</div>
                                <div class="stat-value">${min.toFixed(2)}</div>
                            </div>
                            <div class="metric-stat">
                                <div class="stat-label">Avg</div>
                                <div class="stat-value">${avg.toFixed(2)}</div>
                            </div>
                            <div class="metric-stat">
                                <div class="stat-label">Max</div>
                                <div class="stat-value">${max.toFixed(2)}</div>
                            </div>
                            <div class="metric-stat">
                                <div class="stat-label">Count</div>
                                <div class="stat-value">${values.length}</div>
                            </div>
                        </div>
                        <div class="metric-chart-mini" id="chart-${this.sanitizeId(name)}"></div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = metricsHtml;

        // Render mini charts (simplified sparklines)
        Object.entries(metricsByName).forEach(([name, metrics]) => {
            this.renderMiniChart(name, metrics);
        });
    }

    renderMiniChart(name, metrics) {
        // Simple sparkline visualization using divs
        const chartContainer = document.getElementById(`chart-${this.sanitizeId(name)}`);
        if (!chartContainer) return;

        const values = metrics.map(m => m.value);
        const max = Math.max(...values);
        const min = Math.min(...values);
        const range = max - min || 1;

        const bars = values.slice(-20).map(value => {
            const height = ((value - min) / range) * 100;
            return `<div class="sparkline-bar" style="height: ${height}%"></div>`;
        }).join('');

        chartContainer.innerHTML = `<div class="sparkline">${bars}</div>`;
    }

    async loadAlerts() {
        try {
            const params = new URLSearchParams({
                limit: '50',
            });

            const response = await fetch(`${API_BASE}/alerts?${params}`);
            if (!response.ok) throw new Error('Failed to load alerts');

            const data = await response.json();
            this.alerts = data.alerts;

            this.renderAlerts();
        } catch (error) {
            console.error('Error loading alerts:', error);
            showToast('Failed to load alerts', 'error');
        }
    }

    renderAlerts() {
        const container = document.getElementById('alerts-container');
        if (!container) return;

        const activeAlerts = this.alerts.filter(a => !a.resolved);
        const resolvedAlerts = this.alerts.filter(a => a.resolved);

        if (this.alerts.length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No alerts</div>';
            return;
        }

        const alertsHtml = `
            ${activeAlerts.length > 0 ? `
                <h6 class="mb-3">Active Alerts (${activeAlerts.length})</h6>
                ${activeAlerts.map(alert => this.renderAlertCard(alert)).join('')}
            ` : ''}
            ${resolvedAlerts.length > 0 ? `
                <h6 class="mb-3 mt-4">Resolved Alerts (${resolvedAlerts.length})</h6>
                ${resolvedAlerts.map(alert => this.renderAlertCard(alert)).join('')}
            ` : ''}
        `;

        container.innerHTML = alertsHtml;

        // Add resolve handlers
        container.querySelectorAll('.resolve-alert-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const alertId = e.target.getAttribute('data-alert-id');
                this.resolveAlert(alertId);
            });
        });
    }

    renderAlertCard(alert) {
        const severityClass = {
            info: 'info',
            warning: 'warning',
            error: 'danger',
            critical: 'danger',
        }[alert.severity];

        const timestamp = new Date(alert.timestamp).toLocaleString();

        return `
            <div class="alert alert-${severityClass} alert-dismissible" role="alert">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="alert-heading mb-1">${alert.title}</h6>
                        <p class="mb-2">${alert.description}</p>
                        <small class="text-muted">
                            ${timestamp} - Source: ${alert.source}
                            ${alert.resolved ? `- Resolved at: ${new Date(alert.resolved_at).toLocaleString()}` : ''}
                        </small>
                    </div>
                    ${!alert.resolved ? `
                        <button type="button" class="btn btn-sm btn-outline-${severityClass} resolve-alert-btn" data-alert-id="${alert.alert_id}">
                            Resolve
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    async resolveAlert(alertId) {
        try {
            const response = await fetch(`${API_BASE}/alerts/${alertId}/resolve`, {
                method: 'POST',
            });

            if (!response.ok) throw new Error('Failed to resolve alert');

            showToast('Alert resolved successfully', 'success');
            await this.loadAlerts();
        } catch (error) {
            console.error('Error resolving alert:', error);
            showToast('Failed to resolve alert', 'error');
        }
    }

    async loadErrors() {
        try {
            const response = await fetch(`${API_BASE}/errors/summary?hours=24`);
            if (!response.ok) throw new Error('Failed to load errors');

            const data = await response.json();
            this.errors = data;

            this.renderErrors();
        } catch (error) {
            console.error('Error loading errors:', error);
            showToast('Failed to load error summary', 'error');
        }
    }

    renderErrors() {
        const container = document.getElementById('errors-container');
        if (!container) return;

        if (this.errors.total_errors === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No errors in the last 24 hours</div>';
            return;
        }

        const errorsHtml = `
            <div class="mb-4">
                <h6>Error Summary (24 hours)</h6>
                <div class="row">
                    <div class="col-md-6">
                        <div class="stat-box">
                            <div class="stat-label">Total Errors</div>
                            <div class="stat-value text-danger">${this.errors.total_errors}</div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="stat-box">
                            <div class="stat-label">Unique Error Types</div>
                            <div class="stat-value">${this.errors.unique_errors}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="error-types-list">
                ${Object.entries(this.errors.by_type).map(([errorType, count]) => `
                    <div class="error-type-item" data-error-type="${errorType}">
                        <div class="error-type-name">${errorType}</div>
                        <div class="error-type-count badge bg-danger">${count}</div>
                        <button class="btn btn-sm btn-outline-secondary view-samples-btn" data-error-type="${errorType}">
                            View Samples
                        </button>
                    </div>
                `).join('')}
            </div>
        `;

        container.innerHTML = errorsHtml;

        // Add view samples handlers
        container.querySelectorAll('.view-samples-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const errorType = e.target.getAttribute('data-error-type');
                this.viewErrorSamples(errorType);
            });
        });
    }

    async viewErrorSamples(errorType) {
        try {
            const response = await fetch(`${API_BASE}/errors/${encodeURIComponent(errorType)}`);
            if (!response.ok) throw new Error('Failed to load error samples');

            const data = await response.json();
            this.showErrorSamplesModal(errorType, data.samples);
        } catch (error) {
            console.error('Error loading error samples:', error);
            showToast('Failed to load error samples', 'error');
        }
    }

    showErrorSamplesModal(errorType, samples) {
        const modal = new bootstrap.Modal(document.getElementById('error-samples-modal'));
        const container = document.getElementById('error-samples-container');

        const samplesHtml = samples.map(sample => `
            <div class="error-sample card mb-3">
                <div class="card-header">
                    <small class="text-muted">${sample.timestamp}</small>
                    ${sample.source ? `<span class="badge bg-secondary">${sample.source}</span>` : ''}
                </div>
                <div class="card-body">
                    <p class="mb-2"><strong>${sample.message}</strong></p>
                    ${sample.context ? `
                        <details>
                            <summary>Context</summary>
                            <pre>${this.escapeHtml(JSON.stringify(sample.context, null, 2))}</pre>
                        </details>
                    ` : ''}
                    ${sample.traceback ? `
                        <details>
                            <summary>Traceback</summary>
                            <pre>${this.escapeHtml(sample.traceback)}</pre>
                        </details>
                    ` : ''}
                </div>
            </div>
        `).join('');

        document.getElementById('error-type-display').textContent = errorType;
        container.innerHTML = samplesHtml;
        modal.show();
    }

    onTabChange(target) {
        // Load data when tab is shown
        if (target === '#logs-tab') {
            this.loadLogs();
        } else if (target === '#metrics-tab') {
            this.loadMetrics();
        } else if (target === '#alerts-tab') {
            this.loadAlerts();
        } else if (target === '#errors-tab') {
            this.loadErrors();
        }
    }

    startAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        this.refreshTimer = setInterval(() => {
            if (this.autoRefresh) {
                this.loadDashboard();
            }
        }, this.refreshInterval);
    }

    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    sanitizeId(str) {
        return str.replace(/[^a-zA-Z0-9]/g, '-');
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}

// Helper functions
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.querySelector('.loading-message').textContent = message;
        overlay.classList.add('show');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;

    const bgClass = {
        success: 'bg-success',
        error: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-info',
    }[type] || 'bg-info';

    const toastHtml = `
        <div class="toast align-items-center text-white ${bgClass} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.observabilityDashboard = new ObservabilityDashboard();
});
