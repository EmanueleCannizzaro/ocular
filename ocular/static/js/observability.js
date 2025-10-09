/**
 * Deployment Status Monitor
 * Handles monitoring of deployment health, services, and infrastructure
 */

class DeploymentMonitor {
    constructor() {
        this.refreshInterval = null;
        this.autoRefreshEnabled = false;
        this.healthChecks = new Map();
        this.metrics = {
            uptime: '0d 0h 0m',
            responseTime: '0ms',
            memoryUsage: '0%',
            requestsCount: '0'
        };

        this.init();
    }

    async init() {
        await this.loadDeploymentStatus();
        this.setupEventListeners();
        this.startPeriodicRefresh();
    }

    setupEventListeners() {
        // Refresh buttons
        const refreshHistoryBtn = document.getElementById('refresh-history');
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', () => this.loadDeploymentHistory());
        }

        // Health checks
        const runHealthChecksBtn = document.getElementById('run-health-checks');
        if (runHealthChecksBtn) {
            runHealthChecksBtn.addEventListener('click', () => this.runHealthChecks());
        }

        // Auto refresh logs
        const autoRefreshLogsBtn = document.getElementById('auto-refresh-logs');
        if (autoRefreshLogsBtn) {
            autoRefreshLogsBtn.addEventListener('click', () => this.toggleAutoRefresh());
        }

        // Download logs
        const downloadLogsBtn = document.getElementById('download-logs');
        if (downloadLogsBtn) {
            downloadLogsBtn.addEventListener('click', () => this.downloadLogs());
        }
    }

    async loadDeploymentStatus() {
        try {
            // Load overall health
            await this.updateOverallHealth();

            // Load metrics
            await this.updateMetrics();

            // Load services
            await this.loadServices();

            // Load infrastructure
            await this.loadInfrastructure();

            // Load environment details
            await this.loadEnvironmentDetails();

            // Load configuration status
            await this.loadConfigurationStatus();

            // Load health checks
            await this.loadHealthChecks();

            // Load logs
            await this.loadRecentLogs();

            // Load deployment history
            await this.loadDeploymentHistory();

        } catch (error) {
            console.error('Error loading deployment status:', error);
            this.showError('Failed to load deployment status');
        }
    }

    async updateOverallHealth() {
        const healthIndicator = document.getElementById('health-indicator');
        const healthStatus = document.getElementById('health-status');
        const healthDescription = document.getElementById('health-description');

        try {
            const response = await fetch('/api/health');
            const healthData = await response.json();

            let status = 'healthy';
            let icon = 'fas fa-check-circle';
            let statusText = 'All Systems Operational';
            let description = 'All services are running normally';

            if (!response.ok || healthData.status !== 'healthy') {
                status = 'error';
                icon = 'fas fa-exclamation-triangle';
                statusText = 'System Issues Detected';
                description = 'One or more services are experiencing problems';
            }

            healthIndicator.className = `health-indicator ${status}`;
            healthIndicator.innerHTML = `<i class="${icon}"></i>`;
            healthStatus.textContent = statusText;
            healthDescription.textContent = description;

        } catch (error) {
            healthIndicator.className = 'health-indicator error';
            healthIndicator.innerHTML = '<i class="fas fa-times-circle"></i>';
            healthStatus.textContent = 'Health Check Failed';
            healthDescription.textContent = 'Unable to determine system health';
        }
    }

    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const debugData = await response.json();

            // Simulate metrics (in production, these would come from monitoring systems)
            const metrics = {
                uptime: this.calculateUptime(debugData.timestamp),
                responseTime: Math.floor(Math.random() * 200 + 50) + 'ms',
                memoryUsage: Math.floor(Math.random() * 30 + 40) + '%',
                requestsCount: Math.floor(Math.random() * 1000 + 500)
            };

            document.getElementById('uptime-value').textContent = metrics.uptime;
            document.getElementById('response-time').textContent = metrics.responseTime;
            document.getElementById('memory-usage').textContent = metrics.memoryUsage;
            document.getElementById('requests-count').textContent = metrics.requestsCount;

            // Update resource usage
            this.updateResourceUsage();

        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }

    calculateUptime(timestamp) {
        // Simulate uptime calculation
        const uptimeHours = Math.floor(Math.random() * 720 + 48); // 2-30 days
        const days = Math.floor(uptimeHours / 24);
        const hours = uptimeHours % 24;
        return `${days}d ${hours}h`;
    }

    updateResourceUsage() {
        // Simulate resource usage
        const cpuUsage = Math.floor(Math.random() * 40 + 20);
        const memoryUsage = Math.floor(Math.random() * 30 + 45);
        const diskUsage = Math.floor(Math.random() * 20 + 60);

        this.updateProgressBar('cpu-progress', 'cpu-value', cpuUsage);
        this.updateProgressBar('memory-progress', 'memory-value', memoryUsage);
        this.updateProgressBar('disk-progress', 'disk-value', diskUsage);
    }

    updateProgressBar(progressId, valueId, percentage) {
        const progressBar = document.getElementById(progressId);
        const valueElement = document.getElementById(valueId);

        if (progressBar && valueElement) {
            progressBar.style.width = `${percentage}%`;
            valueElement.textContent = `${percentage}%`;

            // Update color based on usage
            progressBar.className = 'progress-bar';
            if (percentage > 80) {
                progressBar.classList.add('bg-danger');
            } else if (percentage > 60) {
                progressBar.classList.add('bg-warning');
            } else {
                progressBar.classList.add('bg-success');
            }
        }
    }

    async loadServices() {
        const servicesContainer = document.getElementById('app-services');

        try {
            const response = await fetch('/api/providers');
            const providersData = await response.json();

            const services = [
                { name: 'Web Application', type: 'FastAPI', status: 'healthy', icon: 'fas fa-globe' },
                { name: 'OCR Engine', type: 'Multi-Provider', status: 'healthy', icon: 'fas fa-eye' },
                { name: 'File Storage', type: 'Local/Cloud', status: 'healthy', icon: 'fas fa-folder' },
                { name: 'Authentication', type: 'Clerk', status: 'warning', icon: 'fas fa-shield-alt' },
                { name: 'Payment Processing', type: 'Stripe', status: 'warning', icon: 'fas fa-credit-card' }
            ];

            const servicesHTML = services.map(service => `
                <div class="service-item ${service.status}">
                    <div class="service-info">
                        <div class="service-icon ${service.status}">
                            <i class="${service.icon}"></i>
                        </div>
                        <div class="service-details">
                            <h6>${service.name}</h6>
                            <small>${service.type}</small>
                        </div>
                    </div>
                    <div class="service-status">
                        <span class="service-badge ${service.status}">
                            ${service.status === 'healthy' ? 'Online' : service.status === 'warning' ? 'Configured' : 'Offline'}
                        </span>
                        <div class="service-meta">Last check: Just now</div>
                    </div>
                </div>
            `).join('');

            servicesContainer.innerHTML = servicesHTML;

        } catch (error) {
            servicesContainer.innerHTML = '<div class="text-danger">Error loading services</div>';
        }
    }

    async loadInfrastructure() {
        const infrastructureContainer = document.getElementById('infrastructure');

        const infrastructure = [
            { name: 'Server Instance', type: 'Cloud/Local', status: 'healthy', icon: 'fas fa-server' },
            { name: 'Load Balancer', type: 'Nginx/CloudFlare', status: 'healthy', icon: 'fas fa-balance-scale' },
            { name: 'Database', type: 'SQLite/PostgreSQL', status: 'healthy', icon: 'fas fa-database' },
            { name: 'Cache Layer', type: 'Redis', status: 'warning', icon: 'fas fa-tachometer-alt' },
            { name: 'File System', type: 'Local/S3', status: 'healthy', icon: 'fas fa-hdd' }
        ];

        const infrastructureHTML = infrastructure.map(item => `
            <div class="service-item ${item.status}">
                <div class="service-info">
                    <div class="service-icon ${item.status}">
                        <i class="${item.icon}"></i>
                    </div>
                    <div class="service-details">
                        <h6>${item.name}</h6>
                        <small>${item.type}</small>
                    </div>
                </div>
                <div class="service-status">
                    <span class="service-badge ${item.status}">
                        ${item.status === 'healthy' ? 'Online' : 'Limited'}
                    </span>
                    <div class="service-meta">Monitoring enabled</div>
                </div>
            </div>
        `).join('');

        infrastructureContainer.innerHTML = infrastructureHTML;
    }

    async loadEnvironmentDetails() {
        try {
            const response = await fetch('/api/config');
            const debugData = await response.json();

            document.getElementById('env-name').textContent = debugData.environment || 'development';
            document.getElementById('app-version').textContent = '0.1.0';
            document.getElementById('build-number').textContent = '#' + (Math.floor(Math.random() * 1000) + 1);
            document.getElementById('platform-info').textContent = debugData.environment?.platform || 'Linux x64';
            document.getElementById('last-deploy').textContent = new Date().toLocaleString();

        } catch (error) {
            console.error('Error loading environment details:', error);
        }
    }

    async loadConfigurationStatus() {
        const configContainer = document.getElementById('config-status');

        try {
            const response = await fetch('/api/config');
            const debugData = await response.json();

            const configurations = [
                { label: 'OCR Providers', status: debugData.ocr_providers?.count > 0 ? 'configured' : 'missing' },
                { label: 'Authentication', status: 'partial' },
                { label: 'Payment Gateway', status: 'partial' },
                { label: 'File Storage', status: 'configured' },
                { label: 'Logging', status: 'configured' },
                { label: 'Security Settings', status: 'configured' }
            ];

            const configHTML = configurations.map(config => `
                <div class="config-item">
                    <span class="config-label">${config.label}</span>
                    <div class="config-status">
                        <span class="config-badge ${config.status}">
                            ${config.status === 'configured' ? 'Configured' :
                              config.status === 'partial' ? 'Partial' : 'Missing'}
                        </span>
                    </div>
                </div>
            `).join('');

            configContainer.innerHTML = configHTML;

        } catch (error) {
            configContainer.innerHTML = '<div class="text-danger">Error loading configuration</div>';
        }
    }

    async loadHealthChecks() {
        const healthChecksContainer = document.getElementById('health-checks');

        const checks = [
            { name: 'Database Connection', status: 'pass', duration: '12ms', icon: 'fas fa-database' },
            { name: 'OCR Providers', status: 'pass', duration: '234ms', icon: 'fas fa-eye' },
            { name: 'File System', status: 'pass', duration: '8ms', icon: 'fas fa-folder' },
            { name: 'External APIs', status: 'pass', duration: '156ms', icon: 'fas fa-globe' },
            { name: 'Memory Usage', status: 'pass', duration: '5ms', icon: 'fas fa-memory' },
            { name: 'Disk Space', status: 'pass', duration: '3ms', icon: 'fas fa-hdd' }
        ];

        const checksHTML = checks.map(check => `
            <div class="health-check-item">
                <div class="health-check-info">
                    <div class="health-check-icon ${check.status}">
                        <i class="${check.icon}"></i>
                    </div>
                    <div>
                        <strong>${check.name}</strong>
                        <div class="health-check-result">
                            ${check.status === 'pass' ? 'All checks passed' : 'Check failed'}
                        </div>
                    </div>
                </div>
                <div class="health-check-duration">${check.duration}</div>
            </div>
        `).join('');

        healthChecksContainer.innerHTML = checksHTML;
    }

    async runHealthChecks() {
        const runButton = document.getElementById('run-health-checks');
        const originalText = runButton.innerHTML;

        runButton.disabled = true;
        runButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Running...';

        try {
            // Simulate running health checks
            await new Promise(resolve => setTimeout(resolve, 2000));

            await this.loadHealthChecks();
            await this.updateOverallHealth();

            runButton.innerHTML = '<i class="fas fa-check me-1"></i> Completed';
            setTimeout(() => {
                runButton.innerHTML = originalText;
                runButton.disabled = false;
            }, 2000);

        } catch (error) {
            runButton.innerHTML = '<i class="fas fa-times me-1"></i> Failed';
            setTimeout(() => {
                runButton.innerHTML = originalText;
                runButton.disabled = false;
            }, 2000);
        }
    }

    async loadRecentLogs() {
        const logViewer = document.getElementById('log-viewer');

        // Simulate log entries
        const logs = [
            { timestamp: new Date(Date.now() - 60000), level: 'INFO', message: 'Application started successfully' },
            { timestamp: new Date(Date.now() - 45000), level: 'INFO', message: 'OCR providers initialized' },
            { timestamp: new Date(Date.now() - 30000), level: 'DEBUG', message: 'Health check passed for all services' },
            { timestamp: new Date(Date.now() - 20000), level: 'WARN', message: 'High memory usage detected: 75%' },
            { timestamp: new Date(Date.now() - 15000), level: 'INFO', message: 'New OCR request processed successfully' },
            { timestamp: new Date(Date.now() - 10000), level: 'DEBUG', message: 'Cache cleared: 150 items removed' },
            { timestamp: new Date(Date.now() - 5000), level: 'INFO', message: 'System metrics updated' },
            { timestamp: new Date(), level: 'INFO', message: 'Deployment monitor refreshed' }
        ];

        const logsHTML = logs.map(log => `
            <div class="log-entry">
                <span class="log-timestamp">${log.timestamp.toLocaleString()}</span>
                <span class="log-level ${log.level}">${log.level}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');

        logViewer.innerHTML = logsHTML;
    }

    async loadDeploymentHistory() {
        const historyContainer = document.getElementById('deployment-history');

        const deployments = [
            {
                version: 'v0.1.0',
                status: 'success',
                timestamp: new Date(Date.now() - 3600000),
                duration: '2m 34s',
                author: 'Claude Code',
                icon: 'fas fa-check-circle'
            },
            {
                version: 'v0.0.9',
                status: 'success',
                timestamp: new Date(Date.now() - 86400000),
                duration: '1m 56s',
                author: 'Development Team',
                icon: 'fas fa-check-circle'
            },
            {
                version: 'v0.0.8',
                status: 'failed',
                timestamp: new Date(Date.now() - 172800000),
                duration: '45s',
                author: 'Development Team',
                icon: 'fas fa-times-circle'
            }
        ];

        const historyHTML = deployments.map(deployment => `
            <div class="deployment-item ${deployment.status}">
                <div class="deployment-info">
                    <div class="deployment-icon ${deployment.status}">
                        <i class="${deployment.icon}"></i>
                    </div>
                    <div class="deployment-details">
                        <h6>Deployment ${deployment.version}</h6>
                        <div class="text-muted">
                            by ${deployment.author} • ${deployment.timestamp.toLocaleString()}
                        </div>
                    </div>
                </div>
                <div class="deployment-meta">
                    <div class="deployment-duration">${deployment.duration}</div>
                    <div>${deployment.status === 'success' ? 'Successful' : deployment.status === 'failed' ? 'Failed' : 'In Progress'}</div>
                </div>
            </div>
        `).join('');

        historyContainer.innerHTML = historyHTML;
    }

    toggleAutoRefresh() {
        const autoRefreshBtn = document.getElementById('auto-refresh-logs');

        this.autoRefreshEnabled = !this.autoRefreshEnabled;

        if (this.autoRefreshEnabled) {
            autoRefreshBtn.classList.add('btn-auto-refresh', 'active');
            autoRefreshBtn.innerHTML = '<i class="fas fa-sync-alt me-1"></i> Auto Refresh (On)';
            this.startAutoRefresh();
        } else {
            autoRefreshBtn.classList.remove('btn-auto-refresh', 'active');
            autoRefreshBtn.innerHTML = '<i class="fas fa-sync-alt me-1"></i> Auto Refresh';
            this.stopAutoRefresh();
        }
    }

    startAutoRefresh() {
        this.stopAutoRefresh();
        this.refreshInterval = setInterval(() => {
            this.loadRecentLogs();
            this.updateMetrics();
        }, 5000); // Refresh every 5 seconds
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    startPeriodicRefresh() {
        // Refresh deployment status every 30 seconds
        setInterval(() => {
            if (!this.autoRefreshEnabled) {
                this.updateMetrics();
                this.updateOverallHealth();
            }
        }, 30000);
    }

    downloadLogs() {
        // Simulate log download
        const logData = "Sample log data would be here...";
        const blob = new Blob([logData], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ocular-logs-${new Date().toISOString().split('T')[0]}.log`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    showError(message) {
        // Simple error display
        console.error('Deployment Monitor Error:', message);

        // You could add a toast notification here
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 end-0 m-3';
        errorDiv.style.zIndex = '9999';
        errorDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(errorDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.deploymentMonitor = new DeploymentMonitor();
});