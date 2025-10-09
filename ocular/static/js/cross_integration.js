/**
 * Cross-Template Integration Utilities
 * Manages state, navigation, and workflow continuity across templates
 */

class CrossIntegrationManager {
    constructor() {
        this.STORAGE_PREFIX = 'ocular_';
        this.WORKFLOW_KEY = 'current_workflow';
        this.TAB_STATE_KEY = 'tab_states';
        this.RECENT_DOCUMENTS_KEY = 'recent_documents';
        this.init();
    }

    init() {
        this.setupNavigationTracking();
        this.setupWorkflowPersistence();
        this.setupQuickActions();
        this.setupContextualHelp();
        this.highlightActiveNav();
    }

    // ========================================================================
    // Navigation Tracking
    // ========================================================================

    setupNavigationTracking() {
        // Track page views for analytics
        this.recordPageView();

        // Setup navigation event listeners
        document.addEventListener('click', (e) => {
            const navLink = e.target.closest('a[href^="/"]');
            if (navLink) {
                this.recordNavigation(navLink.href, navLink.textContent.trim());
            }
        });
    }

    highlightActiveNav() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.sidebar .nav-link, .navbar .nav-link');

        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPath || (currentPath !== '/' && href !== '/' && currentPath.startsWith(href))) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'page');

                // Scroll sidebar to active item
                if (link.closest('.sidebar')) {
                    link.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            } else {
                link.classList.remove('active');
                link.removeAttribute('aria-current');
            }
        });
    }

    recordPageView() {
        const pageViews = this.getStorage('page_views') || {};
        const currentPage = window.location.pathname;

        pageViews[currentPage] = {
            count: (pageViews[currentPage]?.count || 0) + 1,
            lastVisited: new Date().toISOString(),
            title: document.title
        };

        this.setStorage('page_views', pageViews);
    }

    recordNavigation(url, label) {
        const navHistory = this.getStorage('nav_history') || [];
        navHistory.unshift({
            url,
            label,
            timestamp: new Date().toISOString(),
            from: window.location.pathname
        });

        // Keep only last 50 navigations
        if (navHistory.length > 50) {
            navHistory.pop();
        }

        this.setStorage('nav_history', navHistory);
    }

    // ========================================================================
    // Workflow Persistence
    // ========================================================================

    setupWorkflowPersistence() {
        // Restore workflow state on page load
        this.restoreWorkflowState();

        // Save workflow state before page unload
        window.addEventListener('beforeunload', () => {
            this.saveWorkflowState();
        });

        // Auto-save every 30 seconds
        setInterval(() => this.saveWorkflowState(), 30000);
    }

    saveWorkflowState() {
        const workflow = {
            page: window.location.pathname,
            timestamp: new Date().toISOString(),
            formData: this.extractFormData(),
            scrollPosition: window.scrollY,
            activeTab: this.getActiveTab(),
            selectedProvider: this.getSelectedProvider(),
            uploadedFiles: this.getUploadedFilesInfo()
        };

        this.setStorage(this.WORKFLOW_KEY, workflow);
    }

    restoreWorkflowState() {
        const workflow = this.getStorage(this.WORKFLOW_KEY);
        if (!workflow) return;

        // Only restore if on same page and recent (within 1 hour)
        const isRecent = (new Date() - new Date(workflow.timestamp)) < 3600000;
        if (workflow.page !== window.location.pathname || !isRecent) return;

        // Restore scroll position
        if (workflow.scrollPosition) {
            window.scrollTo(0, workflow.scrollPosition);
        }

        // Restore active tab
        if (workflow.activeTab) {
            this.activateTab(workflow.activeTab);
        }

        // Restore form data
        if (workflow.formData) {
            this.populateFormData(workflow.formData);
        }

        // Show restore notification
        this.showNotification('Workflow restored from previous session', 'info');
    }

    extractFormData() {
        const forms = document.querySelectorAll('form');
        const formData = {};

        forms.forEach((form, index) => {
            const data = {};
            new FormData(form).forEach((value, key) => {
                // Skip file inputs and sensitive data
                if (!key.includes('password') && !key.includes('file')) {
                    data[key] = value;
                }
            });
            formData[`form_${index}`] = data;
        });

        return formData;
    }

    populateFormData(formData) {
        Object.entries(formData).forEach(([formId, data]) => {
            const formIndex = parseInt(formId.split('_')[1]);
            const form = document.querySelectorAll('form')[formIndex];

            if (form) {
                Object.entries(data).forEach(([key, value]) => {
                    const input = form.querySelector(`[name="${key}"]`);
                    if (input && input.type !== 'file') {
                        input.value = value;
                    }
                });
            }
        });
    }

    // ========================================================================
    // Tab State Management
    // ========================================================================

    getActiveTab() {
        const activeTab = document.querySelector('.nav-link.active[data-bs-toggle="tab"]');
        return activeTab ? activeTab.id : null;
    }

    activateTab(tabId) {
        const tab = document.querySelector(`#${tabId}`);
        if (tab) {
            const bootstrapTab = new bootstrap.Tab(tab);
            bootstrapTab.show();
        }
    }

    saveTabState(tabId) {
        const tabStates = this.getStorage(this.TAB_STATE_KEY) || {};
        tabStates[window.location.pathname] = tabId;
        this.setStorage(this.TAB_STATE_KEY, tabStates);
    }

    // ========================================================================
    // Quick Actions
    // ========================================================================

    setupQuickActions() {
        this.renderQuickActionsMenu();
        this.setupKeyboardShortcuts();
    }

    renderQuickActionsMenu() {
        const quickActionsData = this.getContextualQuickActions();

        // Create quick actions button if not exists
        if (!document.getElementById('quick-actions-btn') && quickActionsData.length > 0) {
            const button = document.createElement('button');
            button.id = 'quick-actions-btn';
            button.className = 'btn btn-primary btn-floating position-fixed';
            button.style.cssText = 'bottom: 20px; right: 20px; z-index: 1000; border-radius: 50%; width: 56px; height: 56px;';
            button.innerHTML = '<i class="fas fa-bolt"></i>';
            button.title = 'Quick Actions (Alt+Q)';

            button.addEventListener('click', () => this.showQuickActionsModal());
            document.body.appendChild(button);
        }
    }

    getContextualQuickActions() {
        const currentPage = window.location.pathname;
        const actions = {
            '/': [
                { icon: 'fas fa-tasks', label: 'Start Batch Processing', url: '/batch', color: 'primary' },
                { icon: 'fas fa-history', label: 'View Recent Documents', url: '/document-history', color: 'info' },
                { icon: 'fas fa-rocket', label: 'Deploy New Model', url: '/deployments', color: 'success' }
            ],
            '/batch': [
                { icon: 'fas fa-upload', label: 'Quick Upload', url: '/', color: 'primary' },
                { icon: 'fas fa-chart-line', label: 'View Analytics', url: '/analytics', color: 'info' },
                { icon: 'fas fa-download', label: 'Export Results', action: 'exportBatchResults', color: 'success' }
            ],
            '/deployments': [
                { icon: 'fas fa-plus', label: 'Create Deployment', action: 'openDeploymentModal', color: 'success' },
                { icon: 'fas fa-cube', label: 'Manage Models', url: '/models', color: 'primary' },
                { icon: 'fas fa-eye', label: 'Monitor Health', url: '/observability', color: 'warning' }
            ],
            '/analytics': [
                { icon: 'fas fa-download', label: 'Export Report', action: 'exportAnalytics', color: 'success' },
                { icon: 'fas fa-calendar', label: 'Custom Date Range', action: 'selectDateRange', color: 'info' },
                { icon: 'fas fa-share', label: 'Share Dashboard', action: 'shareDashboard', color: 'primary' }
            ]
        };

        return actions[currentPage] || [];
    }

    showQuickActionsModal() {
        const actions = this.getContextualQuickActions();
        if (actions.length === 0) return;

        let html = `
            <div class="modal fade" id="quickActionsModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"><i class="fas fa-bolt me-2"></i>Quick Actions</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="list-group">
        `;

        actions.forEach(action => {
            const onclick = action.url ? `window.location.href='${action.url}'` : action.action ? `${action.action}()` : '';
            html += `
                <a href="${action.url || '#'}" ${!action.url ? `onclick="${onclick}; return false;"` : ''}
                   class="list-group-item list-group-item-action d-flex align-items-center">
                    <div class="icon-circle bg-${action.color} bg-opacity-10 text-${action.color} p-2 rounded-circle me-3">
                        <i class="${action.icon}"></i>
                    </div>
                    <span>${action.label}</span>
                </a>
            `;
        });

        html += `
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal
        const existingModal = document.getElementById('quickActionsModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add new modal
        document.body.insertAdjacentHTML('beforeend', html);
        const modal = new bootstrap.Modal(document.getElementById('quickActionsModal'));
        modal.show();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Alt+Q: Quick Actions
            if (e.altKey && e.key === 'q') {
                e.preventDefault();
                this.showQuickActionsModal();
            }

            // Alt+H: Toggle Help
            if (e.altKey && e.key === 'h') {
                e.preventDefault();
                this.toggleContextualHelp();
            }

            // Alt+N: Show Navigation History
            if (e.altKey && e.key === 'n') {
                e.preventDefault();
                this.showNavigationHistory();
            }
        });
    }

    // ========================================================================
    // Contextual Help
    // ========================================================================

    setupContextualHelp() {
        // Add help icon to page
        this.addHelpIcon();
    }

    addHelpIcon() {
        if (document.getElementById('contextual-help-btn')) return;

        const button = document.createElement('button');
        button.id = 'contextual-help-btn';
        button.className = 'btn btn-info btn-sm position-fixed';
        button.style.cssText = 'bottom: 90px; right: 20px; z-index: 1000; border-radius: 50%; width: 45px; height: 45px;';
        button.innerHTML = '<i class="fas fa-question"></i>';
        button.title = 'Contextual Help (Alt+H)';

        button.addEventListener('click', () => this.toggleContextualHelp());
        document.body.appendChild(button);
    }

    toggleContextualHelp() {
        let helpPanel = document.getElementById('contextual-help-panel');

        if (helpPanel) {
            helpPanel.remove();
            return;
        }

        const tips = this.getContextualTips();
        if (tips.length === 0) return;

        helpPanel = document.createElement('div');
        helpPanel.id = 'contextual-help-panel';
        helpPanel.className = 'card shadow-lg position-fixed';
        helpPanel.style.cssText = 'bottom: 140px; right: 20px; z-index: 1000; max-width: 350px;';

        helpPanel.innerHTML = `
            <div class="card-header bg-info text-white d-flex justify-content-between align-items-center">
                <span><i class="fas fa-lightbulb me-2"></i>Quick Tips</span>
                <button class="btn btn-sm btn-link text-white" onclick="this.closest('#contextual-help-panel').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="card-body">
                <ul class="list-unstyled mb-0">
                    ${tips.map(tip => `<li class="mb-2"><i class="fas fa-check-circle text-success me-2"></i>${tip}</li>`).join('')}
                </ul>
            </div>
        `;

        document.body.appendChild(helpPanel);
    }

    getContextualTips() {
        const currentPage = window.location.pathname;
        const tips = {
            '/': [
                'Drag and drop multiple files for bulk processing',
                'Use the fallback strategy for maximum reliability',
                'Add custom prompts to extract specific data formats'
            ],
            '/batch': [
                'Schedule batch jobs during off-peak hours for faster processing',
                'Use filters to quickly find specific document types',
                'Export results as CSV or JSON for further analysis'
            ],
            '/deployments': [
                'Test deployments with sample documents before production',
                'Monitor GPU usage to optimize costs',
                'Use auto-scaling for variable workloads'
            ],
            '/analytics': [
                'Compare provider performance to optimize costs',
                'Track confidence scores to identify problematic documents',
                'Set up alerts for processing anomalies'
            ],
            '/providers': [
                'Configure fallback chains for maximum uptime',
                'Test each provider with sample documents',
                'Monitor API quotas to avoid service interruptions'
            ]
        };

        return tips[currentPage] || [];
    }

    // ========================================================================
    // Recent Documents
    // ========================================================================

    addRecentDocument(doc) {
        const recentDocs = this.getStorage(this.RECENT_DOCUMENTS_KEY) || [];

        recentDocs.unshift({
            id: doc.id || Date.now(),
            name: doc.name,
            type: doc.type,
            timestamp: new Date().toISOString(),
            url: doc.url,
            thumbnail: doc.thumbnail
        });

        // Keep only last 20 documents
        if (recentDocs.length > 20) {
            recentDocs.pop();
        }

        this.setStorage(this.RECENT_DOCUMENTS_KEY, recentDocs);
    }

    getRecentDocuments() {
        return this.getStorage(this.RECENT_DOCUMENTS_KEY) || [];
    }

    // ========================================================================
    // Navigation History
    // ========================================================================

    showNavigationHistory() {
        const history = this.getStorage('nav_history') || [];
        if (history.length === 0) {
            this.showNotification('No navigation history available', 'info');
            return;
        }

        let html = `
            <div class="modal fade" id="navHistoryModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"><i class="fas fa-history me-2"></i>Navigation History</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="list-group">
        `;

        history.slice(0, 20).forEach(item => {
            const timeAgo = this.getTimeAgo(new Date(item.timestamp));
            html += `
                <a href="${item.url}" class="list-group-item list-group-item-action">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${item.label}</strong>
                            <br><small class="text-muted">${item.url}</small>
                        </div>
                        <small class="text-muted">${timeAgo}</small>
                    </div>
                </a>
            `;
        });

        html += `
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const existingModal = document.getElementById('navHistoryModal');
        if (existingModal) {
            existingModal.remove();
        }

        document.body.insertAdjacentHTML('beforeend', html);
        const modal = new bootstrap.Modal(document.getElementById('navHistoryModal'));
        modal.show();
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    getSelectedProvider() {
        const providerSelect = document.querySelector('select[name="provider"], #provider');
        return providerSelect ? providerSelect.value : null;
    }

    getUploadedFilesInfo() {
        const fileList = document.getElementById('fileList');
        if (!fileList) return null;

        const files = [];
        fileList.querySelectorAll('.file-item').forEach(item => {
            files.push({
                name: item.dataset.filename,
                size: item.dataset.filesize
            });
        });

        return files.length > 0 ? files : null;
    }

    showNotification(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    getTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);
        const intervals = {
            year: 31536000,
            month: 2592000,
            week: 604800,
            day: 86400,
            hour: 3600,
            minute: 60
        };

        for (const [unit, secondsInUnit] of Object.entries(intervals)) {
            const interval = Math.floor(seconds / secondsInUnit);
            if (interval >= 1) {
                return `${interval} ${unit}${interval !== 1 ? 's' : ''} ago`;
            }
        }

        return 'just now';
    }

    // Storage helpers
    setStorage(key, value) {
        try {
            localStorage.setItem(this.STORAGE_PREFIX + key, JSON.stringify(value));
        } catch (e) {
            console.error('Storage error:', e);
        }
    }

    getStorage(key) {
        try {
            const item = localStorage.getItem(this.STORAGE_PREFIX + key);
            return item ? JSON.parse(item) : null;
        } catch (e) {
            console.error('Storage error:', e);
            return null;
        }
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.crossIntegration = new CrossIntegrationManager();
    });
} else {
    window.crossIntegration = new CrossIntegrationManager();
}

// Expose global functions for inline onclick handlers
window.saveTabState = (tabId) => window.crossIntegration.saveTabState(tabId);
window.addRecentDocument = (doc) => window.crossIntegration.addRecentDocument(doc);
