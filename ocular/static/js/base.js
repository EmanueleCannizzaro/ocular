// Main JavaScript for Ocular OCR Service

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    initializeUploadDashboard();
    initializeDocumentTracking();
    initializeBatchProcessing();
    initializeBillingDashboard();
    initializeOlmocrProcessor();
    initializeUserProfilePage();
}

function initializeUploadDashboard() {
    const dashboard = document.getElementById('upload-dashboard');
    if (!dashboard) {
        return;
    }

    const fileUploadArea = dashboard.querySelector('#fileUploadArea');
    const fileInput = dashboard.querySelector('#fileInput');
    const fileList = dashboard.querySelector('#fileList');
    const uploadForm = dashboard.querySelector('#uploadForm');
    const processingOverlay = dashboard.querySelector('#processingOverlay');
    const resultsContainer = dashboard.querySelector('#resultsContainer');
    const resultsSection = dashboard.querySelector('#results');

    const fileHandler = new FileHandler();
    let selectedFiles = [];

    if (fileUploadArea && fileInput) {
        fileUploadArea.addEventListener('click', () => fileInput.click());

        fileUploadArea.addEventListener('dragover', (event) => {
            event.preventDefault();
            fileUploadArea.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', (event) => {
            event.preventDefault();
            fileUploadArea.classList.remove('dragover');
            handleFiles(event.dataTransfer.files);
        });

        fileInput.addEventListener('change', (event) => {
            handleFiles(event.target.files);
        });
    }

    dashboard.querySelectorAll('.provider-card').forEach((card) => {
        card.addEventListener('click', () => {
            const checkbox = card.querySelector('.provider-checkbox');
            checkbox.checked = !checkbox.checked;
            card.classList.toggle('selected', checkbox.checked);
        });
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        displayFileList();
    }

    function displayFileList() {
        if (!fileList) {
            return;
        }

        if (!selectedFiles.length) {
            fileList.innerHTML = '';
            return;
        }

        const listHTML = selectedFiles
            .map((file, index) => `
                <div class="file-item">
                    <i class="fas fa-file file-icon"></i>
                    <span class="file-name">${escapeHtml(file.name)}</span>
                    <span class="file-size">${fileHandler.formatFileSize(file.size)}</span>
                    <button type="button" class="remove-file" title="Remove file" data-index="${index}">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `)
            .join('');

        fileList.innerHTML = `<h6 class="mt-3">${escapeHtml('Selected Files:')}</h6>${listHTML}`;

        fileList.querySelectorAll('.remove-file').forEach((button) => {
            button.addEventListener('click', (event) => {
                const index = Number(event.currentTarget.getAttribute('data-index'));
                selectedFiles.splice(index, 1);
                displayFileList();
            });
        });
    }

    window.removeFile = function removeFile(index) {
        selectedFiles.splice(index, 1);
        displayFileList();
    };

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            if (!selectedFiles.length) {
                showAlert('Please select at least one file', 'warning');
                return;
            }

            const selectedProviders = Array.from(
                dashboard.querySelectorAll('.provider-checkbox:checked')
            ).map((checkbox) => checkbox.value);

            if (!selectedProviders.length) {
                showAlert('Please select at least one OCR provider', 'warning');
                return;
            }

            showProcessingOverlay(true);

            const formData = new FormData();
            selectedFiles.forEach((file) => formData.append('files', file));
            formData.append('strategy', dashboard.querySelector('#strategy')?.value || 'fallback');
            formData.append('providers', selectedProviders.join(','));
            formData.append('prompt', dashboard.querySelector('#prompt')?.value || '');

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const payload = await response.json();
                displayResults(payload);
            } catch (error) {
                console.error('Processing error:', error);
                showAlert(`Processing failed: ${error.message || error}`, 'danger');
            } finally {
                showProcessingOverlay(false);
            }
        });
    }

    function showProcessingOverlay(show) {
        if (processingOverlay) {
            processingOverlay.style.display = show ? 'flex' : 'none';
        }
    }

    function displayResults(result) {
        if (!resultsContainer || !resultsSection) {
            return;
        }

        if (!result || (!result.success && !result.results?.length)) {
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <h5><i class="fas fa-exclamation-circle me-2"></i>${escapeHtml('Processing Failed')}</h5>
                    <p class="mb-0">${escapeHtml('All files failed to process. Please check your files and try again.')}</p>
                </div>
            `;
        } else {
            const cards = (result.results || []).map((entry, index) => {
                if (entry.success) {
                    const metadata = entry.metadata
                        ? `
                            <div class="mt-3">
                                <button class="btn btn-sm btn-outline-secondary" type="button"
                                        data-bs-toggle="collapse" data-bs-target="#metadata${index}">
                                    <i class="fas fa-info-circle me-1"></i>${escapeHtml('Show Metadata')}
                                </button>
                                <div class="collapse mt-2" id="metadata${index}">
                                    <div class="bg-light p-2 rounded">
                                        <pre class="mb-0 small">${escapeHtml(JSON.stringify(entry.metadata, null, 2))}</pre>
                                    </div>
                                </div>
                            </div>
                        `
                        : '';

                    return `
                        <div class="result-card card">
                            <div class="card-header">
                                <h6 class="mb-0">
                                    <i class="fas fa-file-alt me-2"></i>${escapeHtml(entry.filename || 'Unknown file')}
                                </h6>
                                <small class="text-muted">
                                    ${escapeHtml(
                                        entry.provider_used
                                            ? `Processed with ${entry.provider_used} in ${(entry.processing_time || 0).toFixed(2)}s`
                                            : 'Processing complete'
                                    )}
                                </small>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <strong>${escapeHtml('Extracted Text:')}</strong>
                                    <div class="bg-light p-3 rounded mt-2" style="max-height: 300px; overflow-y: auto;">
                                        <pre class="mb-0">${escapeHtml(entry.extracted_text || 'No text extracted')}</pre>
                                    </div>
                                </div>
                                ${metadata}
                            </div>
                        </div>
                    `;
                }

                return `
                    <div class="error-card card">
                        <div class="card-header">
                            <h6 class="mb-0">
                                <i class="fas fa-exclamation-triangle me-2"></i>${escapeHtml(entry.filename || 'Unknown file')}
                            </h6>
                        </div>
                        <div class="card-body">
                            <div class="alert alert-danger mb-0">
                                ${escapeHtml(entry.error || 'Unknown error occurred')}
                            </div>
                        </div>
                    </div>
                `;
            });

            resultsContainer.innerHTML = cards.join('');
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function initializeDocumentTracking() {
    const page = document.getElementById('document-tracking-page');
    if (!page) {
        return;
    }

    const API_BASE_URL = window.location.origin;
    let currentPage = 1;
    let documentsData = [];

    async function loadStatistics() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/document-tracking/stats`);
            const stats = await response.json();

            document.getElementById('total-documents').textContent = stats.total_documents || 0;
            document.getElementById('unique-documents').textContent = stats.unique_documents || 0;
            document.getElementById('duplicate-count').textContent = stats.duplicate_count || 0;
            document.getElementById('cost-saved').textContent = `$${(stats.estimated_cost_saved || 0).toFixed(2)}`;
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    }

    async function loadDocuments(pageNumber = 1) {
        try {
            currentPage = pageNumber;
            const filters = getFilters();
            const queryParams = new URLSearchParams({
                limit: 20,
                offset: (pageNumber - 1) * 20,
                ...filters
            });

            const response = await fetch(`${API_BASE_URL}/api/document-tracking/processed-documents?${queryParams}`);
            documentsData = await response.json();

            renderDocumentsTable(documentsData);
            updatePagination(documentsData.length);
        } catch (error) {
            console.error('Error loading documents:', error);
            document.getElementById('documents-table-body').innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-danger py-4">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Failed to load documents
                    </td>
                </tr>
            `;
        }
    }

    function renderDocumentsTable(documents) {
        const tbody = document.getElementById('documents-table-body');

        if (documents.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-muted py-4">
                        <i class="fas fa-inbox me-2"></i>
                        No documents found
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = documents.map(doc => `
            <tr>
                <td>
                    <i class="fas fa-file-${getFileIcon(doc.mime_type)} me-2 text-primary"></i>
                    ${doc.file_name || 'Unknown'}
                    ${doc.processing_count > 1 ? `<span class="badge bg-warning duplicate-badge ms-2">Duplicate x${doc.processing_count}</span>` : ''}
                </td>
                <td>
                    <span class="hash-text" title="${doc.file_hash}">
                        ${doc.file_hash.substring(0, 12)}...
                    </span>
                </td>
                <td>${new Date(doc.first_processed_at).toLocaleString()}</td>
                <td>
                    <span class="badge bg-${doc.processing_count > 1 ? 'warning' : 'primary'}">
                        ${doc.processing_count}
                    </span>
                </td>
                <td>
                    <span class="badge bg-info">${doc.provider_used || 'N/A'}</span>
                </td>
                <td>$${(doc.cost_usd || 0).toFixed(4)}</td>
                <td>
                    <span class="badge bg-${doc.processing_status === 'completed' ? 'success' : 'secondary'}">
                        ${doc.processing_status}
                    </span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="showDocumentDetails(${doc.id})">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        document.getElementById('showing-count').textContent = documents.length;
        document.getElementById('total-count').textContent = documents.length;
    }

    function getFileIcon(mimeType) {
        if (!mimeType) return 'file';
        if (mimeType.includes('pdf')) return 'pdf';
        if (mimeType.includes('image')) return 'image';
        return 'file';
    }

    async function showDocumentDetails(docId) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/document-tracking/processed-documents/${docId}`);
            const doc = await response.json();

            const content = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Document Information</h6>
                        <dl class="row">
                            <dt class="col-sm-5">File Name:</dt>
                            <dd class="col-sm-7">${doc.file_name}</dd>

                            <dt class="col-sm-5">File Size:</dt>
                            <dd class="col-sm-7">${formatFileSize(doc.file_size_bytes)}</dd>

                            <dt class="col-sm-5">MIME Type:</dt>
                            <dd class="col-sm-7">${doc.mime_type}</dd>

                            <dt class="col-sm-5">SHA-256 Hash:</dt>
                            <dd class="col-sm-7"><code class="small">${doc.file_hash}</code></dd>
                        </dl>
                    </div>
                    <div class="col-md-6">
                        <h6>Processing Information</h6>
                        <dl class="row">
                            <dt class="col-sm-6">First Processed:</dt>
                            <dd class="col-sm-6">${new Date(doc.first_processed_at).toLocaleString()}</dd>

                            <dt class="col-sm-6">Last Processed:</dt>
                            <dd class="col-sm-6">${new Date(doc.last_processed_at).toLocaleString()}</dd>

                            <dt class="col-sm-6">Times Processed:</dt>
                            <dd class="col-sm-6">${doc.processing_count}</dd>

                            <dt class="col-sm-6">Provider:</dt>
                            <dd class="col-sm-6">${doc.provider_used}</dd>

                            <dt class="col-sm-6">Processing Time:</dt>
                            <dd class="col-sm-6">${doc.processing_time_seconds}s</dd>

                            <dt class="col-sm-6">Total Cost:</dt>
                            <dd class="col-sm-6">$${(doc.cost_usd || 0).toFixed(4)}</dd>
                        </dl>
                    </div>
                </div>
                <hr>
                <h6>Extracted Text Preview</h6>
                <div class="card bg-light">
                    <div class="card-body">
                        <pre class="mb-0 small">${doc.extracted_text_preview || 'No text preview available'}</pre>
                    </div>
                </div>
            `;

            document.getElementById('document-details-content').innerHTML = content;
            new bootstrap.Modal(document.getElementById('documentDetailsModal')).show();
        } catch (error) {
            console.error('Error loading document details:', error);
            alert('Failed to load document details');
        }
    }

    window.showDocumentDetails = showDocumentDetails;

    async function checkForDuplicate() {
        const fileInput = document.getElementById('duplicate-check-file');
        const file = fileInput.files[0];

        if (!file) {
            alert('Please select a file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/api/document-tracking/check-duplicate`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            const resultDiv = document.getElementById('duplicate-result');
            const contentDiv = document.getElementById('duplicate-result-content');

            if (result.is_duplicate) {
                const existing = result.existing_record;
                contentDiv.innerHTML = `
                    <p><strong>⚠️ This document has been processed before!</strong></p>
                    <ul class="mb-0">
                        <li>First processed: ${new Date(existing.first_processed_at).toLocaleString()}</li>
                        <li>Times processed: ${existing.processing_count}</li>
                        <li>Provider used: ${existing.provider_used}</li>
                        <li>Estimated cost saved: $${(existing.cost_usd || 0).toFixed(4)}</li>
                    </ul>
                `;
                resultDiv.className = 'alert alert-warning';
            } else {
                contentDiv.innerHTML = `
                    <p><strong>✅ This is a new document!</strong></p>
                    <p class="mb-0">This file has not been processed before.</p>
                `;
                resultDiv.className = 'alert alert-success';
            }

            resultDiv.style.display = 'block';
        } catch (error) {
            console.error('Error checking for duplicate:', error);
            alert('Failed to check for duplicate');
        }
    }

    window.checkForDuplicate = checkForDuplicate;

    function getFilters() {
        return {
            file_name: document.getElementById('filter-filename').value,
            provider: document.getElementById('filter-provider').value,
            date_from: document.getElementById('filter-date-from').value,
            date_to: document.getElementById('filter-date-to').value
        };
    }

    function applyFilters() {
        loadDocuments(1);
    }

    window.applyFilters = applyFilters;

    function clearFilters() {
        document.getElementById('filter-filename').value = '';
        document.getElementById('filter-provider').value = '';
        document.getElementById('filter-date-from').value = '';
        document.getElementById('filter-date-to').value = '';
        loadDocuments(1);
    }

    window.clearFilters = clearFilters;

    function refreshDocuments() {
        loadStatistics();
        loadDocuments(currentPage);
    }

    window.refreshDocuments = refreshDocuments;

    async function cleanupOldDocuments() {
        if (!confirm('Delete documents older than 90 days?')) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/document-tracking/cleanup?days_old=90`, {
                method: 'DELETE'
            });
            const result = await response.json();
            alert(`Cleaned up ${result.deleted_count} old documents`);
            refreshDocuments();
        } catch (error) {
            console.error('Error cleaning up documents:', error);
            alert('Failed to cleanup old documents');
        }
    }

    window.cleanupOldDocuments = cleanupOldDocuments;

    function updatePagination(totalItems) {
        const totalPages = Math.ceil(totalItems / 20);
        const pagination = document.getElementById('pagination');

        let html = '';
        for (let i = 1; i <= totalPages; i++) {
            html += `
                <li class="page-item ${i === currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a>
                </li>
            `;
        }
        pagination.innerHTML = html;
    }

    function changePage(pageNumber) {
        loadDocuments(pageNumber);
    }

    window.changePage = changePage;

    function formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    function exportToCSV() {
        const csv = [
            ['File Name', 'Hash', 'First Processed', 'Times Processed', 'Provider', 'Cost', 'Status'],
            ...documentsData.map(doc => [
                doc.file_name,
                doc.file_hash,
                new Date(doc.first_processed_at).toISOString(),
                doc.processing_count,
                doc.provider_used,
                doc.cost_usd,
                doc.processing_status
            ])
        ].map(row => row.join(',')).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `document-history-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
    }

    window.exportToCSV = exportToCSV;

    function startAutoRefresh() {
        setInterval(() => {
            loadStatistics();
        }, 30000);
    }

    loadStatistics();
    loadDocuments();
    startAutoRefresh();
}

function initializeBatchProcessing() {
    const page = document.getElementById('batch-processing-page');
    if (!page) {
        return;
    }

    const API_BASE_URL = window.location.origin;

    class BatchManager {
        constructor() {
            this.jobs = [];
            this.schedules = [];
            this.refreshInterval = null;
            this.init();
        }

        async init() {
            await this.loadJobs();
            await this.loadSchedules();
            this.setupEventListeners();
            this.startAutoRefresh();
            this.updateStats();
        }

        setupEventListeners() {
            // Create buttons
            document.getElementById('create-batch-btn').addEventListener('click', () => this.showCreateBatchModal());
            document.getElementById('create-schedule-btn').addEventListener('click', () => this.showCreateScheduleModal());
            document.getElementById('refresh-btn').addEventListener('click', () => this.refreshAll());

            // Submit buttons
            document.getElementById('submit-batch-btn').addEventListener('click', () => this.submitBatchJob());
            document.getElementById('submit-schedule-btn').addEventListener('click', () => this.submitSchedule());

            // Filters
            document.getElementById('status-filter').addEventListener('change', () => this.filterJobs());
            document.getElementById('priority-filter').addEventListener('change', () => this.filterJobs());
            document.getElementById('search-jobs').addEventListener('input', () => this.filterJobs());
            document.getElementById('clear-filters-btn').addEventListener('click', () => this.clearFilters());

            // Schedule type change
            document.getElementById('schedule-type').addEventListener('change', (e) => this.updateScheduleConfig(e.target.value));
        }

        async loadJobs() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs`);
                if (!response.ok) throw new Error('Failed to load jobs');

                this.jobs = await response.json();
                this.renderJobs();
            } catch (error) {
                console.error('Error loading jobs:', error);
                this.showError('Failed to load batch jobs');
            }
        }

        async loadSchedules() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/schedules`);
                if (!response.ok) throw new Error('Failed to load schedules');

                this.schedules = await response.json();
                this.renderSchedules();
            } catch (error) {
                console.error('Error loading schedules:', error);
                this.showError('Failed to load schedules');
            }
        }

        renderJobs() {
            const container = document.getElementById('jobs-container');

            if (this.jobs.length === 0) {
                container.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <i class="fas fa-inbox fa-3x text-muted mb-3"></i>
                        <p class="text-muted">No batch jobs yet. Create your first batch job!</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = this.jobs.map(job => this.createJobCard(job)).join('');
        }

        createJobCard(job) {
            const statusClass = job.status.toLowerCase();
            const priorityClass = job.priority.toLowerCase();
            const progressPercent = job.progress_percent || 0;

            return `
                <div class="col-md-6 col-lg-4">
                    <div class="card job-card ${statusClass} h-100">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-3">
                                <h5 class="card-title mb-0">${job.job_name}</h5>
                                <span class="badge priority-badge ${priorityClass}">${job.priority}</span>
                            </div>

                            <div class="mb-3">
                                <div class="d-flex justify-content-between mb-1">
                                    <span class="text-muted">Status:</span>
                                    <span class="badge bg-${this.getStatusColor(job.status)}">${job.status}</span>
                                </div>
                                <div class="d-flex justify-content-between mb-1">
                                    <span class="text-muted">Progress:</span>
                                    <span>${progressPercent.toFixed(1)}%</span>
                                </div>
                                <div class="progress" style="height: 6px;">
                                    <div class="progress-bar bg-${this.getStatusColor(job.status)}"
                                         style="width: ${progressPercent}%"></div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <small class="text-muted">
                                    <i class="fas fa-file me-1"></i>
                                    ${job.completed_count}/${job.total_files} files
                                    ${job.failed_count > 0 ? `<span class="text-danger ms-2">(${job.failed_count} failed)</span>` : ''}
                                </small>
                            </div>

                            <div class="d-flex gap-2">
                                <button class="btn btn-sm btn-outline-primary flex-grow-1" onclick="batchManager.viewJobDetails('${job.job_id}')">
                                    <i class="fas fa-eye"></i> Details
                                </button>
                                ${this.getJobActions(job)}
                            </div>
                        </div>
                        <div class="card-footer text-muted">
                            <small>
                                <i class="fas fa-clock me-1"></i>
                                ${new Date(job.created_at).toLocaleString()}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }

        getJobActions(job) {
            if (job.status === 'PENDING' || job.status === 'PAUSED') {
                return `<button class="btn btn-sm btn-success" onclick="batchManager.startJob('${job.job_id}')">
                            <i class="fas fa-play"></i>
                        </button>`;
            } else if (job.status === 'PROCESSING') {
                return `<button class="btn btn-sm btn-warning" onclick="batchManager.pauseJob('${job.job_id}')">
                            <i class="fas fa-pause"></i>
                        </button>`;
            } else if (job.status === 'QUEUED') {
                return `<button class="btn btn-sm btn-danger" onclick="batchManager.cancelJob('${job.job_id}')">
                            <i class="fas fa-times"></i>
                        </button>`;
            }
            return '';
        }

        renderSchedules() {
            const container = document.getElementById('schedules-container');

            if (this.schedules.length === 0) {
                container.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <i class="fas fa-calendar-alt fa-3x text-muted mb-3"></i>
                        <p class="text-muted">No schedules yet. Create a scheduled job!</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = this.schedules.map(schedule => this.createScheduleCard(schedule)).join('');
        }

        createScheduleCard(schedule) {
            const statusClass = schedule.enabled ? 'enabled' : 'disabled';
            const nextRun = schedule.next_run ? new Date(schedule.next_run).toLocaleString() : 'N/A';

            return `
                <div class="col-md-6 col-lg-4">
                    <div class="card schedule-card ${statusClass} h-100">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-3">
                                <h5 class="card-title mb-0">${schedule.schedule_name}</h5>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox"
                                           ${schedule.enabled ? 'checked' : ''}
                                           onchange="batchManager.toggleSchedule('${schedule.schedule_id}', this.checked)">
                                </div>
                            </div>

                            <div class="mb-3">
                                <span class="badge bg-info">${schedule.schedule_type}</span>
                                <span class="badge bg-secondary ms-2">${schedule.provider}</span>
                            </div>

                            <div class="mb-2">
                                <small class="text-muted">
                                    <i class="fas fa-clock me-1"></i>
                                    Next run: ${nextRun}
                                </small>
                            </div>

                            <div class="mb-3">
                                <small class="text-muted">
                                    <i class="fas fa-database me-1"></i>
                                    ${schedule.configuration?.input_source || 'Local upload'}
                                </small>
                            </div>

                            <div class="d-flex gap-2">
                                <button class="btn btn-sm btn-outline-primary flex-grow-1" onclick="batchManager.viewScheduleDetails('${schedule.schedule_id}')">
                                    <i class="fas fa-eye"></i> Details
                                </button>
                                <button class="btn btn-sm btn-danger" onclick="batchManager.deleteSchedule('${schedule.schedule_id}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                        <div class="card-footer text-muted">
                            <small>
                                <i class="fas fa-sync me-1"></i>
                                Last run: ${schedule.last_run ? new Date(schedule.last_run).toLocaleString() : 'Never'}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }

        async submitBatchJob() {
            const form = document.getElementById('create-batch-form');
            const formData = new FormData(form);

            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Failed to create batch job');

                this.showSuccess('Batch job created successfully');
                this.loadJobs();
                this.hideModal('createBatchModal');

            } catch (error) {
                console.error('Error creating batch job:', error);
                this.showError('Failed to create batch job');
            }
        }

        async submitSchedule() {
            const form = document.getElementById('create-schedule-form');
            const formData = new FormData(form);

            const payload = {
                schedule_name: formData.get('schedule_name'),
                schedule_type: formData.get('schedule_type'),
                provider: formData.get('provider'),
                configuration: this.collectScheduleConfiguration(formData)
            };

            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/schedules`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error('Failed to create schedule');

                this.showSuccess('Schedule created successfully');
                this.loadSchedules();
                this.hideModal('createScheduleModal');

            } catch (error) {
                console.error('Error creating schedule:', error);
                this.showError('Failed to create schedule');
            }
        }

        collectScheduleConfiguration(formData) {
            const type = formData.get('schedule_type');
            const config = {};

            if (type === 'CRON') {
                config.cron_expression = formData.get('cron_expression');
            } else if (type === 'INTERVAL') {
                config.interval_minutes = parseInt(formData.get('interval_minutes'), 10) || 60;
            } else if (type === 'DAILY') {
                config.time_of_day = formData.get('time_of_day');
            }

            config.input_source = formData.get('input_source');
            config.output_format = formData.get('output_format');
            config.notify_email = formData.get('notify_email') === 'on';

            return config;
        }

        filterJobs() {
            const statusFilter = document.getElementById('status-filter').value;
            const priorityFilter = document.getElementById('priority-filter').value;
            const searchTerm = document.getElementById('search-jobs').value.toLowerCase();

            const filteredJobs = this.jobs.filter(job => {
                const matchesStatus = !statusFilter || job.status === statusFilter;
                const matchesPriority = !priorityFilter || job.priority === priorityFilter;
                const matchesSearch = job.job_name.toLowerCase().includes(searchTerm);
                return matchesStatus && matchesPriority && matchesSearch;
            });

            const container = document.getElementById('jobs-container');
            container.innerHTML = filteredJobs.map(job => this.createJobCard(job)).join('');
        }

        clearFilters() {
            document.getElementById('status-filter').value = '';
            document.getElementById('priority-filter').value = '';
            document.getElementById('search-jobs').value = '';
            this.renderJobs();
        }

        updateScheduleConfig(type) {
            const cronConfig = document.getElementById('cron-config');
            const intervalConfig = document.getElementById('interval-config');
            const dailyConfig = document.getElementById('daily-config');

            cronConfig.style.display = type === 'CRON' ? 'block' : 'none';
            intervalConfig.style.display = type === 'INTERVAL' ? 'block' : 'none';
            dailyConfig.style.display = type === 'DAILY' ? 'block' : 'none';
        }

        async startJob(jobId) {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}/start`, {
                    method: 'POST'
                });
                if (!response.ok) throw new Error('Failed to start job');

                this.showSuccess('Job started');
                this.loadJobs();
            } catch (error) {
                console.error('Error starting job:', error);
                this.showError('Failed to start job');
            }
        }

        async pauseJob(jobId) {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}/pause`, {
                    method: 'POST'
                });
                if (!response.ok) throw new Error('Failed to pause job');

                this.showSuccess('Job paused');
                this.loadJobs();
            } catch (error) {
                console.error('Error pausing job:', error);
                this.showError('Failed to pause job');
            }
        }

        async cancelJob(jobId) {
            if (!confirm('Cancel this batch job?')) return;

            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}/cancel`, {
                    method: 'POST'
                });
                if (!response.ok) throw new Error('Failed to cancel job');

                this.showSuccess('Job cancelled');
                this.loadJobs();
            } catch (error) {
                console.error('Error cancelling job:', error);
                this.showError('Failed to cancel job');
            }
        }

        async toggleSchedule(scheduleId, enabled) {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/schedules/${scheduleId}/${enabled ? 'enable' : 'disable'}`, {
                    method: 'POST'
                });
                if (!response.ok) throw new Error('Failed to update schedule');

                this.showSuccess(`Schedule ${enabled ? 'enabled' : 'disabled'}`);
                this.loadSchedules();
            } catch (error) {
                console.error('Error updating schedule:', error);
                this.showError(`Failed to ${enabled ? 'enable' : 'disable'} schedule`);
            }
        }

        async deleteSchedule(scheduleId) {
            if (!confirm('Delete this schedule?')) return;

            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/schedules/${scheduleId}`, {
                    method: 'DELETE'
                });
                if (!response.ok) throw new Error('Failed to delete schedule');

                this.showSuccess('Schedule deleted');
                this.loadSchedules();
            } catch (error) {
                console.error('Error deleting schedule:', error);
                this.showError('Failed to delete schedule');
            }
        }

        async viewJobDetails(jobId) {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}`);
                if (!response.ok) throw new Error('Failed to load job details');

                const job = await response.json();
                this.showJobDetailsModal(job);
            } catch (error) {
                console.error('Error loading job details:', error);
                this.showError('Failed to load job details');
            }
        }

        showJobDetailsModal(job) {
            const modalHtml = `
                <div class="modal fade" id="jobDetailsModal" tabindex="-1">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Batch Job: ${job.job_name}</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6>Job Details</h6>
                                        <dl>
                                            <dt>Status</dt>
                                            <dd><span class="badge bg-${this.getStatusColor(job.status)}">${job.status}</span></dd>
                                            <dt>Priority</dt>
                                            <dd>${job.priority}</dd>
                                            <dt>Provider</dt>
                                            <dd>${job.provider}</dd>
                                        </dl>
                                    </div>
                                    <div class="col-md-6">
                                        <h6>Progress</h6>
                                        <p>${job.completed_count}/${job.total_files} files processed</p>
                                        <p>${job.failed_count} failures</p>
                                        <div class="progress" style="height: 20px;">
                                            <div class="progress-bar bg-${this.getStatusColor(job.status)}" style="width: ${(job.progress_percent || 0)}%">
                                                ${(job.progress_percent || 0).toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <hr>
                                <h6>Configuration</h6>
                                <pre class="bg-light p-3 rounded">${JSON.stringify(job.configuration, null, 2)}</pre>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            this.renderModal('jobDetailsModal', modalHtml);
        }

        async viewScheduleDetails(scheduleId) {
            try {
                const response = await fetch(`${API_BASE_URL}/api/batch/schedules/${scheduleId}`);
                if (!response.ok) throw new Error('Failed to load schedule details');

                const schedule = await response.json();
                this.showScheduleDetailsModal(schedule);
            } catch (error) {
                console.error('Error loading schedule details:', error);
                this.showError('Failed to load schedule details');
            }
        }

        showScheduleDetailsModal(schedule) {
            const modalHtml = `
                <div class="modal fade" id="scheduleDetailsModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Schedule: ${schedule.schedule_name}</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <p><strong>Type:</strong> ${schedule.schedule_type}</p>
                                <p><strong>Provider:</strong> ${schedule.provider}</p>
                                <p><strong>Status:</strong> ${schedule.enabled ? 'Enabled' : 'Disabled'}</p>
                                <hr>
                                <h6>Configuration</h6>
                                <pre class="bg-light p-3 rounded">${JSON.stringify(schedule.configuration, null, 2)}</pre>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            this.renderModal('scheduleDetailsModal', modalHtml);
        }

        hideModal(id) {
            const modalEl = document.getElementById(id);
            if (!modalEl) return;
            const modal = bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl);
            modal.hide();
        }

        renderModal(id, html) {
            const existingModal = document.getElementById(id);
            if (existingModal) {
                existingModal.remove();
            }
            document.body.insertAdjacentHTML('beforeend', html);
            const modal = new bootstrap.Modal(document.getElementById(id));
            modal.show();

            document.getElementById(id).addEventListener('hidden.bs.modal', function() {
                this.remove();
            });
        }

        async refreshAll() {
            await Promise.all([this.loadJobs(), this.loadSchedules()]);
            this.updateStats();
        }

        async updateStats() {
            const stats = {
                active: this.jobs.filter(job => job.status === 'PROCESSING').length,
                queued: this.jobs.filter(job => job.status === 'QUEUED').length,
                schedules: this.schedules.length,
                providers: new Set(this.jobs.map(job => job.provider)).size
            };

            document.getElementById('active-jobs-count').textContent = stats.active;
            document.getElementById('queued-jobs-count').textContent = stats.queued;
            document.getElementById('schedules-count').textContent = stats.schedules;
            document.getElementById('providers-count').textContent = stats.providers;
        }

        showCreateBatchModal() {
            const modal = new bootstrap.Modal(document.getElementById('createBatchModal'));
            modal.show();
        }

        showCreateScheduleModal() {
            const modal = new bootstrap.Modal(document.getElementById('createScheduleModal'));
            modal.show();
        }

        startAutoRefresh() {
            this.refreshInterval = setInterval(() => {
                this.loadJobs();
                this.loadSchedules();
                this.updateStats();
            }, 10000);
        }

        getStatusColor(status) {
            const colors = {
                'PENDING': 'secondary',
                'QUEUED': 'info',
                'PROCESSING': 'warning',
                'COMPLETED': 'success',
                'FAILED': 'danger',
                'CANCELLED': 'secondary',
                'PAUSED': 'warning'
            };
            return colors[status] || 'secondary';
        }

        showSuccess(message) {
            console.log('Success:', message);
            alert(message);
        }

        showError(message) {
            console.error('Error:', message);
            alert('Error: ' + message);
        }
    }

    const manager = new BatchManager();
    window.batchManager = manager;
}

function initializeBillingDashboard() {
    const page = document.getElementById('billing-page');
    if (!page) {
        return;
    }

    const state = {
        invoices: [],
        budgetAlerts: [],
        costChart: null
    };

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

            state.invoices = await response.json();
            displayInvoices(state.invoices);

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

        let filtered = state.invoices;
        if (statusFilter) {
            filtered = state.invoices.filter(inv => inv.status === statusFilter);
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

        const existingModal = document.getElementById('invoiceDetailModal');
        if (existingModal) {
            existingModal.remove();
        }

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('invoiceDetailModal'));
        modal.show();
    }

    async function payInvoice(invoiceId) {
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
        const totalCostEl = document.getElementById('currentMonthCost');
        if (totalCostEl) {
            totalCostEl.textContent = `$${data.total_cost.toFixed(2)}`;
        }

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

        if (state.costChart) {
            state.costChart.destroy();
        }

        const labels = Object.keys(data.breakdown);
        const values = Object.values(data.breakdown);

        state.costChart = new Chart(ctx, {
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

            state.budgetAlerts = await response.json();
            displayBudgetAlerts(state.budgetAlerts);

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

            const modal = bootstrap.Modal.getInstance(document.getElementById('createBudgetAlertModal'));
            modal.hide();

            showAlert('success', 'Budget alert created successfully');
            await loadBudgetAlerts();

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

    function setupEventListeners() {
        const alertForm = document.getElementById('createBudgetAlertForm');
        if (alertForm) {
            alertForm.addEventListener('submit', handleCreateBudgetAlert);
        }

        const statusFilter = document.getElementById('invoiceStatusFilter');
        if (statusFilter) {
            statusFilter.addEventListener('change', filterInvoices);
        }
    }

    async function init() {
        await Promise.all([
            loadInvoices(),
            loadCurrentMonthCost(),
            loadCostSummary(),
            loadBudgetAlerts()
        ]);
        setupEventListeners();
    }

    window.viewInvoice = viewInvoice;
    window.payInvoice = payInvoice;
    window.generateMonthlyInvoice = generateMonthlyInvoice;
    window.deactivateBudgetAlert = deactivateBudgetAlert;
    window.checkBudgetAlerts = checkBudgetAlerts;

    init();
}

function initializeOlmocrProcessor() {
    const page = document.getElementById('olmocr-processor-page');
    if (!page) {
        return;
    }

    class OLMocrProcessor {
        constructor() {
            this.currentResult = null;
            this.initializeEventListeners();
            this.checkConfiguration();
        }

        initializeEventListeners() {
            document.getElementById('file-form').addEventListener('submit', (e) => {
                e.preventDefault();
                this.processFile();
            });

            document.getElementById('url-form').addEventListener('submit', (e) => {
                e.preventDefault();
                this.processURL();
            });

            document.getElementById('copy-btn').addEventListener('click', () => {
                this.copyToClipboard();
            });

            document.getElementById('download-btn').addEventListener('click', () => {
                this.downloadMarkdown();
            });

            document.getElementById('file-input').addEventListener('change', (e) => {
                this.validateFile(e.target);
            });

            document.getElementById('url-input-field').addEventListener('input', (e) => {
                this.validateURL(e.target);
            });
        }

        async checkConfiguration() {
            try {
                const response = await fetch('/olmocr/config');
                const config = await response.json();

                const statusDiv = document.getElementById('config-status');
                const messageSpan = document.getElementById('config-message');

                if (config.provider_configured) {
                    statusDiv.className = 'alert alert-success mb-4';
                    messageSpan.innerHTML = `<strong>✓ Ready:</strong> ${config.provider} integration configured successfully`;
                } else {
                    statusDiv.className = 'alert alert-warning mb-4';
                    const providerType = config.provider_type || 'runpod';
                    if (providerType === 'vastai') {
                        messageSpan.innerHTML = '<strong>⚠ Warning:</strong> Vast.ai API credentials not configured. Please set VASTAI_INSTANCE_ID, VASTAI_API_KEY, and VASTAI_ENDPOINT_URL environment variables.';
                    } else {
                        messageSpan.innerHTML = '<strong>⚠ Warning:</strong> RunPod API credentials not configured. Please set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables.';
                    }
                }

                statusDiv.style.display = 'block';
            } catch (error) {
                console.error('Error checking configuration:', error);
                this.showToast('Failed to check configuration', 'warning');
            }
        }

        validateFile(input) {
            const file = input.files[0];
            if (!file) return false;

            const maxSize = 50 * 1024 * 1024;
            const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];

            if (file.size > maxSize) {
                this.showError('File size exceeds 50MB limit. Please choose a smaller file.');
                input.value = '';
                return false;
            }

            if (!allowedTypes.includes(file.type)) {
                this.showError('Invalid file type. Please upload a PDF, PNG, or JPEG file.');
                input.value = '';
                return false;
            }

            this.hideError();
            return true;
        }

        validateURL(input) {
            const url = input.value.trim();
            if (!url) return false;

            try {
                new URL(url);
                const supportedExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
                const hasValidExtension = supportedExtensions.some(ext =>
                    url.toLowerCase().includes(ext)
                );

                if (!hasValidExtension) {
                    this.showError('URL should point to a PDF, PNG, or JPEG file.');
                    return false;
                }

                this.hideError();
                return true;
            } catch {
                this.showError('Please enter a valid URL.');
                return false;
            }
        }

        async processFile() {
            const fileInput = document.getElementById('file-input');
            const includeMetadata = document.getElementById('file-metadata').checked;

            if (!this.validateFile(fileInput)) {
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('include_metadata', includeMetadata);

            await this.submitRequest('/olmocr/file', formData);
        }

        async processURL() {
            const urlInput = document.getElementById('url-input-field');
            const includeMetadata = document.getElementById('url-metadata').checked;

            if (!this.validateURL(urlInput)) {
                return;
            }

            const formData = new FormData();
            formData.append('url', urlInput.value);
            formData.append('include_metadata', includeMetadata);

            await this.submitRequest('/olmocr/url', formData);
        }

        async submitRequest(endpoint, formData) {
            this.showProcessing();
            this.hideResults();
            this.hideError();

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
                }

                const result = await response.json();

                if (result.success) {
                    this.showResults(result);
                    this.showSuccessToast();
                } else {
                    throw new Error(result.error || 'Processing failed');
                }

            } catch (error) {
                console.error('Processing error:', error);
                this.showError(error.message || 'Processing failed');
            } finally {
                this.hideProcessing();
            }
        }

        showProcessing() {
            document.getElementById('results-section').style.display = 'none';
            document.getElementById('error-section').style.display = 'none';

            const processingBtn = document.querySelector('#file-form button[type="submit"], #url-form button[type="submit"]');
            if (processingBtn) {
                processingBtn.disabled = true;
                processingBtn.classList.add('loading');
            }

            const processingIndicator = document.getElementById('processing-indicator');
            if (processingIndicator) {
                processingIndicator.style.display = 'block';
            }
        }

        hideProcessing() {
            const processingBtn = document.querySelector('#file-form button[type="submit"], #url-form button[type="submit"]');
            if (processingBtn) {
                processingBtn.disabled = false;
                processingBtn.classList.remove('loading');
            }

            const processingIndicator = document.getElementById('processing-indicator');
            if (processingIndicator) {
                processingIndicator.style.display = 'none';
            }
        }

        showResults(result) {
            this.currentResult = result;

            const resultSection = document.getElementById('results-section');
            resultSection.style.display = 'block';

            const markdownContent = document.getElementById('markdown-content');
            markdownContent.textContent = result.markdown_content || 'No content available';

            const metadataSection = document.getElementById('metadata-section');
            const metadataContent = document.getElementById('metadata-content');

            if (result.metadata && Object.keys(result.metadata).length > 0) {
                metadataSection.style.display = 'block';
                metadataContent.innerHTML = this.formatMetadata(result.metadata);
            } else {
                metadataSection.style.display = 'none';
            }
        }

        formatMetadata(metadata) {
            return Object.entries(metadata).map(([key, value]) => {
                if (typeof value === 'object' && value !== null) {
                    return `<div class="mb-2"><strong>${this.formatKey(key)}:</strong><pre class="bg-light p-2 rounded">${JSON.stringify(value, null, 2)}</pre></div>`;
                }
                return `<div class="mb-1"><strong>${this.formatKey(key)}:</strong> ${value}</div>`;
            }).join('');
        }

        formatKey(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
        }

        showError(message) {
            const errorSection = document.getElementById('error-section');
            const errorMessage = document.getElementById('error-message');
            errorMessage.textContent = message;
            errorSection.style.display = 'block';
        }

        hideError() {
            const errorSection = document.getElementById('error-section');
            if (errorSection) {
                errorSection.style.display = 'none';
            }
        }

        hideResults() {
            const resultSection = document.getElementById('results-section');
            if (resultSection) {
                resultSection.style.display = 'none';
            }
        }

        async copyToClipboard() {
            if (!this.currentResult || !this.currentResult.markdown_content) {
                this.showToast('No content to copy', 'warning');
                return;
            }

            try {
                await navigator.clipboard.writeText(this.currentResult.markdown_content);
                this.showToast('Content copied to clipboard!', 'success');
            } catch (error) {
                console.error('Copy failed:', error);
                this.showToast('Failed to copy content', 'danger');
            }
        }

        downloadMarkdown() {
            if (!this.currentResult || !this.currentResult.markdown_content) {
                this.showToast('No content to download', 'warning');
                return;
            }

            const blob = new Blob([this.currentResult.markdown_content], { type: 'text/markdown' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            const timestamp = new Date().toISOString().split('T')[0];
            const markdownFilename = `olmocr-output-${timestamp}.md`;

            a.href = url;
            a.download = markdownFilename;
            a.click();
            window.URL.revokeObjectURL(url);

            this.showToast(`Downloaded as ${markdownFilename}`, 'success');
        }

        showToast(message, type = 'info') {
            let container = document.getElementById('toast-container');
            if (!container) {
                container = document.createElement('div');
                container.id = 'toast-container';
                container.className = 'toast-container position-fixed top-0 end-0 p-3';
                container.style.zIndex = '1100';
                document.body.appendChild(container);
            }

            const toastId = `toast-${Date.now()}`;
            const toastHtml = `
                <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="d-flex">
                        <div class="toast-body">
                            <i class="fas fa-${this.getToastIcon(type)} me-2"></i>${message}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                    </div>
                </div>
            `;

            container.insertAdjacentHTML('beforeend', toastHtml);
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, { delay: 4000 });
            toast.show();

            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        }

        getToastIcon(type) {
            const icons = {
                success: 'check-circle',
                warning: 'exclamation-triangle',
                danger: 'exclamation-circle',
                info: 'info-circle'
            };
            return icons[type] || 'info-circle';
        }

        showSuccessToast() {
            this.showToast('Document processed successfully!', 'success');
        }
    }

    window.olmocrProcessor = new OLMocrProcessor();
}

function initializeUserProfilePage() {
    const page = document.getElementById('user-profile-page');
    if (!page) {
        return;
    }

    const state = {
        currentUser: null,
        apiKeys: []
    };

    async function loadCurrentUser() {
        try {
            const response = await fetch('/api/auth/me', {
                headers: {
                    'Authorization': `Bearer ${getAuthToken()}`
                }
            });

            if (!response.ok) {
                if (response.status === 401) {
                    window.location.href = '/login';
                    return;
                }
                throw new Error('Failed to load user profile');
            }

            state.currentUser = await response.json();
            displayUserProfile(state.currentUser);

        } catch (error) {
            console.error('Error loading user:', error);
            showAlert('error', 'Failed to load user profile');
        }
    }

    function displayUserProfile(user) {
        const emailInput = document.getElementById('email');
        if (emailInput) emailInput.value = user.email || '';

        const usernameInput = document.getElementById('username');
        if (usernameInput) usernameInput.value = user.username || '';

        const firstNameInput = document.getElementById('firstName');
        if (firstNameInput) firstNameInput.value = user.first_name || '';

        const lastNameInput = document.getElementById('lastName');
        if (lastNameInput) lastNameInput.value = user.last_name || '';

        const emailDisplay = document.getElementById('userEmail');
        if (emailDisplay) emailDisplay.textContent = user.email;

        const roleDisplay = document.getElementById('userRole');
        if (roleDisplay) roleDisplay.textContent = user.role;

        const tierDisplay = document.getElementById('subscriptionTier');
        if (tierDisplay) tierDisplay.textContent = user.subscription_tier;

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
            state.currentUser = updatedUser;
            displayUserProfile(updatedUser);

            showAlert('success', 'Profile updated successfully');

        } catch (error) {
            console.error('Error updating profile:', error);
            showAlert('error', error.message || 'Failed to update profile');
        }
    }

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
        const quotaUsed = document.getElementById('quotaUsed');
        if (quotaUsed) quotaUsed.textContent = quotaData.used.toLocaleString();

        const quotaLimit = document.getElementById('quotaLimit');
        if (quotaLimit) quotaLimit.textContent = quotaData.quota.toLocaleString();

        const quotaRemaining = document.getElementById('quotaRemaining');
        if (quotaRemaining) quotaRemaining.textContent = quotaData.remaining.toLocaleString();

        const progressBar = document.getElementById('quotaProgressBar');
        if (progressBar) {
            const percentage = quotaData.percentage;
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
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

            state.apiKeys = await response.json();
            displayAPIKeys(state.apiKeys);

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
                                    <strong>Created:</strong> ${formatDateTime(key.created_at)}
                                    ${key.expires_at ? `<br><strong>Expires:</strong> ${formatDateTime(key.expires_at)}` : ''}
                                    ${key.last_used_at ? `<br><strong>Last used:</strong> ${formatDateTime(key.last_used_at)}` : ''}
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

    async function handleCreateAPIKey(event) {
        event.preventDefault();

        const formData = {
            key_name: document.getElementById('keyName').value,
            scopes: getSelectedScopes(),
            expires_in_days: document.getElementById('expiresInDays').value ?
                parseInt(document.getElementById('expiresInDays').value, 10) : null,
            rate_limit_per_minute: document.getElementById('rateLimitPerMinute').value ?
                parseInt(document.getElementById('rateLimitPerMinute').value, 10) : null,
            rate_limit_per_hour: document.getElementById('rateLimitPerHour').value ?
                parseInt(document.getElementById('rateLimitPerHour').value, 10) : null
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

            const modal = bootstrap.Modal.getInstance(document.getElementById('createAPIKeyModal'));
            modal.hide();

            showAPIKeyResult(result.full_key, result.warning);

            await loadAPIKeys();
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

        const existingModal = document.getElementById('apiKeyResultModal');
        if (existingModal) {
            existingModal.remove();
        }

        document.body.insertAdjacentHTML('beforeend', modalHtml);

        const modal = new bootstrap.Modal(document.getElementById('apiKeyResultModal'));
        modal.show();

        document.getElementById('apiKeyResultModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }

    function copyAPIKey() {
        const input = document.getElementById('generatedAPIKey');
        if (!input) {
            showAlert('error', 'Nothing to copy');
            return;
        }

        input.select();
        input.setSelectionRange(0, 99999);

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

    function setupEventListeners() {
        const profileForm = document.getElementById('profileUpdateForm');
        if (profileForm) {
            profileForm.addEventListener('submit', handleProfileUpdate);
        }

        const apiKeyForm = document.getElementById('createAPIKeyForm');
        if (apiKeyForm) {
            apiKeyForm.addEventListener('submit', handleCreateAPIKey);
        }
    }

    function formatDateTime(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleString();
    }

    async function init() {
        await Promise.all([
            loadCurrentUser(),
            loadAPIKeys(),
            loadQuotaInfo()
        ]);
        setupEventListeners();
    }

    window.copyAPIKey = copyAPIKey;
    window.revokeAPIKey = revokeAPIKey;
    window.deleteAPIKey = deleteAPIKey;

    init();
}

// File handling utilities
class FileHandler {
    constructor() {
        this.maxFileSize = 10 * 1024 * 1024; // 10MB
        this.allowedTypes = [
            'application/pdf',
            'image/jpeg',
            'image/jpg', 
            'image/png',
            'image/bmp',
            'image/tiff',
            'image/webp'
        ];
    }

    validateFile(file) {
        const errors = [];
        
        // Check file size
        if (file.size > this.maxFileSize) {
            errors.push(`File size exceeds 10MB limit`);
        }
        
        // Check file type
        if (!this.allowedTypes.includes(file.type) && !this.isValidExtension(file.name)) {
            errors.push(`Unsupported file type: ${file.type}`);
        }
        
        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    isValidExtension(filename) {
        const ext = filename.toLowerCase().split('.').pop();
        const validExts = ['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'];
        return validExts.includes(ext);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getFileIcon(file) {
        const type = file.type.toLowerCase();
        const ext = file.name.toLowerCase().split('.').pop();
        
        if (type.includes('pdf') || ext === 'pdf') {
            return 'fas fa-file-pdf text-danger';
        } else if (type.includes('image')) {
            return 'fas fa-file-image text-info';
        }
        return 'fas fa-file text-secondary';
    }
}

// OCR Processing utilities
class OCRProcessor {
    constructor() {
        this.isProcessing = false;
        this.currentRequest = null;
    }

    async processFiles(files, options = {}) {
        if (this.isProcessing) {
            throw new Error('Processing already in progress');
        }

        this.isProcessing = true;
        
        try {
            const formData = new FormData();
            
            // Add files
            files.forEach(file => {
                formData.append('files', file);
            });
            
            // Add options
            formData.append('strategy', options.strategy || 'fallback');
            formData.append('providers', options.providers?.join(',') || 'mistral');
            
            if (options.prompt) {
                formData.append('prompt', options.prompt);
            }

            // Make request with progress tracking
            this.currentRequest = fetch('/process', {
                method: 'POST',
                body: formData
            });

            const response = await this.currentRequest;
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server error: ${response.status}`);
            }

            const result = await response.json();
            return result;
            
        } finally {
            this.isProcessing = false;
            this.currentRequest = null;
        }
    }

    cancelProcessing() {
        if (this.currentRequest) {
            // Note: fetch doesn't support cancellation directly
            // In a real implementation, you might use AbortController
            this.currentRequest = null;
        }
        this.isProcessing = false;
    }
}

// UI utilities
class UIManager {
    static showNotification(message, type = 'info', duration = 5000) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Find or create notification container
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }

        container.appendChild(alertDiv);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    const bsAlert = new bootstrap.Alert(alertDiv);
                    bsAlert.close();
                }
            }, duration);
        }
    }

    static showModal(title, content, options = {}) {
        const modalId = 'dynamic-modal';
        let modal = document.getElementById(modalId);
        
        if (modal) {
            modal.remove();
        }

        const modalHTML = `
            <div class="modal fade" id="${modalId}" tabindex="-1">
                <div class="modal-dialog ${options.size || ''}">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${content}
                        </div>
                        ${options.footer ? `<div class="modal-footer">${options.footer}</div>` : ''}
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);
        modal = document.getElementById(modalId);
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Clean up on hide
        modal.addEventListener('hidden.bs.modal', () => {
            modal.remove();
        });

        return bsModal;
    }

    static updateProcessingStatus(status, percentage = null) {
        const statusElement = document.getElementById('processingStatus');
        if (statusElement) {
            statusElement.textContent = status;
        }

        const progressBar = document.getElementById('processingProgress');
        if (progressBar && percentage !== null) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }
    }

    static animateElement(element, animation = 'fadeIn') {
        element.classList.add('animate__animated', `animate__${animation}`);
        
        element.addEventListener('animationend', () => {
            element.classList.remove('animate__animated', `animate__${animation}`);
        }, { once: true });
    }
}

// Export utilities for global use
window.FileHandler = FileHandler;
window.OCRProcessor = OCRProcessor;
window.UIManager = UIManager;

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    UIManager.showNotification('An unexpected error occurred', 'danger');
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    UIManager.showNotification('An unexpected error occurred', 'danger');
    event.preventDefault();
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Copy to clipboard utility
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text).then(() => {
            UIManager.showNotification('Copied to clipboard!', 'success', 2000);
        }).catch(err => {
            console.error('Failed to copy to clipboard:', err);
            UIManager.showNotification('Failed to copy to clipboard', 'danger');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();

        try {
            document.execCommand('copy');
            UIManager.showNotification('Copied to clipboard!', 'success', 2000);
        } catch (err) {
            console.error('Failed to copy to clipboard:', err);
            UIManager.showNotification('Failed to copy to clipboard', 'danger');
        }

        document.body.removeChild(textArea);
    }
}

// ============================================================================
// Sidebar and Navigation functionality
// ============================================================================

function initializeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mainContent = document.getElementById('mainContent');

    if (!sidebar || !sidebarToggle) return;

    // Load sidebar state from localStorage
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (sidebarCollapsed) {
        sidebar.classList.add('collapsed');
    }

    // Toggle sidebar
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('collapsed');
        const isCollapsed = sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed);

        // Update icon
        const icon = this.querySelector('i');
        icon.className = isCollapsed ? 'fas fa-bars' : 'fas fa-times';
    });

    // Highlight active nav link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    navLinks.forEach(link => {
        const linkPath = new URL(link.href).pathname;
        if (linkPath === currentPath || (currentPath === '/' && linkPath === '/')) {
            link.classList.add('active');
        }
    });

    // Mobile: Close sidebar when clicking outside
    document.addEventListener('click', function(event) {
        if (window.innerWidth <= 768) {
            const isClickInsideSidebar = sidebar.contains(event.target);
            const isClickOnToggle = sidebarToggle.contains(event.target);

            if (!isClickInsideSidebar && !isClickOnToggle && sidebar.classList.contains('show')) {
                sidebar.classList.remove('show');
            }
        }
    });

    // Mobile: Show sidebar on toggle
    if (window.innerWidth <= 768) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('show');
        });
    }
}

// Language selector functionality
function initializeLanguageSelector() {
    const languageSelector = document.getElementById('language-selector');
    if (!languageSelector) return;

    languageSelector.addEventListener('change', function() {
        const selectedLang = this.value;
        console.log('Language changed to:', selectedLang);

        // Add language parameter to current URL
        const url = new URL(window.location.href);
        url.searchParams.set('lang', selectedLang);

        // Reload page with new language
        window.location.href = url.toString();
    });
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl+B or Cmd+B to toggle sidebar
        if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
            event.preventDefault();
            const sidebarToggle = document.getElementById('sidebarToggle');
            if (sidebarToggle) {
                sidebarToggle.click();
            }
        }

        // Ctrl+/ or Cmd+/ to show keyboard shortcuts help (future)
        if ((event.ctrlKey || event.metaKey) && event.key === '/') {
            event.preventDefault();
            console.log('Keyboard shortcuts: Ctrl+B = Toggle sidebar');
        }
    });
}

// ============================================================================
// Notifications functionality
// ============================================================================

async function loadNotifications() {
    try {
        const response = await fetch('/api/notifications/unread');
        if (!response.ok) throw new Error('Failed to load notifications');

        const notifications = await response.json();
        updateNotificationBadge(notifications.length);
        renderNotifications(notifications);
    } catch (error) {
        console.error('Error loading notifications:', error);
        // Show fallback if API not available
        const notificationsList = document.getElementById('notifications-list');
        if (notificationsList) {
            notificationsList.innerHTML = `
                <li class="text-center py-3">
                    <small class="text-muted">No new notifications</small>
                </li>
            `;
        }
    }
}

function updateNotificationBadge(count) {
    const badge = document.getElementById('notification-badge');
    if (!badge) return;

    if (count > 0) {
        badge.textContent = count > 99 ? '99+' : count;
        badge.style.display = 'inline-block';
    } else {
        badge.style.display = 'none';
    }
}

function renderNotifications(notifications) {
    const listContainer = document.getElementById('notifications-list');
    if (!listContainer) return;

    if (notifications.length === 0) {
        listContainer.innerHTML = `
            <li class="text-center py-3">
                <small class="text-muted">No new notifications</small>
            </li>
        `;
        return;
    }

    listContainer.innerHTML = notifications.map(notif => `
        <li>
            <a class="dropdown-item ${notif.read ? '' : 'bg-light'}" href="${notif.link || '#'}" onclick="markAsRead('${notif.id}')">
                <div class="d-flex align-items-start">
                    <i class="fas fa-${getNotificationIcon(notif.type)} me-2 mt-1"></i>
                    <div class="flex-grow-1">
                        <small>${notif.message}</small>
                        <br>
                        <small class="text-muted">${formatTimeAgo(notif.created_at)}</small>
                    </div>
                </div>
            </a>
        </li>
    `).join('');
}

function getNotificationIcon(type) {
    const icons = {
        'deployment': 'rocket',
        'batch': 'tasks',
        'alert': 'exclamation-triangle',
        'success': 'check-circle',
        'error': 'times-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'bell';
}

function formatTimeAgo(dateString) {
    const date = new Date(dateString);
    const seconds = Math.floor((new Date() - date) / 1000);

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
    return Math.floor(seconds / 86400) + 'd ago';
}

async function markAsRead(notificationId) {
    try {
        await fetch(`/api/notifications/${notificationId}/read`, { method: 'POST' });
        loadNotifications();
    } catch (error) {
        console.error('Error marking notification as read:', error);
    }
}

async function markNotificationsAsRead() {
    try {
        await fetch('/api/notifications/mark-all-read', { method: 'POST' });
        loadNotifications();
    } catch (error) {
        console.error('Error marking all notifications as read:', error);
    }
}

// ============================================================================
// Shared Helpers
// ============================================================================

function getAuthToken() {
    if (window.Clerk && window.Clerk.session) {
        try {
            return window.Clerk.session.getToken();
        } catch (error) {
            console.warn('Unable to retrieve Clerk token:', error);
        }
    }

    return localStorage.getItem('auth_token') || '';
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
        return 'N/A';
    }
    return date.toLocaleDateString();
}

function escapeHtml(text) {
    if (text === undefined || text === null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function showAlert(type, message) {
    const normalizedType = type === 'error' ? 'danger' : type;
    const alertHtml = `
        <div class="alert alert-${normalizedType} alert-dismissible fade show" role="alert">
            ${escapeHtml(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const container = document.getElementById('alertContainer') || document.querySelector('.container');
    if (container) {
        container.insertAdjacentHTML('afterbegin', alertHtml);

        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                const bsAlert = bootstrap.Alert.getInstance(alert) || new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}

// ============================================================================
// Authentication (Clerk)
// ============================================================================

async function handleLogout() {
    try {
        const response = await fetch('/api/auth/logout', { method: 'POST' });
        if (response.ok) {
            window.location.href = '/';
        } else {
            console.error('Logout failed');
        }
    } catch (error) {
        console.error('Error during logout:', error);
    }
}

// ============================================================================
// Initialize navigation and sidebar features
// ============================================================================

// Extend the existing DOMContentLoaded to include navigation features
document.addEventListener('DOMContentLoaded', function() {
    // Initialize sidebar
    initializeSidebar();

    // Initialize language selector
    initializeLanguageSelector();

    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();

    // Load notifications
    loadNotifications();

    // Auto-refresh notifications every 30 seconds
    setInterval(loadNotifications, 30000);
});
