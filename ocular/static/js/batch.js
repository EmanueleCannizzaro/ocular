/* Batch Processing JavaScript */

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
                                <i class="fas fa-chart-line me-1"></i>
                                ${schedule.total_runs} runs (${schedule.successful_runs} successful)
                            </small>
                        </div>

                        <div class="d-flex gap-2">
                            <button class="btn btn-sm btn-outline-primary flex-grow-1" onclick="batchManager.editSchedule('${schedule.schedule_id}')">
                                <i class="fas fa-edit"></i> Edit
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="batchManager.deleteSchedule('${schedule.schedule_id}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showCreateBatchModal() {
        const modal = new bootstrap.Modal(document.getElementById('createBatchModal'));
        modal.show();
    }

    showCreateScheduleModal() {
        const modal = new bootstrap.Modal(document.getElementById('createScheduleModal'));
        this.updateScheduleConfig('DAILY'); // Default
        modal.show();
    }

    updateScheduleConfig(scheduleType) {
        const container = document.getElementById('schedule-config-container');

        const configs = {
            'ONCE': `
                <label class="form-label">Run Date & Time</label>
                <input type="datetime-local" class="form-control" id="schedule-datetime" required>
            `,
            'HOURLY': `
                <label class="form-label">Run Every (hours)</label>
                <input type="number" class="form-control" id="schedule-interval" min="1" max="24" value="1" required>
            `,
            'DAILY': `
                <label class="form-label">Run Time</label>
                <input type="time" class="form-control" id="schedule-time" value="00:00" required>
            `,
            'WEEKLY': `
                <label class="form-label">Day of Week</label>
                <select class="form-select" id="schedule-day" required>
                    <option value="monday">Monday</option>
                    <option value="tuesday">Tuesday</option>
                    <option value="wednesday">Wednesday</option>
                    <option value="thursday">Thursday</option>
                    <option value="friday">Friday</option>
                    <option value="saturday">Saturday</option>
                    <option value="sunday">Sunday</option>
                </select>
                <label class="form-label mt-2">Time</label>
                <input type="time" class="form-control" id="schedule-time" value="00:00" required>
            `,
            'MONTHLY': `
                <label class="form-label">Day of Month</label>
                <input type="number" class="form-control" id="schedule-day" min="1" max="31" value="1" required>
                <label class="form-label mt-2">Time</label>
                <input type="time" class="form-control" id="schedule-time" value="00:00" required>
            `,
            'CRON': `
                <label class="form-label">CRON Expression</label>
                <input type="text" class="form-control" id="schedule-cron" placeholder="0 0 * * *" required>
                <div class="form-text">
                    Examples: <code>0 */6 * * *</code> (every 6 hours),
                    <code>0 9 * * 1-5</code> (9 AM weekdays)
                </div>
            `
        };

        container.innerHTML = configs[scheduleType] || configs['DAILY'];
    }

    async submitBatchJob() {
        const formData = new FormData();
        const files = document.getElementById('batch-files').files;

        // Add files
        for (let file of files) {
            formData.append('files', file);
        }

        // Add other fields
        formData.append('job_name', document.getElementById('batch-job-name').value);
        formData.append('provider', document.getElementById('batch-provider').value);
        formData.append('strategy', document.getElementById('batch-strategy').value);
        formData.append('priority', document.getElementById('batch-priority').value);

        try {
            const response = await fetch(`${API_BASE_URL}/api/batch/upload-batch`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Failed to create batch job');

            const result = await response.json();

            bootstrap.Modal.getInstance(document.getElementById('createBatchModal')).hide();
            this.showSuccess(`Batch job created: ${result.job_id}`);

            await this.loadJobs();
            this.updateStats();
        } catch (error) {
            console.error('Error creating batch job:', error);
            this.showError('Failed to create batch job');
        }
    }

    async submitSchedule() {
        const scheduleType = document.getElementById('schedule-type').value;
        const scheduleConfig = this.getScheduleConfig(scheduleType);

        const payload = {
            schedule_name: document.getElementById('schedule-name').value,
            schedule_type: scheduleType,
            schedule_config: scheduleConfig,
            source_path: document.getElementById('schedule-source-path').value,
            provider: document.getElementById('schedule-provider').value,
            strategy: document.getElementById('schedule-strategy').value,
            enabled: document.getElementById('schedule-enabled').checked
        };

        try {
            const response = await fetch(`${API_BASE_URL}/api/batch/schedules`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Failed to create schedule');

            const result = await response.json();

            bootstrap.Modal.getInstance(document.getElementById('createScheduleModal')).hide();
            this.showSuccess(`Schedule created: ${result.schedule_id}`);

            await this.loadSchedules();
            this.updateStats();
        } catch (error) {
            console.error('Error creating schedule:', error);
            this.showError('Failed to create schedule');
        }
    }

    getScheduleConfig(scheduleType) {
        const configs = {
            'ONCE': {
                datetime: document.getElementById('schedule-datetime')?.value
            },
            'HOURLY': {
                interval_hours: parseInt(document.getElementById('schedule-interval')?.value || 1)
            },
            'DAILY': {
                time: document.getElementById('schedule-time')?.value
            },
            'WEEKLY': {
                day_of_week: document.getElementById('schedule-day')?.value,
                time: document.getElementById('schedule-time')?.value
            },
            'MONTHLY': {
                day_of_month: parseInt(document.getElementById('schedule-day')?.value || 1),
                time: document.getElementById('schedule-time')?.value
            },
            'CRON': {
                cron_expression: document.getElementById('schedule-cron')?.value
            }
        };

        return configs[scheduleType] || {};
    }

    async startJob(jobId) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}/start`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to start job');

            this.showSuccess('Job started successfully');
            await this.loadJobs();
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
            await this.loadJobs();
        } catch (error) {
            console.error('Error pausing job:', error);
            this.showError('Failed to pause job');
        }
    }

    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;

        try {
            const response = await fetch(`${API_BASE_URL}/api/batch/jobs/${jobId}/cancel`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to cancel job');

            this.showSuccess('Job cancelled');
            await this.loadJobs();
        } catch (error) {
            console.error('Error cancelling job:', error);
            this.showError('Failed to cancel job');
        }
    }

    async toggleSchedule(scheduleId, enabled) {
        const endpoint = enabled ? 'enable' : 'disable';

        try {
            const response = await fetch(`${API_BASE_URL}/api/batch/schedules/${scheduleId}/${endpoint}`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error(`Failed to ${endpoint} schedule`);

            this.showSuccess(`Schedule ${enabled ? 'enabled' : 'disabled'}`);
            await this.loadSchedules();
        } catch (error) {
            console.error(`Error ${endpoint} schedule:`, error);
            this.showError(`Failed to ${endpoint} schedule`);
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
        document.getElementById('job-details-title').innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            ${job.job_name}
        `;

        const filesHtml = job.files.map(file => `
            <tr>
                <td><small>${file.file_path}</small></td>
                <td><span class="badge bg-${this.getStatusColor(file.status)}">${file.status}</span></td>
                <td>${file.processing_time ? file.processing_time.toFixed(2) + 's' : '-'}</td>
            </tr>
        `).join('');

        document.getElementById('job-details-body').innerHTML = `
            <div class="row mb-4">
                <div class="col-md-6">
                    <h6>Job Information</h6>
                    <table class="table table-sm">
                        <tr><th>Status:</th><td><span class="badge bg-${this.getStatusColor(job.status)}">${job.status}</span></td></tr>
                        <tr><th>Priority:</th><td>${job.priority}</td></tr>
                        <tr><th>Provider:</th><td>${job.provider}</td></tr>
                        <tr><th>Strategy:</th><td>${job.strategy}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Progress</h6>
                    <table class="table table-sm">
                        <tr><th>Total Files:</th><td>${job.total_files}</td></tr>
                        <tr><th>Completed:</th><td>${job.completed_count}</td></tr>
                        <tr><th>Failed:</th><td>${job.failed_count}</td></tr>
                        <tr><th>Progress:</th><td>${job.progress_percent.toFixed(1)}%</td></tr>
                    </table>
                </div>
            </div>

            <h6>Files</h6>
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead>
                        <tr>
                            <th>File Path</th>
                            <th>Status</th>
                            <th>Processing Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${filesHtml}
                    </tbody>
                </table>
            </div>
        `;

        const modal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
        modal.show();
    }

    filterJobs() {
        const status = document.getElementById('status-filter').value;
        const priority = document.getElementById('priority-filter').value;
        const search = document.getElementById('search-jobs').value.toLowerCase();

        const filtered = this.jobs.filter(job => {
            if (status && job.status !== status) return false;
            if (priority && job.priority !== priority) return false;
            if (search && !job.job_name.toLowerCase().includes(search)) return false;
            return true;
        });

        const container = document.getElementById('jobs-container');
        container.innerHTML = filtered.map(job => this.createJobCard(job)).join('');
    }

    clearFilters() {
        document.getElementById('status-filter').value = '';
        document.getElementById('priority-filter').value = '';
        document.getElementById('search-jobs').value = '';
        this.filterJobs();
    }

    async refreshAll() {
        await Promise.all([this.loadJobs(), this.loadSchedules()]);
        this.updateStats();
        this.showSuccess('Refreshed successfully');
    }

    updateStats() {
        const activeJobs = this.jobs.filter(j => j.status === 'PROCESSING').length;
        const queuedJobs = this.jobs.filter(j => j.status === 'QUEUED').length;
        const filesProcessed = this.jobs.reduce((sum, j) => sum + j.completed_count, 0);

        document.getElementById('active-jobs-count').textContent = activeJobs;
        document.getElementById('queued-jobs-count').textContent = queuedJobs;
        document.getElementById('schedules-count').textContent = this.schedules.length;
        document.getElementById('files-processed-count').textContent = filesProcessed;
    }

    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            this.loadJobs();
            this.loadSchedules();
            this.updateStats();
        }, 10000); // Refresh every 10 seconds
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
        // You can implement a toast notification here
        console.log('Success:', message);
        alert(message);
    }

    showError(message) {
        // You can implement a toast notification here
        console.error('Error:', message);
        alert('Error: ' + message);
    }
}

// Initialize when page loads
let batchManager;
document.addEventListener('DOMContentLoaded', () => {
    batchManager = new BatchManager();
});
