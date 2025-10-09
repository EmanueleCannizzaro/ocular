/**
 * Model Testing Interface
 * Handles testing models with file upload and prompt input
 */

let currentTestingModel = null;
let testHistory = [];

/**
 * Show test modal for a model
 */
function showTestModal(modelId) {
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        showNotification('Model not found', 'danger');
        return;
    }

    currentTestingModel = model;

    // Create test modal
    const modalHtml = `
        <div class="modal fade" id="testModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-vial me-2"></i>
                            Test Model: ${model.display_name || model.name}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <!-- Left Panel: Input -->
                            <div class="col-md-6">
                                <h6 class="mb-3">
                                    <i class="fas fa-file-upload me-2"></i>
                                    Input
                                </h6>

                                <form id="test-form">
                                    <div class="mb-3">
                                        <label class="form-label">Upload File (Optional)</label>
                                        <input type="file" class="form-control" id="test-file"
                                               accept="image/*,.pdf">
                                        <small class="text-muted">Supported: Images (JPEG, PNG, WebP), PDF</small>
                                    </div>

                                    <div class="mb-3">
                                        <label class="form-label">Prompt / Instructions</label>
                                        <textarea class="form-control" id="test-prompt" rows="4"
                                                  placeholder="Enter your prompt or instructions for the model..."></textarea>
                                        <small class="text-muted">Provide context or specific instructions</small>
                                    </div>

                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Max Tokens</label>
                                            <input type="number" class="form-control" id="test-max-tokens"
                                                   value="1000" min="1" max="4000">
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Temperature</label>
                                            <input type="number" class="form-control" id="test-temperature"
                                                   value="0.7" min="0" max="2" step="0.1">
                                        </div>
                                    </div>

                                    <div class="d-grid">
                                        <button type="button" class="btn btn-primary" onclick="runModelTest()">
                                            <i class="fas fa-play me-2"></i>
                                            Run Test
                                        </button>
                                    </div>
                                </form>

                                <!-- Model Info Card -->
                                <div class="card mt-3 bg-light">
                                    <div class="card-body p-3">
                                        <h6 class="card-title mb-2">
                                            <i class="fas fa-info-circle me-1"></i>
                                            Model Information
                                        </h6>
                                        <ul class="list-unstyled mb-0 small">
                                            <li><strong>Version:</strong> ${model.version || 'N/A'}</li>
                                            <li><strong>VRAM:</strong> ${model.min_vram_gb || 0} GB</li>
                                            <li><strong>vLLM:</strong> ${model.supports_vllm ? 'Yes' : 'No'}</li>
                                            <li><strong>Status:</strong> ${model.status}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>

                            <!-- Right Panel: Results -->
                            <div class="col-md-6">
                                <h6 class="mb-3">
                                    <i class="fas fa-clipboard-check me-2"></i>
                                    Results
                                </h6>

                                <!-- Loading State -->
                                <div id="test-loading" style="display: none;">
                                    <div class="text-center py-5">
                                        <div class="spinner-border text-primary mb-3" role="status">
                                            <span class="visually-hidden">Testing...</span>
                                        </div>
                                        <p class="text-muted">Running model test...</p>
                                    </div>
                                </div>

                                <!-- Results Container -->
                                <div id="test-results" style="display: none;">
                                    <!-- Performance Metrics -->
                                    <div class="card mb-3">
                                        <div class="card-header bg-success text-white py-2">
                                            <i class="fas fa-tachometer-alt me-2"></i>
                                            Performance Metrics
                                        </div>
                                        <div class="card-body p-3">
                                            <div class="row text-center">
                                                <div class="col-4">
                                                    <div class="metric-value" id="test-time">-</div>
                                                    <small class="text-muted">Time (s)</small>
                                                </div>
                                                <div class="col-4">
                                                    <div class="metric-value" id="test-tokens">-</div>
                                                    <small class="text-muted">Tokens</small>
                                                </div>
                                                <div class="col-4">
                                                    <div class="metric-value" id="test-tokens-per-sec">-</div>
                                                    <small class="text-muted">Tokens/s</small>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Output Text -->
                                    <div class="card">
                                        <div class="card-header py-2 d-flex justify-content-between align-items-center">
                                            <span>
                                                <i class="fas fa-file-alt me-2"></i>
                                                Output
                                            </span>
                                            <button class="btn btn-sm btn-outline-secondary" onclick="copyTestResults()">
                                                <i class="fas fa-copy me-1"></i> Copy
                                            </button>
                                        </div>
                                        <div class="card-body p-3">
                                            <pre id="test-output" class="mb-0" style="max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;"></pre>
                                        </div>
                                    </div>
                                </div>

                                <!-- Error State -->
                                <div id="test-error" style="display: none;">
                                    <div class="alert alert-danger">
                                        <h6 class="alert-heading">
                                            <i class="fas fa-exclamation-triangle me-2"></i>
                                            Test Failed
                                        </h6>
                                        <p id="test-error-message" class="mb-0"></p>
                                    </div>
                                </div>

                                <!-- Initial State -->
                                <div id="test-initial" class="text-center py-5 text-muted">
                                    <i class="fas fa-vial fa-3x mb-3 opacity-25"></i>
                                    <p>Configure test parameters and click "Run Test" to begin.</p>
                                </div>
                            </div>
                        </div>

                        <!-- Test History -->
                        <div class="mt-4" id="test-history-section" style="display: none;">
                            <h6 class="mb-3">
                                <i class="fas fa-history me-2"></i>
                                Test History
                            </h6>
                            <div class="table-responsive">
                                <table class="table table-sm table-hover">
                                    <thead class="table-light">
                                        <tr>
                                            <th>Time</th>
                                            <th>Prompt</th>
                                            <th>Time (s)</th>
                                            <th>Tokens</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody id="test-history-tbody">
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-success" onclick="exportTestResults()">
                            <i class="fas fa-download me-1"></i> Export Results
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existingModal = document.getElementById('testModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Add CSS for metrics
    const style = document.createElement('style');
    style.textContent = `
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #198754;
        }
    `;
    document.head.appendChild(style);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('testModal'));
    modal.show();

    // Load test history for this model
    loadTestHistory(modelId);
}

/**
 * Run model test
 */
async function runModelTest() {
    if (!currentTestingModel) return;

    const file = document.getElementById('test-file').files[0];
    const prompt = document.getElementById('test-prompt').value;
    const maxTokens = parseInt(document.getElementById('test-max-tokens').value);
    const temperature = parseFloat(document.getElementById('test-temperature').value);

    if (!file && !prompt) {
        showNotification('Please provide either a file or a prompt', 'warning');
        return;
    }

    // Show loading state
    document.getElementById('test-initial').style.display = 'none';
    document.getElementById('test-results').style.display = 'none';
    document.getElementById('test-error').style.display = 'none';
    document.getElementById('test-loading').style.display = 'block';

    const formData = new FormData();
    if (file) {
        formData.append('file', file);
    }
    formData.append('prompt', prompt);
    formData.append('max_tokens', maxTokens);
    formData.append('temperature', temperature);

    try {
        const response = await fetch(`${API_BASE_URL}/api/models/${currentTestingModel.id}/test`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Hide loading
        document.getElementById('test-loading').style.display = 'none';

        if (result.success) {
            // Show results
            document.getElementById('test-results').style.display = 'block';

            // Update performance metrics
            document.getElementById('test-time').textContent = result.performance.total_time_seconds || '-';
            document.getElementById('test-tokens').textContent = result.performance.estimated_tokens || '-';
            document.getElementById('test-tokens-per-sec').textContent = result.performance.tokens_per_second || '-';

            // Update output
            document.getElementById('test-output').textContent = result.result || 'No output';

            // Add to test history
            addToTestHistory({
                timestamp: new Date().toISOString(),
                modelId: currentTestingModel.id,
                modelName: currentTestingModel.display_name || currentTestingModel.name,
                prompt: prompt,
                hasFile: !!file,
                performance: result.performance,
                result: result.result
            });

            showNotification('Test completed successfully!', 'success');
        } else {
            // Show error
            document.getElementById('test-error').style.display = 'block';
            document.getElementById('test-error-message').textContent = result.error || 'Unknown error occurred';
            showNotification('Test failed: ' + (result.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        // Hide loading
        document.getElementById('test-loading').style.display = 'none';

        // Show error
        document.getElementById('test-error').style.display = 'block';
        document.getElementById('test-error-message').textContent = error.message;

        showNotification('Test failed: ' + error.message, 'danger');
    }
}

/**
 * Add test result to history
 */
function addToTestHistory(testResult) {
    testHistory.unshift(testResult);

    // Keep only last 10 tests
    if (testHistory.length > 10) {
        testHistory = testHistory.slice(0, 10);
    }

    // Update history display
    updateTestHistoryDisplay();

    // Save to localStorage
    localStorage.setItem('ocular_test_history', JSON.stringify(testHistory));
}

/**
 * Load test history from localStorage
 */
function loadTestHistory(modelId) {
    const stored = localStorage.getItem('ocular_test_history');
    if (stored) {
        try {
            testHistory = JSON.parse(stored);
            updateTestHistoryDisplay();
        } catch (error) {
            console.warn('Failed to load test history:', error);
        }
    }
}

/**
 * Update test history display
 */
function updateTestHistoryDisplay() {
    const tbody = document.getElementById('test-history-tbody');
    const section = document.getElementById('test-history-section');

    if (!tbody) return;

    if (testHistory.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';

    tbody.innerHTML = testHistory.map((test, index) => `
        <tr>
            <td class="small">${new Date(test.timestamp).toLocaleString()}</td>
            <td class="small">${test.prompt.substring(0, 50)}${test.prompt.length > 50 ? '...' : ''}</td>
            <td class="small">${test.performance.total_time_seconds || '-'}</td>
            <td class="small">${test.performance.estimated_tokens || '-'}</td>
            <td>
                <button class="btn btn-xs btn-outline-primary" onclick="viewTestResult(${index})" title="View">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * View a test result from history
 */
function viewTestResult(index) {
    const test = testHistory[index];

    if (!test) return;

    // Update the results panel
    document.getElementById('test-initial').style.display = 'none';
    document.getElementById('test-loading').style.display = 'none';
    document.getElementById('test-error').style.display = 'none';
    document.getElementById('test-results').style.display = 'block';

    // Update metrics
    document.getElementById('test-time').textContent = test.performance.total_time_seconds || '-';
    document.getElementById('test-tokens').textContent = test.performance.estimated_tokens || '-';
    document.getElementById('test-tokens-per-sec').textContent = test.performance.tokens_per_second || '-';

    // Update output
    document.getElementById('test-output').textContent = test.result || 'No output';

    // Update prompt field
    document.getElementById('test-prompt').value = test.prompt;
}

/**
 * Copy test results to clipboard
 */
function copyTestResults() {
    const output = document.getElementById('test-output').textContent;

    navigator.clipboard.writeText(output).then(() => {
        showNotification('Results copied to clipboard!', 'success');
    }).catch(err => {
        showNotification('Failed to copy results', 'danger');
    });
}

/**
 * Export test results as JSON
 */
function exportTestResults() {
    if (testHistory.length === 0) {
        showNotification('No test history to export', 'warning');
        return;
    }

    const exportData = {
        exported_at: new Date().toISOString(),
        model: currentTestingModel ? {
            id: currentTestingModel.id,
            name: currentTestingModel.name,
            display_name: currentTestingModel.display_name,
            version: currentTestingModel.version
        } : null,
        test_count: testHistory.length,
        tests: testHistory
    };

    // Create JSON blob
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    // Create download link
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ocular-test-results-${new Date().toISOString().split('T')[0]}.json`;

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('Test results exported successfully', 'success');
}
