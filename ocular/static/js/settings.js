/* Settings Page JavaScript - Manages .env file updates */

const API_BASE_URL = window.location.origin;

class SettingsManager {
    constructor() {
        this.settings = {};
        this.originalSettings = {};
        this.init();
    }

    async init() {
        await this.loadSettings();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Save button
        document.getElementById('save-all-btn').addEventListener('click', () => {
            this.saveAllSettings();
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSettings();
        });
    }

    async loadSettings() {
        try {
            showLoading('Loading settings...');

            const response = await fetch(`${API_BASE_URL}/api/settings`);
            if (!response.ok) throw new Error('Failed to load settings');

            const data = await response.json();
            this.settings = data.settings;
            this.originalSettings = JSON.parse(JSON.stringify(this.settings));

            this.populateFormFields();

            hideLoading();
            console.log('Settings loaded successfully');
        } catch (error) {
            console.error('Error loading settings:', error);
            hideLoading();
            showToast('Failed to load settings: ' + error.message, 'error');
        }
    }

    populateFormFields() {
        // Application Settings
        this.setFieldValue('app-title', this.settings.APP_TITLE);
        this.setFieldValue('app-description', this.settings.APP_DESCRIPTION);
        this.setFieldValue('app-version', this.settings.APP_VERSION);
        this.setFieldValue('app-environment', this.settings.ENVIRONMENT);
        this.setChecked('app-debug', this.settings.DEBUG === 'True' || this.settings.DEBUG === 'true');
        this.setFieldValue('app-host', this.settings.HOST);
        this.setFieldValue('app-port', this.settings.PORT);
        this.setFieldValue('app-workers', this.settings.WORKERS);
        this.setFieldValue('app-max-file-size', this.settings.MAX_FILE_SIZE_MB);
        this.setFieldValue('app-max-files', this.settings.MAX_FILES_PER_REQUEST);
        this.setFieldValue('app-supported-formats', this.settings.SUPPORTED_FORMATS);
        this.setFieldValue('app-default-strategy', this.settings.DEFAULT_STRATEGY);
        this.setFieldValue('app-default-providers', this.settings.DEFAULT_PROVIDERS);
        this.setFieldValue('app-max-concurrent', this.settings.MAX_CONCURRENT_REQUESTS);
        this.setFieldValue('app-templates-dir', this.settings.TEMPLATES_DIR);
        this.setFieldValue('app-static-dir', this.settings.STATIC_DIR);
        this.setChecked('app-enable-static', this.settings.ENABLE_STATIC_FILES === 'True' || this.settings.ENABLE_STATIC_FILES === 'true');

        // Database Settings
        this.setFieldValue('db-url', this.settings.DATABASE_URL);
        this.setFieldValue('db-pool-size', this.settings.DB_POOL_SIZE);
        this.setFieldValue('db-max-overflow', this.settings.DB_MAX_OVERFLOW);

        // Authentication
        this.setFieldValue('clerk-secret-key', this.settings.CLERK_SECRET_KEY);
        this.setFieldValue('clerk-publishable-key', this.settings.CLERK_PUBLISHABLE_KEY);
        this.setFieldValue('jwt-secret-key', this.settings.JWT_SECRET_KEY);
        this.setFieldValue('session-timeout', this.settings.AUTH_SESSION_TIMEOUT_MINUTES);

        // Payments
        this.setFieldValue('stripe-secret-key', this.settings.STRIPE_SECRET_KEY);
        this.setFieldValue('stripe-publishable-key', this.settings.STRIPE_PUBLISHABLE_KEY);
        this.setFieldValue('stripe-webhook-secret', this.settings.STRIPE_WEBHOOK_SECRET);

        // OLMocr
        this.setFieldValue('olmocr-provider', this.settings.OLMOCR_PROVIDER);
        this.setFieldValue('runpod-endpoint-id', this.settings.RUNPOD_ENDPOINT_ID);
        this.setFieldValue('runpod-api-key', this.settings.RUNPOD_API_KEY);
        this.setFieldValue('runpod-access-key-id', this.settings.RUNPOD_ACCESS_KEY_ID);
        this.setFieldValue('runpod-secret-access-key', this.settings.RUNPOD_SECRET_ACCESS_KEY);
        this.setFieldValue('runpod-bucket-endpoint', this.settings.RUNPOD_BUCKET_ENDPOINT_URL);
        this.setFieldValue('runpod-network-volume-id', this.settings.RUNPOD_NETWORK_VOLUME_ID);
        this.setFieldValue('runpod-region', this.settings.RUNPOD_REGION);
        this.setFieldValue('vastai-instance-id', this.settings.VASTAI_INSTANCE_ID);
        this.setFieldValue('vastai-api-key', this.settings.VASTAI_API_KEY);
        this.setFieldValue('vastai-endpoint-url', this.settings.VASTAI_ENDPOINT_URL);
        this.setFieldValue('deepinfra-api-key', this.settings.DEEPINFRA_TOKEN);
        this.setFieldValue('deepinfra-model', this.settings.DEEPINFRA_MODEL);
        this.setFieldValue('deepinfra-base-url', this.settings.DEEPINFRA_BASE_URL);
        this.setFieldValue('olmocr-model-name', this.settings.OLMOCR_MODEL_NAME);
        this.setFieldValue('olmocr-max-file-size', this.settings.OLMOCR_MAX_FILE_SIZE_MB);
        this.setFieldValue('olmocr-timeout', this.settings.OLMOCR_TIMEOUT_SECONDS);
        this.setFieldValue('olmocr-allowed-extensions', this.settings.OLMOCR_ALLOWED_EXTENSIONS);

        // OCR Providers
        this.setFieldValue('mistral-api-key', this.settings.MISTRAL_API_KEY);
        this.setFieldValue('mistral-model', this.settings.MISTRAL_MODEL);
        this.setFieldValue('google-credentials', this.settings.GOOGLE_APPLICATION_CREDENTIALS);
        this.setFieldValue('google-project-id', this.settings.GOOGLE_PROJECT_ID);
        this.setFieldValue('aws-access-key-id', this.settings.AWS_ACCESS_KEY_ID);
        this.setFieldValue('aws-secret-access-key', this.settings.AWS_SECRET_ACCESS_KEY);
        this.setFieldValue('aws-region', this.settings.AWS_REGION);
        this.setFieldValue('azure-endpoint', this.settings.AZURE_DOC_INTEL_ENDPOINT);
        this.setFieldValue('azure-api-key', this.settings.AZURE_DOC_INTEL_API_KEY);
        this.setFieldValue('tesseract-language', this.settings.TESSERACT_LANGUAGE);
        this.setFieldValue('tesseract-dpi', this.settings.TESSERACT_DPI);
        this.setFieldValue('hf-token', this.settings.HF_TOKEN);

        // Logging
        this.setFieldValue('log-level', this.settings.LOG_LEVEL);
        this.setFieldValue('log-format', this.settings.LOG_FORMAT);
        this.setFieldValue('logfire-token', this.settings.LOGFIRE_TOKEN);
        this.setChecked('enable-file-logging', this.settings.ENABLE_FILE_LOGGING === 'True' || this.settings.ENABLE_FILE_LOGGING === 'true');
        this.setFieldValue('log-file-path', this.settings.LOG_FILE_PATH);

        // Security
        this.setFieldValue('cors-origins', this.settings.CORS_ORIGINS);
        this.setFieldValue('max-request-size', this.settings.MAX_REQUEST_SIZE_MB);
        this.setFieldValue('rate-limit-rpm', this.settings.RATE_LIMIT_RPM);
        this.setFieldValue('api-secret-key', this.settings.API_SECRET_KEY);
        this.setFieldValue('encryption-key', this.settings.ENCRYPTION_KEY);
        this.setChecked('enable-https-only', this.settings.ENABLE_HTTPS_ONLY === 'True' || this.settings.ENABLE_HTTPS_ONLY === 'true');

        // Cache
        this.setFieldValue('redis-url', this.settings.REDIS_URL);
        this.setFieldValue('cache-ttl', this.settings.CACHE_TTL_SECONDS);
        this.setChecked('enable-caching', this.settings.ENABLE_CACHING === 'True' || this.settings.ENABLE_CACHING === 'true');

        // Trigger OLMocr provider toggle
        if (typeof toggleOLMocrProvider === 'function') {
            toggleOLMocrProvider();
        }
    }

    setFieldValue(id, value) {
        const element = document.getElementById(id);
        if (element && value !== undefined && value !== null) {
            element.value = value;
        }
    }

    setChecked(id, value) {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.checked = value === true || value === 'true' || value === 'True';
        }
    }

    async saveAllSettings() {
        try {
            showLoading('Saving settings to .env file...');

            // Collect all form values
            const updatedSettings = this.collectFormValues();

            // Send to API
            const response = await fetch(`${API_BASE_URL}/api/settings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    settings: updatedSettings,
                    create_backup: true
                })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.message || result.detail || 'Failed to save settings');
            }

            hideLoading();

            // Show success message with details
            let message = `Successfully saved ${result.updated_count} settings to .env file`;
            if (result.backup_created) {
                message += ' (backup created)';
            }

            showToast(message, 'success');

            // Update original settings
            this.originalSettings = JSON.parse(JSON.stringify(updatedSettings));

            // Show reload warning
            this.showReloadWarning();

        } catch (error) {
            console.error('Error saving settings:', error);
            hideLoading();
            showToast('Failed to save settings: ' + error.message, 'error');
        }
    }

    collectFormValues() {
        const settings = {};

        // Helper function to get value
        const getValue = (id) => {
            const el = document.getElementById(id);
            return el ? el.value : '';
        };

        const getChecked = (id) => {
            const el = document.getElementById(id);
            return el ? el.checked : false;
        };

        // Application Settings
        settings.APP_TITLE = getValue('app-title');
        settings.APP_DESCRIPTION = getValue('app-description');
        settings.APP_VERSION = getValue('app-version');
        settings.ENVIRONMENT = getValue('app-environment');
        settings.DEBUG = String(getChecked('app-debug'));
        settings.HOST = getValue('app-host');
        settings.PORT = getValue('app-port');
        settings.WORKERS = getValue('app-workers');
        settings.MAX_FILE_SIZE_MB = getValue('app-max-file-size');
        settings.MAX_FILES_PER_REQUEST = getValue('app-max-files');
        settings.SUPPORTED_FORMATS = getValue('app-supported-formats');
        settings.DEFAULT_STRATEGY = getValue('app-default-strategy');
        settings.DEFAULT_PROVIDERS = getValue('app-default-providers');
        settings.MAX_CONCURRENT_REQUESTS = getValue('app-max-concurrent');
        settings.TEMPLATES_DIR = getValue('app-templates-dir');
        settings.STATIC_DIR = getValue('app-static-dir');
        settings.ENABLE_STATIC_FILES = String(getChecked('app-enable-static'));

        // Database
        settings.DATABASE_URL = getValue('db-url');
        settings.DB_POOL_SIZE = getValue('db-pool-size');
        settings.DB_MAX_OVERFLOW = getValue('db-max-overflow');

        // Authentication
        settings.CLERK_SECRET_KEY = getValue('clerk-secret-key');
        settings.CLERK_PUBLISHABLE_KEY = getValue('clerk-publishable-key');
        settings.JWT_SECRET_KEY = getValue('jwt-secret-key');
        settings.AUTH_SESSION_TIMEOUT_MINUTES = getValue('session-timeout');

        // Payments
        settings.STRIPE_SECRET_KEY = getValue('stripe-secret-key');
        settings.STRIPE_PUBLISHABLE_KEY = getValue('stripe-publishable-key');
        settings.STRIPE_WEBHOOK_SECRET = getValue('stripe-webhook-secret');

        // OLMocr
        settings.OLMOCR_PROVIDER = getValue('olmocr-provider');
        settings.RUNPOD_ENDPOINT_ID = getValue('runpod-endpoint-id');
        settings.RUNPOD_API_KEY = getValue('runpod-api-key');
        settings.RUNPOD_ACCESS_KEY_ID = getValue('runpod-access-key-id');
        settings.RUNPOD_SECRET_ACCESS_KEY = getValue('runpod-secret-access-key');
        settings.RUNPOD_BUCKET_ENDPOINT_URL = getValue('runpod-bucket-endpoint');
        settings.RUNPOD_NETWORK_VOLUME_ID = getValue('runpod-network-volume-id');
        settings.RUNPOD_REGION = getValue('runpod-region');
        settings.VASTAI_INSTANCE_ID = getValue('vastai-instance-id');
        settings.VASTAI_API_KEY = getValue('vastai-api-key');
        settings.VASTAI_ENDPOINT_URL = getValue('vastai-endpoint-url');
        settings.DEEPINFRA_TOKEN = getValue('deepinfra-api-key');
        settings.DEEPINFRA_MODEL = getValue('deepinfra-model');
        settings.DEEPINFRA_BASE_URL = getValue('deepinfra-base-url');
        settings.OLMOCR_MODEL_NAME = getValue('olmocr-model-name');
        settings.OLMOCR_MAX_FILE_SIZE_MB = getValue('olmocr-max-file-size');
        settings.OLMOCR_TIMEOUT_SECONDS = getValue('olmocr-timeout');
        settings.OLMOCR_ALLOWED_EXTENSIONS = getValue('olmocr-allowed-extensions');

        // OCR Providers
        settings.MISTRAL_API_KEY = getValue('mistral-api-key');
        settings.MISTRAL_MODEL = getValue('mistral-model');
        settings.GOOGLE_APPLICATION_CREDENTIALS = getValue('google-credentials');
        settings.GOOGLE_PROJECT_ID = getValue('google-project-id');
        settings.AWS_ACCESS_KEY_ID = getValue('aws-access-key-id');
        settings.AWS_SECRET_ACCESS_KEY = getValue('aws-secret-access-key');
        settings.AWS_REGION = getValue('aws-region');
        settings.AZURE_DOC_INTEL_ENDPOINT = getValue('azure-endpoint');
        settings.AZURE_DOC_INTEL_API_KEY = getValue('azure-api-key');
        settings.TESSERACT_LANGUAGE = getValue('tesseract-language');
        settings.TESSERACT_DPI = getValue('tesseract-dpi');
        settings.HF_TOKEN = getValue('hf-token');

        // Logging
        settings.LOG_LEVEL = getValue('log-level');
        settings.LOG_FORMAT = getValue('log-format');
        settings.LOGFIRE_TOKEN = getValue('logfire-token');
        settings.ENABLE_FILE_LOGGING = String(getChecked('enable-file-logging'));
        settings.LOG_FILE_PATH = getValue('log-file-path');

        // Security
        settings.CORS_ORIGINS = getValue('cors-origins');
        settings.MAX_REQUEST_SIZE_MB = getValue('max-request-size');
        settings.RATE_LIMIT_RPM = getValue('rate-limit-rpm');
        settings.API_SECRET_KEY = getValue('api-secret-key');
        settings.ENCRYPTION_KEY = getValue('encryption-key');
        settings.ENABLE_HTTPS_ONLY = String(getChecked('enable-https-only'));

        // Cache
        settings.REDIS_URL = getValue('redis-url');
        settings.CACHE_TTL_SECONDS = getValue('cache-ttl');
        settings.ENABLE_CACHING = String(getChecked('enable-caching'));

        return settings;
    }

    resetSettings() {
        if (confirm('Reset all settings to their original values?')) {
            this.settings = JSON.parse(JSON.stringify(this.originalSettings));
            this.populateFormFields();
            showToast('Settings reset to original values', 'info');
        }
    }

    showReloadWarning() {
        const message = `
            <div class="alert alert-warning alert-dismissible fade show" role="alert">
                <strong>⚠️ Settings Updated</strong><br>
                Changes have been saved to .env file.<br>
                <strong>You must restart the application</strong> for changes to take effect.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        const container = document.querySelector('.container-fluid');
        if (container) {
            const warningDiv = document.createElement('div');
            warningDiv.innerHTML = message;
            container.insertBefore(warningDiv.firstElementChild, container.firstChild);
        }
    }
}

// Helper functions
function showLoading(message = 'Loading...') {
    const loadingHtml = `
        <div id="loading-overlay" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9999; display: flex; align-items: center; justify-content: center;">
            <div class="bg-white p-4 rounded shadow">
                <div class="spinner-border text-primary me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>${message}</span>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', loadingHtml);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

function showToast(message, type = 'info') {
    const bgColor = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'warning': 'bg-warning',
        'info': 'bg-info'
    }[type] || 'bg-info';

    const toastHtml = `
        <div class="toast align-items-center text-white ${bgColor} border-0" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 10000;">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', toastHtml);
    const toastElement = document.body.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();

    // Remove from DOM after hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function togglePassword(fieldId) {
    const field = document.getElementById(fieldId);
    const button = field.nextElementSibling;
    const icon = button.querySelector('i');

    if (field.type === 'password') {
        field.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        field.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

function toggleOLMocrProvider() {
    const provider = document.getElementById('olmocr-provider');
    if (!provider) return;

    const selectedProvider = provider.value;

    // Hide all provider configs
    const runpodConfig = document.getElementById('runpod-config');
    const vastaiConfig = document.getElementById('vastai-config');
    const deepinfraConfig = document.getElementById('deepinfra-config');

    if (runpodConfig) runpodConfig.style.display = 'none';
    if (vastaiConfig) vastaiConfig.style.display = 'none';
    if (deepinfraConfig) deepinfraConfig.style.display = 'none';

    // Show selected provider config
    if (selectedProvider === 'runpod' && runpodConfig) {
        runpodConfig.style.display = 'block';
    } else if (selectedProvider === 'vastai' && vastaiConfig) {
        vastaiConfig.style.display = 'block';
    } else if (selectedProvider === 'deepinfra' && deepinfraConfig) {
        deepinfraConfig.style.display = 'block';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.settingsManager = new SettingsManager();
});
