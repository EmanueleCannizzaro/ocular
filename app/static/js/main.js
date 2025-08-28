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