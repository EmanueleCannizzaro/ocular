/**
 * Analytics Dashboard JavaScript
 * Handles data visualization and real-time metrics updates
 */

const API_BASE_URL = window.location.origin;
let charts = {};
let refreshInterval = null;

/**
 * Initialize dashboard on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Analytics dashboard loaded');
    initializeDashboard();
    startAutoRefresh();
});

/**
 * Initialize all dashboard components
 */
async function initializeDashboard() {
    await loadDashboardSummary();
    await loadDeploymentTimeline();
    await loadGPUDistribution();
    await loadCostBreakdown();
    await loadModelUsage();
    await loadPerformanceTrends();
}

/**
 * Load dashboard summary metrics
 */
async function loadDashboardSummary() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/analytics/dashboard`);
        const data = await response.json();

        // Update summary cards
        document.getElementById('stat-active-deployments').textContent = data.active_deployments;
        document.getElementById('stat-recent-deployments').textContent = `+${data.recent_deployments_24h}`;
        document.getElementById('stat-gpu-utilization').textContent = `${data.gpu_utilization_percent}%`;
        document.getElementById('stat-gpu-count').textContent = `${data.active_gpus}/${data.total_gpus}`;
        document.getElementById('stat-daily-cost').textContent = `$${data.estimated_daily_cost.toFixed(2)}`;
        document.getElementById('stat-total-models').textContent = data.total_models;
        document.getElementById('stat-recent-failures').textContent = data.recent_failures_24h;

    } catch (error) {
        console.error('Error loading dashboard summary:', error);
    }
}

/**
 * Load and render deployment timeline chart
 */
async function loadDeploymentTimeline() {
    try {
        const days = document.getElementById('period-selector').value;
        const response = await fetch(`${API_BASE_URL}/api/analytics/deployment-timeline?days=${days}`);
        const data = await response.json();

        const ctx = document.getElementById('deployment-timeline-chart');
        if (!ctx) return;

        // Prepare data
        const labels = data.timeline.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        const runningData = data.timeline.map(d => d.running);
        const stoppedData = data.timeline.map(d => d.stopped);
        const failedData = data.timeline.map(d => d.failed);

        // Destroy existing chart if it exists
        if (charts.deploymentTimeline) {
            charts.deploymentTimeline.destroy();
        }

        // Create new chart
        charts.deploymentTimeline = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Running',
                        data: runningData,
                        borderColor: 'rgb(25, 135, 84)',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Stopped',
                        data: stoppedData,
                        borderColor: 'rgb(108, 117, 125)',
                        backgroundColor: 'rgba(108, 117, 125, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Failed',
                        data: failedData,
                        borderColor: 'rgb(220, 53, 69)',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y + ' deployments';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading deployment timeline:', error);
    }
}

/**
 * Load and render GPU distribution donut chart
 */
async function loadGPUDistribution() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/analytics/gpu-utilization`);
        const data = await response.json();

        const ctx = document.getElementById('gpu-distribution-chart');
        if (!ctx) return;

        // Prepare data
        const gpuTypes = Object.keys(data.by_gpu_type);
        const gpuCounts = gpuTypes.map(type => data.by_gpu_type[type].total);
        const colors = [
            'rgb(13, 110, 253)',
            'rgb(25, 135, 84)',
            'rgb(255, 193, 7)',
            'rgb(220, 53, 69)',
            'rgb(111, 66, 193)'
        ];

        // Destroy existing chart if it exists
        if (charts.gpuDistribution) {
            charts.gpuDistribution.destroy();
        }

        // Create new chart
        charts.gpuDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: gpuTypes,
                datasets: [{
                    data: gpuCounts,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const utilization = data.by_gpu_type[label].utilization_percent;
                                return `${label}: ${value} GPUs (${utilization}% utilized)`;
                            }
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading GPU distribution:', error);
    }
}

/**
 * Load and render cost breakdown chart
 */
async function loadCostBreakdown() {
    try {
        const days = 30;
        const response = await fetch(`${API_BASE_URL}/api/analytics/cost-analytics?days=${days}`);
        const data = await response.json();

        const ctx = document.getElementById('cost-breakdown-chart');
        if (!ctx) return;

        // Prepare data
        const labels = data.daily_breakdown.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        const costs = data.daily_breakdown.map(d => d.cost);

        // Destroy existing chart if it exists
        if (charts.costBreakdown) {
            charts.costBreakdown.destroy();
        }

        // Create new chart
        charts.costBreakdown = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Daily Cost (USD)',
                    data: costs,
                    backgroundColor: 'rgba(255, 193, 7, 0.7)',
                    borderColor: 'rgb(255, 193, 7)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });

        // Update total cost display (could add this to the card)
        console.log(`Total cost for ${days} days: $${data.total_cost.toFixed(2)}`);

    } catch (error) {
        console.error('Error loading cost breakdown:', error);
    }
}

/**
 * Load and render model usage chart
 */
async function loadModelUsage() {
    try {
        const days = document.getElementById('period-selector').value;
        const response = await fetch(`${API_BASE_URL}/api/analytics/model-usage?days=${days}`);
        const data = await response.json();

        const ctx = document.getElementById('model-usage-chart');
        if (!ctx) return;

        // Prepare data (top 10 models)
        const topModels = data.models.slice(0, 10);
        const labels = topModels.map(m => m.display_name);
        const counts = topModels.map(m => m.deployment_count);

        // Destroy existing chart if it exists
        if (charts.modelUsage) {
            charts.modelUsage.destroy();
        }

        // Create new chart
        charts.modelUsage = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Deployments',
                    data: counts,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgb(13, 110, 253)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.x + ' deployments';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading model usage:', error);
    }
}

/**
 * Load and render performance trends chart
 */
async function loadPerformanceTrends() {
    try {
        const days = document.getElementById('period-selector').value;
        const response = await fetch(`${API_BASE_URL}/api/analytics/performance-trends?days=${days}`);
        const data = await response.json();

        const ctx = document.getElementById('performance-trends-chart');
        if (!ctx) return;

        // Prepare data
        const labels = data.daily_trends.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        const successRates = data.daily_trends.map(d => d.success_rate_percent);
        const totalDeployments = data.daily_trends.map(d => d.total_deployments);

        // Destroy existing chart if it exists
        if (charts.performanceTrends) {
            charts.performanceTrends.destroy();
        }

        // Create new chart
        charts.performanceTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Success Rate (%)',
                        data: successRates,
                        borderColor: 'rgb(25, 135, 84)',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Total Deployments',
                        data: totalDeployments,
                        borderColor: 'rgb(13, 110, 253)',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Success Rate (%)'
                        },
                        beginAtZero: true,
                        max: 100
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Total Deployments'
                        },
                        beginAtZero: true,
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading performance trends:', error);
    }
}

/**
 * Refresh entire dashboard
 */
async function refreshDashboard() {
    await initializeDashboard();
}

/**
 * Start auto-refresh interval
 */
function startAutoRefresh() {
    // Refresh every 5 minutes
    refreshInterval = setInterval(() => {
        loadDashboardSummary();
    }, 5 * 60 * 1000);
}

/**
 * Stop auto-refresh interval
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});
