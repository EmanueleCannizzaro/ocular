#!/bin/bash

# Deploy OLMocr vLLM to DigitalOcean App Platform with GPU support
# This script creates and deploys an app using the DigitalOcean CLI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="olmocr-vllm-app"
REGION="nyc"
REPO_URL=""
BRANCH="main"
INSTANCE_SIZE="gpu-h100-1x"  # gpu-rtx6000-1x, gpu-h100-1x
HF_TOKEN=""
DO_TOKEN=""
CONFIG_FILE=".do/app.yaml"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy OLMocr vLLM to DigitalOcean App Platform with GPU support

OPTIONS:
    -n, --app-name NAME      App name (default: olmocr-vllm-app)
    -r, --repo-url URL       GitHub repository URL (required)
    -b, --branch BRANCH      Git branch (default: main)
    -s, --instance-size SIZE GPU instance size (default: gpu-h100-1x)
    -t, --hf-token TOKEN     Hugging Face token
    --do-token TOKEN         DigitalOcean API token (or use DIGITALOCEAN_ACCESS_TOKEN env var)
    --region REGION          DigitalOcean region (default: nyc)
    --config-file FILE       App configuration file (default: .do/app.yaml)
    --update                 Update existing app instead of creating new one
    --destroy                Destroy the app
    -h, --help               Show this help message

EXAMPLES:
    # Create new app
    $0 --repo-url https://github.com/user/repo --hf-token your_token

    # Update existing app
    $0 --repo-url https://github.com/user/repo --update

    # Use custom configuration
    $0 --repo-url https://github.com/user/repo --config-file custom-app.yaml

    # Destroy app
    $0 --app-name my-app --destroy

INSTANCE SIZES:
    gpu-rtx6000-1x  - RTX 6000 GPU (24GB VRAM)
    gpu-h100-1x     - H100 GPU (80GB VRAM) - Premium tier

REQUIREMENTS:
    - DigitalOcean CLI (doctl) installed and authenticated
    - GitHub repository with OLMocr application code
    - Hugging Face token for model access

EOF
}

# Function to validate requirements
validate_requirements() {
    print_status "Validating requirements..."

    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        print_error "DigitalOcean CLI (doctl) is not installed"
        print_status "Install it from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        exit 1
    fi

    # Check if authenticated
    if [[ -n "$DO_TOKEN" ]]; then
        export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"
    fi

    if ! doctl auth list | grep -q "current context"; then
        print_error "Not authenticated with DigitalOcean"
        print_status "Run: doctl auth init"
        exit 1
    fi

    # Check repository URL
    if [[ -z "$REPO_URL" && "$DESTROY" != "true" ]]; then
        print_error "Repository URL is required"
        show_usage
        exit 1
    fi

    print_success "Requirements validated"
}

# Function to create app configuration
create_app_config() {
    print_status "Creating app configuration..."

    # Update configuration file with provided values
    if [[ -f "$CONFIG_FILE" ]]; then
        # Create temporary config with substitutions
        local temp_config="/tmp/app-config-$(date +%s).yaml"
        cp "$CONFIG_FILE" "$temp_config"

        # Replace placeholders
        sed -i "s|name: .*|name: $APP_NAME|g" "$temp_config"
        sed -i "s|region: .*|region: $REGION|g" "$temp_config"
        sed -i "s|instance_size_slug: .*|instance_size_slug: $INSTANCE_SIZE|g" "$temp_config"

        if [[ -n "$REPO_URL" ]]; then
            sed -i "s|repo_clone_url: .*|repo_clone_url: $REPO_URL|g" "$temp_config"
            sed -i "s|branch: .*|branch: $BRANCH|g" "$temp_config"
        fi

        CONFIG_FILE="$temp_config"
    else
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi

    print_success "App configuration created"
}

# Function to set environment variables
set_environment_variables() {
    local app_id="$1"

    print_status "Setting environment variables..."

    if [[ -n "$HF_TOKEN" ]]; then
        print_status "Setting HF_TOKEN..."
        doctl apps update "$app_id" --spec - <<EOF
name: $APP_NAME
services:
- name: olmocr-api
  envs:
  - key: HF_TOKEN
    value: $HF_TOKEN
    type: SECRET
EOF
    fi

    print_success "Environment variables set"
}

# Function to create app
create_app() {
    print_status "Creating DigitalOcean App..."

    # Create the app
    local app_output=$(doctl apps create --spec "$CONFIG_FILE" --format json)
    local app_id=$(echo "$app_output" | jq -r '.id')

    if [[ "$app_id" == "null" || -z "$app_id" ]]; then
        print_error "Failed to create app"
        echo "$app_output"
        exit 1
    fi

    print_success "App created with ID: $app_id"

    # Set environment variables
    if [[ -n "$HF_TOKEN" ]]; then
        set_environment_variables "$app_id"
    fi

    # Wait for deployment
    print_status "Waiting for deployment to complete..."

    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        local status=$(doctl apps get "$app_id" --format json | jq -r '.last_deployment_active_at')

        if [[ "$status" != "null" && -n "$status" ]]; then
            print_success "Deployment completed!"
            break
        fi

        print_status "Deployment in progress... (attempt $((attempt + 1))/$max_attempts)"
        sleep 30
        ((attempt++))
    done

    if [[ $attempt -eq $max_attempts ]]; then
        print_warning "Deployment taking longer than expected"
    fi

    # Get app info
    get_app_info "$app_id"
}

# Function to update existing app
update_app() {
    print_status "Updating existing app..."

    # Get app ID
    local app_id=$(doctl apps list --format json | jq -r ".[] | select(.spec.name == \"$APP_NAME\") | .id")

    if [[ -z "$app_id" || "$app_id" == "null" ]]; then
        print_error "App '$APP_NAME' not found"
        print_status "Available apps:"
        doctl apps list --format "ID,Name,Status"
        exit 1
    fi

    print_status "Found app: $app_id"

    # Update the app
    doctl apps update "$app_id" --spec "$CONFIG_FILE"

    print_success "App update initiated"

    # Get updated app info
    get_app_info "$app_id"
}

# Function to destroy app
destroy_app() {
    print_status "Destroying app '$APP_NAME'..."

    # Get app ID
    local app_id=$(doctl apps list --format json | jq -r ".[] | select(.spec.name == \"$APP_NAME\") | .id")

    if [[ -z "$app_id" || "$app_id" == "null" ]]; then
        print_error "App '$APP_NAME' not found"
        exit 1
    fi

    # Confirm destruction
    read -p "Are you sure you want to destroy app '$APP_NAME' (ID: $app_id)? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Aborted"
        exit 0
    fi

    # Destroy the app
    doctl apps delete "$app_id" --force

    print_success "App destroyed"
}

# Function to get app information
get_app_info() {
    local app_id="$1"

    print_status "Getting app information..."

    local app_info=$(doctl apps get "$app_id" --format json)
    local app_url=$(echo "$app_info" | jq -r '.default_ingress')
    local status=$(echo "$app_info" | jq -r '.phase')

    print_success "App deployment information:"
    echo "  App ID: $app_id"
    echo "  Name: $APP_NAME"
    echo "  Status: $status"
    echo "  URL: $app_url"
    echo "  Health Check: $app_url/health"
    echo "  API Endpoint: $app_url/v1/chat/completions"

    # Show deployment logs
    print_status "Recent deployment logs:"
    doctl apps logs "$app_id" --type deploy --tail 20 || true
}

# Function to test deployment
test_deployment() {
    local app_url="$1"

    print_status "Testing deployment..."

    # Test health endpoint
    if curl -f -s "$app_url/health" > /dev/null; then
        print_success "Health check passed"
    else
        print_warning "Health check failed - service may still be starting"
    fi
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--app-name)
                APP_NAME="$2"
                shift 2
                ;;
            -r|--repo-url)
                REPO_URL="$2"
                shift 2
                ;;
            -b|--branch)
                BRANCH="$2"
                shift 2
                ;;
            -s|--instance-size)
                INSTANCE_SIZE="$2"
                shift 2
                ;;
            -t|--hf-token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --do-token)
                DO_TOKEN="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --config-file)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --update)
                UPDATE="true"
                shift
                ;;
            --destroy)
                DESTROY="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    print_status "DigitalOcean App Platform deployment"
    print_status "App: $APP_NAME"
    print_status "Region: $REGION"
    print_status "Instance: $INSTANCE_SIZE"

    # Validate requirements
    validate_requirements

    if [[ "$DESTROY" == "true" ]]; then
        destroy_app
        exit 0
    fi

    # Create app configuration
    create_app_config

    if [[ "$UPDATE" == "true" ]]; then
        update_app
    else
        create_app
    fi

    cat << EOF

${GREEN}=== DEPLOYMENT COMPLETE ===${NC}

Your OLMocr service is deployed on DigitalOcean App Platform!

Monitor your app:
doctl apps list
doctl apps get [APP_ID]
doctl apps logs [APP_ID] --type run --follow

Update your app:
$0 --app-name $APP_NAME --repo-url $REPO_URL --update

EOF
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi