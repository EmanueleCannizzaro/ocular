#!/bin/bash

# OLMocr RunPod Unified Deployment Script
# Supports both vLLM (via CLI) and Custom Docker deployments
# This script automates the deployment of AllenAI's OLMocr model on RunPod serverless platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="olmocr-serverless"
MODEL_NAME="allenai/olmOCR-7B-0825"
ENDPOINT_NAME="olmocr-endpoint"

# Default values
DEPLOYMENT_TYPE="docker"  # docker or vllm
DEPLOYMENT_MODE="quick"   # quick, custom, or template (for vLLM)
GPU_TYPE="NVIDIA RTX 4090"
REGION="US-OR-1"
DRY_RUN=false
VERBOSE=false

# Docker-specific defaults
DOCKER_USERNAME="${DOCKER_USERNAME:-ocular}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-30}"
MIN_WORKERS="${MIN_WORKERS:-0}"
MAX_WORKERS="${MAX_WORKERS:-3}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-30}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-600}"

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

Deploy OLMocr on RunPod Serverless - Supports Docker and vLLM deployments

DEPLOYMENT TYPES:
    docker              Custom Docker container with S3 integration (default)
    vllm                vLLM-optimized deployment via RunPod CLI

OPTIONS:
    -t, --type TYPE          Deployment type: docker or vllm (default: docker)
    -m, --mode MODE          vLLM deployment mode: quick, custom, template (default: quick)
    -g, --gpu GPU_TYPE       GPU type to use (default: "NVIDIA RTX 4090")
    -r, --region REGION      Region to deploy to (default: US-OR-1)
    -n, --name NAME          Custom endpoint name (default: olmocr-endpoint)
    -k, --api-key KEY        RunPod API key (can also use RUNPOD_API_KEY env var)
    -u, --docker-user USER   Docker Hub username (default: ocular)
    -d, --dry-run            Show what would be deployed without actually deploying
    -v, --verbose            Enable verbose output
    -h, --help               Show this help message

DOCKER-SPECIFIC OPTIONS:
    --min-workers N          Minimum workers (default: 0)
    --max-workers N          Maximum workers (default: 3)
    --idle-timeout N         Idle timeout in seconds (default: 30)
    --execution-timeout N    Execution timeout in seconds (default: 600)
    --container-disk N       Container disk in GB (default: 30)

EXAMPLES:
    # Docker deployment (default)
    $0 --type docker --docker-user myusername

    # vLLM quick deployment
    $0 --type vllm --api-key your_runpod_api_key

    # Custom vLLM deployment with specific GPU
    $0 --type vllm --mode custom --gpu "NVIDIA A100" --region "EU-RO-1"

    # Dry run to see what would be deployed
    $0 --dry-run --verbose

ENVIRONMENT VARIABLES:
    RUNPOD_API_KEY              RunPod API key
    RUNPOD_ACCESS_KEY_ID        RunPod S3 access key (for Docker deployment)
    RUNPOD_SECRET_ACCESS_KEY    RunPod S3 secret key (for Docker deployment)
    RUNPOD_BUCKET_ENDPOINT_URL  RunPod S3 endpoint (for Docker deployment)
    RUNPOD_NETWORK_VOLUME_ID    RunPod network volume ID
    DOCKER_USERNAME             Docker Hub username
    HF_TOKEN                    Hugging Face token (for vLLM)

EOF
}

# Function to check prerequisites for vLLM deployment
check_vllm_prerequisites() {
    print_status "Checking vLLM prerequisites..."

    # Check if runpodctl is installed
    if ! command -v runpodctl &> /dev/null; then
        print_error "runpodctl CLI not found. Installing..."
        wget -qO- cli.runpod.net | sudo bash
        print_success "runpodctl installed successfully"
    else
        print_success "runpodctl found: $(runpodctl --version)"
    fi

    # Check API key
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        print_error "RUNPOD_API_KEY not set. Please provide it via --api-key or environment variable."
        exit 1
    fi

    print_success "vLLM prerequisites check completed"
}

# Function to check prerequisites for Docker deployment
check_docker_prerequisites() {
    print_status "Checking Docker prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi

    print_success "Docker found: $(docker --version)"

    # Check if .env file exists
    if [ -f "$SCRIPT_DIR/../../.env" ]; then
        print_status "Loading environment from .env file..."
        source "$SCRIPT_DIR/../../.env"
    elif [ -f "$SCRIPT_DIR/.env" ]; then
        print_status "Loading environment from local .env file..."
        source "$SCRIPT_DIR/.env"
    else
        print_warning "No .env file found. Using environment variables or defaults."
    fi

    # Verify S3 credentials for Docker deployment
    if [[ -z "$RUNPOD_ACCESS_KEY_ID" ]]; then
        print_warning "RUNPOD_ACCESS_KEY_ID not set"
    fi

    if [[ -z "$RUNPOD_SECRET_ACCESS_KEY" ]]; then
        print_warning "RUNPOD_SECRET_ACCESS_KEY not set"
    fi

    print_success "Docker prerequisites check completed"
}

# Function to configure runpodctl
configure_runpodctl() {
    print_status "Configuring runpodctl..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would configure runpodctl with API key"
        return
    fi

    runpodctl config --apiKey "$RUNPOD_API_KEY"
    print_success "runpodctl configured successfully"
}

# Function to build and push Docker image
build_and_push_docker() {
    local image_name="${DOCKER_USERNAME}/olmocr-runpod"
    local full_image="${image_name}:${IMAGE_TAG}"

    print_status "Docker image: $full_image"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would build Docker image: $full_image"
        print_status "[DRY RUN] Would push to Docker Hub"
        return
    fi

    # Step 1: Build Docker image
    print_status "Building Docker image..."
    cd "$SCRIPT_DIR/olmocr" || cd "$SCRIPT_DIR"
    docker build -t "$full_image" . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    print_success "Docker image built successfully"

    # Step 2: Login to Docker Hub
    print_status "Logging in to Docker Hub..."
    docker login || {
        print_error "Failed to login to Docker Hub"
        exit 1
    }
    print_success "Logged in to Docker Hub"

    # Step 3: Push to Docker Hub
    print_status "Pushing image to Docker Hub (this may take several minutes)..."
    docker push "$full_image" || {
        print_error "Failed to push Docker image"
        exit 1
    }
    print_success "Image pushed to Docker Hub"

    echo "$full_image"
}

# Function to deploy using Docker
deploy_docker() {
    print_status "Deploying OLMocr using custom Docker container..."

    local full_image=$(build_and_push_docker)

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would deploy Docker image: $full_image"
        print_status "[DRY RUN] Configuration:"
        print_status "  GPU: $GPU_TYPE"
        print_status "  Container Disk: ${CONTAINER_DISK_GB}GB"
        print_status "  Workers: Min=$MIN_WORKERS, Max=$MAX_WORKERS"
        print_status "  Timeouts: Idle=${IDLE_TIMEOUT}s, Execution=${EXECUTION_TIMEOUT}s"
        return
    fi

    # Deploy using Python API script if available
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        print_warning "RUNPOD_API_KEY not set - cannot deploy automatically"
        print_status ""
        print_status "To deploy manually:"
        print_status "1. Go to https://console.runpod.io/serverless"
        print_status "2. Click 'Deploy Endpoint'"
        print_status "3. Use image: $full_image"
        print_status "4. Configure environment variables from .env file"
    else
        # Deploy using Python script
        local deploy_script="$SCRIPT_DIR/olmocr/deploy_olmocr_api.py"
        if [[ ! -f "$deploy_script" ]]; then
            deploy_script="$SCRIPT_DIR/deploy_olmocr_api.py"
        fi

        if [[ -f "$deploy_script" ]]; then
            print_status "Deploying via API..."
            python3 "$deploy_script" || {
                print_warning "Automatic deployment failed - deploy manually via console"
            }
        else
            print_warning "Deploy API script not found. Please deploy manually via console."
        fi
    fi

    print_success "Docker deployment process completed"
}

# Function to deploy using vLLM Quick Deploy
deploy_vllm_quick() {
    print_status "Deploying OLMocr using vLLM Quick Deploy..."

    local deploy_cmd="runpodctl deploy"
    deploy_cmd+=" --name '$ENDPOINT_NAME'"
    deploy_cmd+=" --image 'runpod/worker-vllm:stable'"
    deploy_cmd+=" --gpu '$GPU_TYPE'"
    deploy_cmd+=" --region '$REGION'"
    deploy_cmd+=" --env MODEL_NAME='$MODEL_NAME'"
    deploy_cmd+=" --env MODEL_REVISION='main'"
    deploy_cmd+=" --env BASE_PATH='/runpod-volume'"
    deploy_cmd+=" --env WORKER_CUDA_VERSION='12.1.0'"
    deploy_cmd+=" --env MAX_MODEL_LEN='4096'"
    deploy_cmd+=" --env GPU_MEMORY_UTILIZATION='0.9'"
    deploy_cmd+=" --env DTYPE='bfloat16'"
    deploy_cmd+=" --env TRUST_REMOTE_CODE='true'"

    if [[ -n "$HF_TOKEN" ]]; then
        deploy_cmd+=" --env HF_TOKEN='$HF_TOKEN'"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would execute: $deploy_cmd"
        return
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        print_status "Executing: $deploy_cmd"
    fi

    eval "$deploy_cmd"
    print_success "vLLM Quick deployment initiated"
}

# Function to deploy using vLLM custom configuration
deploy_vllm_custom() {
    print_status "Deploying OLMocr using vLLM custom configuration..."

    # Check if runpod.toml exists
    local config_file="$SCRIPT_DIR/runpod-config/runpod.toml"
    if [[ ! -f "$config_file" ]]; then
        print_error "runpod.toml not found at $config_file"
        exit 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would create project from $config_file"
        print_status "[DRY RUN] Would execute: runpodctl project deploy"
        return
    fi

    # Create temporary project directory
    local temp_dir=$(mktemp -d)
    local project_dir="$temp_dir/$PROJECT_NAME"

    print_status "Creating project structure in $project_dir..."
    runpodctl project create --name "$PROJECT_NAME" --path "$temp_dir"

    # Copy configuration files
    cp "$SCRIPT_DIR/runpod-config/runpod.toml" "$project_dir/"
    cp "$SCRIPT_DIR/runpod-config/requirements.txt" "$project_dir/"
    cp "$SCRIPT_DIR/runpod-config/.runpodignore" "$project_dir/"

    # Update environment variables in runpod.toml
    if [[ -n "$HF_TOKEN" ]]; then
        sed -i "s/\${HF_TOKEN}/$HF_TOKEN/g" "$project_dir/runpod.toml"
    fi

    cd "$project_dir"
    print_status "Deploying from $project_dir..."
    runpodctl project deploy

    # Cleanup
    cd "$SCRIPT_DIR"
    rm -rf "$temp_dir"

    print_success "vLLM custom deployment completed"
}

# Function to deploy using vLLM template
deploy_vllm_template() {
    print_status "Deploying OLMocr using vLLM template..."

    local template_cmd="runpodctl template deploy"
    template_cmd+=" --template 'runpod/worker-vllm'"
    template_cmd+=" --name '$ENDPOINT_NAME'"
    template_cmd+=" --gpu '$GPU_TYPE'"
    template_cmd+=" --region '$REGION'"
    template_cmd+=" --env MODEL_NAME='$MODEL_NAME'"

    if [[ -n "$HF_TOKEN" ]]; then
        template_cmd+=" --env HF_TOKEN='$HF_TOKEN'"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would execute: $template_cmd"
        return
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        print_status "Executing: $template_cmd"
    fi

    eval "$template_cmd"
    print_success "vLLM template deployment initiated"
}

# Function to check deployment status
check_deployment_status() {
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would check deployment status"
        return
    fi

    print_status "Checking deployment status..."
    sleep 5  # Wait for deployment to start

    if [[ "$DEPLOYMENT_TYPE" == "vllm" ]]; then
        # List endpoints to find our deployment
        runpodctl list endpoints | grep "$ENDPOINT_NAME" || {
            print_warning "Endpoint not found in list. It may still be initializing."
        }
    fi
}

# Function to show post-deployment information
show_deployment_info() {
    print_success "Deployment process completed!"

    cat << EOF

${GREEN}=== DEPLOYMENT INFORMATION ===${NC}

Endpoint Name: $ENDPOINT_NAME
Model: $MODEL_NAME
GPU Type: $GPU_TYPE
Region: $REGION
Deployment Type: $DEPLOYMENT_TYPE

${YELLOW}=== NEXT STEPS ===${NC}

1. Check your RunPod dashboard at: https://console.runpod.io/serverless
2. Wait for the endpoint to become "Running" (this may take several minutes)
3. Once running, get your endpoint ID and test it

${BLUE}Testing the endpoint:${NC}

EOF

    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        cat << EOF
   # Using the test script
   python test_endpoint.py YOUR_ENDPOINT_ID

   # Using curl
   curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \\
     -H "Authorization: Bearer \$RUNPOD_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input": {
         "image": "base64_encoded_image_here",
         "prompt": "Extract all text from this document.",
         "upload_result": true
       }
     }'

EOF
    else
        cat << EOF
   # Using curl for vLLM endpoint
   curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \\
     -H "Authorization: Bearer \$RUNPOD_API_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input": {
         "model": "$MODEL_NAME",
         "messages": [
           {
             "role": "user",
             "content": [
               {"type": "text", "text": "Convert this document to markdown."},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
             ]
           }
         ],
         "max_tokens": 4096
       }
     }'

EOF
    fi

    cat << EOF
${YELLOW}=== USEFUL COMMANDS ===${NC}
- List endpoints: runpodctl list endpoints
- Get endpoint logs: runpodctl logs ENDPOINT_ID
- Update endpoint: runpodctl update endpoint ENDPOINT_ID
- Delete endpoint: runpodctl delete endpoint ENDPOINT_ID

EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            -g|--gpu)
                GPU_TYPE="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -n|--name)
                ENDPOINT_NAME="$2"
                shift 2
                ;;
            -k|--api-key)
                RUNPOD_API_KEY="$2"
                shift 2
                ;;
            -u|--docker-user)
                DOCKER_USERNAME="$2"
                shift 2
                ;;
            --min-workers)
                MIN_WORKERS="$2"
                shift 2
                ;;
            --max-workers)
                MAX_WORKERS="$2"
                shift 2
                ;;
            --idle-timeout)
                IDLE_TIMEOUT="$2"
                shift 2
                ;;
            --execution-timeout)
                EXECUTION_TIMEOUT="$2"
                shift 2
                ;;
            --container-disk)
                CONTAINER_DISK_GB="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
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

    # Validate deployment type
    if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|vllm)$ ]]; then
        print_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        print_error "Valid types: docker, vllm"
        exit 1
    fi

    # Validate deployment mode (for vLLM)
    if [[ "$DEPLOYMENT_TYPE" == "vllm" ]] && [[ ! "$DEPLOYMENT_MODE" =~ ^(quick|custom|template)$ ]]; then
        print_error "Invalid deployment mode: $DEPLOYMENT_MODE"
        print_error "Valid modes: quick, custom, template"
        exit 1
    fi

    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  OLMocr RunPod Unified Deployment${NC}"
    echo -e "${BLUE}  Model: $MODEL_NAME${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""

    print_status "Deployment Type: $DEPLOYMENT_TYPE"
    if [[ "$DEPLOYMENT_TYPE" == "vllm" ]]; then
        print_status "Deployment Mode: $DEPLOYMENT_MODE"
    fi
    print_status "GPU: $GPU_TYPE"
    print_status "Region: $REGION"
    print_status "Endpoint: $ENDPOINT_NAME"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY RUN MODE - No actual deployment will occur"
    fi

    echo ""

    # Execute deployment pipeline based on type
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        check_docker_prerequisites
        deploy_docker
    else
        check_vllm_prerequisites
        configure_runpodctl

        case "$DEPLOYMENT_MODE" in
            quick)
                deploy_vllm_quick
                ;;
            custom)
                deploy_vllm_custom
                ;;
            template)
                deploy_vllm_template
                ;;
        esac
    fi

    check_deployment_status
    show_deployment_info
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
