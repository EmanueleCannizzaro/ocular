#!/bin/bash

# Deploy OLMocr vLLM to Google Cloud Run with GPU support
# This script builds and deploys the container to Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ID=""
REGION="us-central1"
SERVICE_NAME="olmocr-vllm"
IMAGE_NAME="olmocr-vllm"
MEMORY="32Gi"
CPU="8"
MAX_INSTANCES="10"
CONCURRENCY="80"
TIMEOUT="540s"
GPU_TYPE="nvidia-l4"
GPU_COUNT="1"
ENV_FILE=""
HF_TOKEN=""

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

Deploy OLMocr vLLM to Google Cloud Run with GPU support

OPTIONS:
    -p, --project-id ID      Google Cloud Project ID (required)
    -r, --region REGION      Cloud Run region (default: us-central1)
    -s, --service-name NAME  Service name (default: olmocr-vllm)
    -i, --image-name NAME    Image name (default: olmocr-vllm)
    -m, --memory SIZE        Memory allocation (default: 32Gi)
    -c, --cpu COUNT          CPU allocation (default: 8)
    -g, --gpu-type TYPE      GPU type (default: nvidia-l4)
    --gpu-count COUNT        GPU count (default: 1)
    --max-instances N        Max instances (default: 10)
    --concurrency N          Concurrency (default: 80)
    --timeout TIME           Timeout (default: 540s)
    --env-file FILE          Environment file with variables
    --hf-token TOKEN         Hugging Face token
    --build-only             Only build image, don't deploy
    --deploy-only            Only deploy (image must exist)
    -h, --help               Show this help message

EXAMPLES:
    # Basic deployment
    $0 --project-id my-project --hf-token your_token

    # Custom configuration
    $0 -p my-project -r us-west1 -s my-olmocr --memory 64Gi --cpu 16

    # Build and deploy with environment file
    $0 -p my-project --env-file .env.production

    # Only build image
    $0 -p my-project --build-only

GPU TYPES:
    nvidia-l4      - Cost-effective, good performance
    nvidia-t4      - Basic GPU for lighter workloads
    nvidia-v100    - High-performance training/inference
    nvidia-a100    - Top-tier performance (if available)

REGIONS WITH GPU SUPPORT:
    us-central1, us-west1, us-west4, europe-west1, europe-west4, asia-east1

EOF
}

# Function to validate requirements
validate_requirements() {
    print_status "Validating requirements..."

    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed"
        print_status "Install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        print_status "Install it from: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with gcloud"
        print_status "Run: gcloud auth login"
        exit 1
    fi

    # Check project ID
    if [[ -z "$PROJECT_ID" ]]; then
        print_error "Project ID is required"
        show_usage
        exit 1
    fi

    # Set project
    gcloud config set project "$PROJECT_ID"

    print_success "Requirements validated"
}

# Function to enable required APIs
enable_apis() {
    print_status "Enabling required APIs..."

    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com

    print_success "APIs enabled"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."

    local image_tag="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

    # Build image with Cloud Build
    print_status "Building with Cloud Build..."
    gcloud builds submit \
        --tag "$image_tag" \
        --timeout=20m \
        --machine-type=e2-highcpu-8 \
        .

    print_success "Image built: $image_tag"
}

# Function to deploy to Cloud Run
deploy_service() {
    print_status "Deploying to Cloud Run..."

    local image_tag="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

    # Prepare environment variables
    local env_vars=""
    if [[ -n "$HF_TOKEN" ]]; then
        env_vars="--set-env-vars HF_TOKEN=$HF_TOKEN"
    fi

    if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
        env_vars="$env_vars --env-vars-file $ENV_FILE"
    fi

    # Add default environment variables
    env_vars="$env_vars --set-env-vars MODEL_NAME=allenai/olmOCR-7B-0225-preview"
    env_vars="$env_vars --set-env-vars GPU_MEMORY_UTILIZATION=0.9"
    env_vars="$env_vars --set-env-vars DTYPE=bfloat16"
    env_vars="$env_vars --set-env-vars TRUST_REMOTE_CODE=true"

    # Deploy service
    gcloud run deploy "$SERVICE_NAME" \
        --image "$image_tag" \
        --platform managed \
        --region "$REGION" \
        --gpu "$GPU_COUNT" \
        --gpu-type "$GPU_TYPE" \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --timeout "$TIMEOUT" \
        --concurrency "$CONCURRENCY" \
        --max-instances "$MAX_INSTANCES" \
        --min-instances 0 \
        --allow-unauthenticated \
        --port 8080 \
        $env_vars

    # Get service URL
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format="value(status.url)")

    print_success "Deployment completed!"
    print_status "Service URL: $service_url"
    print_status "Health check: $service_url/health"
    print_status "API endpoint: $service_url/v1/chat/completions"
}

# Function to test deployment
test_deployment() {
    print_status "Testing deployment..."

    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format="value(status.url)")

    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -f -s "$service_url/health" > /dev/null; then
        print_success "Health check passed"
    else
        print_warning "Health check failed - service may still be starting"
    fi

    print_status "Service is deployed at: $service_url"
}

# Function to cleanup on failure
cleanup() {
    if [[ $? -ne 0 ]]; then
        print_error "Deployment failed"
        print_status "Check the logs with:"
        print_status "  gcloud run services logs read $SERVICE_NAME --region $REGION"
    fi
}

# Main function
main() {
    # Set cleanup trap
    trap cleanup EXIT

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -s|--service-name)
                SERVICE_NAME="$2"
                shift 2
                ;;
            -i|--image-name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -m|--memory)
                MEMORY="$2"
                shift 2
                ;;
            -c|--cpu)
                CPU="$2"
                shift 2
                ;;
            -g|--gpu-type)
                GPU_TYPE="$2"
                shift 2
                ;;
            --gpu-count)
                GPU_COUNT="$2"
                shift 2
                ;;
            --max-instances)
                MAX_INSTANCES="$2"
                shift 2
                ;;
            --concurrency)
                CONCURRENCY="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --hf-token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --deploy-only)
                DEPLOY_ONLY=true
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

    print_status "Starting Cloud Run deployment..."
    print_status "Project: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Service: $SERVICE_NAME"
    print_status "GPU: $GPU_TYPE (${GPU_COUNT}x)"

    # Validate requirements
    validate_requirements

    # Enable APIs
    enable_apis

    # Build image (unless deploy-only)
    if [[ "$DEPLOY_ONLY" != "true" ]]; then
        build_image
    fi

    # Deploy service (unless build-only)
    if [[ "$BUILD_ONLY" != "true" ]]; then
        deploy_service
        test_deployment

        cat << EOF

${GREEN}=== DEPLOYMENT COMPLETE ===${NC}

Your OLMocr service is deployed and ready!

Service URL: $(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)")

Test the service:
curl -X POST "$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)")/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "allenai/olmOCR-7B-0225-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract text from this image"},
          {"type": "image_url", {"image_url": {"url": "data:image/jpeg;base64,..."}}}
        ]
      }
    ]
  }'

Monitor logs:
gcloud run services logs read $SERVICE_NAME --region $REGION --follow

EOF
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi