#!/bin/bash

# Deploy OLMocr vLLM to Lambda Labs GPU instances
# This script launches and configures GPU instances for AI model deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTANCE_NAME="olmocr-vllm-lambda"
INSTANCE_TYPE="gpu_1x_a100_sxm4"  # gpu_1x_a6000, gpu_1x_a100_sxm4, gpu_1x_h100_pcie
REGION="us-west-2"
SSH_KEY_NAME="default"
FILE_SYSTEM_NAME="shared"
LAMBDA_API_KEY=""
HF_TOKEN=""
REPO_URL=""
BRANCH="main"

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

Deploy OLMocr vLLM to Lambda Labs GPU instances

OPTIONS:
    -n, --instance-name NAME Instance name (default: olmocr-vllm-lambda)
    -t, --instance-type TYPE GPU instance type (default: gpu_1x_a100_sxm4)
    -r, --region REGION      Region (default: us-west-2)
    -k, --ssh-key-name NAME  SSH key name (default: default)
    -f, --file-system NAME   File system name (default: shared)
    --lambda-key KEY         Lambda Labs API key (or use LAMBDA_API_KEY env var)
    --hf-token TOKEN         Hugging Face token
    --repo-url URL           GitHub repository URL
    --branch BRANCH          Git branch (default: main)
    --terminate              Terminate the instance
    --status                 Show instance status
    --ssh                    SSH into the instance
    --logs                   Show instance logs
    -h, --help               Show this help message

EXAMPLES:
    # Launch new instance
    $0 --lambda-key your_key --hf-token your_hf_token

    # Launch with custom configuration
    $0 -n my-olmocr -t gpu_1x_h100_pcie --repo-url https://github.com/user/repo

    # Check instance status
    $0 --status

    # SSH into instance
    $0 --ssh

    # Terminate instance
    $0 --terminate

INSTANCE TYPES:
    gpu_1x_a6000      - A6000 GPU (48GB VRAM) - Cost effective
    gpu_1x_a100_sxm4  - A100 SXM4 (80GB VRAM) - High performance
    gpu_1x_h100_pcie  - H100 PCIe (80GB VRAM) - Latest generation
    gpu_2x_a100_sxm4  - 2x A100 SXM4 (160GB VRAM) - Multi-GPU
    gpu_4x_a100_sxm4  - 4x A100 SXM4 (320GB VRAM) - Large scale

REGIONS:
    us-west-2, us-east-1, europe-west-1

REQUIREMENTS:
    - Lambda Labs account and API key
    - SSH key configured in Lambda Labs
    - curl and jq installed

EOF
}

# Function to validate requirements
validate_requirements() {
    print_status "Validating requirements..."

    # Check if curl is installed
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed"
        exit 1
    fi

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        print_error "jq is not installed"
        print_status "Install with: sudo apt-get install jq"
        exit 1
    fi

    # Check API key
    if [[ -z "$LAMBDA_API_KEY" ]]; then
        if [[ -n "$LAMBDA_LABS_API_KEY" ]]; then
            LAMBDA_API_KEY="$LAMBDA_LABS_API_KEY"
        else
            print_error "Lambda Labs API key is required"
            print_status "Get your API key from: https://cloud.lambdalabs.com/api-keys"
            exit 1
        fi
    fi

    print_success "Requirements validated"
}

# Function to make Lambda Labs API call
lambda_api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"

    local url="https://cloud.lambdalabs.com/api/v1${endpoint}"

    if [[ -n "$data" ]]; then
        curl -s -X "$method" \
            -H "Authorization: Bearer $LAMBDA_API_KEY" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$url"
    else
        curl -s -X "$method" \
            -H "Authorization: Bearer $LAMBDA_API_KEY" \
            "$url"
    fi
}

# Function to get instance info
get_instance_info() {
    print_status "Getting instance information..."

    local instances=$(lambda_api_call "GET" "/instances")
    local instance_id=$(echo "$instances" | jq -r ".data[] | select(.name == \"$INSTANCE_NAME\") | .id")

    if [[ -z "$instance_id" || "$instance_id" == "null" ]]; then
        print_error "Instance '$INSTANCE_NAME' not found"
        print_status "Available instances:"
        echo "$instances" | jq -r '.data[] | "\(.name) (\(.id)) - \(.status)"'
        return 1
    fi

    local instance=$(lambda_api_call "GET" "/instances/$instance_id")
    echo "$instance"
}

# Function to launch instance
launch_instance() {
    print_status "Launching Lambda Labs instance..."

    # Create cloud-init script
    local cloud_init_script="$(cat << 'EOF'
#!/bin/bash

# Update system
apt-get update
apt-get install -y docker.io nvidia-docker2 git curl wget

# Restart Docker with GPU support
systemctl restart docker

# Set up environment
export HF_TOKEN="${HF_TOKEN}"
export MODEL_NAME="allenai/olmOCR-7B-0225-preview"

# Create application directory
mkdir -p /app
cd /app

# Clone repository if provided
if [[ -n "${REPO_URL}" ]]; then
    git clone "${REPO_URL}" .
    git checkout "${BRANCH}"
fi

# Create Docker run script
cat > run_olmocr.sh << 'DOCKER_SCRIPT'
#!/bin/bash

docker run -d \
    --name olmocr-vllm \
    --gpus all \
    --restart unless-stopped \
    -p 8080:8080 \
    -e MODEL_NAME=allenai/olmOCR-7B-0225-preview \
    -e HF_TOKEN=${HF_TOKEN} \
    -e GPU_MEMORY_UTILIZATION=0.9 \
    -e DTYPE=bfloat16 \
    -e TRUST_REMOTE_CODE=true \
    -e MAX_MODEL_LEN=4096 \
    -v /shared:/shared \
    -v /app:/app \
    vllm/vllm-openai:latest \
    --model allenai/olmOCR-7B-0225-preview \
    --served-model-name allenai/olmOCR-7B-0225-preview \
    --host 0.0.0.0 \
    --port 8080 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --trust-remote-code

echo "OLMocr vLLM container started"
echo "Health check: curl http://localhost:8080/health"
echo "API endpoint: http://localhost:8080/v1/chat/completions"
DOCKER_SCRIPT

chmod +x run_olmocr.sh

# Start the service
./run_olmocr.sh

# Create status check script
cat > check_status.sh << 'STATUS_SCRIPT'
#!/bin/bash

echo "=== Docker Container Status ==="
docker ps -a | grep olmocr-vllm

echo -e "\n=== Container Logs (last 20 lines) ==="
docker logs olmocr-vllm --tail 20

echo -e "\n=== Health Check ==="
curl -s http://localhost:8080/health | jq . || echo "Health check failed"

echo -e "\n=== GPU Status ==="
nvidia-smi
STATUS_SCRIPT

chmod +x check_status.sh

echo "Setup complete! Use ./check_status.sh to monitor the service"
EOF
)"

    # Substitute environment variables in cloud-init script
    cloud_init_script=$(echo "$cloud_init_script" | \
        sed "s/\${HF_TOKEN}/$HF_TOKEN/g" | \
        sed "s/\${REPO_URL}/$REPO_URL/g" | \
        sed "s/\${BRANCH}/$BRANCH/g")

    # Create launch request
    local launch_data=$(cat << EOF
{
    "region_name": "$REGION",
    "instance_type_name": "$INSTANCE_TYPE",
    "ssh_key_names": ["$SSH_KEY_NAME"],
    "file_system_names": ["$FILE_SYSTEM_NAME"],
    "name": "$INSTANCE_NAME",
    "cloud_init": $(echo "$cloud_init_script" | jq -R -s .)
}
EOF
)

    print_status "Launch configuration:"
    echo "  Name: $INSTANCE_NAME"
    echo "  Type: $INSTANCE_TYPE"
    echo "  Region: $REGION"
    echo "  SSH Key: $SSH_KEY_NAME"

    # Launch the instance
    local result=$(lambda_api_call "POST" "/instance-operations/launch" "$launch_data")

    local instance_ids=$(echo "$result" | jq -r '.data.instance_ids[]?' 2>/dev/null)

    if [[ -z "$instance_ids" ]]; then
        print_error "Failed to launch instance"
        echo "$result" | jq '.error // .'
        exit 1
    fi

    print_success "Instance launched successfully!"
    print_status "Instance ID: $instance_ids"

    # Wait for instance to be running
    print_status "Waiting for instance to be ready..."

    local max_attempts=20
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        local instance_info=$(lambda_api_call "GET" "/instances/$instance_ids")
        local status=$(echo "$instance_info" | jq -r '.data.status')
        local ip=$(echo "$instance_info" | jq -r '.data.ip')

        if [[ "$status" == "active" ]]; then
            print_success "Instance is ready!"
            print_status "IP Address: $ip"
            print_status "SSH: ssh ubuntu@$ip"
            print_status "Service URL: http://$ip:8080"
            break
        fi

        print_status "Instance status: $status (attempt $((attempt + 1))/$max_attempts)"
        sleep 30
        ((attempt++))
    done

    if [[ $attempt -eq $max_attempts ]]; then
        print_warning "Instance taking longer than expected to start"
    fi
}

# Function to terminate instance
terminate_instance() {
    print_status "Terminating instance '$INSTANCE_NAME'..."

    local instance_info=$(get_instance_info)
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    local instance_id=$(echo "$instance_info" | jq -r '.data.id')

    # Confirm termination
    read -p "Are you sure you want to terminate instance '$INSTANCE_NAME' (ID: $instance_id)? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Aborted"
        exit 0
    fi

    # Terminate the instance
    local result=$(lambda_api_call "POST" "/instance-operations/terminate" "{\"instance_ids\": [\"$instance_id\"]}")

    if echo "$result" | jq -e '.data' >/dev/null; then
        print_success "Instance termination initiated"
    else
        print_error "Failed to terminate instance"
        echo "$result" | jq '.error // .'
        exit 1
    fi
}

# Function to show instance status
show_status() {
    local instance_info=$(get_instance_info)
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    print_success "Instance Status:"
    echo "$instance_info" | jq -r '
        "  Name: " + .data.name +
        "\n  ID: " + .data.id +
        "\n  Status: " + .data.status +
        "\n  Type: " + .data.instance_type.name +
        "\n  Region: " + .data.region.name +
        "\n  IP: " + (.data.ip // "N/A") +
        "\n  Created: " + .data.created_timestamp
    '

    local ip=$(echo "$instance_info" | jq -r '.data.ip // empty')
    if [[ -n "$ip" ]]; then
        print_status "Service endpoints:"
        echo "  Health: http://$ip:8080/health"
        echo "  API: http://$ip:8080/v1/chat/completions"
        echo "  SSH: ssh ubuntu@$ip"
    fi
}

# Function to SSH into instance
ssh_instance() {
    local instance_info=$(get_instance_info)
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    local ip=$(echo "$instance_info" | jq -r '.data.ip // empty')
    local status=$(echo "$instance_info" | jq -r '.data.status')

    if [[ "$status" != "active" ]]; then
        print_error "Instance is not active (status: $status)"
        exit 1
    fi

    if [[ -z "$ip" ]]; then
        print_error "Instance IP not available"
        exit 1
    fi

    print_status "Connecting to instance..."
    print_status "IP: $ip"

    ssh ubuntu@"$ip"
}

# Function to show instance logs
show_logs() {
    local instance_info=$(get_instance_info)
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    local ip=$(echo "$instance_info" | jq -r '.data.ip // empty')
    local status=$(echo "$instance_info" | jq -r '.data.status')

    if [[ "$status" != "active" ]]; then
        print_error "Instance is not active (status: $status)"
        exit 1
    fi

    if [[ -z "$ip" ]]; then
        print_error "Instance IP not available"
        exit 1
    fi

    print_status "Fetching logs from instance..."

    ssh ubuntu@"$ip" 'bash -s' << 'EOF'
echo "=== OLMocr Container Status ==="
docker ps -a | grep olmocr-vllm || echo "Container not found"

echo -e "\n=== Container Logs ==="
docker logs olmocr-vllm --tail 50 2>/dev/null || echo "No logs available"

echo -e "\n=== System Status ==="
uptime
df -h
free -h

echo -e "\n=== GPU Status ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--instance-name)
                INSTANCE_NAME="$2"
                shift 2
                ;;
            -t|--instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -k|--ssh-key-name)
                SSH_KEY_NAME="$2"
                shift 2
                ;;
            -f|--file-system)
                FILE_SYSTEM_NAME="$2"
                shift 2
                ;;
            --lambda-key)
                LAMBDA_API_KEY="$2"
                shift 2
                ;;
            --hf-token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --repo-url)
                REPO_URL="$2"
                shift 2
                ;;
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --terminate)
                TERMINATE="true"
                shift
                ;;
            --status)
                STATUS="true"
                shift
                ;;
            --ssh)
                SSH="true"
                shift
                ;;
            --logs)
                LOGS="true"
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

    # Validate requirements
    validate_requirements

    if [[ "$TERMINATE" == "true" ]]; then
        terminate_instance
        exit 0
    fi

    if [[ "$STATUS" == "true" ]]; then
        show_status
        exit 0
    fi

    if [[ "$SSH" == "true" ]]; then
        ssh_instance
        exit 0
    fi

    if [[ "$LOGS" == "true" ]]; then
        show_logs
        exit 0
    fi

    print_status "Lambda Labs GPU instance deployment"
    print_status "Instance: $INSTANCE_NAME"
    print_status "Type: $INSTANCE_TYPE"
    print_status "Region: $REGION"

    # Launch instance
    launch_instance

    cat << EOF

${GREEN}=== DEPLOYMENT COMPLETE ===${NC}

Your OLMocr instance is launching on Lambda Labs!

Monitor your instance:
$0 --status
$0 --logs

Connect to your instance:
$0 --ssh

Terminate your instance:
$0 --terminate

EOF
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi