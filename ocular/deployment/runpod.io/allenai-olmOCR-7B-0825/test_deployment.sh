#!/bin/bash

# Test OLMocr vLLM Deployment on RunPod
# This script tests the deployed endpoint to ensure it's working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENDPOINT_ID=""
API_KEY=""
VERBOSE=false
OUTPUT_FILE=""

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

Test OLMocr vLLM deployment on RunPod

OPTIONS:
    -e, --endpoint-id ID     RunPod endpoint ID (required)
    -k, --api-key KEY        RunPod API key (or use RUNPOD_API_KEY env var)
    -o, --output FILE        Save test results to file
    -v, --verbose            Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    # Basic test
    $0 --endpoint-id abc123 --api-key your_key

    # Test with verbose output and save results
    $0 -e abc123 -k your_key -v -o test_results.json

    # Use environment variables
    export RUNPOD_API_KEY=your_key
    $0 --endpoint-id abc123

EOF
}

# Function to create test image (base64 encoded sample)
create_test_image() {
    # This is a minimal 1x1 PNG in base64 - replace with actual test image
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}

# Function to test endpoint health
test_health() {
    print_status "Testing endpoint health..."

    local health_url="https://api.runpod.ai/v2/${ENDPOINT_ID}/health"

    if [[ "$VERBOSE" == "true" ]]; then
        print_status "Health check URL: $health_url"
    fi

    local response=$(curl -s -w "%{http_code}" -H "Authorization: Bearer $API_KEY" "$health_url" -o /tmp/health_response.json)
    local http_code="${response: -3}"

    if [[ "$http_code" == "200" ]]; then
        print_success "Endpoint is healthy"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "Health response:"
            cat /tmp/health_response.json | jq . 2>/dev/null || cat /tmp/health_response.json
            echo
        fi
        return 0
    else
        print_error "Health check failed with HTTP $http_code"
        cat /tmp/health_response.json 2>/dev/null
        return 1
    fi
}

# Function to test basic text completion
test_text_completion() {
    print_status "Testing text completion..."

    local test_payload='{
        "input": {
            "model": "allenai/olmOCR-7B-0225-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract text from this simple test image and convert to markdown format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,'$(create_test_image)'"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": false
        }
    }'

    local url="https://api.runpod.ai/v2/${ENDPOINT_ID}/run"

    if [[ "$VERBOSE" == "true" ]]; then
        print_status "Test URL: $url"
        print_status "Payload: $test_payload"
    fi

    print_status "Sending test request (this may take 30-60 seconds)..."

    local start_time=$(date +%s)
    local response=$(curl -s -w "%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "$url" \
        -o /tmp/test_response.json)
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    local http_code="${response: -3}"

    if [[ "$http_code" == "200" ]]; then
        print_success "Text completion test passed (${duration}s)"

        # Parse and display response
        if command -v jq &> /dev/null; then
            local output_text=$(cat /tmp/test_response.json | jq -r '.output.choices[0].message.content // .output.text // .output.generated_text // "No text found"')
            print_status "Generated text: $output_text"
        else
            print_warning "jq not available - showing raw response"
            cat /tmp/test_response.json
        fi

        if [[ "$VERBOSE" == "true" ]]; then
            echo
            echo "Full response:"
            cat /tmp/test_response.json | jq . 2>/dev/null || cat /tmp/test_response.json
            echo
        fi

        return 0
    else
        print_error "Text completion test failed with HTTP $http_code"
        cat /tmp/test_response.json 2>/dev/null
        return 1
    fi
}

# Function to test endpoint performance
test_performance() {
    print_status "Testing endpoint performance..."

    local num_requests=3
    local total_time=0
    local successful_requests=0

    for i in $(seq 1 $num_requests); do
        print_status "Performance test $i/$num_requests..."

        local start_time=$(date +%s.%N)
        if test_text_completion &>/dev/null; then
            local end_time=$(date +%s.%N)
            local request_time=$(echo "$end_time - $start_time" | bc)
            total_time=$(echo "$total_time + $request_time" | bc)
            successful_requests=$((successful_requests + 1))
            print_success "Request $i completed in ${request_time}s"
        else
            print_error "Request $i failed"
        fi
    done

    if [[ $successful_requests -gt 0 ]]; then
        local avg_time=$(echo "scale=2; $total_time / $successful_requests" | bc)
        print_success "Performance test completed: $successful_requests/$num_requests successful"
        print_status "Average response time: ${avg_time}s"
    else
        print_error "All performance tests failed"
        return 1
    fi
}

# Function to save test results
save_results() {
    if [[ -n "$OUTPUT_FILE" ]]; then
        print_status "Saving test results to $OUTPUT_FILE..."

        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

        cat > "$OUTPUT_FILE" << EOF
{
    "timestamp": "$timestamp",
    "endpoint_id": "$ENDPOINT_ID",
    "tests": {
        "health_check": "$(test_health &>/dev/null && echo 'passed' || echo 'failed')",
        "text_completion": "$(test_text_completion &>/dev/null && echo 'passed' || echo 'failed')"
    },
    "raw_responses": {
        "health": $(cat /tmp/health_response.json 2>/dev/null || echo '{}'),
        "test": $(cat /tmp/test_response.json 2>/dev/null || echo '{}')
    }
}
EOF

        print_success "Results saved to $OUTPUT_FILE"
    fi
}

# Function to cleanup temporary files
cleanup() {
    rm -f /tmp/health_response.json /tmp/test_response.json
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--endpoint-id)
                ENDPOINT_ID="$2"
                shift 2
                ;;
            -k|--api-key)
                API_KEY="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
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

    # Validate required parameters
    if [[ -z "$ENDPOINT_ID" ]]; then
        print_error "Endpoint ID is required"
        show_usage
        exit 1
    fi

    if [[ -z "$API_KEY" ]]; then
        if [[ -n "$RUNPOD_API_KEY" ]]; then
            API_KEY="$RUNPOD_API_KEY"
        else
            print_error "API key is required (use --api-key or RUNPOD_API_KEY env var)"
            exit 1
        fi
    fi

    print_status "Starting OLMocr endpoint tests..."
    print_status "Endpoint ID: $ENDPOINT_ID"

    # Set up cleanup trap
    trap cleanup EXIT

    # Run tests
    local test_results=0

    if ! test_health; then
        print_error "Health check failed - endpoint may not be ready"
        test_results=1
    fi

    if ! test_text_completion; then
        print_error "Text completion test failed"
        test_results=1
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        test_performance || true  # Don't fail on performance test errors
    fi

    # Save results if requested
    save_results

    if [[ $test_results -eq 0 ]]; then
        print_success "All tests passed! Endpoint is working correctly."

        cat << EOF

${GREEN}=== ENDPOINT READY ===${NC}

Your OLMocr endpoint is ready for use!

Endpoint URL: https://api.runpod.ai/v2/$ENDPOINT_ID/runsync
API Key: ${API_KEY:0:8}...

Add these to your .env file:
RUNPOD_ENDPOINT_ID=$ENDPOINT_ID
RUNPOD_API_KEY=$API_KEY

EOF
    else
        print_error "Some tests failed. Check the endpoint configuration and logs."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi