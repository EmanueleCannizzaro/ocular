"""
Modal AI Model Deployment Template
Deploys OLMocr with vLLM on Modal's serverless GPU platform
"""

import modal
from typing import Dict, List, Any

# Modal app configuration
app = modal.App("olmocr-vllm-modal")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "vllm>=0.4.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "fastapi>=0.104.0",
        "pillow>=10.0.0",
        "huggingface-hub>=0.19.0",
        "tokenizers>=0.15.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "requests>=2.31.0"
    ])
    .env({"HF_HOME": "/cache/huggingface"})
)

# Volume for model caching
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# GPU configuration - using A100 for optimal performance
gpu_config = modal.gpu.A100(count=1, memory=40)

# Model loading function (runs once per container)
@app.function(
    image=image,
    gpu=gpu_config,
    cpu=8.0,
    memory=32768,  # 32GB RAM
    timeout=300,   # 5 minutes timeout
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    keep_warm=1,   # Keep 1 instance warm to reduce cold starts
    allow_concurrent_inputs=10
)
class VLLMModel:
    """vLLM model wrapper for Modal deployment"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    @modal.enter()
    def load_model(self):
        """Load the model when the container starts"""
        import vllm
        from transformers import AutoTokenizer

        model_name = "allenai/olmOCR-7B-0225-preview"

        print(f"Loading model: {model_name}")

        # Initialize vLLM engine
        self.model = vllm.LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache/models",
            max_model_len=4096,
            tensor_parallel_size=1,
            disable_log_stats=False
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/cache/huggingface",
            trust_remote_code=True
        )

        print("Model loaded successfully")

    @modal.method()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response using the loaded model"""
        from vllm import SamplingParams
        import base64
        from PIL import Image
        import io

        try:
            # Process messages to extract text and images
            prompt_text = ""
            images = []

            for message in messages:
                if isinstance(message.get("content"), list):
                    for content_item in message["content"]:
                        if content_item.get("type") == "text":
                            prompt_text += content_item.get("text", "")
                        elif content_item.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = content_item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                # Decode base64 image
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(io.BytesIO(image_bytes))
                                images.append(image)
                elif isinstance(message.get("content"), str):
                    prompt_text += message["content"]

            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|im_end|>"]
            )

            # For multimodal models, we need to format the prompt appropriately
            if images:
                # Convert images to the format expected by the model
                formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            # Generate response
            outputs = self.model.generate([formatted_prompt], sampling_params)

            # Format response in OpenAI-compatible format
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": outputs[0].outputs[0].text.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                    "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(outputs[0].outputs[0].token_ids)
                },
                "model": "allenai/olmOCR-7B-0225-preview"
            }

            return response

        except Exception as e:
            print(f"Error during generation: {e}")
            return {
                "error": str(e),
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "model_name": "allenai/olmOCR-7B-0225-preview"
        }

# Create model instance
model = VLLMModel()

# Web endpoints
@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=30
)
@modal.web_endpoint(method="GET", label="health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "olmocr-vllm-modal"}

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=30
)
@modal.web_endpoint(method="POST", label="chat-completions")
def chat_completions(request_data: Dict[str, Any]):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Extract parameters from request
        messages = request_data.get("messages", [])
        max_tokens = request_data.get("max_tokens", 4096)
        temperature = request_data.get("temperature", 0.1)
        top_p = request_data.get("top_p", 0.9)
        stream = request_data.get("stream", False)

        # Validate input
        if not messages:
            return {"error": "Messages are required"}, 400

        # Generate response using the model
        response = model.generate.remote(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream
        )

        return response

    except Exception as e:
        print(f"Error in chat_completions: {e}")
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=30
)
@modal.web_endpoint(method="GET", label="models")
def list_models():
    """List available models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "allenai/olmOCR-7B-0225-preview",
                "object": "model",
                "created": 1677610602,
                "owned_by": "allenai",
                "permission": [],
                "root": "allenai/olmOCR-7B-0225-preview",
                "parent": None
            }
        ]
    }

# Local development and testing
@app.local_entrypoint()
def main():
    """Local testing entrypoint"""

    # Test the model with a sample request
    test_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract text from this document and convert to markdown format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
                    }
                }
            ]
        }
    ]

    print("Testing model deployment...")

    # Test health check
    health_result = model.health_check.remote()
    print(f"Health check: {health_result}")

    # Test generation
    if health_result.get("model_loaded"):
        response = model.generate.remote(
            messages=test_messages,
            max_tokens=512,
            temperature=0.1
        )
        print(f"Generation test: {response}")
    else:
        print("Model not loaded, skipping generation test")

# Deployment configuration
if __name__ == "__main__":
    print("Deploy this app using: modal deploy modal_app.py")
    print("Or run locally: modal run modal_app.py")
