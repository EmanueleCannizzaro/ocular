# OLMocr RunPod Serverless Deployment
# Base: PyTorch with CUDA support for GPU acceleration
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to reduce cold start time (optional but recommended)
# Uncomment the following lines to bake the model into the image
# RUN python -c "from transformers import AutoModel, AutoProcessor; \
#     model = AutoModel.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True); \
#     processor = AutoProcessor.from_pretrained('allenai/olmOCR-7B-0825', trust_remote_code=True)"

# Copy handler
COPY handler.py .

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Set Hugging Face cache directory to network volume if available
ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache

# Command to run handler
CMD ["python", "-u", "handler.py"]
