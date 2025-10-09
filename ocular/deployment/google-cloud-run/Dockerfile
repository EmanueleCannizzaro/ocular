# Google Cloud Run Dockerfile for OLMocr with vLLM
# Optimized for GPU instances (L4, T4, V100, A100)

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_NAME="allenai/olmOCR-7B-0225-preview"
ENV MODEL_REVISION="main"
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV DTYPE="bfloat16"
ENV TRUST_REMOTE_CODE=true
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install core dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and related packages
RUN pip install --no-cache-dir \
    vllm==0.4.0 \
    transformers==4.35.0 \
    tokenizers==0.15.0 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.0 \
    safetensors==0.4.0

# Install web framework and utilities
RUN pip install --no-cache-dir \
    fastapi==0.104.0 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    pillow==10.0.0 \
    numpy==1.24.0 \
    scipy==1.11.0 \
    requests==2.31.0 \
    pdf2image==1.16.3 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1

# Copy application code
COPY cloud_run_app.py /app/main.py
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY requirements.txt /app/

# Create cache directories
RUN mkdir -p /app/cache/models /app/cache/huggingface /tmp/uploads

# Set proper permissions
RUN chmod -R 755 /app && chmod 777 /tmp/uploads

# Pre-download model (optional - comment out to reduce image size)
# RUN python3 -c "
# from transformers import AutoTokenizer, AutoModelForCausalLM;
# model_name = '${MODEL_NAME}';
# print(f'Pre-downloading {model_name}...');
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/app/cache/huggingface', trust_remote_code=True);
# print('Model pre-download completed')
# "

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["python", "main.py"]