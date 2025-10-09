# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and git
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install uv, the fast Python package installer
RUN pip install uv

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy the extraction script into the container
COPY extract_data.py .

# Command to run the script when the container launches
CMD ["python3", "extract_data.py"]