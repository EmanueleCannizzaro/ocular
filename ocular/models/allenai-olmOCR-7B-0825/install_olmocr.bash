#!/bin/bash

# --- Installation Script for olmocr using uv ---

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"

# 2. Create a virtual environment
echo "Creating virtual environment in ./.venv"
uv venv

# 3. Activate the environment and install packages
echo "Installing dependencies from requirements.txt..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "✅ Installation complete. Activate the environment with 'source .venv/bin/activate'"