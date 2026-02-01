#!/bin/bash
set -e

echo "=================================="
echo "Local LLM Server Setup for 3090"
echo "=================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install uv if not already installed
echo ""
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✓ uv already installed"
fi

# Create Python virtual environment with uv
echo ""
echo "Creating Python virtual environment with uv..."
uv venv

# Install Python dependencies with uv
echo ""
echo "Installing Python dependencies with uv..."
uv pip install -r requirements.txt

# Install Ollama
echo ""
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama already installed"
fi

# Create directories
mkdir -p logs
mkdir -p models

# Make scripts executable
chmod +x start_ollama.sh
chmod +x start_vllm.sh
chmod +x test_ollama.sh
chmod +x test_vllm.sh
chmod +x pull_models.sh

echo ""
echo "=================================="
echo "✓ Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Pull a model: ./pull_models.sh"
echo "2. Start server: ./start_ollama.sh or ./start_vllm.sh"
echo "3. Test: ./test_ollama.sh or ./test_vllm.sh"
