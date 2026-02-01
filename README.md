# Local LLM Server for 3090

A complete setup for running state-of-the-art LLMs on your RTX 3090 for local network access.

## Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Choose your server:

# Option A: Ollama (easiest)
./start_ollama.sh

# Option B: vLLM (faster, OpenAI-compatible)
./start_vllm.sh
```

## What's Included

- **Ollama**: Easy-to-use local LLM server
- **vLLM**: High-performance inference with OpenAI API compatibility
- **Python environment**: All dependencies managed
- **Test scripts**: Verify your setup works

## Network Access

Both servers are configured to accept connections from your local network.

- Ollama: `http://YOUR_IP:11434`
- vLLM: `http://YOUR_IP:8000`

## Hardware Requirements

- NVIDIA RTX 3090 (24GB VRAM)
- CUDA drivers installed
- 100GB+ free disk space for models
