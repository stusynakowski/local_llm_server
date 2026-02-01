#!/bin/bash

echo "Starting vLLM server on local network..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):8000"

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
HOST=${HOST:-0.0.0.0}
PORT=${VLLM_PORT:-8000}
MODEL=${VLLM_MODEL:-meta-llama/Llama-3.3-70B-Instruct}
QUANTIZATION=${VLLM_QUANTIZATION:-awq}
GPU_UTIL=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_LEN=${MAX_MODEL_LEN:-4096}

echo ""
echo "Model: $MODEL"
echo "Quantization: $QUANTIZATION"
echo "Max Length: $MAX_LEN"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANTIZATION" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --trust-remote-code \
    2>&1 | tee logs/vllm_$(date +%Y%m%d_%H%M%S).log
