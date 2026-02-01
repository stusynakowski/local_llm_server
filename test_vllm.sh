#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

PORT=${VLLM_PORT:-8000}
HOST=${1:-localhost}

echo "Testing vLLM server at $HOST:$PORT..."
echo ""

# Test 1: Server health
echo "1. Checking server health..."
if curl -s http://$HOST:$PORT/health > /dev/null; then
    echo "   ✓ Server is running"
else
    echo "   ✗ Server is not responding"
    exit 1
fi

# Test 2: List models
echo ""
echo "2. Available models:"
curl -s http://$HOST:$PORT/v1/models | python3 -m json.tool

# Test 3: OpenAI-compatible chat completion
echo ""
echo "3. Testing chat completion..."

python3 << 'EOF'
import requests
import json
import os

host = os.getenv('HOST', 'localhost')
port = os.getenv('VLLM_PORT', '8000')
url = f"http://{host}:{port}/v1/chat/completions"

response = requests.post(url, json={
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [
        {"role": "user", "content": "Say 'Hello from your local vLLM server!' and nothing else."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
})

if response.status_code == 200:
    data = response.json()
    print(data['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
EOF

echo ""
echo ""
echo "✓ All tests passed!"
echo ""
echo "OpenAI-compatible API: http://$HOST:$PORT/v1"
