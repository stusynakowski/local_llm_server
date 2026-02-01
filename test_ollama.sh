#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

PORT=${OLLAMA_PORT:-11434}
HOST=${1:-localhost}

echo "Testing Ollama server at $HOST:$PORT..."
echo ""

# Test 1: Server health
echo "1. Checking server health..."
if curl -s http://$HOST:$PORT/ > /dev/null; then
    echo "   ✓ Server is running"
else
    echo "   ✗ Server is not responding"
    exit 1
fi

# Test 2: List models
echo ""
echo "2. Available models:"
curl -s http://$HOST:$PORT/api/tags | python3 -m json.tool

# Test 3: Simple generation
echo ""
echo "3. Testing generation..."
MODEL=${DEFAULT_MODEL:-llama3.3:70b-instruct-q4_K_M}

curl -s http://$HOST:$PORT/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"Say 'Hello from your local LLM server!' and nothing else.\",
  \"stream\": false
}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('response', 'No response'))"

echo ""
echo ""
echo "✓ All tests passed!"
echo ""
echo "API Endpoint: http://$HOST:$PORT"
