#!/bin/bash

echo "Starting Ollama server on local network..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):11434"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set Ollama to listen on all interfaces
export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT:-11434}

# Start Ollama server
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

ollama serve
