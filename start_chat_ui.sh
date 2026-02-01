#!/bin/bash

echo "Starting Local LLM Chat UI..."

# Activate virtual environment if using vLLM Python deps
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Chat UI will be available at:"
echo "  http://localhost:3000"
echo "  http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo "Make sure Ollama is running on port 11434"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 chat_ui.py
