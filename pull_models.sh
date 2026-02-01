#!/bin/bash

echo "=================================="
echo "Model Downloader for 3090 (24GB)"
echo "=================================="
echo ""
echo "Recommended models for your GPU:"
echo ""
echo "1. Llama 3.3 70B (Recommended) - Best overall"
echo "2. Qwen2.5 72B - Excellent for coding"
echo "3. Llama 3.1 70B - Stable choice"
echo "4. DeepSeek V2.5 - Great for reasoning"
echo ""
read -p "Enter choice (1-4) or 'all': " choice

pull_llama33() {
    echo "Pulling Llama 3.3 70B (Q4_K_M - ~40GB)..."
    ollama pull llama3.3:70b-instruct-q4_K_M
}

pull_qwen() {
    echo "Pulling Qwen2.5 72B (Q4_K_M - ~42GB)..."
    ollama pull qwen2.5:72b-instruct-q4_K_M
}

pull_llama31() {
    echo "Pulling Llama 3.1 70B (Q4_K_M - ~40GB)..."
    ollama pull llama3.1:70b-instruct-q4_K_M
}

pull_deepseek() {
    echo "Pulling DeepSeek V2.5 (Q4_K_M)..."
    ollama pull deepseek-v2.5:latest
}

case $choice in
    1)
        pull_llama33
        ;;
    2)
        pull_qwen
        ;;
    3)
        pull_llama31
        ;;
    4)
        pull_deepseek
        ;;
    all)
        pull_llama33
        pull_qwen
        pull_llama31
        pull_deepseek
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Model(s) downloaded successfully!"
echo ""
echo "Available models:"
ollama list
