#!/bin/bash

echo "=================================="
echo "Install LLM Server as System Service"
echo "=================================="
echo ""
echo "Choose which server to install:"
echo "1. Ollama"
echo "2. vLLM"
echo "3. Both"
echo ""
read -p "Enter choice (1-3): " choice

install_ollama_service() {
    echo "Installing Ollama service..."
    sudo cp systemd/ollama.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ollama.service
    echo "✓ Ollama service installed"
    echo ""
    echo "Commands:"
    echo "  Start:   sudo systemctl start ollama"
    echo "  Stop:    sudo systemctl stop ollama"
    echo "  Status:  sudo systemctl status ollama"
    echo "  Logs:    sudo journalctl -u ollama -f"
}

install_vllm_service() {
    echo "Installing vLLM service..."
    sudo cp systemd/vllm.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable vllm.service
    echo "✓ vLLM service installed"
    echo ""
    echo "Commands:"
    echo "  Start:   sudo systemctl start vllm"
    echo "  Stop:    sudo systemctl stop vllm"
    echo "  Status:  sudo systemctl status vllm"
    echo "  Logs:    sudo journalctl -u vllm -f"
}

case $choice in
    1)
        install_ollama_service
        ;;
    2)
        install_vllm_service
        ;;
    3)
        install_ollama_service
        echo ""
        install_vllm_service
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Installation complete!"
