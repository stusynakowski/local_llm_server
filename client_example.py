#!/usr/bin/env python3
"""
Example client for accessing the local LLM server from another machine.
"""

import requests
import json
from typing import Optional

class OllamaClient:
    """Client for Ollama API"""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
    
    def generate(self, prompt: str, model: str = "llama3.3:70b-instruct-q4_K_M", 
                 stream: bool = False) -> str:
        """Generate a response from the model"""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        if stream:
            return response
        else:
            return response.json()["response"]
    
    def chat(self, messages: list, model: str = "llama3.3:70b-instruct-q4_K_M") -> str:
        """Chat with the model"""
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["message"]["content"]


class VLLMClient:
    """Client for vLLM OpenAI-compatible API"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}/v1"
    
    def chat(self, messages: list, model: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Chat completion using OpenAI format"""
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": model or "meta-llama/Llama-3.3-70B-Instruct",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def complete(self, prompt: str, model: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Text completion"""
        url = f"{self.base_url}/completions"
        
        data = {
            "model": model or "meta-llama/Llama-3.3-70B-Instruct",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]


def example_ollama():
    """Example using Ollama"""
    print("=" * 50)
    print("Ollama Example")
    print("=" * 50)
    
    # Replace with your server IP
    client = OllamaClient(host="192.168.1.100", port=11434)
    
    # Simple generation
    response = client.generate("Explain quantum computing in one sentence.")
    print(f"\nResponse: {response}")
    
    # Chat
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = client.chat(messages)
    print(f"\nChat Response: {response}")


def example_vllm():
    """Example using vLLM"""
    print("\n" + "=" * 50)
    print("vLLM Example (OpenAI-compatible)")
    print("=" * 50)
    
    # Replace with your server IP
    client = VLLMClient(host="192.168.1.100", port=8000)
    
    # Chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about programming."}
    ]
    response = client.chat(messages, temperature=0.8)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    print("Local LLM Server Client Examples")
    print("Update the host IP addresses in the code before running!")
    print()
    
    choice = input("Test (1) Ollama or (2) vLLM? ")
    
    if choice == "1":
        example_ollama()
    elif choice == "2":
        example_vllm()
    else:
        print("Invalid choice")
