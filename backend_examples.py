#!/usr/bin/env python3
"""
Backend API Client Examples for Ollama Server

Use these examples in your backend service to call the local LLM server.
"""

import requests
import json
from typing import List, Dict, Optional, Generator


# =============================================================================
# SIMPLE HTTP REQUESTS (No Dependencies)
# =============================================================================

def simple_generate(prompt: str, server_url: str = "http://192.168.1.100:11434") -> str:
    """
    Simple text generation with a single prompt.
    
    Args:
        prompt: Your input text
        server_url: Your Ollama server URL
        
    Returns:
        Generated text response
    """
    response = requests.post(
        f"{server_url}/api/generate",
        json={
            "model": "llama3.3:70b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def simple_chat(messages: List[Dict[str, str]], server_url: str = "http://192.168.1.100:11434") -> str:
    """
    Chat with conversation history.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
                 Example: [{"role": "user", "content": "Hello"}]
        server_url: Your Ollama server URL
        
    Returns:
        Assistant's response
    """
    response = requests.post(
        f"{server_url}/api/chat",
        json={
            "model": "llama3.3:70b-instruct-q4_K_M",
            "messages": messages,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def streaming_chat(messages: List[Dict[str, str]], server_url: str = "http://192.168.1.100:11434") -> Generator[str, None, None]:
    """
    Streaming chat - yields tokens as they're generated.
    
    Args:
        messages: Conversation history
        server_url: Your Ollama server URL
        
    Yields:
        Individual text chunks as they arrive
    """
    response = requests.post(
        f"{server_url}/api/chat",
        json={
            "model": "llama3.3:70b-instruct-q4_K_M",
            "messages": messages,
            "stream": True
        },
        stream=True
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]


# =============================================================================
# REUSABLE CLIENT CLASS
# =============================================================================

class OllamaBackendClient:
    """
    Production-ready Ollama client for backend services.
    """
    
    def __init__(self, 
                 host: str = "192.168.1.100", 
                 port: int = 11434,
                 model: str = "llama3.3:70b-instruct-q4_K_M",
                 timeout: int = 300):
        """
        Initialize the client.
        
        Args:
            host: Ollama server hostname/IP
            port: Ollama server port
            model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.timeout = timeout
    
    def generate(self, 
                 prompt: str, 
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             system_prompt: Optional[str] = None) -> str:
        """
        Chat with conversation history.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            system_prompt: Optional system message
            
        Returns:
            Assistant's response
        """
        # Prepend system message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    def chat_stream(self, 
                    messages: List[Dict[str, str]], 
                    temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Streaming chat - yields tokens as generated.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            
        Yields:
            Text chunks
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    if content:
                        yield content
    
    def embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text (useful for RAG, semantic search).
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def is_healthy(self) -> bool:
        """
        Check if the server is responding.
        
        Returns:
            True if server is healthy
        """
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except:
            return False


# =============================================================================
# ASYNC VERSION (for FastAPI/async frameworks)
# =============================================================================

import aiohttp
from typing import AsyncGenerator

class AsyncOllamaClient:
    """
    Async Ollama client for async frameworks like FastAPI.
    """
    
    def __init__(self, 
                 host: str = "192.168.1.100", 
                 port: int = 11434,
                 model: str = "llama3.3:70b-instruct-q4_K_M"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
    
    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Async chat."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            ) as response:
                data = await response.json()
                return data["message"]["content"]
    
    async def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Async streaming chat."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature}
                }
            ) as response:
                async for line in response.content:
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                yield content


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_1_simple_request():
    """Example 1: Simple one-off request"""
    prompt = "Explain what a REST API is in one sentence."
    response = simple_generate(prompt, server_url="http://192.168.1.100:11434")
    print(f"Response: {response}")


def example_2_chat_conversation():
    """Example 2: Multi-turn conversation"""
    messages = [
        {"role": "user", "content": "What's the capital of France?"},
    ]
    
    response = simple_chat(messages, server_url="http://192.168.1.100:11434")
    print(f"Assistant: {response}")
    
    # Continue conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What's its population?"})
    
    response = simple_chat(messages, server_url="http://192.168.1.100:11434")
    print(f"Assistant: {response}")


def example_3_streaming():
    """Example 3: Streaming responses"""
    messages = [
        {"role": "user", "content": "Write a short poem about coding."}
    ]
    
    print("Assistant: ", end="", flush=True)
    for chunk in streaming_chat(messages, server_url="http://192.168.1.100:11434"):
        print(chunk, end="", flush=True)
    print()


def example_4_client_class():
    """Example 4: Using the client class"""
    client = OllamaBackendClient(
        host="192.168.1.100",
        port=11434,
        model="llama3.3:70b-instruct-q4_K_M"
    )
    
    # Check if server is up
    if not client.is_healthy():
        print("Server is not responding!")
        return
    
    # Simple generation
    result = client.generate(
        "What is machine learning?",
        temperature=0.7,
        max_tokens=100
    )
    print(f"Generated: {result}")
    
    # Chat with system prompt
    response = client.chat(
        messages=[
            {"role": "user", "content": "How do I sort a list in Python?"}
        ],
        system_prompt="You are a helpful Python programming assistant. Give concise answers with code examples.",
        temperature=0.3
    )
    print(f"Response: {response}")


def example_5_fastapi_integration():
    """Example 5: FastAPI endpoint integration"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    client = OllamaBackendClient(host="192.168.1.100", port=11434)
    
    class ChatRequest(BaseModel):
        message: str
        conversation_history: List[Dict[str, str]] = []
    
    @app.post("/chat")
    async def chat_endpoint(request: ChatRequest):
        """Your backend API endpoint"""
        messages = request.conversation_history + [
            {"role": "user", "content": request.message}
        ]
        
        response = client.chat(messages)
        
        return {
            "response": response,
            "conversation": messages + [{"role": "assistant", "content": response}]
        }
    
    # To run: uvicorn your_file:app --reload


def example_6_async_fastapi():
    """Example 6: Async FastAPI with streaming"""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    
    app = FastAPI()
    client = AsyncOllamaClient(host="192.168.1.100", port=11434)
    
    @app.post("/chat/stream")
    async def chat_stream_endpoint(messages: List[Dict[str, str]]):
        """Streaming chat endpoint"""
        async def generate():
            async for chunk in client.chat_stream(messages):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")


def example_7_error_handling():
    """Example 7: Production error handling"""
    client = OllamaBackendClient(host="192.168.1.100", port=11434, timeout=30)
    
    try:
        # Check server health first
        if not client.is_healthy():
            print("Error: LLM server is not available")
            return
        
        # Make request with error handling
        response = client.chat(
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7
        )
        print(f"Success: {response}")
        
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server")
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_8_embeddings_rag():
    """Example 8: Using embeddings for RAG/semantic search"""
    client = OllamaBackendClient(host="192.168.1.100", port=11434)
    
    # Get embeddings for documents
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "Machine learning models require training data."
    ]
    
    doc_embeddings = [client.embeddings(doc) for doc in documents]
    
    # Get embedding for query
    query = "What is Python?"
    query_embedding = client.embeddings(query)
    
    # Find most similar document (cosine similarity)
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    most_similar_idx = np.argmax(similarities)
    
    print(f"Most relevant document: {documents[most_similar_idx]}")


if __name__ == "__main__":
    print("=" * 70)
    print("Ollama Backend Client Examples")
    print("=" * 70)
    print("\nUpdate the server URL (192.168.1.100) to your actual server IP!\n")
    
    # Run examples
    print("\n--- Example 1: Simple Request ---")
    # example_1_simple_request()
    
    print("\n--- Example 2: Conversation ---")
    # example_2_chat_conversation()
    
    print("\n--- Example 3: Streaming ---")
    # example_3_streaming()
    
    print("\n--- Example 4: Client Class ---")
    # example_4_client_class()
    
    print("\n--- Example 7: Error Handling ---")
    # example_7_error_handling()
    
    print("\nUncomment the examples you want to run!")
