# Local LLM Server - System Design

**Last Updated:** February 1, 2026  
**Author:** Development Team  
**Hardware:** NVIDIA RTX 3090 (24GB VRAM)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It All Works](#how-it-all-works)
4. [Model Quantization Explained](#model-quantization-explained)
5. [FastAPI Implementation](#fastapi-implementation)
6. [API Flow](#api-flow)
7. [Memory Management](#memory-management)
8. [Performance Characteristics](#performance-characteristics)

---

## Overview

This system provides a production-ready local LLM server running Llama 3.3 70B on an RTX 3090, accessible via REST API and a web-based chat interface.

### Key Question: How does 40GB+ model fit in 24GB VRAM?

**Short Answer:** Quantization reduces the model from ~140GB (full precision) down to ~22GB in VRAM.

**The Math:**
- Llama 3.3 70B has 70 billion parameters
- Full precision (FP16): 70B × 2 bytes = 140GB
- Q4 quantization: 70B × 0.5 bytes = 35GB on disk → ~22GB in VRAM after optimization
- Q4_K_M variant: Mixed 4-bit + 6-bit quantization for optimal quality/size

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LOCAL NETWORK                          │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │   Browser    │ ───────>│  FastAPI UI  │                │
│  │  (Port 3000) │         │  chat_ui.py  │                │
│  └──────────────┘         └──────┬───────┘                │
│                                   │                         │
│  ┌──────────────┐                │                         │
│  │  Your Backend│ ─────────┐     │                         │
│  │   Service    │          │     │                         │
│  └──────────────┘          ▼     ▼                         │
│                        ┌──────────────┐                     │
│                        │    Ollama    │                     │
│                        │    Server    │                     │
│                        │ (Port 11434) │                     │
│                        └──────┬───────┘                     │
│                               │                             │
│                        ┌──────▼───────┐                     │
│                        │   RTX 3090   │                     │
│                        │   24GB VRAM  │                     │
│                        │              │                     │
│                        │ Llama 3.3 70B│                     │
│                        │   (Q4_K_M)   │                     │
│                        └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Ollama Server** (Core)
   - Handles model loading, inference, and memory management
   - Exposes REST API on port 11434
   - Manages CUDA kernels and GPU memory

2. **FastAPI UI** (Optional)
   - Web-based chat interface
   - Runs on port 3000
   - Proxies requests to Ollama server
   - Handles streaming responses

3. **Backend Clients** (Integration)
   - Python clients for external services
   - Can be sync or async
   - Connect directly to Ollama API

---

## How It All Works

### 1. Model Loading Process

```
User runs: ./start_ollama.sh
    │
    ├─> Ollama server starts
    │
    ├─> Waits for first request
    │
User sends first message
    │
    ├─> Ollama loads model from disk (~/.ollama/models/)
    │   ├─> Reads quantized weights (~40GB file)
    │   ├─> Decompresses into GPU memory
    │   └─> Final VRAM usage: ~22GB
    │
    ├─> Model ready for inference (30 seconds)
    │
    └─> Future requests use cached model (instant)
```

### 2. Request Flow

```
Client Request (JSON)
    │
    ├─> HTTP POST to localhost:11434/api/chat
    │   {
    │     "model": "llama3.3:70b-instruct-q4_K_M",
    │     "messages": [{"role": "user", "content": "Hello"}],
    │     "stream": true/false
    │   }
    │
    ├─> Ollama receives request
    │
    ├─> Tokenizes input text
    │   └─> Converts text to token IDs
    │
    ├─> GPU Inference (CUDA)
    │   ├─> Forward pass through 80 layers
    │   ├─> Attention calculations
    │   ├─> Matrix multiplications (quantized)
    │   └─> Generates next token
    │
    ├─> Decodes token to text
    │
    ├─> Streams or buffers response
    │
    └─> Returns JSON response
```

---

## Model Quantization Explained

### What is Quantization?

Quantization reduces the precision of model weights from high-precision floats to lower-precision integers.

### Llama 3.3 70B Size Breakdown

| Format | Precision | Size | VRAM Usage | Quality |
|--------|-----------|------|------------|---------|
| FP32 | 32-bit | 280GB | 280GB | 100% |
| FP16 | 16-bit | 140GB | 140GB | 99.9% |
| Q8 | 8-bit | 70GB | 72GB | 99% |
| **Q4_K_M** | **4-6 bit mixed** | **~40GB** | **~22GB** | **95-97%** |
| Q3 | 3-bit | 26GB | 18GB | 90-93% |
| Q2 | 2-bit | 18GB | 14GB | 80-85% |

### Why Q4_K_M?

- **K** = k-quants (improved quantization method)
- **M** = Medium (balanced quality/size)
- **Mixed precision**: Important weights stay at 6-bit, others at 4-bit
- **Best for 3090**: Maximizes quality while fitting in 24GB

### The Compression Process

```
Original Model (FP16)
    │
    ├─> Weight: 0.8472651 (16-bit float)
    │
Quantization
    │
    ├─> Clustered into 16 bins (4-bit)
    ├─> Weight becomes: bin #13 (4 bits)
    ├─> Lookup table stores bin centers
    │
    └─> Weight: ~0.85 (4-bit quantized)
```

**VRAM Efficiency:**
- Disk: 40GB (compressed weights + lookup tables)
- VRAM: 22GB (decompressed during inference + KV cache)
- KV Cache: ~2-4GB (stores conversation context)

---

## FastAPI Implementation

### How FastAPI is Used

FastAPI powers the web-based chat UI (`chat_ui.py`). It acts as a proxy/interface between the browser and Ollama server.

### Why FastAPI?

1. **Async Support** - Handles concurrent chat sessions
2. **Streaming** - Server-Sent Events (SSE) for real-time responses
3. **Type Safety** - Pydantic models for request validation
4. **Fast** - Built on Starlette/Uvicorn (high performance)
5. **Simple** - Minimal code to create REST endpoints

### Code Breakdown

```python
# chat_ui.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests

app = FastAPI()

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"

@app.post("/api/chat")
async def chat(request: Request):
    """
    This endpoint receives chat messages from the browser
    and streams responses from Ollama back to the client.
    """
    data = await request.json()  # Get messages from browser
    
    async def generate():
        # Make streaming request to Ollama
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=data,
            stream=True  # Enable streaming
        )
        
        # Stream each chunk back to browser
        for line in response.iter_lines():
            if line:
                yield f"data: {line}\n\n"  # SSE format
    
    # Return Server-Sent Events stream
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream"
    )
```

### Request Flow with FastAPI

```
Browser JavaScript
    │
    ├─> fetch("http://localhost:3000/api/chat", {
    │     method: "POST",
    │     body: JSON.stringify({messages: [...]})
    │   })
    │
    ▼
FastAPI (chat_ui.py)
    │
    ├─> @app.post("/api/chat")
    ├─> Validates request
    ├─> Forwards to Ollama
    │
    ▼
Ollama Server (localhost:11434)
    │
    ├─> Processes with LLM
    ├─> Generates tokens
    │
    ◄── Streams back to FastAPI
    │
FastAPI
    │
    ├─> Converts to SSE format
    ├─> data: {"content": "Hello"}
    ├─> data: {"content": " there"}
    │
    ◄── Streams to Browser
    │
Browser
    │
    └─> Displays real-time response
```

### Server-Sent Events (SSE)

FastAPI uses SSE to stream responses:

```
HTTP Response:
Content-Type: text/event-stream

data: {"content": "The"}
data: {"content": " capital"}
data: {"content": " of"}
data: {"content": " France"}
data: {"content": " is"}
data: {"content": " Paris"}
data: [DONE]
```

Browser receives each chunk as it's generated → smooth typing effect.

---

## API Flow

### Non-Streaming Request

```python
# Client sends
POST http://localhost:11434/api/chat
{
  "model": "llama3.3:70b-instruct-q4_K_M",
  "messages": [{"role": "user", "content": "Hi"}],
  "stream": false
}

# Server processes (2-5 seconds)

# Server responds
{
  "model": "llama3.3:70b-instruct-q4_K_M",
  "created_at": "2026-02-01T...",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true
}
```

### Streaming Request

```python
# Client sends
POST http://localhost:11434/api/chat
{
  "model": "llama3.3:70b-instruct-q4_K_M",
  "messages": [{"role": "user", "content": "Hi"}],
  "stream": true  # Enable streaming
}

# Server streams back (multiple responses)
{"message": {"content": "Hello"}}
{"message": {"content": "!"}}
{"message": {"content": " How"}}
{"message": {"content": " can"}}
{"message": {"content": " I"}}
...
{"done": true}
```

### API Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/api/generate` | Text completion | One-shot generation |
| `/api/chat` | Conversational | Multi-turn chat |
| `/api/tags` | List models | Get available models |
| `/api/embeddings` | Vector embeddings | RAG, semantic search |

---

## Memory Management

### VRAM Breakdown (During Inference)

```
┌─────────────────────────────────────┐
│         RTX 3090 24GB VRAM          │
├─────────────────────────────────────┤
│ Model Weights:        ~18GB (75%)   │  ← Quantized parameters
│ KV Cache:             ~3GB  (12%)   │  ← Conversation context
│ Activations:          ~2GB  (8%)    │  ← Intermediate calculations
│ CUDA Overhead:        ~1GB  (5%)    │  ← Driver, kernels
├─────────────────────────────────────┤
│ Total Used:           ~24GB (100%)  │
└─────────────────────────────────────┘
```

### KV Cache Explained

The Key-Value cache stores computed attention keys/values for previous tokens, so they don't need to be recalculated.

```
Without KV Cache:
- Generate token 1: Process tokens [1]
- Generate token 2: Process tokens [1, 2]  ← Recalculates token 1
- Generate token 3: Process tokens [1, 2, 3]  ← Recalculates 1, 2
Time: O(n²)

With KV Cache:
- Generate token 1: Process [1], cache results
- Generate token 2: Use cache [1], process [2]
- Generate token 3: Use cache [1,2], process [3]
Time: O(n)
```

### Context Length on RTX 3090

**Model Capability:** Llama 3.3 supports up to 128K tokens (~96,000 words)

**3090 Reality Check:**

| Context Length | KV Cache Size | Total VRAM | Status | Use Case |
|----------------|---------------|------------|--------|----------|
| 2K tokens | ~1.5GB | ~20GB | ✅ Excellent | Short chats |
| 4K tokens | ~3GB | ~22GB | ✅ Optimal | Most conversations |
| 8K tokens | ~6GB | ~25GB | ⚠️ Tight | Long chats |
| 16K tokens | ~12GB | ~31GB | ❌ OOM | Won't fit |
| 32K+ tokens | ~24GB+ | ~43GB+ | ❌ OOM | Impossible |

**Practical Recommendations:**

1. **Default: 4K tokens** (3,000 words)
   - Fits comfortably in 24GB VRAM
   - Handles most conversations
   - Good balance of context and safety

2. **Extended: 8K tokens** (6,000 words)
   - Possible but risky
   - Leaves only ~2GB headroom
   - May OOM with long responses
   - Use `--max-model-len 8192` flag

3. **Never exceed 8K** on 3090 with 70B Q4 model

**Why the Limitation?**

KV Cache grows quadratically with context length:
```
Context: 4K tokens
KV Cache = 4,096 × 80 layers × 8 heads × 128 dim × 2 (K+V) × 2 bytes (FP16)
        = ~3GB VRAM

Context: 16K tokens  
KV Cache = 16,384 × ... = ~12GB VRAM
```

**Solutions for Longer Context:**

1. **Use a smaller model**
   - Llama 3.1 8B: Can handle 32K tokens on 3090
   - Llama 3.1 13B: Can handle 16K tokens

2. **Quantize KV cache** (Q8)
   - Reduces cache by 50%
   - Ollama doesn't support this yet
   - vLLM has experimental support

3. **Sliding window**
   - Keep only last N tokens
   - Summarize old context
   - Implemented at application level

**Configuration:**

```bash
# Set max context in vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --max-model-len 4096  # Safe limit for 3090

# Ollama (auto-managed, but can set)
ollama run llama3.3:70b-instruct-q4_K_M \
    --num-ctx 4096
```

**Monitoring Context Usage:**

```python
# Check conversation length
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")  # Similar tokenization

messages = [...]  # Your conversation
tokens = sum(len(enc.encode(m['content'])) for m in messages)
print(f"Current context: {tokens} tokens")

if tokens > 3500:
    print("⚠️ Approaching 4K limit, consider summarizing")
```

---

## Performance Characteristics

### Inference Speed

| Metric | Value | Notes |
|--------|-------|-------|
| First token latency | 2-3s | Initial processing |
| Generation speed | 15-20 tok/s | On RTX 3090 |
| Model load time | 30s | First request only |
| Memory bandwidth | ~936 GB/s | 3090 spec |

### Bottlenecks

1. **Memory Bandwidth** (Primary)
   - Fetching weights from VRAM
   - 3090: 936 GB/s
   - 4090: 1008 GB/s (7% faster)

2. **Compute** (Secondary)
   - Matrix multiplications
   - 3090: 35.6 TFLOPS (FP32)

3. **Quantization Overhead**
   - Dequantizing 4-bit → 16-bit during inference
   - ~10-15% slower than FP16 (but fits in memory!)

### Scaling Considerations

**Concurrent Users:**
```
1 user:  ~22GB VRAM, 15-20 tok/s
2 users: ~24GB VRAM, 7-10 tok/s per user
3 users: OOM (Out of Memory)
```

**Solution for Multiple Users:**
- Use smaller model (13B, 7B)
- Reduce context length
- Queue requests
- Use vLLM with continuous batching

---

## Network Stack

### Protocol Layers

```
Browser (JavaScript)
    ↕ HTTP/JSON
FastAPI (Python)
    ↕ HTTP/JSON
Ollama (Go)
    ↕ C++ bindings
llama.cpp (C++)
    ↕ CUDA calls
NVIDIA Driver
    ↕ PCIe 4.0
RTX 3090 GPU
```

### Port Configuration

- **11434** - Ollama API (HTTP REST)
- **3000** - Chat UI (FastAPI)
- **8000** - vLLM (Alternative, OpenAI-compatible)

### Security Considerations

⚠️ **Current Setup: No Authentication**

For production:
```python
# Add API key authentication
from fastapi import Header, HTTPException

@app.post("/api/chat")
async def chat(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(401, "Invalid API key")
    # ... rest of code
```

---

## Technology Stack

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Engine | llama.cpp | Model inference |
| Server | Ollama (Go) | API server, model management |
| UI | FastAPI (Python) | Web interface |
| Frontend | HTML/JS/CSS | Browser interface |
| GPU | CUDA | Parallel computing |
| Quantization | GGUF format | Compressed models |

### Dependencies

```
Python Environment (uv):
- vllm         → Alternative inference engine
- fastapi      → Web framework
- uvicorn      → ASGI server
- requests     → HTTP client
- aiohttp      → Async HTTP

System:
- CUDA 12.x    → GPU drivers
- Ollama       → Model server
- NVIDIA Driver 550+
```

---

## Why This Architecture?

### Design Decisions

1. **Ollama over vLLM**
   - Easier setup (no Python environment for inference)
   - Better memory management
   - Automatic model downloading
   - Built-in quantization support

2. **FastAPI for UI**
   - Fast development
   - Built-in async support
   - Easy streaming
   - Type safety

3. **Q4_K_M Quantization**
   - Best quality/size tradeoff for 24GB
   - 95-97% of original quality
   - Fits comfortably with headroom

4. **Local-only (no cloud)**
   - Data privacy
   - No API costs
   - Low latency
   - Full control

---

## Performance Optimization

### Current Optimizations

1. **Quantization** - Reduces memory by 85%
2. **KV Cache** - Speeds up sequential generation
3. **CUDA Kernels** - GPU-accelerated operations
4. **Streaming** - Perceived latency reduction

### Potential Improvements

1. **Flash Attention** - 2-4x faster attention (requires newer GPUs)
2. **Continuous Batching** - Handle multiple users efficiently (vLLM)
3. **Model Caching** - Keep model loaded (systemd service)
4. **Speculative Decoding** - 2x faster generation (needs draft model)

---

## Troubleshooting

### Common Issues

**OOM (Out of Memory)**
```
Symptom: CUDA out of memory error
Cause: Model + KV cache exceeds 24GB
Solution: 
- Use Q3 quantization
- Reduce max_tokens
- Shorter context length
```

**Slow Generation**
```
Symptom: <5 tokens/second
Cause: CPU bottleneck or thermal throttling
Solution:
- Check GPU temperature (nvidia-smi)
- Ensure PCIe 4.0 connection
- Close other GPU applications
```

**Port Already in Use**
```
Symptom: Address already in use (11434)
Cause: Ollama already running
Solution: pkill ollama
```

---

## Future Enhancements

- [ ] Add authentication to API
- [ ] Implement request queueing
- [ ] Add monitoring dashboard (Grafana)
- [ ] Support for multiple models
- [ ] Fine-tuning pipeline
- [ ] RAG (Retrieval Augmented Generation) integration
- [ ] Docker containerization
- [ ] Load balancing across multiple GPUs

---

## Conclusion

This system demonstrates how modern quantization techniques enable running state-of-the-art 70B parameter models on consumer hardware. The combination of:

- **Q4 quantization** → Fits in 24GB VRAM
- **Ollama** → Easy model management
- **FastAPI** → Simple web interface
- **RTX 3090** → Strong compute + memory bandwidth

Creates a production-ready local LLM server that rivals cloud-based solutions in quality while maintaining complete privacy and control.

---

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp Quantization](https://github.com/ggerganov/llama.cpp)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Llama 3.3 Model Card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
