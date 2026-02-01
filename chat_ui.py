#!/usr/bin/env python3
"""
Local Chat UI for Ollama Server
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import requests
import json
import asyncio
from typing import AsyncGenerator

app = FastAPI(title="Local LLM Chat")

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.3:70b-instruct-q4_K_M"

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chat interface"""
    return HTML_CONTENT

@app.get("/api/models")
async def get_models():
    """Get available models"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/chat")
async def chat(request: Request):
    """Stream chat responses"""
    data = await request.json()
    messages = data.get("messages", [])
    model = data.get("model", DEFAULT_MODEL)
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                },
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    
                    if chunk.get("done", False):
                        yield "data: [DONE]\n\n"
                        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local LLM Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }
        
        .model-select {
            padding: 8px 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 14px;
            cursor: pointer;
        }
        
        .model-select option {
            background: #764ba2;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #f7f8fc;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 12px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            flex-shrink: 0;
        }
        
        .message.user .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant .avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .message-content {
            max-width: 70%;
            padding: 16px 20px;
            border-radius: 16px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .input-container {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e1e4e8;
            display: flex;
            gap: 12px;
        }
        
        #messageInput {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #e1e4e8;
            border-radius: 12px;
            font-size: 15px;
            font-family: inherit;
            resize: none;
            outline: none;
            transition: border-color 0.2s;
        }
        
        #messageInput:focus {
            border-color: #667eea;
        }
        
        #sendButton {
            padding: 14px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        #sendButton:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        #sendButton:active {
            transform: translateY(0);
        }
        
        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 16px 20px;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .empty-state {
            text-align: center;
            color: #888;
            padding: 60px 20px;
        }
        
        .empty-state h2 {
            font-size: 28px;
            margin-bottom: 12px;
            color: #667eea;
        }
        
        .empty-state p {
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Local LLM Chat</h1>
            <select class="model-select" id="modelSelect">
                <option value="llama3.3:70b-instruct-q4_K_M">Llama 3.3 70B</option>
            </select>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="empty-state">
                <h2>Welcome to Your Local LLM</h2>
                <p>Start chatting with your state-of-the-art language model running on your RTX 3090</p>
            </div>
        </div>
        
        <div class="input-container">
            <textarea 
                id="messageInput" 
                placeholder="Type your message..." 
                rows="1"
                onkeydown="if(event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"
            ></textarea>
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        let messages = [];
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const modelSelect = document.getElementById('modelSelect');
        
        // Load available models
        fetch('/api/models')
            .then(r => r.json())
            .then(data => {
                if (data.models) {
                    modelSelect.innerHTML = data.models.map(m => 
                        `<option value="${m.name}">${m.name}</option>`
                    ).join('');
                }
            });
        
        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? 'U' : 'AI';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            
            // Remove empty state
            const emptyState = chatContainer.querySelector('.empty-state');
            if (emptyState) emptyState.remove();
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return contentDiv;
        }
        
        function addTypingIndicator() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = 'typing-indicator';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = 'AI';
            
            const indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.innerHTML = '<span></span><span></span><span></span>';
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(indicator);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function removeTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) indicator.remove();
        }
        
        async function sendMessage() {
            const content = messageInput.value.trim();
            if (!content) return;
            
            // Add user message
            messages.push({ role: 'user', content });
            addMessage('user', content);
            messageInput.value = '';
            
            // Disable input
            sendButton.disabled = true;
            messageInput.disabled = true;
            
            // Add typing indicator
            addTypingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages,
                        model: modelSelect.value
                    })
                });
                
                removeTypingIndicator();
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let contentDiv = null;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;
                            
                            try {
                                const json = JSON.parse(data);
                                if (json.content) {
                                    assistantMessage += json.content;
                                    
                                    if (!contentDiv) {
                                        contentDiv = addMessage('assistant', assistantMessage);
                                    } else {
                                        contentDiv.textContent = assistantMessage;
                                    }
                                    
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
                
                messages.push({ role: 'assistant', content: assistantMessage });
                
            } catch (error) {
                removeTypingIndicator();
                addMessage('assistant', 'Error: ' + error.message);
            }
            
            // Re-enable input
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🤖 Local LLM Chat UI")
    print("=" * 60)
    print(f"Starting chat interface on http://0.0.0.0:3000")
    print(f"Ollama server: {OLLAMA_HOST}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=3000)
