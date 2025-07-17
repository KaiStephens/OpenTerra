#!/usr/bin/env python3
"""
OpenTerra Chat Web UI

A FastAPI-based web interface for chatting with the latest AI models including Claude 4.
Features real-time chat, API key management, provider switching, and custom model input.
Updated January 2025 with latest AI models.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import AI SDKs
try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not found. Install with: pip install anthropic")
    anthropic = None

try:
    import openai
except ImportError:
    print("Warning: openai package not found. Install with: pip install openai")
    openai = None

from provider_config import (
    PROVIDERS,
    get_available_providers, 
    get_provider_models, 
    get_provider_config,
    get_model_display_name,
    DEFAULT_MODELS,
    is_custom_model,
    validate_custom_model_format
)

from agent_system import Agent, validate_workspace_directory

# Pydantic Models
class ChatMessage(BaseModel):
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = None

class ChatRequest(BaseModel):
    message: str
    settings: Dict[str, Any]
    chatId: Optional[str] = None
    messages: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    chatId: Optional[str] = None
    modelUsed: Optional[str] = None

class ChatSettings(BaseModel):
    provider: str
    apiKey: str
    model: Optional[str] = None
    customModel: Optional[str] = ""  # For custom model input

class AgentExecuteRequest(BaseModel):
    taskId: str
    instruction: str
    settings: Dict[str, Any]
    workspaceDir: Optional[str] = None

class AgentExecuteResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    actions: List[str] = []
    primaryAction: Optional[str] = None
    iterations: Optional[int] = None

class ProviderInfo(BaseModel):
    id: str
    name: str
    apiKeyLabel: str
    defaultModel: str
    models: List[Dict[str, str]]

# FastAPI app setup
app = FastAPI(title="OpenTerra Chat", description="Chat interface for latest AI models including Claude 4")

# Setup static files and templates
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Chat management
class ChatManager:
    def __init__(self):
        self.websocket_connections: List[WebSocket] = []
        self.chats_dir = Path("chat_history")
        self.chats_dir.mkdir(exist_ok=True)
    
    async def broadcast_message(self, message_type: str, data: Any):
        """Broadcast a message to all connected clients."""
        if self.websocket_connections:
            message = {"type": message_type, "data": data}
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
            
            for ws in disconnected:
                self.websocket_connections.remove(ws)

    def get_api_client(self, provider: str, api_key: str):
        """Get the appropriate API client for the provider"""
        provider_config = get_provider_config(provider)
        
        if provider == "anthropic":
            return anthropic.Anthropic(api_key=api_key)
        elif provider in ["openrouter", "moonshot"]:
            # Both use OpenRouter API
            return openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def send_chat_message(self, settings: ChatSettings, user_message: str, chat_history: List[Dict] = None) -> ChatResponse:
        """Send a message to AI and get response."""
        try:
            # Validate settings
            if not settings.provider or not settings.apiKey:
                            return ChatResponse(
                response=None,
                error="Please configure your API key in settings",
                chatId=None
            )
            
            # Determine which model to use
            final_model = settings.model or DEFAULT_MODELS.get(settings.provider, "")
            
            # Use custom model if provided and valid
            if settings.customModel and validate_custom_model_format(settings.customModel):
                final_model = settings.customModel.strip()
                
            if not final_model:
                return ChatResponse(
                    response=None,
                    error="Please select a model or enter a custom model",
                    chatId=None
                )
            
            # Create API client
            client = self.get_api_client(settings.provider, settings.apiKey)
            
            # Prepare messages
            messages = self._prepare_messages(user_message, chat_history or [])
            
            # Send message based on provider
            if settings.provider == "anthropic":
                response_text = await self._send_anthropic_message(client, final_model, messages)
            else:
                # OpenRouter or MoonshotAI
                response_text = await self._send_openrouter_message(client, final_model, messages)
            
            return ChatResponse(
                response=response_text,
                error=None,
                chatId=None,
                modelUsed=final_model
            )
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                error_msg = "Invalid API key. Please check your credentials."
            elif "model" in error_msg.lower() or "not found" in error_msg.lower():
                error_msg = f"Model '{final_model}' not available. Please check your model selection."
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                error_msg = "Rate limit exceeded. Please try again later."
            elif "billing" in error_msg.lower():
                error_msg = "Billing issue detected. Please check your account."
            
            return ChatResponse(
                response=None,
                error=error_msg,
                chatId=None,
                modelUsed=final_model if 'final_model' in locals() else None
            )
    
    async def _send_anthropic_message(self, client, model: str, messages: List[Dict]) -> str:
        """Send message using Anthropic API"""
        # Convert OpenAI format to Anthropic format
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        kwargs = {
            "model": model,
            "max_tokens": 4000,  # Fixed smart token limit
            "temperature": 0.7,  # Fixed optimal temperature
            "messages": anthropic_messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = client.messages.create(**kwargs)
        
        # Extract response text
        response_text = ""
        for content in response.content:
            if hasattr(content, 'text'):
                response_text += content.text
                
        return response_text
    
    async def _send_openrouter_message(self, client, model: str, messages: List[Dict]) -> str:
        """Send message using OpenRouter API (for OpenRouter and MoonshotAI)"""
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,  # Fixed smart token limit
            temperature=0.7,  # Fixed optimal temperature
            extra_headers={
                "HTTP-Referer": "https://openterra.ai",
                "X-Title": "OpenTerra Chat"
            }
        )
        
        return response.choices[0].message.content
    
    def _prepare_messages(self, user_message: str, chat_history: List[Dict]) -> List[Dict]:
        """Prepare messages for API."""
        messages = []
        
        # Add recent chat history (limit to prevent token overflow)
        for msg in chat_history[-10:]:  # Last 10 messages for context
            if msg.get('role') in ['user', 'assistant', 'system']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    async def validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate API key for the given provider"""
        try:
            client = self.get_api_client(provider, api_key)
            
            if provider == "anthropic":
                # Test with a simple message for Anthropic
                response = client.messages.create(
                    model="claude-3-7-sonnet",  # Updated to current model
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                return True
            else:
                # Test with OpenRouter (for openrouter and moonshot)
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",  # Use a small, cheap model for testing
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                return True
        except Exception as e:
            print(f"API key validation failed for {provider}: {e}")
            return False

chat_manager = ChatManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main chat interface page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle chat messages."""
    try:
        # Convert settings dict to ChatSettings
        settings = ChatSettings(**request.settings)
        
        # Send message to AI
        response = await chat_manager.send_chat_message(
            settings=settings,
            user_message=request.message,
            chat_history=request.messages
        )
        
        response.chatId = request.chatId or f"chat-{uuid.uuid4().hex[:8]}"
        
        # Broadcast to websocket clients if needed
        await chat_manager.broadcast_message("chat_response", response.model_dump())
        
        return response
        
    except Exception as e:
        return ChatResponse(
            response=None,
            error=f"Server error: {str(e)}",
            chatId=request.chatId or f"chat-{uuid.uuid4().hex[:8]}"
        )

@app.get("/api/providers")
async def get_providers():
    """Get all available providers and their models."""
    providers_data = []
    
    for provider_name in get_available_providers():
        config = get_provider_config(provider_name)
        models = get_provider_models(provider_name)
        
        # Add display names for models
        models_with_display = []
        for model in models:
            models_with_display.append({
                "id": model,
                "name": get_model_display_name(model)
            })
        
        providers_data.append({
            "id": provider_name,
            "name": config["name"],
            "apiKeyLabel": config["api_key_label"],
            "defaultModel": config["default_model"],
            "models": models_with_display,
            "supportsCustomModels": provider_name == "openrouter"
        })
    
    return {"providers": providers_data}

@app.post("/api/validate_key")
async def validate_key_endpoint(request: dict):
    """Validate API key for a provider."""
    provider = request.get("provider")
    api_key = request.get("apiKey")
    
    if not provider or not api_key:
        return {"valid": False, "error": "Provider and API key required"}
    
    try:
        is_valid = await chat_manager.validate_api_key(provider, api_key)
        return {"valid": is_valid}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.post("/api/validate_directory")
async def validate_directory_endpoint(request: dict):
    """Validate a workspace directory path."""
    directory_path = request.get("directoryPath", "")
    
    validation_result = validate_workspace_directory(directory_path)
    
    return {
        "valid": validation_result["valid"],
        "path": validation_result["path"],
        "message": validation_result["message"],
        "error": validation_result.get("error")
    }

@app.post("/api/agent/execute")
async def agent_execute_endpoint(request: AgentExecuteRequest) -> AgentExecuteResponse:
    """Execute a task using the AI agent with real tools."""
    try:
        # Convert settings dict to ChatSettings
        settings = ChatSettings(**request.settings)
        
        # Validate settings
        if not settings.provider or not settings.apiKey or not settings.model:
            return AgentExecuteResponse(
                success=False,
                error="Please configure your API key and model in settings",
                actions=[],
                primaryAction="error"
            )
        
        # Determine which model to use
        final_model = settings.model
        if settings.customModel and validate_custom_model_format(settings.customModel):
            final_model = settings.customModel.strip()
        
        # Create and execute agent
        agent = Agent(
            provider=settings.provider,
            api_key=settings.apiKey,
            model=final_model
        )
        
        # Execute the task
        result = await agent.execute_task(
            instruction=request.instruction,
            workspace_dir=request.workspaceDir
        )
        
        return AgentExecuteResponse(
            success=result["success"],
            result=result.get("result"),
            error=result.get("error"),
            actions=result.get("actions", []),
            primaryAction=result.get("primaryAction"),
            iterations=result.get("iterations")
        )
        
    except Exception as e:
        return AgentExecuteResponse(
            success=False,
            error=f"Agent execution failed: {str(e)}",
            actions=[],
            primaryAction="error"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    chat_manager.websocket_connections.append(websocket)
    
    try:
        # Send initial data
        providers_response = await get_providers()
        await websocket.send_json({
            "type": "initial_data",
            "data": providers_response
        })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()
                # Handle any client messages if needed
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send periodic ping
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        chat_manager.websocket_connections.remove(websocket)
    except Exception as e:
        if websocket in chat_manager.websocket_connections:
            chat_manager.websocket_connections.remove(websocket)

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "OpenTerra Chat",
        "version": "2.0.0",
        "providers_available": len(get_available_providers()),
        "websocket_connections": len(chat_manager.websocket_connections),
        "models_supported": sum(len(get_provider_models(p)) for p in get_available_providers())
    }

# Legacy endpoints for backwards compatibility
@app.get("/api/agents")
async def get_agents():
    """Legacy endpoint - returns empty for compatibility."""
    return {"agents": []}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenTerra Chat Web UI - 2025 Edition")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🚀 Starting OpenTerra Agents - 2025 Edition...")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print("🤖 Agent Interface: http://127.0.0.1:8000")
    print(f"🛠️  Available providers: {', '.join(get_available_providers())}")
    print("🔥 Latest models: Claude 4 Opus/Sonnet, DeepSeek R1, GPT-4o, Grok 3, Kimi K2!")
    print("✨ AI Agents with REAL tools:")
    print("   • File editing & creation")
    print("   • Code analysis & refactoring") 
    print("   • Terminal command execution")
    print("   • Git operations")
    print("   • Directory workspace management")
    
    uvicorn.run(
        "web_ui:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    ) 