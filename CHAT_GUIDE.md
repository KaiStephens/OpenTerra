# OpenTerra Chat - User Guide

## Welcome to OpenTerra Chat! 🚀

OpenTerra Chat is a modern, web-based interface for chatting with Claude AI and other top language models. It features a clean design similar to ChatGPT/Claude with support for the best AI providers and simplified configuration.

## 🚀 Quick Start

### 1. Start the Chat Interface

```bash
# Option 1: Use the launcher script (recommended)
python start_web_ui.py

# Option 2: Start directly
python web_ui.py --host 0.0.0.0
```

### 2. Open Your Browser

Navigate to: **http://localhost:8000**

## ⚙️ Initial Setup

### Configure Your API Key

1. **Click the Settings button** (gear icon) in the sidebar
2. **Select your provider**:
   - **Anthropic** - Direct Claude API with latest Claude 3.7 Sonnet
   - **OpenRouter** - Access to all top models (Claude, GPT-4o, DeepSeek R1, Grok, etc.)
   - **Moonshot** - Kimi K2 (excellent for coding & reasoning tasks)

3. **Enter your API key**:
   - For Anthropic: Your `ANTHROPIC_API_KEY`
   - For OpenRouter: Your `OPENROUTER_API_KEY`
   - For Moonshot: Your `OPENROUTER_API_KEY` (uses OpenRouter)

4. **Choose a model** (optional):
   - Each provider offers different models
   - Leave blank to use the default (latest recommended model)

5. **Save Settings**

## 💬 Using the Chat Interface

### Chat Features

- **Clean, modern interface** similar to Claude/ChatGPT
- **Real-time messaging** with typing indicators
- **Message history** preserved during your session
- **Responsive design** works on desktop and mobile
- **Auto-resizing text input** with support for multi-line messages
- **Optimized settings** - temperature fixed at 0.7, smart token limits

### Chat Controls

- **Send Message**: Type your message and press Enter
- **New Line**: Shift + Enter for multi-line messages
- **New Chat**: Click "New Chat" in the sidebar
- **Settings**: Click the gear icon to modify configuration

### Keyboard Shortcuts

- `Enter` - Send message
- `Shift + Enter` - New line in message

## 🔧 Provider Configuration

### Anthropic (Direct API)
```
Provider: anthropic
API Key: Your Anthropic API key
Default Model: claude-3.7-sonnet (latest)
```

**Available Models:**
- `claude-3.7-sonnet` - Latest and greatest (Feb 2025)
- `claude-3.7-sonnet:thinking` - Reasoning mode
- `claude-3.5-sonnet-20241022` - Previous version
- `claude-3.5-haiku-20241022` - Fast model
- And older Claude models

### OpenRouter (All Top Models)
```
Provider: openrouter
API Key: Your OpenRouter API key
Default Model: anthropic/claude-3.7-sonnet
```

**Available Models:**
- **Latest Claude:** `anthropic/claude-3.7-sonnet`, `anthropic/claude-3.7-sonnet:thinking`
- **DeepSeek R1:** `deepseek/deepseek-r1`, `deepseek/deepseek-r1:free` (o1-level reasoning)
- **GPT-4o:** `openai/gpt-4o-search-preview`, `openai/gpt-4o-mini`
- **Grok:** `x-ai/grok-3-mini-beta`, `x-ai/grok-2-1212`
- **Qwen:** `qwen/qwen-max`, `qwen/qwen3-32b`
- **Kimi K2:** `moonshotai/kimi-k2`
- **Gemini:** `google/gemini-2.5-flash`

### Moonshot AI (Kimi K2)
```
Provider: moonshot
API Key: Your OpenRouter API key
Default Model: moonshotai/kimi-k2
```

**About Kimi K2:**
- 1 trillion parameters, 32B active
- Excellent for coding and reasoning
- Strong tool use capabilities
- 128K context window

## 🎛️ Chat Settings

### Optimized Configuration
- **Temperature: 0.7** (fixed) - Perfect balance between creativity and consistency
- **Smart Token Limits** - Automatically optimized for each model
- **No manual tuning needed** - Just select your provider and start chatting

## 🔍 Troubleshooting

### Common Issues

**"Please configure your API key in settings"**
- Click Settings and enter your API key
- Make sure you've selected the correct provider
- Verify your API key is valid

**"Invalid API key"**
- Double-check your API key
- For Anthropic: Use your ANTHROPIC_API_KEY
- For OpenRouter/Moonshot: Use your OPENROUTER_API_KEY

**"Failed to send message"**
- Check your internet connection
- Verify the provider's service status
- Try refreshing the page

**Connection Issues**
- Ensure the server is running on port 8000
- Check if any firewall is blocking the connection
- Try using `127.0.0.1:8000` instead of `localhost:8000`

### Server Logs

If you're having issues, check the server console for error messages. The server will display helpful debugging information.

## 🌟 Features

### Current Features
- ✅ Real-time chat with latest AI models
- ✅ 3 top providers (Anthropic, OpenRouter, Moonshot)
- ✅ Latest models (Claude 3.7, DeepSeek R1, Grok 3, GPT-4o, Kimi K2)
- ✅ API key management in web interface
- ✅ Model selection per provider
- ✅ Optimized settings (temp 0.7, smart tokens)
- ✅ Responsive design
- ✅ WebSocket real-time updates
- ✅ Chat history (session-based)

### Top Models Available (2025)
- **Claude 3.7 Sonnet** - Latest from Anthropic with reasoning
- **DeepSeek R1** - Open source, o1-level performance
- **Grok 3 Mini** - Latest from xAI
- **GPT-4o Search** - OpenAI with search capabilities
- **Kimi K2** - Excellent for coding and agentic tasks
- **Qwen-Max** - Top Chinese model
- And many more via OpenRouter

## 🛠️ Development

### File Structure
```
OpenTerra/
├── web_ui.py          # Main FastAPI application
├── templates/
│   └── dashboard.html # Chat interface HTML
├── start_web_ui.py    # Launcher script
├── provider_config.py # Provider management (3 providers)
└── requirements.txt   # Python dependencies
```

### API Endpoints
- `GET /` - Chat interface
- `POST /api/chat` - Send chat message
- `GET /api/providers` - List 3 available providers
- `GET /api/health` - Health check
- `WebSocket /ws` - Real-time updates

## 📝 Notes

- **API Keys are stored locally** in your browser's localStorage
- **Chat history** is currently session-based (not persistent)
- **Temperature is fixed at 0.7** for optimal results
- **Token limits are automatically optimized** per model
- **Only 3 providers** for simplicity and quality

## 🆘 Support

If you encounter issues:

1. Check this guide first
2. Look at the browser console for errors (F12)
3. Check the server console for error messages
4. Verify your API key and provider configuration
5. Try refreshing the page or restarting the server

---

**Available Providers:**
- 🤖 **Anthropic** - Direct Claude API
- 🌐 **OpenRouter** - Access to all top models
- 🚀 **Moonshot** - Kimi K2 specialist

**Enjoy chatting with the latest AI models! 🎉** 