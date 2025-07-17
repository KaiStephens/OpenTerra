# 🌐 Claude Parallel Runner - Web UI Guide

A comprehensive guide to using the web interface for managing parallel Claude agents with real-time monitoring and background execution.

## 🚀 Quick Start

### Launch the Web UI

```bash
# Activate virtual environment
source venv/bin/activate

# Start the web server
python start_web_ui.py

# Or with custom options
python start_web_ui.py --port 8080 --open-browser
```

**Access the dashboard**: http://127.0.0.1:8000

## ✨ New Features

### 🎯 Model Selection
- **Dynamic Model Switching**: Change models per provider in real-time
- **Provider-Specific Models**: Each provider shows its available models
- **Model Override**: Override default models when creating agents

### 🔄 Real-Time Web Interface
- **Live Agent Monitoring**: Watch agents execute tasks in real-time
- **WebSocket Updates**: Instant status updates without page refresh
- **Background Execution**: Agents run in the background while you browse
- **Multi-Agent Management**: Create and monitor multiple agents simultaneously

## 🎛️ Web UI Features

### Provider Management
- **Visual Provider Selection**: Click to switch between providers
- **Model Configuration**: Set primary and small/fast models
- **Real-Time Validation**: Instant feedback on provider status
- **API Key Detection**: Automatic detection of configured providers

### Agent Creation
- **Multi-Prompt Support**: Add multiple tasks per agent
- **Custom Configuration**: Set concurrency, timeouts, and priorities
- **Provider Override**: Use different providers per agent
- **Model Override**: Specify models per agent

### Real-Time Monitoring
- **Live Progress Bars**: Visual progress tracking
- **Status Indicators**: Color-coded status badges
- **Task Statistics**: Completed/failed task counts
- **Error Display**: Real-time error messages
- **Execution Time**: Track task duration

### Results Management
- **Detailed Results**: View full task results and messages
- **JSON Export**: Download results as JSON files
- **Result History**: Access historical execution data
- **Error Analysis**: Detailed error information

## 🔧 CLI Enhancements

### Model Management Commands

```bash
# List available models for current provider
python cli.py config models

# List models for specific provider
python cli.py config models --provider openrouter

# Set model for a provider
python cli.py config set-model anthropic --model claude-3-5-sonnet-20241022

# Set both primary and small/fast models
python cli.py config set-model openrouter \
  --model claude-sonnet-4-20250514 \
  --small-fast-model google/gemini-2.5-flash

# Switch provider with model override
python cli.py config set anthropic \
  --model claude-3-5-sonnet-20241022 \
  --small-fast-model claude-3-5-haiku-20241022
```

### Enhanced Run Commands

```bash
# Run with specific provider and model
python cli.py run \
  --prompts "Task 1" "Task 2" \
  --provider openrouter \
  --model claude-sonnet-4-20250514

# Run with custom configuration
python cli.py run \
  --prompts "Complex analysis task" \
  --provider anthropic \
  --max-concurrent 2 \
  --timeout 600 \
  --output detailed_results.json
```

## 🎨 Web UI Interface Guide

### 1. Provider Configuration Panel
Located at the top of the dashboard:
- **Provider Cards**: Visual representation of all available providers
- **Current Provider**: Highlighted with blue border and checkmark
- **Model Dropdowns**: Select primary and small/fast models
- **Real-Time Updates**: Changes apply immediately

### 2. Agent Creation Form
Create new background agents:
- **Agent ID**: Optional custom identifier
- **Provider Selection**: Override current provider
- **Concurrency Settings**: Max parallel tasks (1-10)
- **Timeout Configuration**: Per-task timeout (30-3600 seconds)
- **Multi-Prompt Input**: Add/remove prompts dynamically

### 3. Active Agents Grid
Monitor running and completed agents:
- **Status Badges**: Pending, Running, Completed, Failed
- **Progress Bars**: Visual progress indication
- **Task Statistics**: Success/failure counts
- **Action Buttons**: Stop agents, view results
- **Real-Time Updates**: Live status changes

### 4. Results Modal
Detailed result viewing:
- **Task Breakdown**: Individual task results
- **Execution Metrics**: Timing and performance data
- **Message History**: Full conversation logs
- **Error Details**: Comprehensive error information

## 🔄 WebSocket Real-Time Updates

The web UI uses WebSocket connections for real-time updates:

- **Agent Status Changes**: Instant status updates
- **Progress Updates**: Live progress bar changes
- **Provider Changes**: Real-time provider switching
- **Error Notifications**: Immediate error display
- **Connection Status**: Visual connection indicator

## 🛠️ API Reference

### Provider Endpoints

```bash
# Get all providers
GET /api/providers

# Get models for provider
GET /api/providers/{provider_name}/models

# Set current provider
POST /api/providers/set
{
  "name": "openrouter",
  "model": "claude-sonnet-4-20250514",
  "small_fast_model": "google/gemini-2.5-flash"
}

# Create new provider
POST /api/providers
{
  "name": "custom_provider",
  "provider_type": "custom_router",
  "base_url": "https://my-router.com",
  "api_key_env_var": "MY_API_KEY",
  "model": "my-model",
  "description": "My custom provider"
}
```

### Agent Endpoints

```bash
# Get all agents
GET /api/agents

# Create new agent
POST /api/agents
{
  "agent_id": "my_agent",
  "prompts": ["Task 1", "Task 2"],
  "provider": "openrouter",
  "model": "claude-sonnet-4-20250514",
  "max_concurrent": 3,
  "timeout": 300.0,
  "priority": 1
}

# Get specific agent
GET /api/agents/{agent_id}

# Stop agent
DELETE /api/agents/{agent_id}

# Get agent results
GET /api/agents/{agent_id}/results
```

### WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Handle messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  switch (message.type) {
    case 'agent_update':
      updateAgentUI(message.data);
      break;
    case 'provider_changed':
      updateProviderUI(message.data);
      break;
  }
};
```

## 🎯 Usage Scenarios

### 1. A/B Testing Different Models

```bash
# Create multiple agents with different models
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_claude",
    "prompts": ["Explain quantum computing"],
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022"
  }'

curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "test_gpt",
    "prompts": ["Explain quantum computing"],
    "provider": "openrouter", 
    "model": "openai/gpt-4-turbo"
  }'
```

### 2. Batch Processing with Different Providers

Use the web UI to:
1. Create multiple agents with different provider/model combinations
2. Monitor progress in real-time
3. Compare results and performance
4. Export results for analysis

### 3. Development and Testing

```bash
# Start with auto-reload for development
python start_web_ui.py --reload --open-browser

# Test different configurations
# Create agents with various settings
# Monitor real-time behavior
```

## 🔍 Monitoring and Debugging

### Connection Status
- **Green dot**: WebSocket connected
- **Red dot**: Disconnected (auto-reconnect attempts)

### Agent Status Colors
- **Yellow**: Pending
- **Blue**: Running  
- **Green**: Completed
- **Red**: Failed

### Error Handling
- **Toast Notifications**: Success/error messages
- **Error Details**: Full error messages in agent cards
- **Connection Recovery**: Automatic WebSocket reconnection

## 🚀 Performance Tips

### Web UI Optimization
1. **Close unused tabs**: Reduces WebSocket connections
2. **Monitor concurrent agents**: Don't exceed API limits
3. **Use appropriate timeouts**: Prevent hanging tasks
4. **Regular cleanup**: Remove completed agents

### Background Agent Management
1. **Batch similar tasks**: Group related prompts
2. **Use priorities**: Important tasks first
3. **Monitor resource usage**: Watch CPU/memory
4. **Set reasonable concurrency**: Based on API limits

## 🛡️ Security Considerations

### API Key Safety
- **Environment variables**: Store keys in environment, not code
- **Local access**: Web UI binds to localhost by default
- **No key exposure**: Keys never sent to frontend

### Network Security
- **Local binding**: Default host is 127.0.0.1
- **HTTPS ready**: Can be configured with reverse proxy
- **WebSocket security**: WSS support available

## 📊 Comparison: CLI vs Web UI

| Feature | CLI | Web UI |
|---------|-----|--------|
| **Quick tasks** | ✅ Fast | ⚠️ More setup |
| **Multiple agents** | ❌ Sequential | ✅ Parallel |
| **Real-time monitoring** | ❌ Limited | ✅ Full |
| **Visual feedback** | ⚠️ Text only | ✅ Rich UI |
| **Background execution** | ❌ Blocking | ✅ Non-blocking |
| **Result comparison** | ⚠️ Manual | ✅ Visual |
| **Automation** | ✅ Scriptable | ⚠️ Manual |
| **Remote access** | ✅ SSH | ✅ Web browser |

## 🎉 Best Practices

### Model Selection
1. **Start with fast models** for quick tests
2. **Use powerful models** for complex tasks
3. **Match model to task complexity**
4. **Consider cost vs quality tradeoffs**

### Agent Management
1. **Use descriptive agent IDs**
2. **Set realistic timeouts**
3. **Monitor resource usage**
4. **Clean up completed agents**

### Provider Strategy
1. **Have backup providers** configured
2. **Use OpenRouter** for model variety
3. **Use Anthropic direct** for reliability
4. **Monitor rate limits** across providers

---

**🎯 Ready to start?** Launch the web UI and begin creating your parallel Claude agents!

```bash
python start_web_ui.py --open-browser
``` 