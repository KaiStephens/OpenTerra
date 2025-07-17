# OpenTerra Chat - Latest 2025 AI Models Guide

## 🔥 What's New in 2025 Edition

OpenTerra Chat has been completely updated with the latest and most powerful AI models available as of January 2025. You now have access to cutting-edge models including **Claude 4**, **DeepSeek R1**, **GPT-4o**, **Grok 3**, and **Kimi K2**.

## 📊 Provider Overview

### Anthropic (Direct API)
- **Claude 4 Opus** - Flagship 2025 model with superior reasoning
- **Claude 4 Sonnet** - Balanced performance and speed
- **Claude 3.7 Sonnet** - Hybrid reasoning capabilities
- **Claude 3.5 Series** - Previous generation models

### OpenRouter (31 Models Available)
Access to the largest collection of models through a single API:

#### 🧠 Reasoning Models
- **DeepSeek R1** & **R1:free** - o1-level reasoning performance (open source)
- **DeepSeek R1-0528-Qwen3-8B** - Distilled 8B parameter reasoning model
- **OpenAI o3-mini** & **o1-pro** - Latest OpenAI reasoning models

#### 🤖 Latest Language Models
- **Claude 4 Opus/Sonnet** via OpenRouter
- **GPT-4o (2025-01-15)** - Latest OpenAI flagship
- **Grok 3 Mini Beta** & **Grok 2 1212** - xAI models
- **Gemini 2.0 Flash Experimental** - Google's latest

#### 🔨 Coding Specialists
- **Qwen 2.5 Coder 32B** - Advanced coding model
- **Mistral Codestral** - Specialized for code generation
- **Llama 3.3 70B** - Meta's latest coding-optimized model

### MoonshotAI (Kimi K2)
- **Kimi K2 (Free)** - 1 trillion parameters, 32B active (FREE tier!)
- **Kimi K2 (Paid)** - Full performance tier
- **Moonshot v1 Series** - Various context lengths (8k, 32k, 128k)

## ✨ New Features

### 🎯 Custom Model Input
- Paste any model name to use custom or experimental models
- Format: `provider/model-name` or `custom-model-name`
- Examples: `anthropic/claude-4-opus`, `custom/my-model`

### ⚙️ Optimized Settings
- **Temperature**: Fixed at 0.7 for optimal performance
- **Max Tokens**: Smart automatic limits (4000 tokens)
- **No Manual Tuning**: Settings optimized for coding and reasoning

### 🔌 Multi-Provider Support
- Switch between providers seamlessly
- Unified API interface
- Real-time model switching

### 💾 Enhanced Chat Experience
- Chat history persistence
- Export conversations
- Real-time typing indicators
- Model badges showing which AI responded

## 🚀 Getting Started

### 1. Start the Server
```bash
python web_ui.py --port 8001
```

### 2. Access the Interface
Open your browser to: http://localhost:8001

### 3. Configure Your API Keys
- **Anthropic**: Get your `ANTHROPIC_API_KEY` from console.anthropic.com
- **OpenRouter**: Get your `OPENROUTER_API_KEY` from openrouter.ai
- **MoonshotAI**: Uses OpenRouter API (same key)

### 4. Select Your Model
Choose from 44+ available models or paste a custom model name.

## 🏆 Top Recommended Models for 2025

### For Coding
1. **Claude 4 Opus** - Best overall coding performance
2. **DeepSeek R1** - Excellent reasoning, open source
3. **Qwen 2.5 Coder 32B** - Specialized code generation
4. **Claude 3.7 Sonnet** - Hybrid reasoning for complex problems

### For General Chat
1. **Claude 4 Sonnet** - Balanced performance and speed
2. **GPT-4o (2025)** - Latest OpenAI capabilities
3. **Kimi K2 (Free)** - Amazing performance at no cost
4. **Grok 3 Mini** - Fast and capable

### For Reasoning & Analysis
1. **DeepSeek R1** - o1-level reasoning performance
2. **Claude 4 Opus** - Superior analytical capabilities
3. **OpenAI o1-pro** - Advanced reasoning model
4. **Claude 3.7 Sonnet** - Visible thinking process

## 💡 Pro Tips

### Custom Models
- Use the custom model field to access any model on OpenRouter
- Try experimental models: `deepseek/deepseek-r1-zero:free`
- Access preview models: `google/gemini-2.0-flash-exp`

### Free Options
- **DeepSeek R1:free** - Powerful reasoning model at no cost
- **Kimi K2:free** - 1T parameter model with free tier
- **Many OpenRouter models** - Free tiers available

### Performance Optimization
- Temperature is fixed at 0.7 for optimal results
- Smart token limits prevent cutoffs
- No need to adjust settings manually

## 📈 Model Comparison

| Model | Provider | Strengths | Best For |
|-------|----------|-----------|----------|
| Claude 4 Opus | Anthropic | Superior reasoning, long context | Complex coding, analysis |
| Claude 4 Sonnet | Anthropic | Balanced speed/quality | General purpose |
| DeepSeek R1 | OpenRouter | o1-level reasoning, open source | Problem solving |
| GPT-4o (2025) | OpenRouter | Latest capabilities | General chat |
| Kimi K2 (Free) | MoonshotAI | 1T params, free tier | High-quality free option |
| Grok 3 Mini | OpenRouter | Fast responses | Quick tasks |

## 🔧 Technical Details

### Architecture
- **FastAPI** backend with real-time WebSocket support
- **AlpineJS** frontend for reactive UI
- **TailwindCSS** for modern styling

### API Compatibility
- OpenAI-compatible API for all providers
- Unified request/response format
- Automatic provider routing

### Security
- API keys stored locally only
- Real-time key validation
- Secure provider switching

## 🆘 Troubleshooting

### Common Issues
1. **"Model not available"** - Check custom model spelling
2. **"API key invalid"** - Verify key and provider selection
3. **Rate limits** - Try free tier models or wait

### Getting Help
- Check the model display name for guidance
- Use the custom model field for experimental models
- Switch providers if one is unavailable

## 📝 What's Next?

OpenTerra Chat will continue to be updated with the latest models as they're released. The custom model input ensures you can always access cutting-edge models even before they're officially added to the interface.

---

**OpenTerra Chat 2025 Edition** - Bringing you the future of AI, today! 🚀 