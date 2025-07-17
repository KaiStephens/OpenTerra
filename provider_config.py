# Provider Configuration for OpenTerra
# Updated January 2025 with latest AI models

PROVIDERS = {
    "anthropic": {
        "name": "Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_label": "ANTHROPIC_API_KEY",
        "default_model": "claude-4-opus",
        "models": [
            # Claude 4 Series (Released May 2025)
            "claude-4-opus",
            "claude-4-sonnet", 
            
            # Claude 3.7 Series (February 2025)
            "claude-3-7-sonnet",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            
            # Legacy Claude 3
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    },
    
    "openrouter": {
        "name": "OpenRouter",
        "api_key_env": "OPENROUTER_API_KEY", 
        "api_key_label": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-4-opus",
        "models": [
            # Latest Claude 4 Models via OpenRouter
            "anthropic/claude-4-opus",
            "anthropic/claude-4-sonnet",
            "anthropic/claude-3-7-sonnet",
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-3-5-haiku",
            
            # Latest OpenAI Models (2025)
            "openai/gpt-4o-2025-01-15",
            "openai/o3-mini",
            "openai/o1-pro", 
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            
            # DeepSeek R1 Series (Latest reasoning models)
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-r1-0528-qwen3-8b",
            "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "deepseek/deepseek-r1-zero",
            "deepseek/deepseek-r1-zero:free",
            
            # xAI Grok Models (2025)
            "x-ai/grok-3-mini-beta",
            "x-ai/grok-2-1212",
            "x-ai/grok-beta",
            
            # Google Gemini 2.0 Series
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-pro-latest",
            "google/gemini-1.5-flash",
            
            # Meta Llama Models
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.2-90b-vision-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            
            # Qwen Latest Models
            "qwen/qwen-2.5-coder-32b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qw2.5-coder-7b-instruct",
            
            # Mistral Latest
            "mistralai/mistral-large-2407",
            "mistralai/codestral-latest",
            "mistralai/pixtral-12b-2409"
        ]
    },
    
    "moonshot": {
        "name": "MoonshotAI", 
        "api_key_env": "OPENROUTER_API_KEY",  # Uses OpenRouter API
        "api_key_label": "OPENROUTER_API_KEY (for MoonshotAI)",
        "default_model": "moonshotai/kimi-k2:free",
        "models": [
            # MoonshotAI Kimi K2 (1 trillion parameters, 32B active)
            "moonshotai/kimi-k2:free",      # Free tier
            "moonshotai/kimi-k2",           # Paid tier
            "moonshotai/moonshot-v1-8k",
            "moonshotai/moonshot-v1-32k",
            "moonshotai/moonshot-v1-128k"
        ]
    }
}

# Model display names for better UX
MODEL_DISPLAY_NAMES = {
    # Claude 4 Series
    "claude-4-opus": "Claude 4 Opus (Flagship 2025)",
    "claude-4-sonnet": "Claude 4 Sonnet (Balanced 2025)",
    "anthropic/claude-4-opus": "Claude 4 Opus (Flagship 2025)",
    "anthropic/claude-4-sonnet": "Claude 4 Sonnet (Balanced 2025)",
    
    # Claude 3.7
    "claude-3-7-sonnet": "Claude 3.7 Sonnet (Hybrid Reasoning)",
    "anthropic/claude-3-7-sonnet": "Claude 3.7 Sonnet (Hybrid Reasoning)",
    
    # OpenAI Latest
    "openai/gpt-4o-2025-01-15": "GPT-4o (Latest 2025)",
    "openai/o3-mini": "o3-mini (Reasoning)",
    "openai/o1-pro": "o1-pro (Advanced Reasoning)",
    
    # DeepSeek R1 Series
    "deepseek/deepseek-r1": "DeepSeek R1 (o1-level Reasoning)",
    "deepseek/deepseek-r1:free": "DeepSeek R1 (Free)",
    "deepseek/deepseek-r1-0528-qwen3-8b": "DeepSeek R1 Qwen3-8B (Fast Reasoning)",
    "deepseek/deepseek-r1-0528-qwen3-8b:free": "DeepSeek R1 Qwen3-8B (Free)",
    
    # MoonshotAI
    "moonshotai/kimi-k2:free": "Kimi K2 (Free - 1T params)",
    "moonshotai/kimi-k2": "Kimi K2 (1T params, 32B active)",
    
    # Grok
    "x-ai/grok-3-mini-beta": "Grok 3 Mini Beta",
    "x-ai/grok-2-1212": "Grok 2 (Dec 2024)",
    
    # Google
    "google/gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)",
    "google/gemini-1.5-pro-latest": "Gemini 1.5 Pro (Latest)",
}

# Default model mappings for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-4-opus",
    "openrouter": "anthropic/claude-4-opus", 
    "moonshot": "moonshotai/kimi-k2:free"
}

def get_model_display_name(model_id):
    """Get a user-friendly display name for a model"""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)

def get_available_providers():
    """Get list of available providers"""
    return list(PROVIDERS.keys())

def get_provider_models(provider_name):
    """Get available models for a provider"""
    if provider_name in PROVIDERS:
        return PROVIDERS[provider_name]["models"]
    return []

def get_provider_config(provider_name):
    """Get configuration for a specific provider"""
    return PROVIDERS.get(provider_name, {})

def validate_model_for_provider(provider_name, model_id):
    """Validate that a model is available for a provider"""
    if provider_name not in PROVIDERS:
        return False
    return model_id in PROVIDERS[provider_name]["models"]

def is_custom_model(model_id):
    """Check if a model_id is a custom model (not in predefined lists)"""
    for provider_config in PROVIDERS.values():
        if model_id in provider_config["models"]:
            return False
    return True

def validate_custom_model_format(model_id):
    """Validate custom model format - should be provider/model or just model name"""
    if not model_id or not isinstance(model_id, str):
        return False
    
    # Allow alphanumeric, hyphens, underscores, slashes, colons, and dots
    import re
    pattern = r'^[a-zA-Z0-9\-_/.:]+'
    return bool(re.match(pattern, model_id.strip())) 