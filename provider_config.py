# Provider Configuration for OpenTerra
# Updated January 2025 with actual current SOTA models

PROVIDERS = {
    "anthropic": {
        "name": "Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_label": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-7-sonnet",
        "models": [
            # Current SOTA Claude models
            "claude-3-7-sonnet",          # Latest Claude 3.7
            "claude-4-opus"               # Claude 4 if available
        ]
    },
    
    "openrouter": {
        "name": "OpenRouter",
        "api_key_env": "OPENROUTER_API_KEY", 
        "api_key_label": "OPENROUTER_API_KEY",
        "default_model": "x-ai/grok-4",
        "models": [
            # Current SOTA models from each company (2025)
            "x-ai/grok-4",                          # Latest Grok 4
            "deepseek/deepseek-r1",                 # DeepSeek R1 reasoning model
            "google/gemini-2.5-pro",                # Gemini 2.5 Pro
            "anthropic/claude-3-7-sonnet",         # Claude 3.7
            "openai/gpt-4o-2025-01-15",           # Latest GPT-4o
            "openai/o3-mini",                       # OpenAI o3-mini
            "meta-llama/llama-3.3-70b-instruct",   # Latest Llama
            "qwen/qwen-2.5-coder-32b-instruct",    # Latest Qwen coder
            "mistralai/mistral-large-2407",         # Latest Mistral
            "moonshotai/kimi-k2"                    # Kimi K2
        ]
    },
    
    "moonshot": {
        "name": "MoonshotAI", 
        "api_key_env": "OPENROUTER_API_KEY",
        "api_key_label": "OPENROUTER_API_KEY",
        "default_model": "moonshotai/kimi-k2:free",
        "models": [
            # Latest MoonshotAI models
            "moonshotai/kimi-k2:free",    # Free tier
            "moonshotai/kimi-k2"          # Paid tier
        ]
    }
}

# Clean model display names for current SOTA models
MODEL_DISPLAY_NAMES = {
    # Anthropic models
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "claude-4-opus": "Claude 4 Opus",
    
    # xAI models  
    "x-ai/grok-4": "Grok 4",
    
    # DeepSeek models
    "deepseek/deepseek-r1": "DeepSeek R1",
    
    # Google models
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    
    # Anthropic via OpenRouter
    "anthropic/claude-3-7-sonnet": "Claude 3.7 Sonnet",
    
    # OpenAI models
    "openai/gpt-4o-2025-01-15": "GPT-4o 2025",
    "openai/o3-mini": "GPT o3-mini",
    
    # Meta models
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    
    # Qwen models
    "qwen/qwen-2.5-coder-32b-instruct": "Qwen 2.5 Coder 32B",
    
    # Mistral models
    "mistralai/mistral-large-2407": "Mistral Large",
    
    # MoonshotAI models
    "moonshotai/kimi-k2:free": "Kimi K2 Free",
    "moonshotai/kimi-k2": "Kimi K2"
}

# Default model mappings for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-3-7-sonnet",
    "openrouter": "x-ai/grok-4", 
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
    """Validate custom model format for OpenRouter only"""
    if not model_id or not isinstance(model_id, str):
        return False
    
    # Custom models only work with OpenRouter
    # Allow provider/model format for OpenRouter
    import re
    pattern = r'^[a-zA-Z0-9\-_/.:]+'
    return bool(re.match(pattern, model_id.strip()))

def can_use_custom_model(provider_name):
    """Only OpenRouter supports custom models"""
    return provider_name == "openrouter" 