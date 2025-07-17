![OpenTerra Logo](openterra_logo.png)

# Claude Parallel Runner

A powerful Python framework for running multiple Claude Code SDK instances in parallel with seamless provider switching between Anthropic and OpenRouter (via y-router).

## ✨ Features

- **🚀 Parallel Execution**: Run multiple Claude Code SDK instances concurrently
- **🔄 Provider Switching**: Easy toggle between Anthropic and OpenRouter 
- **⚡ y-router Integration**: Use OpenRouter's vast model selection through y-router
- **📊 Progress Tracking**: Rich console output with real-time progress
- **🔒 Rate Limiting**: Respect API limits with built-in throttling
- **🔄 Auto Retry**: Exponential backoff for failed requests
- **💾 Results Storage**: Persistent results with JSON export
- **🎛️ Flexible Configuration**: Customizable concurrency, timeouts, and priorities
- **🐚 Shell Integration**: Quick aliases for provider switching
- **📋 CLI Interface**: Full command-line interface for easy usage

## 🚀 Quick Start

### One-Line Setup

```bash
# Clone and setup everything automatically
git clone <this-repo>
cd OpenTerra
python setup.py
```

### Manual Setup

1. **Install Prerequisites**
   ```bash
   # Ensure Python 3.10+ and Node.js are installed
   python --version  # Should be 3.10+
   node --version    # Any recent version
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   npm install -g @anthropic-ai/claude-code
   ```

3. **Configure Providers**
   ```bash
   # Set up environment variables
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   
   # Configure y-router for OpenRouter (optional - uses shared instance by default)
   export ANTHROPIC_BASE_URL="https://cc.yovy.app"
   ```

## 🎯 Usage Examples

### Command Line Interface

```bash
# Run multiple prompts in parallel
python cli.py run --prompts "Write a hello world function" "Explain async programming" "Create a REST API"

# Use specific provider
python cli.py run --prompts "Task 1" "Task 2" --provider openrouter

# Interactive mode
python cli.py interactive

# List available providers
python cli.py config list

# Switch provider
python cli.py config set anthropic

# Setup new provider
python cli.py setup openrouter --api-key your-key
```

### Python API

```python
from claude_parallel_runner import run_tasks_parallel_sync

# Simple parallel execution
prompts = [
    "Write a Python function to calculate fibonacci numbers",
    "Explain the concept of decorators",
    "Create a simple REST API using FastAPI"
]

results = run_tasks_parallel_sync(
    prompts=prompts,
    max_concurrent=3,
    timeout=300.0,
    provider="openrouter"  # or "anthropic"
)

# Check results
for result in results:
    if result.success:
        print(f"✅ {result.task_id}: Completed in {result.execution_time:.2f}s")
    else:
        print(f"❌ {result.task_id}: {result.error}")
```

### Advanced Configuration

```python
from claude_parallel_runner import ClaudeParallelRunner, ParallelRunnerConfig, TaskConfig

# Advanced configuration
config = ParallelRunnerConfig(
    max_concurrent_tasks=5,
    rate_limit_per_minute=30,
    default_timeout=180.0,
    results_output_file="results.json",
    save_intermediate_results=True
)

# Custom tasks with priorities
tasks = [
    TaskConfig(
        id="high_priority_task",
        prompt="Critical analysis of the codebase",
        timeout=120.0,
        priority=3  # Higher priority
    ),
    TaskConfig(
        id="background_task",
        prompt="Generate documentation",
        timeout=300.0,
        priority=1  # Lower priority
    )
]

runner = ClaudeParallelRunner(config, provider_name="anthropic")
results = runner.run_sync(tasks)
```

### Shell Aliases (Quick Provider Switching)

```bash
# Generate shell aliases
python cli.py config aliases

# Source the aliases
source claude_aliases.sh

# Now use quick commands
claude-anthropic "Write a function to sort a list"
claude-openrouter "Explain machine learning concepts"
claude-moonshot "Create a web scraper"
```

## 🔧 Provider Configuration

### Anthropic (Direct API)

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
python cli.py config set anthropic
```

### OpenRouter (via y-router)

```bash
# Using shared y-router instance
export OPENROUTER_API_KEY="your-openrouter-api-key"
export ANTHROPIC_BASE_URL="https://cc.yovy.app"
python cli.py config set openrouter

# Using custom y-router deployment
export ANTHROPIC_BASE_URL="https://your-worker.your-subdomain.workers.dev"
python cli.py config set openrouter_custom
```

### Custom Router

```python
from provider_config import provider_manager, ProviderConfig, ProviderType

# Add custom provider
custom_provider = ProviderConfig(
    name="my_custom_provider",
    provider_type=ProviderType.CUSTOM_ROUTER,
    base_url="https://my-custom-endpoint.com",
    api_key_env_var="MY_CUSTOM_API_KEY",
    model="my-preferred-model",
    description="My custom Claude provider"
)

provider_manager.add_provider(custom_provider)
provider_manager.set_provider("my_custom_provider")
```

## 📊 Configuration Options

### ParallelRunnerConfig

| Option | Default | Description |
|--------|---------|-------------|
| `max_concurrent_tasks` | 3 | Maximum number of parallel tasks |
| `rate_limit_per_minute` | 60 | API calls per minute |
| `default_timeout` | 300.0 | Default timeout per task (seconds) |
| `default_retry_count` | 3 | Number of retries on failure |
| `enable_logging` | True | Enable rich console logging |
| `log_level` | "INFO" | Logging level |
| `results_output_file` | None | Save results to JSON file |
| `save_intermediate_results` | True | Save intermediate results |

### TaskConfig

| Option | Default | Description |
|--------|---------|-------------|
| `id` | Required | Unique task identifier |
| `prompt` | Required | The prompt to send to Claude |
| `options` | {} | Claude Code options |
| `cwd` | None | Working directory for task |
| `timeout` | 300.0 | Task timeout in seconds |
| `retry_count` | 3 | Number of retries on failure |
| `priority` | 1 | Task priority (higher = more priority) |

## 🌐 y-router Integration

This project integrates with [y-router](https://github.com/user/y-router), a Cloudflare Worker that translates between Anthropic's Claude API and OpenAI-compatible APIs.

### Benefits of y-router

- **Access to More Models**: Use OpenRouter's vast selection of models
- **Cost Optimization**: Choose different models based on task complexity
- **Redundancy**: Fall back to different providers if one is unavailable
- **Rate Limit Distribution**: Spread load across multiple providers

### Setting Up y-router

1. **Use Shared Instance** (Quick Start)
   ```bash
   export ANTHROPIC_BASE_URL="https://cc.yovy.app"
   export ANTHROPIC_API_KEY="your-openrouter-api-key"
   ```

2. **Deploy Your Own** (Recommended for Production)
   ```bash
   # Clone y-router
   git clone https://github.com/user/y-router
   cd y-router
   
   # Deploy to Cloudflare Workers
   npm install -g wrangler
   wrangler deploy
   
   # Use your deployment
   export ANTHROPIC_BASE_URL="https://your-worker.your-subdomain.workers.dev"
   ```

## 📁 Project Structure

```
OpenTerra/
├── claude_parallel_runner.py    # Main parallel runner implementation
├── provider_config.py           # Provider management and configuration
├── cli.py                      # Command-line interface
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── examples/
│   └── basic_usage.py          # Usage examples
├── results/                    # Generated results (created at runtime)
├── provider_configs.json       # Custom provider configurations
├── claude_aliases.sh           # Generated shell aliases
└── README.md                   # This file
```

## 🔍 Monitoring and Debugging

### Rich Console Output

The runner provides beautiful console output with:
- Real-time progress tracking
- Colored status indicators
- Execution summaries with statistics
- Error details and retry information

### Logging

```python
# Enable debug logging
config = ParallelRunnerConfig(
    log_level="DEBUG",
    enable_logging=True
)
```

### Results Analysis

```python
# Analyze results
successful_tasks = [r for r in results if r.success]
failed_tasks = [r for r in results if not r.success]

print(f"Success rate: {len(successful_tasks)}/{len(results)}")
print(f"Average execution time: {sum(r.execution_time for r in successful_tasks)/len(successful_tasks):.2f}s")

# Export detailed results
import json
with open("detailed_results.json", "w") as f:
    json.dump([r.model_dump() for r in results], f, indent=2, default=str)
```

## 🚀 Performance Tips

1. **Optimize Concurrency**: Start with 3 concurrent tasks, adjust based on API limits
2. **Use Appropriate Timeouts**: Set realistic timeouts based on task complexity
3. **Implement Priority Queues**: Use task priorities for better resource allocation
4. **Monitor Rate Limits**: Adjust `rate_limit_per_minute` based on your API tier
5. **Batch Similar Tasks**: Group similar tasks together for better efficiency

## 🛠️ Troubleshooting

### Common Issues

1. **"No provider configured"**
   ```bash
   # Set up a provider
   python cli.py setup interactive
   ```

2. **Rate limiting errors**
   ```python
   # Reduce concurrent tasks and rate limit
   config = ParallelRunnerConfig(
       max_concurrent_tasks=2,
       rate_limit_per_minute=30
   )
   ```

3. **Connection timeouts**
   ```python
   # Increase timeout and retries
   task = TaskConfig(
       prompt="your prompt",
       timeout=600.0,  # 10 minutes
       retry_count=5
   )
   ```

4. **Import errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   npm install -g @anthropic-ai/claude-code
   ```

### Debug Mode

```bash
# Run with verbose output
python cli.py run --prompts "test" --verbose

# Check current configuration
python cli.py config current
python cli.py config validate
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request


**⚠️ Important**: This is an independent tool not affiliated with Anthropic, OpenAI, or OpenRouter. Users are responsible for compliance with all service terms and applicable costs. 