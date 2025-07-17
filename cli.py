#!/usr/bin/env python3
"""
Claude Parallel Runner CLI

A command-line interface for running multiple Claude Code SDK instances in parallel
with support for different providers (Anthropic, OpenRouter, etc.).

Usage:
    python cli.py run --prompts "task1" "task2" "task3"
    python cli.py config --list
    python cli.py config --set-provider openrouter
    python cli.py setup --provider openrouter --api-key your-key
"""

import typer
import asyncio
import json
import os
from typing import List, Optional
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from claude_parallel_runner import (
    ClaudeParallelRunner, 
    ParallelRunnerConfig, 
    TaskConfig,
    run_tasks_parallel_sync
)
from provider_config import (
    provider_manager, 
    quick_setup_anthropic, 
    quick_setup_openrouter,
    ensure_provider_setup,
    ProviderConfig,
    ProviderType
)

app = typer.Typer(help="Claude Parallel Runner - Run multiple Claude instances in parallel")
config_app = typer.Typer(help="Configuration management")
setup_app = typer.Typer(help="Provider setup commands")

app.add_typer(config_app, name="config")
app.add_typer(setup_app, name="setup")

console = Console()


@app.command()
def run(
    prompts: List[str] = typer.Option(..., "--prompts", "-p", help="List of prompts to execute"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to use (anthropic, openrouter, etc.)"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", "-c", help="Maximum concurrent tasks"),
    timeout: float = typer.Option(300.0, "--timeout", "-t", help="Timeout per task in seconds"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Run multiple prompts in parallel."""
    
    if not prompts:
        console.print("❌ No prompts provided. Use --prompts to specify tasks.")
        raise typer.Exit(1)
    
    console.print(f"🚀 Running {len(prompts)} prompts in parallel")
    
    if provider:
        console.print(f"🔧 Using provider: {provider}")
    
    try:
        # Configure output file if specified
        config = ParallelRunnerConfig(
            max_concurrent_tasks=max_concurrent,
            results_output_file=output_file,
            log_level="DEBUG" if verbose else "INFO"
        )
        
        # Run tasks
        results = run_tasks_parallel_sync(
            prompts=prompts,
            max_concurrent=max_concurrent,
            timeout=timeout,
            provider=provider
        )
        
        # Display summary
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        if successful == total:
            console.print(f"🎉 All {total} tasks completed successfully!")
        else:
            console.print(f"⚠️  {successful}/{total} tasks completed successfully")
            
        if output_file:
            console.print(f"💾 Results saved to {output_file}")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def interactive():
    """Run in interactive mode to configure and execute tasks."""
    
    console.print(Panel("🤖 Claude Parallel Runner - Interactive Mode", style="cyan"))
    
    # Check provider setup
    if not ensure_provider_setup():
        console.print("Let's set up a provider first...")
        setup_interactive()
    
    # Get current provider info
    current_provider = provider_manager.get_current_config()
    if current_provider:
        console.print(f"📡 Current provider: {current_provider.name}")
    
    prompts = []
    console.print("\n📝 Enter your prompts (one per line, empty line to finish):")
    
    while True:
        prompt = Prompt.ask(f"Prompt {len(prompts) + 1}", default="")
        if not prompt:
            break
        prompts.append(prompt)
    
    if not prompts:
        console.print("No prompts entered. Exiting.")
        return
    
    # Get execution parameters
    max_concurrent = typer.prompt("Max concurrent tasks", default=3, type=int)
    timeout = typer.prompt("Timeout per task (seconds)", default=300.0, type=float)
    
    save_results = Confirm.ask("Save results to file?", default=False)
    output_file = None
    if save_results:
        output_file = Prompt.ask("Output filename", default="results.json")
    
    # Confirm execution
    console.print(f"\n📊 Ready to execute {len(prompts)} prompts")
    console.print(f"   Max concurrent: {max_concurrent}")
    console.print(f"   Timeout: {timeout}s")
    console.print(f"   Provider: {current_provider.name if current_provider else 'Auto-detect'}")
    
    if not Confirm.ask("Proceed with execution?", default=True):
        console.print("Cancelled.")
        return
    
    # Execute tasks
    try:
        config = ParallelRunnerConfig(
            max_concurrent_tasks=max_concurrent,
            results_output_file=output_file
        )
        
        results = run_tasks_parallel_sync(
            prompts=prompts,
            max_concurrent=max_concurrent,
            timeout=timeout
        )
        
        console.print("✅ Execution completed!")
        
    except Exception as e:
        console.print(f"❌ Error during execution: {str(e)}")


@config_app.command("list")
def list_providers():
    """List available providers."""
    provider_manager.list_providers()


@config_app.command("current")
def show_current():
    """Show current provider configuration."""
    current = provider_manager.get_current_config()
    if current:
        console.print(f"📡 Current provider: {current.name}")
        console.print(f"   Type: {current.provider_type.value}")
        console.print(f"   Base URL: {current.base_url or 'Default'}")
        console.print(f"   Model: {current.model or 'Default'}")
        console.print(f"   Description: {current.description}")
        
        # Validate setup
        if provider_manager.validate_current_setup():
            console.print("✅ Provider is properly configured")
        else:
            console.print("❌ Provider configuration has issues")
    else:
        console.print("❌ No provider currently selected")


@config_app.command("set")
def set_provider(
    name: str = typer.Argument(..., help="Provider name to set as current"),
    model: Optional[str] = typer.Option(None, "--model", help="Override the default model"),
    small_fast_model: Optional[str] = typer.Option(None, "--small-fast-model", help="Override the small/fast model")
):
    """Set the current provider with optional model override."""
    if provider_manager.set_provider(name, model=model, small_fast_model=small_fast_model):
        console.print(f"✅ Provider set to: {name}")
    else:
        console.print(f"❌ Failed to set provider: {name}")
        console.print("Available providers:")
        provider_manager.list_providers()


@config_app.command("validate")
def validate_setup():
    """Validate current provider setup."""
    if provider_manager.validate_current_setup():
        console.print("✅ Current setup is valid")
    else:
        console.print("❌ Current setup has issues")


@config_app.command("models")
def list_models(provider: Optional[str] = typer.Option(None, "--provider", help="Provider to show models for")):
    """List available models for a provider."""
    if not provider:
        # Show models for current provider
        current = provider_manager.get_current_config()
        if current:
            provider = current.name
        else:
            console.print("❌ No current provider. Specify --provider or set a current provider.")
            return
    
    models = provider_manager.get_available_models(provider)
    if not models:
        console.print(f"❌ No models found for provider: {provider}")
        return
    
    console.print(f"🤖 Available models for {provider}:")
    for model in models:
        console.print(f"  • {model}")

@config_app.command("set-model")
def set_model(
    provider: str = typer.Argument(..., help="Provider name"),
    model: Optional[str] = typer.Option(None, "--model", help="Main model to set"),
    small_fast_model: Optional[str] = typer.Option(None, "--small-fast-model", help="Small/fast model to set")
):
    """Update model configuration for a provider."""
    if provider_manager.update_provider_model(provider, model=model, small_fast_model=small_fast_model):
        console.print(f"✅ Updated models for {provider}")
    else:
        console.print(f"❌ Failed to update models for {provider}")

@config_app.command("aliases")
def generate_aliases():
    """Generate shell aliases for quick provider switching."""
    aliases = provider_manager.create_shell_aliases()
    console.print("📝 Shell aliases:")
    console.print(aliases)
    
    if Confirm.ask("Save aliases to file?", default=True):
        filename = "claude_aliases.sh"
        with open(filename, "w") as f:
            f.write(aliases)
        console.print(f"💾 Aliases saved to {filename}")
        console.print(f"Run: source {filename} to activate aliases")


@setup_app.command("anthropic")
def setup_anthropic(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Anthropic API key"),
    set_current: bool = typer.Option(True, "--set-current", help="Set as current provider")
):
    """Set up Anthropic provider."""
    
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = typer.prompt("Anthropic API key", hide_input=True)
    
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    if set_current:
        provider_manager.set_provider("anthropic")
    
    console.print("✅ Anthropic provider configured")
    
    if provider_manager.validate_current_setup():
        console.print("✅ Setup validation passed")
    else:
        console.print("⚠️  Setup validation failed - please check your API key")


@setup_app.command("openrouter")
def setup_openrouter(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenRouter API key"),
    custom_router: bool = typer.Option(False, "--custom", help="Use custom y-router deployment"),
    router_url: Optional[str] = typer.Option(None, "--router-url", help="Custom router URL"),
    model: Optional[str] = typer.Option(None, "--model", help="Specific model to use"),
    set_current: bool = typer.Option(True, "--set-current", help="Set as current provider")
):
    """Set up OpenRouter provider."""
    
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            api_key = typer.prompt("OpenRouter API key", hide_input=True)
    
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    provider_name = "openrouter"
    
    if custom_router:
        provider_name = "openrouter_custom"
        if router_url:
            # Update the custom provider configuration
            custom_config = provider_manager.providers["openrouter_custom"]
            custom_config.base_url = router_url
            console.print(f"📡 Using custom router: {router_url}")
    
    if model:
        provider_manager.providers[provider_name].model = model
        console.print(f"🤖 Using model: {model}")
    
    if set_current:
        provider_manager.set_provider(provider_name)
    
    console.print("✅ OpenRouter provider configured")
    
    if provider_manager.validate_current_setup():
        console.print("✅ Setup validation passed")
    else:
        console.print("⚠️  Setup validation failed - please check your API key")


@setup_app.command("interactive")
def setup_interactive():
    """Interactive provider setup."""
    
    console.print(Panel("🔧 Provider Setup", style="cyan"))
    
    # Show available providers
    provider_manager.list_providers()
    
    provider_choice = Prompt.ask(
        "Select provider", 
        choices=["anthropic", "openrouter", "openrouter_custom", "moonshot"],
        default="anthropic"
    )
    
    if provider_choice == "anthropic":
        api_key = typer.prompt("Anthropic API key", hide_input=True)
        setup_anthropic(api_key=api_key)
        
    elif provider_choice in ["openrouter", "openrouter_custom"]:
        api_key = typer.prompt("OpenRouter API key", hide_input=True)
        
        if provider_choice == "openrouter_custom":
            router_url = Prompt.ask("Custom y-router URL", default="https://your-worker.your-subdomain.workers.dev")
            setup_openrouter(api_key=api_key, custom_router=True, router_url=router_url)
        else:
            setup_openrouter(api_key=api_key)
    
    elif provider_choice == "moonshot":
        api_key = typer.prompt("Moonshot API key", hide_input=True)
        os.environ["MOONSHOT_API_KEY"] = api_key
        provider_manager.set_provider("moonshot")
        console.print("✅ Moonshot provider configured")


@app.command()
def example():
    """Run a quick example with sample prompts."""
    
    console.print("🎯 Running example with sample prompts...")
    
    sample_prompts = [
        "Write a Python function to calculate the factorial of a number",
        "Explain the concept of recursion in programming",
        "Create a simple REST API endpoint using FastAPI",
        "Write a unit test for a calculator function"
    ]
    
    console.print(f"📝 Sample prompts: {len(sample_prompts)}")
    for i, prompt in enumerate(sample_prompts, 1):
        console.print(f"   {i}. {prompt}")
    
    if not Confirm.ask("Run these sample prompts?", default=True):
        console.print("Cancelled.")
        return
    
    try:
        results = run_tasks_parallel_sync(
            prompts=sample_prompts,
            max_concurrent=2,
            timeout=120.0
        )
        
        console.print("🎉 Example completed!")
        console.print("Check the execution summary above for results.")
        
    except Exception as e:
        console.print(f"❌ Example failed: {str(e)}")


if __name__ == "__main__":
    app() 