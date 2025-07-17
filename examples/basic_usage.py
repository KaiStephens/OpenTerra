#!/usr/bin/env python3
"""
Basic Usage Examples for Claude Parallel Runner

This script demonstrates how to use the Claude Parallel Runner with different providers.
"""

import os
from claude_parallel_runner import run_tasks_parallel_sync, TaskConfig, ClaudeParallelRunner, ParallelRunnerConfig
from provider_config import provider_manager, quick_setup_anthropic, quick_setup_openrouter

def example_1_simple_prompts():
    """Example 1: Run simple prompts in parallel."""
    print("🎯 Example 1: Simple parallel prompts")
    
    prompts = [
        "Write a Python function to reverse a string",
        "Explain the difference between lists and tuples in Python",
        "Create a simple class for a bank account with deposit and withdraw methods"
    ]
    
    # Run with default provider (auto-detected)
    results = run_tasks_parallel_sync(
        prompts=prompts,
        max_concurrent=2,
        timeout=120.0
    )
    
    print(f"✅ Completed {len([r for r in results if r.success])}/{len(results)} tasks")


def example_2_provider_switching():
    """Example 2: Switch between providers."""
    print("🎯 Example 2: Provider switching")
    
    # List available providers
    print("\n📋 Available providers:")
    provider_manager.list_providers()
    
    # Switch to OpenRouter (if configured)
    if os.getenv("OPENROUTER_API_KEY"):
        print("\n🔄 Switching to OpenRouter...")
        provider_manager.set_provider("openrouter")
        
        prompts = ["Explain async/await in Python"]
        results = run_tasks_parallel_sync(prompts, provider="openrouter")
        print(f"OpenRouter result: {'✅' if results[0].success else '❌'}")
    
    # Switch to Anthropic (if configured)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n🔄 Switching to Anthropic...")
        provider_manager.set_provider("anthropic")
        
        prompts = ["Explain decorators in Python"]
        results = run_tasks_parallel_sync(prompts, provider="anthropic")
        print(f"Anthropic result: {'✅' if results[0].success else '❌'}")


def example_3_advanced_configuration():
    """Example 3: Advanced configuration with custom settings."""
    print("🎯 Example 3: Advanced configuration")
    
    # Create advanced configuration
    config = ParallelRunnerConfig(
        max_concurrent_tasks=4,
        rate_limit_per_minute=30,  # Slower rate for demonstration
        default_timeout=180.0,
        results_output_file="advanced_results.json",
        save_intermediate_results=True
    )
    
    # Create custom tasks
    tasks = [
        TaskConfig(
            id="code_review_task",
            prompt="Review this Python code for potential improvements: def add(a, b): return a + b",
            timeout=60.0,
            priority=3  # High priority
        ),
        TaskConfig(
            id="documentation_task", 
            prompt="Write comprehensive documentation for a REST API endpoint",
            timeout=120.0,
            priority=1  # Low priority
        ),
        TaskConfig(
            id="testing_task",
            prompt="Create unit tests for a shopping cart class",
            timeout=90.0,
            priority=2  # Medium priority
        )
    ]
    
    # Run with custom configuration
    runner = ClaudeParallelRunner(config)
    results = runner.run_sync(tasks)
    
    print(f"✅ Advanced example completed with {len(results)} tasks")
    print("📁 Results saved to advanced_results.json")


def example_4_error_handling():
    """Example 4: Error handling and retries."""
    print("🎯 Example 4: Error handling")
    
    # Include a potentially problematic prompt
    prompts = [
        "Write a simple hello world function",
        "Explain quantum computing in simple terms",
        "",  # Empty prompt that might cause issues
        "Create a basic HTTP server in Python"
    ]
    
    try:
        results = run_tasks_parallel_sync(
            prompts=prompts,
            max_concurrent=2,
            timeout=60.0
        )
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"📊 Results: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            print("❌ Failed tasks:")
            for result in failed:
                print(f"   • {result.task_id}: {result.error}")
    
    except Exception as e:
        print(f"💥 Error during execution: {e}")


def example_5_background_processing():
    """Example 5: Simulate background processing."""
    print("🎯 Example 5: Background-style processing")
    
    # Simulate multiple batches of work
    batches = [
        ["Write a function to calculate prime numbers", "Explain list comprehensions"],
        ["Create a simple web scraper", "Write a data validation function"],
        ["Implement a binary search algorithm", "Explain Python generators"]
    ]
    
    for i, batch in enumerate(batches, 1):
        print(f"\n🔄 Processing batch {i}/{len(batches)}")
        
        results = run_tasks_parallel_sync(
            prompts=batch,
            max_concurrent=2,
            timeout=90.0
        )
        
        print(f"   Batch {i} completed: {len([r for r in results if r.success])}/{len(results)} successful")


def example_6_model_comparison():
    """Example 6: Compare responses from different models/providers."""
    print("🎯 Example 6: Model comparison")
    
    test_prompt = "Explain the concept of object-oriented programming in 2-3 sentences"
    
    providers_to_test = []
    
    # Check which providers are available
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append("anthropic")
    
    if os.getenv("OPENROUTER_API_KEY"):
        providers_to_test.append("openrouter")
    
    if not providers_to_test:
        print("❌ No providers configured for comparison")
        return
    
    print(f"🔍 Testing prompt with {len(providers_to_test)} providers:")
    print(f"   '{test_prompt}'")
    
    for provider in providers_to_test:
        print(f"\n📡 Testing with {provider}...")
        
        try:
            results = run_tasks_parallel_sync(
                prompts=[test_prompt],
                max_concurrent=1,
                timeout=60.0,
                provider=provider
            )
            
            if results and results[0].success:
                print(f"✅ {provider}: Response received")
                # You could analyze/compare the responses here
            else:
                print(f"❌ {provider}: Failed")
                
        except Exception as e:
            print(f"💥 {provider}: Error - {e}")


if __name__ == "__main__":
    print("🚀 Claude Parallel Runner - Basic Usage Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_1_simple_prompts()
        print("\n" + "-" * 40)
        
        example_2_provider_switching()
        print("\n" + "-" * 40)
        
        example_3_advanced_configuration()
        print("\n" + "-" * 40)
        
        example_4_error_handling()
        print("\n" + "-" * 40)
        
        example_5_background_processing()
        print("\n" + "-" * 40)
        
        example_6_model_comparison()
        
    except KeyboardInterrupt:
        print("\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
    
    print("\n🎉 All examples completed!") 