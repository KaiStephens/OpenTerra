"""
Claude Code SDK Parallel Runner

This module provides a robust framework for running multiple Claude Code SDK instances
in parallel with proper error handling, rate limiting, and resource management.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import json

import anyio
from asyncio_throttle import Throttler
from rich.console import Console
from rich.progress import Progress, TaskID, track
from rich.logging import RichHandler
from rich.table import Table
from pydantic import BaseModel, Field

from claude_code_sdk import query, ClaudeCodeOptions, Message
from provider_config import provider_manager, ensure_provider_setup


# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class TaskConfig(BaseModel):
    """Configuration for a single Claude Code task."""
    id: str = Field(..., description="Unique task identifier")
    prompt: str = Field(..., description="The prompt to send to Claude")
    options: Dict[str, Any] = Field(default_factory=dict, description="Claude Code options")
    cwd: Optional[Union[str, Path]] = Field(None, description="Working directory for this task")
    timeout: Optional[float] = Field(300.0, description="Task timeout in seconds")
    retry_count: int = Field(3, description="Number of retries on failure")
    priority: int = Field(1, description="Task priority (higher = more priority)")


class TaskResult(BaseModel):
    """Result of a Claude Code task execution."""
    task_id: str
    success: bool
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: datetime
    end_time: datetime
    retry_attempts: int = 0


@dataclass
class ParallelRunnerConfig:
    """Configuration for the parallel runner."""
    max_concurrent_tasks: int = 3
    rate_limit_per_minute: int = 60
    default_timeout: float = 300.0
    default_retry_count: int = 3
    enable_logging: bool = True
    log_level: str = "INFO"
    results_output_file: Optional[str] = None
    save_intermediate_results: bool = True


class ClaudeParallelRunner:
    """
    A robust parallel runner for Claude Code SDK instances.
    
    Features:
    - Rate limiting to respect API limits
    - Concurrent execution with configurable limits
    - Automatic retries with exponential backoff
    - Progress tracking and rich console output
    - Resource cleanup and error handling
    - Results persistence and reporting
    """
    
    def __init__(self, config: ParallelRunnerConfig = None, provider_name: Optional[str] = None):
        self.config = config or ParallelRunnerConfig()
        self.throttler = Throttler(rate_limit=self.config.rate_limit_per_minute, period=60)
        self.results: List[TaskResult] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # Setup provider
        if provider_name:
            provider_manager.set_provider(provider_name)
        
        # Ensure a provider is configured
        if not ensure_provider_setup():
            raise RuntimeError("No Claude provider configured. Please set up a provider first.")
        
        # Setup logging
        if self.config.enable_logging:
            logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper()))
            
        # Log current provider
        current_provider = provider_manager.get_current_config()
        if current_provider:
            logger.info(f"🔧 Using provider: {current_provider.name} ({current_provider.description})")
    
    async def execute_single_task(self, task_config: TaskConfig) -> TaskResult:
        """Execute a single Claude Code task with retries and error handling."""
        start_time = datetime.now()
        
        result = TaskResult(
            task_id=task_config.id,
            success=False,
            start_time=start_time,
            end_time=start_time,
            retry_attempts=0
        )
        
        for attempt in range(task_config.retry_count + 1):
            try:
                result.retry_attempts = attempt
                
                # Apply rate limiting
                async with self.throttler:
                    # Get provider-specific options
                    provider_options = provider_manager.get_claude_code_options()
                    
                    # Merge provider options with task-specific options
                    merged_options = {**provider_options, **task_config.options}
                    
                    # Setup Claude Code options
                    options = ClaudeCodeOptions(**merged_options)
                    if task_config.cwd:
                        options.cwd = Path(task_config.cwd)
                    
                    messages: List[Message] = []
                    
                    # Execute with timeout
                    async with asyncio.timeout(task_config.timeout):
                        async for message in query(
                            prompt=task_config.prompt,
                            options=options
                        ):
                            messages.append(message)
                            
                            # Save intermediate results if enabled
                            if self.config.save_intermediate_results:
                                await self._save_intermediate_result(task_config.id, message)
                    
                    # Convert messages to serializable format
                    result.messages = [self._message_to_dict(msg) for msg in messages]
                    result.success = True
                    result.end_time = datetime.now()
                    result.execution_time = (result.end_time - result.start_time).total_seconds()
                    
                    logger.info(f"✅ Task {task_config.id} completed successfully")
                    break
                    
            except asyncio.TimeoutError:
                error_msg = f"Task {task_config.id} timed out after {task_config.timeout}s"
                logger.warning(f"⏰ {error_msg}")
                result.error = error_msg
                
            except Exception as e:
                error_msg = f"Task {task_config.id} failed on attempt {attempt + 1}: {str(e)}"
                logger.warning(f"❌ {error_msg}")
                result.error = error_msg
                
                # Exponential backoff for retries
                if attempt < task_config.retry_count:
                    backoff_time = 2 ** attempt
                    logger.info(f"🔄 Retrying {task_config.id} in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
        
        result.end_time = datetime.now()
        result.execution_time = (result.end_time - result.start_time).total_seconds()
        
        if not result.success:
            logger.error(f"💥 Task {task_config.id} failed after {result.retry_attempts + 1} attempts")
        
        return result
    
    async def run_parallel(self, tasks: List[TaskConfig], 
                          progress_callback: Optional[Callable] = None) -> List[TaskResult]:
        """
        Run multiple Claude Code tasks in parallel.
        
        Args:
            tasks: List of task configurations to execute
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of task results
        """
        if not tasks:
            logger.warning("No tasks provided to execute")
            return []
        
        logger.info(f"🚀 Starting parallel execution of {len(tasks)} tasks")
        logger.info(f"📊 Max concurrent: {self.config.max_concurrent_tasks}, Rate limit: {self.config.rate_limit_per_minute}/min")
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Create progress tracker
        with Progress() as progress:
            task_progress = progress.add_task("[cyan]Executing tasks...", total=len(tasks))
            
            async def run_task_with_semaphore(task_config: TaskConfig) -> TaskResult:
                async with self.semaphore:
                    result = await self.execute_single_task(task_config)
                    progress.update(task_progress, advance=1)
                    
                    if progress_callback:
                        progress_callback(result)
                    
                    return result
            
            # Execute all tasks concurrently
            tasks_coroutines = [run_task_with_semaphore(task) for task in sorted_tasks]
            results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = TaskResult(
                        task_id=sorted_tasks[i].id,
                        success=False,
                        error=str(result),
                        start_time=datetime.now(),
                        end_time=datetime.now()
                    )
                    processed_results.append(error_result)
                    logger.error(f"💥 Unexpected error in task {sorted_tasks[i].id}: {result}")
                else:
                    processed_results.append(result)
        
        self.results.extend(processed_results)
        
        # Save results if configured
        if self.config.results_output_file:
            await self._save_results(processed_results)
        
        # Print summary
        self._print_execution_summary(processed_results)
        
        return processed_results
    
    def run_sync(self, tasks: List[TaskConfig]) -> List[TaskResult]:
        """Synchronous wrapper for running parallel tasks."""
        return anyio.run(self.run_parallel, tasks)
    
    async def _save_intermediate_result(self, task_id: str, message: Message):
        """Save intermediate results during task execution."""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().isoformat()
            filename = results_dir / f"{task_id}_intermediate_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(self._message_to_dict(message), f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save intermediate result for {task_id}: {e}")
    
    async def _save_results(self, results: List[TaskResult]):
        """Save final results to file."""
        try:
            results_data = {
                "execution_time": datetime.now().isoformat(),
                "total_tasks": len(results),
                "successful_tasks": sum(1 for r in results if r.success),
                "failed_tasks": sum(1 for r in results if not r.success),
                "results": [result.model_dump() for result in results]
            }
            
            with open(self.config.results_output_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)
                
            logger.info(f"💾 Results saved to {self.config.results_output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert a Claude Code message to a serializable dictionary."""
        try:
            if hasattr(message, 'model_dump'):
                return message.model_dump()
            elif hasattr(message, '__dict__'):
                return message.__dict__
            else:
                return {"content": str(message)}
        except Exception:
            return {"content": str(message), "type": "unknown"}
    
    def _print_execution_summary(self, results: List[TaskResult]):
        """Print a rich summary of execution results."""
        console.print("\n" + "="*60)
        console.print("📋 [bold cyan]EXECUTION SUMMARY[/bold cyan]")
        console.print("="*60)
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        total_time = sum(r.execution_time for r in results)
        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        
        # Summary stats
        stats_table = Table(title="📊 Execution Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Tasks", str(total_tasks))
        stats_table.add_row("Successful", f"✅ {successful_tasks}")
        stats_table.add_row("Failed", f"❌ {failed_tasks}")
        stats_table.add_row("Success Rate", f"{(successful_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%")
        stats_table.add_row("Total Execution Time", f"{total_time:.2f}s")
        stats_table.add_row("Average Task Time", f"{avg_time:.2f}s")
        
        console.print(stats_table)
        
        # Detailed results
        if failed_tasks > 0:
            console.print("\n❌ [bold red]Failed Tasks:[/bold red]")
            for result in results:
                if not result.success:
                    console.print(f"  • {result.task_id}: {result.error}")
        
        console.print("\n🎉 [bold green]Execution completed![/bold green]")


# Convenience functions for quick usage
async def run_tasks_parallel(prompts: List[str], 
                           max_concurrent: int = 3,
                           timeout: float = 300.0,
                           provider: Optional[str] = None) -> List[TaskResult]:
    """
    Quick function to run multiple prompts in parallel.
    
    Args:
        prompts: List of prompts to execute
        max_concurrent: Maximum number of concurrent tasks
        timeout: Timeout per task in seconds
        provider: Provider name to use (e.g., 'anthropic', 'openrouter')
        
    Returns:
        List of task results
    """
    config = ParallelRunnerConfig(max_concurrent_tasks=max_concurrent)
    runner = ClaudeParallelRunner(config, provider_name=provider)
    
    tasks = [
        TaskConfig(
            id=f"task_{i+1}",
            prompt=prompt,
            timeout=timeout
        )
        for i, prompt in enumerate(prompts)
    ]
    
    return await runner.run_parallel(tasks)


def run_tasks_parallel_sync(prompts: List[str], 
                          max_concurrent: int = 3,
                          timeout: float = 300.0,
                          provider: Optional[str] = None) -> List[TaskResult]:
    """Synchronous version of run_tasks_parallel."""
    return anyio.run(run_tasks_parallel, prompts, max_concurrent, timeout, provider)


if __name__ == "__main__":
    # Example usage
    example_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Create a simple REST API using FastAPI",
        "Explain how to optimize database queries",
        "Write unit tests for a calculator class",
        "Create a Docker configuration for a Python app"
    ]
    
    # Run example
    results = run_tasks_parallel_sync(example_prompts, max_concurrent=2)
    
    # Print results
    for result in results:
        if result.success:
            print(f"✅ {result.task_id}: Completed in {result.execution_time:.2f}s")
        else:
            print(f"❌ {result.task_id}: Failed - {result.error}") 