#!/usr/bin/env python3
"""
OpenTerra Agent System

A comprehensive agent system that gives AI models real tools to:
- Edit and create files
- Analyze codebases
- Run terminal commands
- Manage git repositories
- Install dependencies
"""

import os
import json
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

def validate_workspace_directory(workspace_dir: str) -> Dict[str, Any]:
    """Validate that a workspace directory exists and is accessible"""
    if not workspace_dir:
        return {
            "valid": True,
            "path": os.getcwd(),
            "message": "Using current working directory"
        }
    
    try:
        # Resolve to absolute path
        abs_path = os.path.abspath(workspace_dir.strip())
        
        # Check if directory exists
        if not os.path.exists(abs_path):
            return {
                "valid": False,
                "path": abs_path,
                "error": "Directory does not exist",
                "message": f"Directory {abs_path} does not exist"
            }
        
        # Check if it's actually a directory
        if not os.path.isdir(abs_path):
            return {
                "valid": False,
                "path": abs_path,
                "error": "Path is not a directory",
                "message": f"Path {abs_path} exists but is not a directory"
            }
        
        # Check if we can read the directory
        try:
            os.listdir(abs_path)
        except PermissionError:
            return {
                "valid": False,
                "path": abs_path,
                "error": "Permission denied",
                "message": f"Cannot access directory {abs_path} - permission denied"
            }
        
        # Check if we can write to the directory (for creating files)
        if not os.access(abs_path, os.W_OK):
            return {
                "valid": False,
                "path": abs_path,
                "error": "Directory not writable",
                "message": f"Directory {abs_path} is not writable"
            }
        
        return {
            "valid": True,
            "path": abs_path,
            "message": f"Valid workspace directory: {abs_path}"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "path": workspace_dir,
            "error": str(e),
            "message": f"Error validating directory: {str(e)}"
        }

class AgentTool:
    """Base class for agent tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class FileReadTool(AgentTool):
    """Read file contents"""
    def __init__(self):
        super().__init__(
            "read_file",
            "Read the contents of a file"
        )
    
    async def execute(self, file_path: str, workspace_dir: str = None) -> Dict[str, Any]:
        try:
            if workspace_dir and os.path.isabs(file_path):
                # If file_path is absolute, use it directly (but validate it's within workspace)
                full_path = file_path
                if not full_path.startswith(os.path.abspath(workspace_dir)):
                    return {
                        "success": False,
                        "error": "Access denied: file is outside workspace directory",
                        "message": f"File {file_path} is outside the allowed workspace"
                    }
            elif workspace_dir:
                full_path = os.path.join(workspace_dir, file_path)
            else:
                full_path = file_path
            
            # Resolve any relative paths and ensure they exist
            full_path = os.path.abspath(full_path)
            
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": "File not found",
                    "message": f"File {file_path} does not exist at {full_path}"
                }
            
            if not os.path.isfile(full_path):
                return {
                    "success": False,
                    "error": "Path is not a file",
                    "message": f"Path {full_path} exists but is not a file"
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": full_path,
                "file_size": len(content),
                "message": f"Successfully read {file_path} ({len(content)} characters)"
            }
        except UnicodeDecodeError as e:
            return {
                "success": False,
                "error": f"File encoding error: {str(e)}",
                "message": f"Could not read {file_path} - file may be binary or use unsupported encoding"
            }
        except PermissionError as e:
            return {
                "success": False,
                "error": "Permission denied",
                "message": f"Permission denied reading {file_path}: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to read {file_path}: {str(e)}"
            }

class FileWriteTool(AgentTool):
    """Write or create files"""
    def __init__(self):
        super().__init__(
            "write_file",
            "Write content to a file (creates file if it doesn't exist)"
        )
    
    async def execute(self, file_path: str, content: str, workspace_dir: str = None) -> Dict[str, Any]:
        try:
            if workspace_dir and os.path.isabs(file_path):
                # If file_path is absolute, use it directly (but validate it's within workspace)
                full_path = file_path
                if not full_path.startswith(os.path.abspath(workspace_dir)):
                    return {
                        "success": False,
                        "error": "Access denied: file is outside workspace directory",
                        "message": f"File {file_path} is outside the allowed workspace"
                    }
            elif workspace_dir:
                full_path = os.path.join(workspace_dir, file_path)
            else:
                full_path = file_path
            
            # Resolve to absolute path
            full_path = os.path.abspath(full_path)
            
            # Create directories if they don't exist
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Check if file already exists and get some info
            file_existed = os.path.exists(full_path)
            original_size = 0
            if file_existed:
                original_size = os.path.getsize(full_path)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            new_size = len(content)
            
            return {
                "success": True,
                "file_path": full_path,
                "file_existed": file_existed,
                "original_size": original_size,
                "new_size": new_size,
                "bytes_written": new_size,
                "lines_written": content.count('\n') + 1,
                "message": f"Successfully {'updated' if file_existed else 'created'} {file_path} ({new_size} bytes, {content.count(chr(10)) + 1} lines)"
            }
        except PermissionError as e:
            return {
                "success": False,
                "error": "Permission denied",
                "message": f"Permission denied writing to {file_path}: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to write {file_path}: {str(e)}"
            }

class DirectoryListTool(AgentTool):
    """List directory contents"""
    def __init__(self):
        super().__init__(
            "list_directory",
            "List files and directories in a given path"
        )
    
    async def execute(self, dir_path: str = ".", workspace_dir: str = None) -> Dict[str, Any]:
        try:
            if workspace_dir:
                full_path = os.path.join(workspace_dir, dir_path)
            else:
                full_path = dir_path
            
            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                items.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            
            return {
                "success": True,
                "items": items,
                "path": full_path,
                "message": f"Found {len(items)} items in {dir_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to list directory {dir_path}: {str(e)}"
            }

class RunCommandTool(AgentTool):
    """Execute terminal commands"""
    def __init__(self):
        super().__init__(
            "run_command",
            "Execute a terminal command and return the output"
        )
    
    async def execute(self, command: str, workspace_dir: str = None) -> Dict[str, Any]:
        try:
            # Change to workspace directory if specified
            original_cwd = os.getcwd()
            if workspace_dir and os.path.exists(workspace_dir):
                os.chdir(workspace_dir)
            
            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Restore original directory
            if workspace_dir:
                os.chdir(original_cwd)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "command": command,
                "message": f"Command executed: {command}"
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 30 seconds",
                "message": f"Command timed out: {command}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to run command {command}: {str(e)}"
            }

class CodeAnalysisTool(AgentTool):
    """Analyze code structure and patterns"""
    def __init__(self):
        super().__init__(
            "analyze_code",
            "Analyze code files for structure, dependencies, and patterns"
        )
    
    async def execute(self, file_path: str, workspace_dir: str = None) -> Dict[str, Any]:
        try:
            if workspace_dir:
                full_path = os.path.join(workspace_dir, file_path)
            else:
                full_path = file_path
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic analysis
            lines = content.split('\n')
            analysis = {
                "line_count": len(lines),
                "file_type": os.path.splitext(file_path)[1],
                "imports": [],
                "functions": [],
                "classes": []
            }
            
            # Extract imports, functions, classes based on file type
            if file_path.endswith('.py'):
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        analysis["imports"].append(line)
                    elif line.startswith('def '):
                        analysis["functions"].append(line)
                    elif line.startswith('class '):
                        analysis["classes"].append(line)
            
            return {
                "success": True,
                "analysis": analysis,
                "file_path": full_path,
                "message": f"Successfully analyzed {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to analyze {file_path}: {str(e)}"
            }

class CreateDirectoryTool(AgentTool):
    """Create directories"""
    def __init__(self):
        super().__init__(
            "create_directory",
            "Create a directory and any necessary parent directories"
        )
    
    async def execute(self, dir_path: str, workspace_dir: str = None) -> Dict[str, Any]:
        try:
            if workspace_dir:
                full_path = os.path.join(workspace_dir, dir_path)
            else:
                full_path = dir_path
            
            full_path = os.path.abspath(full_path)
            
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    return {
                        "success": True,
                        "path": full_path,
                        "already_existed": True,
                        "message": f"Directory {dir_path} already exists"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Path exists but is not a directory",
                        "message": f"Path {full_path} exists but is not a directory"
                    }
            
            os.makedirs(full_path, exist_ok=True)
            
            return {
                "success": True,
                "path": full_path,
                "already_existed": False,
                "message": f"Successfully created directory {dir_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create directory {dir_path}: {str(e)}"
            }

class DeleteFileTool(AgentTool):
    """Delete files or directories"""
    def __init__(self):
        super().__init__(
            "delete_file",
            "Delete a file or directory (use with caution)"
        )
    
    async def execute(self, file_path: str, workspace_dir: str = None, confirm: bool = False) -> Dict[str, Any]:
        try:
            if not confirm:
                return {
                    "success": False,
                    "error": "Confirmation required",
                    "message": "Add 'confirm': true to the parameters to confirm deletion"
                }
            
            if workspace_dir:
                full_path = os.path.join(workspace_dir, file_path)
            else:
                full_path = file_path
            
            full_path = os.path.abspath(full_path)
            
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": "File not found",
                    "message": f"File {file_path} does not exist"
                }
            
            is_directory = os.path.isdir(full_path)
            
            if is_directory:
                shutil.rmtree(full_path)
                return {
                    "success": True,
                    "path": full_path,
                    "was_directory": True,
                    "message": f"Successfully deleted directory {file_path}"
                }
            else:
                os.remove(full_path)
                return {
                    "success": True,
                    "path": full_path,
                    "was_directory": False,
                    "message": f"Successfully deleted file {file_path}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete {file_path}: {str(e)}"
            }

class SearchInFilesTool(AgentTool):
    """Search for text in files"""
    def __init__(self):
        super().__init__(
            "search_in_files",
            "Search for text patterns in files within a directory"
        )
    
    async def execute(self, search_term: str, directory: str = ".", file_pattern: str = "*", workspace_dir: str = None) -> Dict[str, Any]:
        try:
            import glob
            import re
            
            if workspace_dir:
                search_dir = os.path.join(workspace_dir, directory)
            else:
                search_dir = directory
            
            search_dir = os.path.abspath(search_dir)
            
            if not os.path.exists(search_dir):
                return {
                    "success": False,
                    "error": "Directory not found",
                    "message": f"Directory {directory} does not exist"
                }
            
            # Find files matching pattern
            pattern_path = os.path.join(search_dir, "**", file_pattern)
            files = glob.glob(pattern_path, recursive=True)
            files = [f for f in files if os.path.isfile(f)]
            
            matches = []
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, 1):
                            if search_term.lower() in line.lower():
                                rel_path = os.path.relpath(file_path, search_dir)
                                matches.append({
                                    "file": rel_path,
                                    "line": line_num,
                                    "content": line.strip(),
                                    "full_path": file_path
                                })
                except (UnicodeDecodeError, PermissionError):
                    # Skip binary files or files we can't read
                    continue
            
            return {
                "success": True,
                "search_term": search_term,
                "directory": search_dir,
                "files_searched": len(files),
                "matches_found": len(matches),
                "matches": matches[:50],  # Limit to first 50 matches
                "message": f"Found {len(matches)} matches for '{search_term}' in {len(files)} files"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Search failed: {str(e)}"
            }

class GitTool(AgentTool):
    """Git operations"""
    def __init__(self):
        super().__init__(
            "git_operation",
            "Perform git operations like status, add, commit, etc."
        )
    
    async def execute(self, operation: str, workspace_dir: str = None, **kwargs) -> Dict[str, Any]:
        try:
            original_cwd = os.getcwd()
            if workspace_dir and os.path.exists(workspace_dir):
                os.chdir(workspace_dir)
            
            if operation == "status":
                result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            elif operation == "add":
                files = kwargs.get("files", ".")
                result = subprocess.run(["git", "add", files], capture_output=True, text=True)
            elif operation == "commit":
                message = kwargs.get("message", "Auto commit by OpenTerra Agent")
                result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
            else:
                result = subprocess.run(["git"] + operation.split(), capture_output=True, text=True)
            
            if workspace_dir:
                os.chdir(original_cwd)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "operation": operation,
                "message": f"Git {operation} completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Git operation failed: {str(e)}"
            }

class Agent:
    """Main agent class that uses tools to complete tasks"""
    
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.tools = {
            "read_file": FileReadTool(),
            "write_file": FileWriteTool(),
            "list_directory": DirectoryListTool(),
            "create_directory": CreateDirectoryTool(),
            "delete_file": DeleteFileTool(),
            "search_in_files": SearchInFilesTool(),
            "run_command": RunCommandTool(),
            "analyze_code": CodeAnalysisTool(),
            "git_operation": GitTool()
        }
        
        # Initialize API client
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider in ["openrouter", "moonshot"]:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def execute_task(self, instruction: str, workspace_dir: str = None) -> Dict[str, Any]:
        """Execute a task using available tools"""
        try:
            # Validate workspace directory first
            workspace_validation = validate_workspace_directory(workspace_dir)
            if not workspace_validation["valid"]:
                return {
                    "success": False,
                    "error": workspace_validation["error"],
                    "actions": [],
                    "primaryAction": "error",
                    "workspace_error": workspace_validation["message"]
                }
            
            # Use the validated workspace path
            validated_workspace = workspace_validation["path"]
            
            # System prompt that explains the agent's capabilities
            system_prompt = f"""You are an AI coding assistant with access to powerful tools. You can:

1. **read_file(file_path)** - Read file contents
2. **write_file(file_path, content)** - Create or update files
3. **list_directory(dir_path)** - List directory contents
4. **create_directory(dir_path)** - Create directories
5. **delete_file(file_path, confirm=true)** - Delete files/directories (requires confirmation)
6. **search_in_files(search_term, directory=".", file_pattern="*")** - Search for text in files
7. **run_command(command)** - Execute terminal commands
8. **analyze_code(file_path)** - Analyze code structure
9. **git_operation(operation, **kwargs)** - Git operations

Working directory: {validated_workspace}

**Important Guidelines:**
- You are an AUTONOMOUS agent - you must complete tasks through tool usage, not just provide suggestions
- Always start by analyzing the workspace structure with list_directory
- Use read_file to understand existing code before making changes
- Make actual file edits using write_file when implementing features
- Test your changes with run_command after making modifications
- Use git_operation to manage version control if needed
- Provide clear explanations of each action you take

**CRITICAL: You must use tools to complete tasks. Do not just provide text explanations.**

**Task Format:**
For each action you take, structure your response as:

ACTION: tool_name
PARAMS: {{"param": "value"}}
REASONING: Brief explanation of why you're taking this action

Example:
ACTION: list_directory
PARAMS: {{"dir_path": "."}}
REASONING: Analyzing workspace structure to understand the project

I will execute the tool and give you the result. You then continue with the next action until the task is fully complete.

**Your mission:** {instruction}

**Start now by analyzing the workspace structure with list_directory.**"""

            messages = [{"role": "user", "content": instruction}]
            actions_taken = []
            max_iterations = 10  # Prevent infinite loops
            
            for iteration in range(max_iterations):
                # Get response from AI
                if self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        system=system_prompt,
                        messages=messages
                    )
                    ai_response = response.content[0].text
                else:
                    # OpenRouter
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": system_prompt}] + messages,
                        max_tokens=4000
                    )
                    ai_response = response.choices[0].message.content
                
                # Parse AI response for actions
                action_result = await self._parse_and_execute_action(ai_response, validated_workspace)
                
                if action_result:
                    actions_taken.append(action_result["action"])
                    # Add the action result to conversation
                    messages.append({"role": "assistant", "content": ai_response})
                    
                    # Format tool result for better AI understanding
                    tool_result = action_result['result']
                    result_message = f"Tool '{action_result['action']}' executed successfully.\n"
                    result_message += f"Reasoning: {action_result.get('reasoning', 'Not provided')}\n"
                    result_message += f"Result: {json.dumps(tool_result, indent=2)}\n"
                    
                    if not tool_result.get('success', True):
                        result_message += "\n**ERROR**: The tool execution failed. Please analyze the error and try a different approach."
                    
                    result_message += "\nContinue with the next action to complete your mission."
                    
                    messages.append({"role": "user", "content": result_message})
                    
                    # Check if task seems complete but be more specific
                    completion_indicators = ["task complete", "mission complete", "finished successfully", "all done"]
                    if any(indicator in ai_response.lower() for indicator in completion_indicators):
                        # Verify the agent explicitly states completion
                        if "task" in ai_response.lower() and "complete" in ai_response.lower():
                            break
                else:
                    # No action found - encourage the agent to take action
                    messages.append({"role": "assistant", "content": ai_response})
                    if iteration < max_iterations - 1:  # Don't add this on the last iteration
                        encouragement = """
No action detected in your response. Remember:
- You must use tools to complete tasks
- Start each action with: ACTION: tool_name
- Follow with: PARAMS: {"param": "value"}
- Add: REASONING: explanation

Please continue with your next action to complete the mission."""
                        messages.append({"role": "user", "content": encouragement})
                    break
            
            return {
                "success": True,
                "result": ai_response,
                "actions": actions_taken,
                "primaryAction": actions_taken[0] if actions_taken else "analysis",
                "iterations": iteration + 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "actions": actions_taken if 'actions_taken' in locals() else [],
                "primaryAction": "error"
            }
    
    async def _parse_and_execute_action(self, ai_response: str, workspace_dir: str = None) -> Optional[Dict[str, Any]]:
        """Parse AI response for actions and execute them"""
        lines = ai_response.split('\n')
        
        action_found = False
        tool_name = None
        params = {}
        reasoning = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('ACTION:'):
                tool_name = line.replace('ACTION:', '').strip()
                action_found = True
                
            elif line.startswith('PARAMS:') and action_found:
                try:
                    params_str = line.replace('PARAMS:', '').strip()
                    params = json.loads(params_str)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for params: {e}")
                    params = {}
                
            elif line.startswith('REASONING:') and action_found:
                reasoning = line.replace('REASONING:', '').strip()
        
        # If we found an action, execute it
        if action_found and tool_name:
            # Add workspace_dir to params
            if workspace_dir:
                params['workspace_dir'] = workspace_dir
            
            # Execute the tool
            if tool_name in self.tools:
                print(f"Executing tool: {tool_name} with params: {params}")
                result = await self.tools[tool_name].execute(**params)
                return {
                    "action": tool_name,
                    "params": params,
                    "result": result,
                    "reasoning": reasoning
                }
            else:
                print(f"Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}")
                return {
                    "action": tool_name,
                    "params": params,
                    "result": {
                        "success": False,
                        "error": f"Unknown tool: {tool_name}",
                        "available_tools": list(self.tools.keys())
                    },
                    "reasoning": reasoning
                }
        
        return None 