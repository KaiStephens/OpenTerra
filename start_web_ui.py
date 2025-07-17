#!/usr/bin/env python3
"""
OpenTerra Chat Launcher

Simple script to start the OpenTerra Chat web interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the OpenTerra Chat web interface."""
    print("🚀 Starting OpenTerra Chat...")
    print("💬 This will start a web-based chat interface for Claude AI")
    print("")
    
    # Check if we're in the right directory
    if not Path("web_ui.py").exists():
        print("❌ Error: web_ui.py not found in current directory")
        print("Please run this script from the OpenTerra project root")
        sys.exit(1)
    
    # Start the web UI
    try:
        print("📡 Starting server on http://localhost:8000")
        print("🌐 Open your browser and go to: http://localhost:8000")
        print("")
        print("💡 To configure your API key:")
        print("   1. Click the Settings button in the sidebar")
        print("   2. Select your provider (Anthropic, OpenRouter, etc.)")
        print("   3. Enter your API key")
        print("   4. Save settings and start chatting!")
        print("")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the web UI
        subprocess.run([
            sys.executable, "web_ui.py",
            "--host", "0.0.0.0",
            "--port", "8000"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down OpenTerra Chat...")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 