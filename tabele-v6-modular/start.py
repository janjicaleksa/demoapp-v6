#!/usr/bin/env python3
"""
Startup script for AI Processor Kupci Dobavljaci (Modular Version)
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print("✓ Loaded environment variables from .env file")

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    
    print(f"🚀 Starting {settings.app_name} (Modular Version)...")
    print(f"📍 Server will be available at: http://{settings.host}:{settings.port}")
    print(f"📚 API Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"🏥 Health Check: http://{settings.host}:{settings.port}/health")
    print(f"🐛 Debug mode: {settings.debug}")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )