#!/usr/bin/env python3
"""
Startup script for AI Processor Kupci Dobavljaci
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

if __name__ == "__main__":
    import uvicorn
    from main import app
    
    # Get configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting AI Processor Kupci Dobavljaci...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"Press Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    ) 