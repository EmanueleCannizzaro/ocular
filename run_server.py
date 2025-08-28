#!/usr/bin/env python3
"""
Development server runner for Ocular OCR web application.
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run the development server."""
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("Warning: .env file not found. Please create one with your API keys.")
        print("Example content:")
        print("MISTRAL_API_KEY=your_mistral_api_key_here")
        print("")
    
    # Set environment for development
    os.environ.setdefault("ENVIRONMENT", "development")
    
    # Run the server
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        reload_dirs=[str(project_root)]
    )

if __name__ == "__main__":
    main()