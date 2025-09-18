#!/usr/bin/env python3
"""
Startup script for the Knowledge Retrieval API
"""

import uvicorn
from config import Config

if __name__ == "__main__":
    print("🚀 Starting Knowledge Retrieval API")
    print(f"📡 Server will be available at: http://{Config.API_HOST}:{Config.API_PORT}")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )
