#!/usr/bin/env python3
"""
Test script to check if the server can start
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing API server import...")
    import api_server
    print("✅ API server imported successfully")
    
    print("Testing FastAPI app...")
    app = api_server.app
    print("✅ FastAPI app created successfully")
    
    print("Testing uvicorn import...")
    import uvicorn
    print("✅ Uvicorn imported successfully")
    
    print("Starting server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)








