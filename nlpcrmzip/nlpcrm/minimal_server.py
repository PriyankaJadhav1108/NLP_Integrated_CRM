#!/usr/bin/env python3
"""
Minimal FastAPI server for testing
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="NLP CRM Test Server")

@app.get("/")
async def root():
    return {"message": "NLP CRM Server is running!", "status": "ok"}

@app.get("/crm", response_class=HTMLResponse)
async def crm_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NLP CRM Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .success { color: green; }
            .info { color: blue; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 NLP CRM Dashboard</h1>
            <div class="card">
                <h2 class="success">✅ Server is Running Successfully!</h2>
                <p><strong>Server URL:</strong> http://127.0.0.1:8000</p>
                <p><strong>Status:</strong> <span class="success">Online</span></p>
            </div>
            
            <div class="card">
                <h3>Enhanced NLP Analysis Features:</h3>
                <ul>
                    <li>📝 Text Summarization</li>
                    <li>🧠 Sentiment Analysis</li>
                    <li>🆔 Named Entity Recognition</li>
                    <li>🔑 Keyword Extraction</li>
                    <li>🎯 Intent Classification</li>
                    <li>⚡ Priority Scoring</li>
                    <li>🌍 Language Detection</li>
                    <li>🎤 Audio Transcription (Whisper)</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>API Endpoints:</h3>
                <ul>
                    <li><code>GET /</code> - Server status</li>
                    <li><code>GET /crm</code> - CRM Dashboard</li>
                    <li><code>POST /v1/answer</code> - Answer queries</li>
                    <li><code>POST /v1/enhanced-nlp/analyze</code> - Enhanced NLP analysis</li>
                    <li><code>POST /v1/asr/transcribe</code> - Audio transcription</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>Next Steps:</h3>
                <ol>
                    <li>Stop this test server (Ctrl+C)</li>
                    <li>Run the full API server: <code>python -m uvicorn api_server:app --host 127.0.0.1 --port 8000</code></li>
                    <li>Access the full CRM dashboard with enhanced NLP features</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting minimal test server...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("CRM Dashboard: http://127.0.0.1:8000/crm")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")








