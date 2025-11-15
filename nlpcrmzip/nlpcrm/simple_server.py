#!/usr/bin/env python3
"""
Simple HTTP server for testing
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>NLP CRM Test Server</title>
            </head>
            <body>
                <h1>🚀 NLP CRM Server is Running!</h1>
                <p>Server is accessible at: <strong>http://127.0.0.1:8000</strong></p>
                <p>If you can see this page, the server is working correctly.</p>
                <hr>
                <h2>Next Steps:</h2>
                <ol>
                    <li>Stop this test server (Ctrl+C)</li>
                    <li>Run the main API server</li>
                    <li>Access the CRM dashboard at: <a href="http://127.0.0.1:8000/crm">http://127.0.0.1:8000/crm</a></li>
                </ol>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            super().do_GET()

if __name__ == "__main__":
    port = 8000
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, CustomHandler)
    
    print(f"🚀 Starting test server on http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✅ Server stopped successfully")
        httpd.shutdown()








