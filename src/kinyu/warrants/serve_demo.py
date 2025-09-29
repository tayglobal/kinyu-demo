#!/usr/bin/env python3
"""
Simple HTTP server to serve the warrants demo locally.
WebAssembly modules require proper MIME types and CORS headers.
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Override to handle cache-busting query parameters."""
        # Parse the URL and remove the query string
        self.path = urlparse(self.path).path
        super().do_GET()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def guess_type(self, path):
        """Override to set correct MIME types for WebAssembly files."""
        result = super().guess_type(path)
        
        # Handle different return types from super().guess_type()
        if isinstance(result, tuple):
            mimetype, encoding = result
        else:
            mimetype = result
            encoding = None
        
        if path.endswith('.wasm'):
            return 'application/wasm'
        elif path.endswith('.js') and 'pkg' in path:
            return 'application/javascript'
        elif path.endswith('.d.ts'):
            return 'text/typescript'
        
        return mimetype

def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # Change to the directory containing the HTML file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Starting server on http://localhost:{port}")
    print("Open http://localhost:8000/warrants_demo.html in your browser")
    print("Press Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
