#!/usr/bin/env python3
"""
Test different ports to see which one works
"""
import socket
import sys

def test_port(host, port):
    """Test if a port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # True if port is available
    except:
        return False

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if test_port('127.0.0.1', port):
            return port
    return None

if __name__ == "__main__":
    print("Testing port availability...")
    
    # Test common ports
    test_ports = [8000, 8001, 8080, 3000, 5000, 9000]
    
    for port in test_ports:
        available = test_port('127.0.0.1', port)
        status = "Available" if available else "In use"
        print(f"Port {port}: {status}")
    
    # Find first available port
    available_port = find_available_port()
    if available_port:
        print(f"\nFirst available port: {available_port}")
        print(f"Try running the server on port {available_port}")
    else:
        print("\nNo available ports found in range 8000-8009")
