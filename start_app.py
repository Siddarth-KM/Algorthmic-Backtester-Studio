#!/usr/bin/env python3
"""
Start script for the Backtesting Application
This script starts the backend API server and provides instructions for the frontend.
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'yfinance', 
        'matplotlib', 'ta', 'xgboost', 'scikit-learn', 
        'numpy', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    else:
        print("✅ All Python dependencies are installed!")
    return True

def start_backend():
    """Start the Flask backend server."""
    print("🚀 Starting backend API server...")
    try:
        # Start the backend server
        subprocess.run([sys.executable, 'backend_api.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend server: {e}")
        return False
    return True

def check_frontend_dependencies():
    """Check if frontend dependencies are installed."""
    frontend_path = Path('frontend')
    if not frontend_path.exists():
        print("❌ Frontend directory not found!")
        return False
    
    node_modules = frontend_path / 'node_modules'
    if not node_modules.exists():
        print("📦 Frontend dependencies not installed. Installing...")
        try:
            subprocess.check_call(['npm', 'install'], cwd=frontend_path)
            print("✅ Frontend dependencies installed!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies.")
            return False
    else:
        print("✅ Frontend dependencies are installed!")
    return True

def main():
    """Main function to start the application."""
    print("🎯 AI-Powered Backtesting Platform")
    print("=" * 50)
    
    # Check and install Python dependencies
    if not check_dependencies():
        return
    
    # Check frontend dependencies
    if not check_frontend_dependencies():
        return
    
    print("\n📋 Starting the application...")
    print("=" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(2)
    
    print("\n🌐 Backend API server is running on http://localhost:5000")
    print("📱 To start the frontend, open a new terminal and run:")
    print("   cd frontend")
    print("   npm start")
    print("\n🔗 The frontend will be available at http://localhost:3000")
    print("\n⏹️  Press Ctrl+C to stop the backend server")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

if __name__ == "__main__":
    main() 