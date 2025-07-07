#!/usr/bin/env python3
"""
Test script to verify the backend API connection
"""

import requests
import json
import time

def test_backend_connection():
    """Test if the backend API is responding."""
    try:
        # Test basic connection
        response = requests.get('http://localhost:5000/')
        print(f"✅ Backend is running! Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start the backend first:")
        print("   py backend_api.py")
        return False

def test_backtest_endpoint():
    """Test the backtest endpoint with a simple request."""
    try:
        # Test data for a simple backtest
        test_data = {
            "strategy": "1",  # SMA Strategy
            "ticker": "AAPL",
            "testPeriod": "1",  # Last 1 month
            "customStart": ""
        }
        
        print("🧪 Testing backtest endpoint...")
        response = requests.post(
            'http://localhost:5000/api/backtest',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backtest endpoint is working!")
            print(f"📊 Received response with keys: {list(result.keys())}")
            return True
        else:
            print(f"❌ Backtest endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend API")
        return False
    except Exception as e:
        print(f"❌ Error testing backtest endpoint: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Testing Backend API Connection")
    print("=" * 40)
    
    # Wait a moment for backend to fully start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Test basic connection
    if not test_backend_connection():
        return
    
    # Test backtest endpoint
    if test_backtest_endpoint():
        print("\n🎉 All tests passed! Your backend is ready.")
        print("\n📱 To start the frontend:")
        print("   cd frontend")
        print("   npm start")
        print("\n🌐 Frontend will be available at: http://localhost:3000")
    else:
        print("\n❌ Some tests failed. Please check the backend logs.")

if __name__ == "__main__":
    main() 