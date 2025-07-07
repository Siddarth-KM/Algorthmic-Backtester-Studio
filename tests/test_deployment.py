#!/usr/bin/env python3
"""
Deployment test script to verify the application works correctly.
"""

import sys
import os
import subprocess
import importlib
import time

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    required_modules = [
        'flask',
        'flask_cors', 
        'pandas',
        'yfinance',
        'matplotlib',
        'ta',
        'xgboost',
        'sklearn',
        'numpy',
        'scipy'
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def test_backend_files():
    """Test that all backend files exist."""
    print("\nTesting backend files...")
    
    required_files = [
        'backend_api.py',
        'backtester_main.py',
        'ML_Project.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            return False
    
    return True

def test_frontend_files():
    """Test that all frontend files exist."""
    print("\nTesting frontend files...")
    
    required_files = [
        'frontend/package.json',
        'frontend/src/App.js',
        'frontend/src/index.js'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            return False
    
    return True

def test_backend_startup():
    """Test that the backend can start without errors."""
    print("\nTesting backend startup...")
    
    try:
        # Import the backend module
        import backend_api
        print("✅ Backend imports successfully")
        
        # Test that the app object exists
        if hasattr(backend_api, 'app'):
            print("✅ Flask app object exists")
        else:
            print("❌ Flask app object missing")
            return False
            
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")
        return False
    
    return True

def test_ml_module():
    """Test that the ML module works."""
    print("\nTesting ML module...")
    
    try:
        from ML_Project import run_ml_backtest
        print("✅ ML module imports successfully")
        
        # Test with a simple call (won't actually run backtest)
        if callable(run_ml_backtest):
            print("✅ ML backtest function is callable")
        else:
            print("❌ ML backtest function is not callable")
            return False
            
    except Exception as e:
        print(f"❌ ML module test failed: {e}")
        return False
    
    return True

def test_backtester_module():
    """Test that the backtester module works."""
    print("\nTesting backtester module...")
    
    try:
        from backtester_main import run_standard_backtest
        print("✅ Backtester module imports successfully")
        
        if callable(run_standard_backtest):
            print("✅ Standard backtest function is callable")
        else:
            print("❌ Standard backtest function is not callable")
            return False
            
    except Exception as e:
        print(f"❌ Backtester module test failed: {e}")
        return False
    
    return True

def main():
    """Run all deployment tests."""
    print("🚀 Algorithmic Backtester Studio - Deployment Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_backend_files,
        test_frontend_files,
        test_backend_startup,
        test_ml_module,
        test_backtester_module
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your application is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 