#!/usr/bin/env python3
"""
Simple test script for AI Processor Kupci Dobavljaci
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import aiofiles
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_main_app():
    """Test if the main application can be imported"""
    try:
        from main import app
        print("✓ Main application imported successfully")
        return True
    except Exception as e:
        print(f"✗ Main app import error: {e}")
        return False

def test_folder_structure():
    """Test if required folders exist"""
    required_folders = ['templates', 'static']
    missing_folders = []
    
    for folder in required_folders:
        if not Path(folder).exists():
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"✗ Missing folders: {missing_folders}")
        return False
    else:
        print("✓ All required folders exist")
        return True

def test_template_file():
    """Test if the main template exists"""
    template_file = Path('templates/index.html')
    if template_file.exists():
        print("✓ Main template file exists")
        return True
    else:
        print("✗ Main template file missing")
        return False

def test_requirements():
    """Test if requirements file exists"""
    if Path('requirements.txt').exists():
        print("✓ Requirements file exists")
        return True
    else:
        print("✗ Requirements file missing")
        return False

def main():
    """Run all tests"""
    print("Testing AI Processor Kupci Dobavljaci...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_main_app,
        test_folder_structure,
        test_template_file,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application:")
        print("  python start.py")
        print("  or")
        print("  uvicorn main:app --reload")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 