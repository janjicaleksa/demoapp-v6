#!/usr/bin/env python3
"""
Basic tests for the modular AI Processor application
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test if all modules can be imported"""
    try:
        # Test core imports
        from app.core.config import settings
        from app.core.logging_config import setup_logging
        from app.core.utils import normalize_client_name
        
        # Test model imports
        from app.models.schemas import ClientResponse, ProcessingResult
        
        # Test service imports
        from app.services.storage_service import StorageService
        from app.services.document_service import DocumentService
        from app.services.ai_service import AIService
        
        # Test route imports
        from app.routes import clients_router, upload_router, main_router
        
        # Test main app
        from main import app
        
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        from app.core.config import settings
        
        # Test that settings object exists and has expected attributes
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'app_version')
        assert hasattr(settings, 'debug')
        
        print(f"✓ Configuration loaded: {settings.app_name} v{settings.app_version}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_services():
    """Test service initialization"""
    try:
        from app.services.storage_service import StorageService
        from app.services.ai_service import AIService
        
        # Test service initialization
        storage_service = StorageService()
        ai_service = AIService()
        
        print("✓ Services initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Service initialization error: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    try:
        from app.core.utils import normalize_client_name, determine_file_type, get_desired_keys
        
        # Test normalize_client_name
        result = normalize_client_name("Test Client 123!")
        assert result == "test-client-123"
        
        # Test determine_file_type
        kupci_type = determine_file_type("kupci_data.xlsx")
        assert kupci_type == "kupci"
        
        dobavljaci_type = determine_file_type("dobavljaci_report.pdf")
        assert dobavljaci_type == "dobavljaci"
        
        # Test get_desired_keys
        kupci_keys = get_desired_keys("kupci")
        assert "konto" in kupci_keys
        assert "naziv_partnera" in kupci_keys
        
        print("✓ Utility functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Utility function error: {e}")
        return False

def test_folder_structure():
    """Test if required folders exist"""
    required_folders = [
        Path('app'),
        Path('app/core'),
        Path('app/models'),
        Path('app/routes'),
        Path('app/services'),
        Path('templates'),
        Path('static'),
        Path('tests')
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not folder.exists():
            missing_folders.append(str(folder))
    
    if missing_folders:
        print(f"✗ Missing folders: {missing_folders}")
        return False
    else:
        print("✓ All required folders exist")
        return True

def test_required_files():
    """Test if required files exist"""
    required_files = [
        Path('main.py'),
        Path('requirements.txt'),
        Path('env_example.txt'),
        Path('start.py'),
        Path('README.md'),
        Path('app/__init__.py'),
        Path('app/core/__init__.py'),
        Path('app/models/__init__.py'),
        Path('app/routes/__init__.py'),
        Path('app/services/__init__.py')
    ]
    
    missing_files = []
    for file in required_files:
        if not file.exists():
            missing_files.append(str(file))
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing AI Processor Kupci Dobavljaci (Modular Version)")
    print("=" * 70)
    
    tests = [
        ("Folder Structure", test_folder_structure),
        ("Required Files", test_required_files),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Services", test_services),
        ("Utilities", test_utilities)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Tests Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular application is ready to run.")
        print("\n🚀 To start the application:")
        print("  python start.py")
        print("  or")
        print("  python main.py")
        print("  or")
        print("  uvicorn main:app --reload")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)