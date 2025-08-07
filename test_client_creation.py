#!/usr/bin/env python3
"""
Test script to verify client creation and folder structure
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import normalize_client_name, create_client_structure, create_period_structure, get_blob_name

def test_client_creation():
    """Test client creation and folder structure"""
    print("Testing client creation and folder structure...")
    
    # Test client name normalization
    test_names = [
        "Test Client",
        "Test-Client",
        "Test Client 123",
        "Test.Client",
        "Test/Client",
        "Test\\Client",
        "Test:Client",
        "Test*Client",
        "Test?Client",
        "Test<Client>",
        "Test\"Client\"",
        "Test|Client",
    ]
    
    print("\nTesting client name normalization:")
    for name in test_names:
        normalized = normalize_client_name(name)
        print(f"  '{name}' -> '{normalized}'")
    
    # Test client structure creation
    print("\nTesting client structure creation:")
    test_client_slug = "test-client"
    
    # Test local storage structure
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ""  # Force local storage
    structure = create_client_structure(test_client_slug)
    print(f"  Local storage structure: {structure}")
    
    # Test Azure storage structure (virtual)
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "dummy"  # Force Azure storage
    structure = create_client_structure(test_client_slug)
    print(f"  Azure storage structure: {structure}")
    
    # Test period structure creation
    print("\nTesting period structure creation:")
    period_date = "2024-01-01"
    
    # Test local storage period structure
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ""  # Force local storage
    period_structure = create_period_structure(test_client_slug, period_date)
    print(f"  Local storage period structure: {period_structure}")
    
    # Test Azure storage period structure (virtual)
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "dummy"  # Force Azure storage
    period_structure = create_period_structure(test_client_slug, period_date)
    print(f"  Azure storage period structure: {period_structure}")
    
    # Test blob name generation
    print("\nTesting blob name generation:")
    file_type = "kupci-prethodna-fiskalna-godina.xlsx"
    period_date = "2024-01-01"
    
    # Test raw file blob name
    blob_name = get_blob_name(test_client_slug, file_type, period_date, is_processed=False)
    print(f"  Raw file blob name: {blob_name}")
    
    # Test processed file blob name
    blob_name = get_blob_name(test_client_slug, file_type, period_date, is_processed=True)
    print(f"  Processed file blob name: {blob_name}")
    
    # Test path parsing
    print("\nTesting path parsing:")
    azure_path = f"azure://clients/{test_client_slug}/raw/{period_date}"
    path_parts = azure_path.split('/')
    print(f"  Azure path: {azure_path}")
    print(f"  Path parts: {path_parts}")
    print(f"  Container name (index 2): {path_parts[2]}")
    print(f"  Client slug (index 3): {path_parts[3]}")
    print(f"  Folder type (index 4): {path_parts[4]}")
    print(f"  Period date (index 5): {path_parts[5]}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_client_creation() 