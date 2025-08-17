"""
Utility functions for Easy Tables
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from ..core.config import settings


def validate_file_extension(filename: str) -> bool:
    """Validate if file extension is allowed"""
    file_path = Path(filename)
    return file_path.suffix.lower() in settings.allowed_extensions


def validate_file_type(file_type: str) -> bool:
    """Validate if file type is allowed"""
    return file_type in settings.file_types


def validate_period(period: str) -> bool:
    """Validate if period is allowed"""
    return period in settings.periods


def sanitize_client_name(client_name: str) -> str:
    """Create a safe slug from client name"""
    # Remove special characters and replace spaces with underscores
    slug = re.sub(r'[^\w\s-]', '', client_name.strip())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug.lower()


def get_storage_path(client_slug: str, file_type: str, period: str, is_iv: bool = False) -> Path:
    """Get organized storage path based on parameters instead of dates"""
    base_path = settings.local_storage_path / client_slug / file_type / period
    
    if is_iv and file_type == "kupci":
        base_path = base_path / "iv"
    
    return base_path


def create_directory_structure(client_slug: str):
    """Create parameter-based directory structure for a client"""
    base_path = settings.local_storage_path / client_slug
    
    structure = {}
    for file_type in settings.file_types:
        structure[file_type] = {}
        for period in settings.periods:
            period_path = base_path / file_type / period
            period_path.mkdir(parents=True, exist_ok=True)
            structure[file_type][period] = str(period_path)
            
            # Create IV subdirectory for kupci
            if file_type == "kupci":
                iv_path = period_path / "iv"
                iv_path.mkdir(parents=True, exist_ok=True)
                structure[file_type][f"{period}_iv"] = str(iv_path)
    
    return structure


def get_desired_headers(file_type: str) -> List[str]:
    """Get desired headers for AI processing based on file type"""
    if file_type == "kupci":
        return ['konto', 'naziv partnera', 'promet duguje', 'saldo']
    elif file_type == "dobavljaci":
        return ['konto', 'naziv partnera', 'promet potrazuje', 'saldo']
    else:
        raise ValueError(f"Unknown file type: {file_type}")


def get_period_display_name(period: str) -> str:
    """Convert period code to display name for AI processing"""
    if period == "prethodna_fiskalna_godina":
        return "kraj prethodne fiskalne godine"
    elif period == "presek_bilansa_tekuca_godina":
        return "presek bilansa tekuće godine"
    else:
        return period