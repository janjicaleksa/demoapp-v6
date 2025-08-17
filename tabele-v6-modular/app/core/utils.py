"""
Utility functions for AI Processor Kupci Dobavljaci
"""

import re
from pathlib import Path
from typing import List, Dict, Any
from ..core.logging_config import get_logger

logger = get_logger(__name__)


def normalize_client_name(name: str) -> str:
    """
    Normalize client name to create a valid folder name
    
    Args:
        name: Client name to normalize
        
    Returns:
        Normalized client name suitable for folder/file names
    """
    if not name or not name.strip():
        return ""
    
    # Remove forbidden characters and convert to lowercase
    normalized = re.sub(r'[<>:"/\\|?*]', '', name.lower())
    # Replace spaces with hyphens
    normalized = re.sub(r'\s+', '-', normalized.strip())
    # Remove multiple hyphens
    normalized = re.sub(r'-+', '-', normalized)
    # Remove leading/trailing hyphens
    normalized = normalized.strip('-')
    
    logger.debug(f"Normalized client name '{name}' to '{normalized}'")
    return normalized


def determine_file_type(filename: str) -> str:
    """
    Determine if file is 'kupci' or 'dobavljaci' based on filename
    
    Args:
        filename: Name of the file to analyze
        
    Returns:
        'kupci' or 'dobavljaci'
        
    Raises:
        ValueError: If file type cannot be determined
    """
    filename_lower = filename.lower()
    
    if 'kupci' in filename_lower:
        return 'kupci'
    elif 'dobavljaci' in filename_lower:
        return 'dobavljaci'
    else:
        logger.error(f"Cannot determine file type for filename: {filename}")
        raise ValueError(f"Cannot determine file type for filename: {filename}")


def get_desired_keys(file_type: str) -> List[str]:
    """
    Get desired mapping keys based on file type
    
    Args:
        file_type: Type of file ('kupci' or 'dobavljaci')
        
    Returns:
        List of desired column keys for the file type
    """
    if file_type == 'kupci':
        return ['konto', 'naziv_partnera', 'promet duguje', 'saldo']
    elif file_type == 'dobavljaci':
        return ['konto', 'naziv_partnera', 'promet potrazuje', 'saldo']
    else:
        logger.warning(f"Unknown file type: {file_type}")
        return []


def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """
    Validate if file extension is allowed
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    is_valid = file_extension in allowed_extensions
    
    if not is_valid:
        logger.warning(f"File extension '{file_extension}' not allowed for file '{filename}'")
    
    return is_valid


def clean_table_headers(headers: List[str]) -> List[str]:
    """
    Clean and normalize table headers
    
    Args:
        headers: List of original headers
        
    Returns:
        List of cleaned headers
    """
    cleaned = []
    for header in headers:
        if header:
            # Remove special characters and normalize
            cleaned_header = re.sub(r'\W+', '', str(header)).strip().lower()
            cleaned.append(cleaned_header)
        else:
            cleaned.append("")
    
    return cleaned


def get_blob_name(client_slug: str, file_type: str, period_date: str = None, 
                  is_extracted: bool = False, is_processed: bool = False) -> str:
    """
    Generate blob name for Azure Blob Storage
    
    Args:
        client_slug: Normalized client name
        file_type: Type/name of the file
        period_date: Period date (optional)
        is_extracted: Whether this is extracted data
        is_processed: Whether this is processed data
        
    Returns:
        Generated blob name
    """
    if is_extracted:
        folder_type = "extracted"
    elif is_processed:          
        folder_type = "processed"
    else:
        folder_type = "raw"
    
    if period_date:
        blob_name = f"{client_slug}/{folder_type}/{period_date}/{file_type}"
    else:
        blob_name = f"{client_slug}/{folder_type}/{file_type}"
    
    logger.debug(f"Generated blob name: {blob_name}")
    return blob_name


def safe_get_dict_value(data: dict, key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with logging
    
    Args:
        data: Dictionary to get value from
        key: Key to look for
        default: Default value if key not found
        
    Returns:
        Value from dictionary or default
    """
    try:
        return data.get(key, default)
    except Exception as e:
        logger.error(f"Error getting key '{key}' from dict: {e}")
        return default


def format_processing_summary(results: List[Dict[str, Any]]) -> str:
    """
    Format processing results into a readable summary
    
    Args:
        results: List of processing results
        
    Returns:
        Formatted summary string
    """
    if not results:
        return "No files processed"
    
    summary_lines = ["Processing Summary:"]
    total_records = 0
    
    for result in results:
        file_type = result.get('file_type', 'Unknown')
        records = result.get('records_processed', 0)
        period = result.get('period', 'Unknown')
        
        summary_lines.append(f"  - {period}: {file_type} ({records} records)")
        total_records += records
    
    summary_lines.append(f"Total records processed: {total_records}")
    
    return "\n".join(summary_lines)