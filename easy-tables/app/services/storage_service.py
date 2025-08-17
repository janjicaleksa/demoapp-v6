"""
Storage service for Easy Tables
Handles file storage organized by parameters instead of dates
"""

import json
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import UploadFile

from ..core.config import settings
from ..core.utils import get_storage_path


class StorageService:
    """Service for handling file storage with parameter-based organization"""
    
    def __init__(self):
        self.base_path = settings.local_storage_path
        self.base_path.mkdir(exist_ok=True)
    
    async def save_uploaded_file(
        self, 
        file: UploadFile, 
        client_slug: str, 
        file_type: str, 
        period: str,
        is_iv: bool = False
    ) -> Path:
        """Save uploaded file to parameter-based directory structure"""
        storage_path = get_storage_path(client_slug, file_type, period, is_iv)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_path / f"raw_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return file_path
    
    async def save_extracted_content(
        self,
        content: str,
        client_slug: str,
        file_type: str,
        period: str,
        original_filename: str,
        is_iv: bool = False
    ) -> Path:
        """Save extracted content from Azure Document Intelligence"""
        storage_path = get_storage_path(client_slug, file_type, period, is_iv)
        
        # Remove extension and add extracted suffix
        base_name = Path(original_filename).stem
        extracted_path = storage_path / f"extracted_{base_name}.txt"
        
        async with aiofiles.open(extracted_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        return extracted_path
    
    async def save_ai_result(
        self,
        ai_result: Dict[str, Any],
        client_slug: str,
        file_type: str,
        period: str,
        original_filename: str,
        is_iv: bool = False
    ) -> Path:
        """Save AI processing result"""
        storage_path = get_storage_path(client_slug, file_type, period, is_iv)
        
        base_name = Path(original_filename).stem
        result_path = storage_path / f"ai_result_{base_name}.json"
        
        async with aiofiles.open(result_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(ai_result, ensure_ascii=False, indent=2))
        
        return result_path
    
    async def read_file_content(self, file_path: Path) -> str:
        """Read file content"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    def get_client_structure(self, client_slug: str) -> Dict[str, Any]:
        """Get the parameter-based directory structure for a client"""
        base_path = self.base_path / client_slug
        
        structure = {}
        for file_type in settings.file_types:
            structure[file_type] = {}
            for period in settings.periods:
                period_path = base_path / file_type / period
                structure[file_type][period] = {
                    "path": str(period_path),
                    "exists": period_path.exists()
                }
                
                if file_type == "kupci":
                    iv_path = period_path / "iv"
                    structure[file_type][f"{period}_iv"] = {
                        "path": str(iv_path),
                        "exists": iv_path.exists()
                    }
        
        return structure