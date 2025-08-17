"""
Pydantic models and data schemas for Easy Tables
Simplified schemas focusing on parameter-based organization
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ClientResponse(BaseModel):
    """Response model for client creation"""
    message: str
    client_slug: str
    structure: Optional[Dict[str, Any]] = None


class FileUploadRequest(BaseModel):
    """Request model for file upload with parameters"""
    file_type: str = Field(..., description="Either 'kupci' or 'dobavljaci'")
    period: str = Field(..., description="Either 'prethodna_fiskalna_godina' or 'presek_bilansa_tekuca_godina'")
    is_iv: bool = Field(default=False, description="Only applicable for kupci file type")
    

class ProcessingResult(BaseModel):
    """Result of document extraction and processing"""
    file_type: str
    period: str
    original_filename: str
    is_iv: bool = False
    raw_path: str
    extracted_path: Optional[str] = None
    processed_path: Optional[str] = None
    records_processed: int = 0


class AIProcessingResult(BaseModel):
    """Result from AI processing with custom prompt"""
    file_type: str
    period_date: str
    datum: str
    ispravka_vrednosti: bool
    tabela: List[Dict[str, Any]]
    

class FileUploadResponse(BaseModel):
    """Response model for file upload and processing"""
    message: str
    client_slug: str
    results: List[ProcessingResult]
    ai_results: Optional[List[AIProcessingResult]] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    error_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TableRecord(BaseModel):
    """Individual table record from AI processing"""
    konto: Optional[str] = None
    naziv_partnera: Optional[str] = None
    promet_duguje: Optional[float] = None
    promet_potrazuje: Optional[float] = None
    saldo: Optional[float] = None