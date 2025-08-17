"""
Pydantic models and data schemas for AI Processor Kupci Dobavljaci
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ClientResponse(BaseModel):
    """Response model for client creation"""
    message: str
    client_slug: str
    structure: Optional[Dict[str, str]] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    error_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class StandardizedRecord(BaseModel):
    """Standardized financial record"""
    konto: Optional[str] = None
    naziv_partnera: Optional[str] = None
    promet_duguje: Optional[str] = None
    promet_potrazuje: Optional[str] = None
    saldo: Optional[str] = None


class ColumnMapping(BaseModel):
    """Column mapping for AI processing"""
    table_index: int
    mapping: Dict[str, Optional[str]]
    confidence: Optional[float] = None


class ProcessingResult(BaseModel):
    """Result of file processing"""
    period: str
    file_type: str
    original_filename: str
    raw_path: str
    extracted_path: Optional[str] = None
    column_mapping_path: Optional[str] = None
    processed_path: Optional[str] = None
    records_processed: int = 0
    mapping_confidence: Optional[str] = None
    error: Optional[str] = None


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    message: str
    period1_date: Optional[str] = None
    period2_date: Optional[str] = None
    period_date: Optional[str] = None  # For single period uploads
    results: List[ProcessingResult]


class AIColumnMappingRequest(BaseModel):
    """Request model for AI column mapping"""
    filename: str
    file_type: str
    desired_keys: List[str]
    table_headers: List[Dict[str, Any]]


class AIColumnMappingResponse(BaseModel):
    """Response model for AI column mapping"""
    filename: str
    file_type: str
    desired_keys: List[str]
    column_mapping: Dict[str, Any]
    tables_analyzed: int
    mapping_confidence: str


class ClientStructure(BaseModel):
    """Client folder structure"""
    client_slug: str
    raw_path: str
    extracted_path: str
    processed_path: str


class PeriodStructure(BaseModel):
    """Period-specific folder structure"""
    raw_period_path: str
    extracted_period_path: str
    processed_period_path: str


class FileTypeMapping(BaseModel):
    """File type mapping configuration"""
    kupci_keys: List[str] = ['konto', 'naziv_partnera', 'promet duguje', 'saldo']
    dobavljaci_keys: List[str] = ['konto', 'naziv_partnera', 'promet potrazuje', 'saldo']


class TableData(BaseModel):
    """Table data structure"""
    headers: List[str]
    rows: List[List[str]]
    table_index: int = 0


class ExtractedTableData(BaseModel):
    """Extracted table data with metadata"""
    tables: List[List[Dict[str, Any]]]
    total_tables: int
    extraction_method: str  # pdf, excel, csv
    file_path: str