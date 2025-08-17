"""
Document processing service for extracting tables from various file formats
"""

import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.utils import clean_table_headers
from ..models.schemas import ExtractedTableData
from .storage_service import StorageService

logger = get_logger(__name__)


class DocumentService:
    """Service for processing documents and extracting tables"""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.document_intelligence_client = None
        
        self._initialize_document_intelligence()
    
    def _initialize_document_intelligence(self):
        """Initialize Azure Document Intelligence client"""
        if (settings.azure_document_intelligence_endpoint and 
            settings.azure_document_intelligence_key):
            try:
                self.document_intelligence_client = DocumentAnalysisClient(
                    endpoint=settings.azure_document_intelligence_endpoint,
                    credential=AzureKeyCredential(settings.azure_document_intelligence_key),
                    logging_enable=False
                )
                logger.info("Azure Document Intelligence client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Document Intelligence client: {e}")
                self.document_intelligence_client = None
        else:
            logger.warning("Azure Document Intelligence credentials not configured, PDF processing will be skipped")
    
    async def extract_tables_from_pdf(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """
        Extract tables from PDF using Azure Document Intelligence
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of tables, each table is a list of dictionaries
        """
        if not self.document_intelligence_client:
            logger.warning("Azure Document Intelligence not configured, skipping PDF processing")
            return []
        
        try:
            # Handle both local files and Azure Blob Storage files
            if file_path.startswith("azure://"):
                # Download from Azure Blob Storage to temporary file
                blob_name = file_path.replace(f"azure://{settings.azure_blob_storage_container_name}/", "")
                content = await self.storage_service.download_from_blob_storage(blob_name)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    with open(temp_file_path, "rb") as document:
                        poller = self.document_intelligence_client.begin_analyze_document(
                            "prebuilt-document", document.read()
                        )
                        result = poller.result()
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            else:
                # Local file
                with open(file_path, "rb") as document:
                    poller = self.document_intelligence_client.begin_analyze_document(
                        "prebuilt-document", document.read()
                    )
                    result = poller.result()
            
            tables = []
            for table in result.tables:
                temp_table = defaultdict(dict)
                for cell in table.cells:
                    temp_table[cell.row_index][cell.column_index] = cell.content

                # Get total size
                num_rows = table.row_count
                num_columns = table.column_count

                # Reconstruct table
                rows = []
                for r in range(num_rows):
                    row = []
                    for c in range(num_columns):
                        row.append(temp_table[r].get(c, ""))
                    rows.append(row)
                
                # Format output: use first row as keys
                if rows and len(rows) > 1:
                    formatted_data = []
                    headers = rows[0]
                    normalized_headers = clean_table_headers(headers)
                    
                    for row in rows[1:]:
                        record = {}
                        for i, key in enumerate(normalized_headers):
                            if i < len(row):
                                record[key] = row[i]
                            else:
                                record[key] = ""
                        formatted_data.append(record)
                    
                    if formatted_data:
                        tables.append(formatted_data)
                    else:
                        logger.warning(f"No data extracted from table {len(tables)}")

            logger.info(f"Extracted {len(tables)} tables from PDF: {file_path}")
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []
    
    def extract_tables_from_excel(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """
        Extract tables from Excel file
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of tables, each table is a list of dictionaries
        """
        try:
            # Handle both local files and Azure Blob Storage files
            if file_path.startswith("azure://"):
                # Download from Azure Blob Storage to temporary file
                import asyncio
                blob_name = file_path.replace(f"azure://{settings.azure_blob_storage_container_name}/", "")
                content = asyncio.run(self.storage_service.download_from_blob_storage(blob_name))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Read all sheets
                    excel_file = pd.ExcelFile(temp_file_path)
                    tables = []
                    
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(temp_file_path, sheet_name=sheet_name)
                        if not df.empty:
                            # Convert DataFrame to list of dictionaries
                            table_data = []
                            for _, row in df.iterrows():
                                row_dict = {}
                                for i, value in enumerate(row):
                                    if pd.notna(value):
                                        row_dict[f"column_{i}"] = str(value)
                                if row_dict:
                                    table_data.append(row_dict)
                            if table_data:
                                tables.append(table_data)
                    
                    return tables
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            else:
                # Local file
                # Read all sheets
                excel_file = pd.ExcelFile(file_path)
                tables = []
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    if not df.empty:
                        # Convert DataFrame to list of dictionaries
                        table_data = []
                        for _, row in df.iterrows():
                            row_dict = {}
                            for i, value in enumerate(row):
                                if pd.notna(value):
                                    row_dict[f"column_{i}"] = str(value)
                            if row_dict:
                                table_data.append(row_dict)
                        if table_data:
                            tables.append(table_data)
                
                logger.info(f"Extracted {len(tables)} tables from Excel: {file_path}")
                return tables
                
        except Exception as e:
            logger.error(f"Error extracting tables from Excel: {e}")
            return []
    
    def extract_tables_from_csv(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """
        Extract tables from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of tables (single table for CSV)
        """
        try:
            # Handle both local files and Azure Blob Storage files
            if file_path.startswith("azure://"):
                # Download from Azure Blob Storage to temporary file
                import asyncio
                blob_name = file_path.replace(f"azure://{settings.azure_blob_storage_container_name}/", "")
                content = asyncio.run(self.storage_service.download_from_blob_storage(blob_name))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    df = pd.read_csv(temp_file_path)
                    if df.empty:
                        return []
                    
                    table_data = []
                    for _, row in df.iterrows():
                        row_dict = {}
                        for i, value in enumerate(row):
                            if pd.notna(value):
                                row_dict[f"column_{i}"] = str(value)
                        if row_dict:
                            table_data.append(row_dict)
                    
                    return [table_data] if table_data else []
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            else:
                # Local file
                df = pd.read_csv(file_path)
                if df.empty:
                    return []
                
                table_data = []
                for _, row in df.iterrows():
                    row_dict = {}
                    for i, value in enumerate(row):
                        if pd.notna(value):
                            row_dict[f"column_{i}"] = str(value)
                    if row_dict:
                        table_data.append(row_dict)
                
                logger.info(f"Extracted table from CSV: {file_path}")
                return [table_data] if table_data else []
                
        except Exception as e:
            logger.error(f"Error extracting tables from CSV: {e}")
            return []
    
    async def extract_tables_from_file(self, file_path: str, file_extension: str) -> ExtractedTableData:
        """
        Extract tables from file based on extension
        
        Args:
            file_path: Path to the file
            file_extension: File extension (e.g., '.pdf', '.xlsx', '.csv')
            
        Returns:
            ExtractedTableData with tables and metadata
        """
        tables = []
        extraction_method = ""
        
        if file_extension == '.pdf':
            tables = await self.extract_tables_from_pdf(file_path)
            extraction_method = "pdf"
        elif file_extension in ['.xlsx', '.xls']:
            tables = self.extract_tables_from_excel(file_path)
            extraction_method = "excel"
        elif file_extension == '.csv':
            tables = self.extract_tables_from_csv(file_path)
            extraction_method = "csv"
        else:
            logger.warning(f"Unsupported file extension for table extraction: {file_extension}")
        
        return ExtractedTableData(
            tables=tables,
            total_tables=len(tables),
            extraction_method=extraction_method,
            file_path=file_path
        )