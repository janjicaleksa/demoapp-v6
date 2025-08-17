"""
Document processing service for Easy Tables
Handles Azure AI Document Intelligence integration
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from ..core.config import settings


logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document processing using Azure AI Document Intelligence"""
    
    def __init__(self):
        self.client = None
        if settings.azure_document_intelligence_endpoint and settings.azure_document_intelligence_key:
            try:
                self.client = DocumentAnalysisClient(
                    endpoint=settings.azure_document_intelligence_endpoint,
                    credential=AzureKeyCredential(settings.azure_document_intelligence_key)
                )
                logger.info("Azure Document Intelligence client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Document Intelligence client: {e}")
        else:
            logger.warning("Azure Document Intelligence credentials not provided")
    
    async def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from document using Azure Document Intelligence"""
        if not self.client:
            raise ValueError("Azure Document Intelligence client not initialized")
        
        try:
            with open(file_path, 'rb') as file:
                # Use general document model for text extraction
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-document",
                    document=file
                )
                result = poller.result()
            
            # Extract text content
            extracted_text = ""
            if result.content:
                extracted_text = result.content
            
            # Also extract tables if present
            if result.tables:
                extracted_text += "\n\n=== TABLES ===\n"
                for i, table in enumerate(result.tables):
                    extracted_text += f"\nTable {i + 1}:\n"
                    
                    # Extract table headers
                    headers = []
                    for cell in table.cells:
                        if cell.row_index == 0:
                            headers.append(cell.content)
                    
                    if headers:
                        extracted_text += " | ".join(headers) + "\n"
                        extracted_text += " | ".join(["-" * len(h) for h in headers]) + "\n"
                    
                    # Extract table rows
                    current_row = -1
                    row_data = []
                    
                    for cell in table.cells:
                        if cell.row_index != current_row:
                            if row_data and current_row > 0:  # Skip header row
                                extracted_text += " | ".join(row_data) + "\n"
                            current_row = cell.row_index
                            row_data = []
                        
                        if current_row > 0:  # Skip header row
                            row_data.append(cell.content or "")
                    
                    if row_data:
                        extracted_text += " | ".join(row_data) + "\n"
            
            logger.info(f"Successfully extracted text from {file_path.name}")
            return extracted_text
            
        except AzureError as e:
            logger.error(f"Azure Document Intelligence error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from {file_path.name}: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Document Intelligence service is available"""
        return self.client is not None