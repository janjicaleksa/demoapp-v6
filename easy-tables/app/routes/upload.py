"""
File upload and processing routes for Easy Tables
Simplified version focused on individual file processing with parameters
"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from ..core.config import settings
from ..core.utils import validate_file_extension, validate_file_type, validate_period
from ..models.schemas import FileUploadResponse, ProcessingResult, AIProcessingResult
from ..services.storage_service import StorageService
from ..services.document_service import DocumentService
from ..services.ai_service import AIService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])

# Initialize services
storage_service = StorageService()
document_service = DocumentService()
ai_service = AIService()


@router.post("/process", response_model=FileUploadResponse)
async def process_file(
    client_slug: str = Form(...),
    file_type: str = Form(...),  # "kupci" or "dobavljaci"
    period: str = Form(...),  # "prethodna_fiskalna_godina" or "presek_bilansa_tekuca_godina"
    is_iv: bool = Form(False),  # Only for kupci
    file: UploadFile = File(...)
):
    """Process a single file with Azure AI Document Intelligence and custom AI prompt"""
    try:
        logger.info(f"Processing file upload for client: {client_slug}")
        logger.info(f"File type: {file_type}, Period: {period}, Is IV: {is_iv}")
        
        # Validate inputs
        if not validate_file_type(file_type):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
        
        if not validate_period(period):
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
        
        if not validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file extension: {file.filename}")
        
        if is_iv and file_type != "kupci":
            raise HTTPException(status_code=400, detail="IV flag is only applicable for kupci file type")
        
        # Check file size
        if file.size > settings.max_file_size:
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed size")
        
        # Save uploaded file
        logger.info(f"Saving file: {file.filename}")
        file_path = await storage_service.save_uploaded_file(
            file, client_slug, file_type, period, is_iv
        )
        
        # Initialize processing result
        processing_result = ProcessingResult(
            file_type=file_type,
            period=period,
            original_filename=file.filename,
            is_iv=is_iv,
            raw_path=str(file_path)
        )
        
        ai_result = None
        
        # Extract text using Azure Document Intelligence
        if document_service.is_available():
            try:
                logger.info(f"Extracting text from: {file.filename}")
                extracted_text = await document_service.extract_text_from_file(file_path)
                
                # Save extracted content
                extracted_path = await storage_service.save_extracted_content(
                    extracted_text, client_slug, file_type, period, file.filename, is_iv
                )
                processing_result.extracted_path = str(extracted_path)
                
                # Process with AI using custom prompt
                if ai_service.is_available():
                    try:
                        logger.info(f"Processing with AI: {file.filename}")
                        ai_response = await ai_service.process_extracted_text(
                            extracted_text, file_type, period, is_iv
                        )
                        
                        # Save AI result
                        ai_result_path = await storage_service.save_ai_result(
                            ai_response, client_slug, file_type, period, file.filename, is_iv
                        )
                        processing_result.processed_path = str(ai_result_path)
                        
                        # Create AI processing result
                        ai_result = AIProcessingResult(
                            file_type=ai_response.get("file_type", file_type),
                            period_date=ai_response.get("period_date", period),
                            datum=ai_response.get("datum", ""),
                            ispravka_vrednosti=ai_response.get("ispravka_vrednosti", False),
                            tabela=ai_response.get("tabela", [])
                        )
                        
                        processing_result.records_processed = len(ai_response.get("tabela", []))
                        
                        logger.info(f"Successfully processed {processing_result.records_processed} records")
                        
                    except Exception as e:
                        logger.error(f"AI processing failed: {e}")
                        # Continue without AI processing
                        
            except Exception as e:
                logger.error(f"Document extraction failed: {e}")
                # Continue without extraction
        
        return FileUploadResponse(
            message=f"File '{file.filename}' processed successfully",
            client_slug=client_slug,
            results=[processing_result],
            ai_results=[ai_result] if ai_result else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=FileUploadResponse)
async def process_batch_files(
    client_slug: str = Form(...),
    # Prethodna fiskalna godina files
    kupci_prethodna: Optional[UploadFile] = File(None),
    dobavljaci_prethodna: Optional[UploadFile] = File(None),
    kupci_prethodna_iv: Optional[UploadFile] = File(None),
    # Presek bilansa files
    kupci_presek: Optional[UploadFile] = File(None),
    dobavljaci_presek: Optional[UploadFile] = File(None),
    kupci_presek_iv: Optional[UploadFile] = File(None)
):
    """Process multiple files in batch"""
    try:
        logger.info(f"Processing batch upload for client: {client_slug}")
        
        results = []
        ai_results = []
        
        # Define file mappings
        file_mappings = [
            (kupci_prethodna, "kupci", "prethodna_fiskalna_godina", False),
            (dobavljaci_prethodna, "dobavljaci", "prethodna_fiskalna_godina", False),
            (kupci_prethodna_iv, "kupci", "prethodna_fiskalna_godina", True),
            (kupci_presek, "kupci", "presek_bilansa_tekuca_godina", False),
            (dobavljaci_presek, "dobavljaci", "presek_bilansa_tekuca_godina", False),
            (kupci_presek_iv, "kupci", "presek_bilansa_tekuca_godina", True)
        ]
        
        for file, file_type, period, is_iv in file_mappings:
            if file and file.filename:
                try:
                    # Process each file individually
                    response = await process_file(
                        client_slug=client_slug,
                        file_type=file_type,
                        period=period,
                        is_iv=is_iv,
                        file=file
                    )
                    
                    results.extend(response.results)
                    if response.ai_results:
                        ai_results.extend(response.ai_results)
                        
                except Exception as e:
                    logger.error(f"Failed to process {file.filename}: {e}")
                    # Continue with other files
        
        if not results:
            raise HTTPException(status_code=400, detail="No files were processed successfully")
        
        return FileUploadResponse(
            message=f"Batch processing completed. {len(results)} files processed.",
            client_slug=client_slug,
            results=results,
            ai_results=ai_results if ai_results else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))