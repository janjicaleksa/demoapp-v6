"""
File upload and processing routes
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.utils import validate_file_extension, get_blob_name
from ..models.schemas import FileUploadResponse, ProcessingResult
from ..services.storage_service import StorageService
from ..services.document_service import DocumentService
from ..services.ai_service import AIService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])

# Initialize services
storage_service = StorageService()
document_service = DocumentService(storage_service)
ai_service = AIService()


@router.post("/fixed", response_model=FileUploadResponse)
async def upload_fixed_files(
    client_slug: str = Form(...),
    period1_date: Optional[str] = Form(None),
    period2_date: Optional[str] = Form(None),
    # Period 1 fields
    kupci_prethodna_fiskalna_godina: Optional[UploadFile] = File(None),
    dobavljaci_prethodna_fiskalna_godina: Optional[UploadFile] = File(None),
    kupci_prethodna_fiskalna_godina_iv: Optional[UploadFile] = File(None),
    # Period 2 fields
    kupci_bilans_preseka: Optional[UploadFile] = File(None),
    dobavljaci_bilans_preseka: Optional[UploadFile] = File(None),
    kupci_bilans_preseka_iv: Optional[UploadFile] = File(None)
):
    """Upload fixed files for processing - handles both periods simultaneously"""
    try:
        logger.info(f"Processing upload for client_slug: {client_slug}")
        logger.info(f"Period 1 date: {period1_date}, Period 2 date: {period2_date}")
        all_results = []
        
        # Read and store file content for all files to avoid consumption issues
        file_contents = {}
        
        # Store Period 1 file contents
        if kupci_prethodna_fiskalna_godina:
            file_contents['kupci_prethodna_fiskalna_godina'] = await kupci_prethodna_fiskalna_godina.read()
        if dobavljaci_prethodna_fiskalna_godina:
            file_contents['dobavljaci_prethodna_fiskalna_godina'] = await dobavljaci_prethodna_fiskalna_godina.read()
        if kupci_prethodna_fiskalna_godina_iv:
            file_contents['kupci_prethodna_fiskalna_godina_iv'] = await kupci_prethodna_fiskalna_godina_iv.read()
        
        # Store Period 2 file contents
        if kupci_bilans_preseka:
            file_contents['kupci_bilans_preseka'] = await kupci_bilans_preseka.read()
        if dobavljaci_bilans_preseka:
            file_contents['dobavljaci_bilans_preseka'] = await dobavljaci_bilans_preseka.read()
        if kupci_bilans_preseka_iv:
            file_contents['kupci_bilans_preseka_iv'] = await kupci_bilans_preseka_iv.read()
        
        # Process Period 1 if files are provided
        period1_required_files = [kupci_prethodna_fiskalna_godina, dobavljaci_prethodna_fiskalna_godina]
        period1_has_files = any(file is not None for file in period1_required_files)
        
        if period1_has_files and period1_date:
            if not all(period1_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 1 must be provided")
            
            period1_file_mappings = {
                "kupci-prethodna-fiskalna-godina": (kupci_prethodna_fiskalna_godina, file_contents.get('kupci_prethodna_fiskalna_godina')),
                "dobavljaci-prethodna-fiskalna-godina": (dobavljaci_prethodna_fiskalna_godina, file_contents.get('dobavljaci_prethodna_fiskalna_godina')),
                "kupci-prethodna-fiskalna-godina-iv": (kupci_prethodna_fiskalna_godina_iv, file_contents.get('kupci_prethodna_fiskalna_godina_iv'))
            }
            
            # Create period 1 structure
            period1_structure = storage_service.create_period_structure(client_slug, period1_date)
            period1_results = await process_files_with_content(period1_file_mappings, period1_structure, "Period 1")
            all_results.extend(period1_results)
        
        # Process Period 2 if files are provided
        period2_required_files = [kupci_bilans_preseka, dobavljaci_bilans_preseka]
        period2_has_files = any(file is not None for file in period2_required_files)
        
        if period2_has_files and period2_date:
            if not all(period2_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 2 must be provided")
            
            period2_file_mappings = {
                "kupci-bilans-preseka": (kupci_bilans_preseka, file_contents.get('kupci_bilans_preseka')),
                "dobavljaci-bilans-preseka": (dobavljaci_bilans_preseka, file_contents.get('dobavljaci_bilans_preseka')),
                "kupci-bilans-preseka-iv": (kupci_bilans_preseka_iv, file_contents.get('kupci_bilans_preseka_iv'))
            }
            
            # Create period 2 structure
            period2_structure = storage_service.create_period_structure(client_slug, period2_date)
            period2_results = await process_files_with_content(period2_file_mappings, period2_structure, "Period 2")
            all_results.extend(period2_results)
        
        if not all_results:
            raise HTTPException(status_code=400, detail="No valid files provided for either period")
        
        return FileUploadResponse(
            message="Files processed successfully",
            period1_date=period1_date,
            period2_date=period2_date,
            results=all_results
        )
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fixed/single", response_model=FileUploadResponse)
async def upload_fixed_files_single_period(
    client_slug: str = Form(...),
    period_date: str = Form(...),
    # Period fields
    kupci_prethodna_fiskalna_godina: Optional[UploadFile] = File(None),
    dobavljaci_prethodna_fiskalna_godina: Optional[UploadFile] = File(None),
    kupci_prethodna_fiskalna_godina_iv: Optional[UploadFile] = File(None),
    kupci_bilans_preseka: Optional[UploadFile] = File(None),
    dobavljaci_bilans_preseka: Optional[UploadFile] = File(None),
    kupci_bilans_preseka_iv: Optional[UploadFile] = File(None)
):
    """Upload fixed files for a single period (for individual period forms)"""
    try:
        # Read and store file content for all files
        file_contents = {}
        
        # Store file contents
        if kupci_prethodna_fiskalna_godina:
            file_contents['kupci_prethodna_fiskalna_godina'] = await kupci_prethodna_fiskalna_godina.read()
        if dobavljaci_prethodna_fiskalna_godina:
            file_contents['dobavljaci_prethodna_fiskalna_godina'] = await dobavljaci_prethodna_fiskalna_godina.read()
        if kupci_prethodna_fiskalna_godina_iv:
            file_contents['kupci_prethodna_fiskalna_godina_iv'] = await kupci_prethodna_fiskalna_godina_iv.read()
        if kupci_bilans_preseka:
            file_contents['kupci_bilans_preseka'] = await kupci_bilans_preseka.read()
        if dobavljaci_bilans_preseka:
            file_contents['dobavljaci_bilans_preseka'] = await dobavljaci_bilans_preseka.read()
        if kupci_bilans_preseka_iv:
            file_contents['kupci_bilans_preseka_iv'] = await kupci_bilans_preseka_iv.read()
        
        # Determine which period type based on the files provided
        period1_required_files = [kupci_prethodna_fiskalna_godina, dobavljaci_prethodna_fiskalna_godina]
        period2_required_files = [kupci_bilans_preseka, dobavljaci_bilans_preseka]
        
        period1_has_files = any(file is not None for file in period1_required_files)
        period2_has_files = any(file is not None for file in period2_required_files)
        
        if period1_has_files and period2_has_files:
            raise HTTPException(status_code=400, detail="Cannot mix Period 1 and Period 2 files in single upload")
        
        if not period1_has_files and not period2_has_files:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        # Process based on which period type
        if period1_has_files:
            if not all(period1_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 1 must be provided")
            
            file_mappings = {
                "kupci-prethodna-fiskalna-godina": (kupci_prethodna_fiskalna_godina, file_contents.get('kupci_prethodna_fiskalna_godina')),
                "dobavljaci-prethodna-fiskalna-godina": (dobavljaci_prethodna_fiskalna_godina, file_contents.get('dobavljaci_prethodna_fiskalna_godina')),
                "kupci-prethodna-fiskalna-godina-iv": (kupci_prethodna_fiskalna_godina_iv, file_contents.get('kupci_prethodna_fiskalna_godina_iv'))
            }
            period_name = "Period 1"
        else:
            if not all(period2_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 2 must be provided")
            
            file_mappings = {
                "kupci-bilans-preseka": (kupci_bilans_preseka, file_contents.get('kupci_bilans_preseka')),
                "dobavljaci-bilans-preseka": (dobavljaci_bilans_preseka, file_contents.get('dobavljaci_bilans_preseka')),
                "kupci-bilans-preseka-iv": (kupci_bilans_preseka_iv, file_contents.get('kupci_bilans_preseka_iv'))
            }
            period_name = "Period 2"
        
        # Create period structure
        period_structure = storage_service.create_period_structure(client_slug, period_date)
        results = await process_files_with_content(file_mappings, period_structure, period_name)
        
        return FileUploadResponse(
            message="Files processed successfully",
            period_date=period_date,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unfixed", response_model=FileUploadResponse)
async def upload_unfixed_files(
    client_slug: str = Form(...),
    period_date: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload unfixed files (placeholder for future implementation)"""
    try:
        # Create period structure
        period_structure = storage_service.create_period_structure(client_slug, period_date)
        
        results = []
        for file in files:
            content = await file.read()
            
            # Save raw file to storage
            raw_file_path = storage_service.get_storage_path(
                client_slug=client_slug,
                file_type=file.filename,
                period_date=period_date
            )
            
            saved_path = await storage_service.save_file_to_storage(raw_file_path, content)
            
            results.append(ProcessingResult(
                period="Unfixed",
                file_type="unfixed",
                original_filename=file.filename,
                raw_path=saved_path,
                records_processed=0,
                error="Processing not implemented yet"
            ))
        
        return FileUploadResponse(
            message="Unfixed files uploaded (processing not implemented)",
            period_date=period_date,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error uploading unfixed files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_files_with_content(file_mappings: Dict[str, tuple], period_structure, period_name: str) -> List[ProcessingResult]:
    """Process files for a specific period using pre-read content"""
    results = []
    
    logger.info(f"Processing {period_name} with {len(file_mappings)} file mappings (using pre-read content)")
    
    for file_type, (file_obj, content) in file_mappings.items():
        # Check if file was actually uploaded (has filename and content)
        file_uploaded = file_obj is not None and file_obj.filename and content is not None
        logger.info(f"Checking {period_name} - {file_type}: {file_uploaded}")
        
        if file_uploaded:
            try:
                # Validate file type
                file_extension = Path(file_obj.filename).suffix.lower()
                
                if not validate_file_extension(file_obj.filename, settings.allowed_extensions):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File type {file_extension} not allowed for {file_type}"
                    )
                
                # Save raw file to storage
                raw_file_path = storage_service.get_storage_path(
                    client_slug=period_structure.raw_period_path.split('/')[-3] if '/' in period_structure.raw_period_path else period_structure.raw_period_path.split('\\')[-3],
                    file_type=f"{file_type}{file_extension}",
                    period_date=period_structure.raw_period_path.split('/')[-1] if '/' in period_structure.raw_period_path else period_structure.raw_period_path.split('\\')[-1]
                )
                
                raw_file_path_str = await storage_service.save_file_to_storage(raw_file_path, content)
                
                # Process file based on type
                extracted_data = await document_service.extract_tables_from_file(raw_file_path_str, file_extension)
                
                # AI Column Mapping and standardization
                if extracted_data.tables:
                    # Save extracted data to storage
                    extracted_file_path = storage_service.get_storage_path(
                        client_slug=period_structure.extracted_period_path.split('/')[-3] if '/' in period_structure.extracted_period_path else period_structure.extracted_period_path.split('\\')[-3],
                        file_type=f"{file_type}-extracted.json",
                        period_date=period_structure.extracted_period_path.split('/')[-1] if '/' in period_structure.extracted_period_path else period_structure.extracted_period_path.split('\\')[-1],
                        is_extracted=True
                    )
                    
                    extracted_content = json.dumps(extracted_data.tables, indent=2, ensure_ascii=False).encode('utf-8')
                    extracted_file_path_str = await storage_service.save_file_to_storage(extracted_file_path, extracted_content)
                    
                    # AI column mapping
                    column_mapping_result = await ai_service.ai_column_mapping(extracted_data.tables, file_obj.filename)
                    
                    # Save column mapping
                    mapping_file_path = storage_service.get_storage_path(
                        client_slug=period_structure.extracted_period_path.split('/')[-3] if '/' in period_structure.extracted_period_path else period_structure.extracted_period_path.split('\\')[-3],
                        file_type=f"{file_type}-mapping.json",
                        period_date=period_structure.extracted_period_path.split('/')[-1] if '/' in period_structure.extracted_period_path else period_structure.extracted_period_path.split('\\')[-1],
                        is_extracted=True
                    )
                    
                    mapping_content = json.dumps(column_mapping_result.dict(), indent=2, ensure_ascii=False).encode('utf-8')
                    column_mapping_path_str = await storage_service.save_file_to_storage(mapping_file_path, mapping_content)
                    
                    # Standardize data using the mapping
                    standardized_data = ai_service.standardize_table_data_with_mapping(
                        extracted_data.tables[0] if extracted_data.tables else [], 
                        column_mapping_result
                    )
                    
                    # Save processed data to storage
                    processed_file_path = storage_service.get_storage_path(
                        client_slug=period_structure.processed_period_path.split('/')[-3] if '/' in period_structure.processed_period_path else period_structure.processed_period_path.split('\\')[-3],
                        file_type=f"{file_type}-processed.json",
                        period_date=period_structure.processed_period_path.split('/')[-1] if '/' in period_structure.processed_period_path else period_structure.processed_period_path.split('\\')[-1],
                        is_processed=True
                    )
                    
                    processed_content = json.dumps(standardized_data, indent=2, ensure_ascii=False).encode('utf-8')
                    processed_file_path_str = await storage_service.save_file_to_storage(processed_file_path, processed_content)
                    
                    results.append(ProcessingResult(
                        period=period_name,
                        file_type=file_type,
                        original_filename=file_obj.filename,
                        raw_path=raw_file_path_str,
                        extracted_path=extracted_file_path_str,
                        column_mapping_path=column_mapping_path_str,
                        processed_path=processed_file_path_str,
                        records_processed=standardized_data.get("total_records", 0),
                        mapping_confidence=column_mapping_result.mapping_confidence
                    ))
                else:
                    # No tables extracted
                    results.append(ProcessingResult(
                        period=period_name,
                        file_type=file_type,
                        original_filename=file_obj.filename,
                        raw_path=raw_file_path_str,
                        records_processed=0,
                        error="No tables could be extracted from the file"
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing file {file_type}: {e}")
                results.append(ProcessingResult(
                    period=period_name,
                    file_type=file_type,
                    original_filename=file_obj.filename if file_obj else "unknown",
                    raw_path="",
                    records_processed=0,
                    error=str(e)
                ))
    
    return results