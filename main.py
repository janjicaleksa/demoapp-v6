from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import aiofiles
from typing import List, Optional
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import logging
import tempfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Processor Kupci Dobavljaci", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Azure Document Intelligence configuration
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_KEY = os.getenv("AZURE_KEY", "")

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME = "clients"

# Initialize Azure Blob Storage client
blob_service_client = None
if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        # Ensure container exists
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()
        logger.info("Azure Blob Storage client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure Blob Storage client: {e}")
        blob_service_client = None
else:
    logger.warning("AZURE_STORAGE_CONNECTION_STRING not configured, using local storage fallback")

# Fallback to local storage if Azure is not configured
USE_LOCAL_STORAGE = blob_service_client is None
BASE_DIR = Path("clients") if USE_LOCAL_STORAGE else None
if USE_LOCAL_STORAGE:
    BASE_DIR.mkdir(exist_ok=True)

def normalize_client_name(name: str) -> str:
    """Normalize client name to create a valid folder name"""
    # Remove forbidden characters and convert to lowercase
    normalized = re.sub(r'[<>:"/\\|?*]', '', name.lower())
    # Replace spaces with hyphens
    normalized = re.sub(r'\s+', '-', normalized.strip())
    # Remove multiple hyphens
    normalized = re.sub(r'-+', '-', normalized)
    # Remove leading/trailing hyphens
    normalized = normalized.strip('-')
    return normalized

async def upload_to_blob_storage(blob_name: str, content: bytes, content_type: str = None) -> str:
    """Upload content to Azure Blob Storage"""
    if not blob_service_client:
        raise Exception("Azure Blob Storage not configured")
    
    logger.info(f"Uploading to Azure Blob Storage: container={CONTAINER_NAME}, blob={blob_name}")
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    blob_client.upload_blob(content, overwrite=True, content_settings=None if not content_type else 
                           blob_client.get_blob_properties().content_settings)
    result_path = f"azure://{CONTAINER_NAME}/{blob_name}"
    logger.info(f"Successfully uploaded to: {result_path}")
    return result_path

async def download_from_blob_storage(blob_name: str) -> bytes:
    """Download content from Azure Blob Storage"""
    if not blob_service_client:
        raise Exception("Azure Blob Storage not configured")
    
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    download_stream = blob_client.download_blob()
    return download_stream.readall()

async def save_file_to_storage(file_path: str, content: bytes) -> str:
    """Save file to storage (Azure Blob Storage or local fallback)"""
    if USE_LOCAL_STORAGE:
        # Local storage fallback
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, 'wb') as f:
            await f.write(content)
        return str(path)
    else:
        # Azure Blob Storage
        return await upload_to_blob_storage(file_path, content)

async def read_file_from_storage(file_path: str) -> bytes:
    """Read file from storage (Azure Blob Storage or local fallback)"""
    if USE_LOCAL_STORAGE:
        # Local storage fallback
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    else:
        # Azure Blob Storage
        return await download_from_blob_storage(file_path)

def get_blob_name(client_slug: str, file_type: str, period_date: str = None, is_processed: bool = False) -> str:
    """Generate blob name for Azure Blob Storage"""
    if period_date:
        folder_type = "processed" if is_processed else "raw"
        blob_name = f"{client_slug}/{folder_type}/{period_date}/{file_type}"
        logger.info(f"Generated blob name: {blob_name} (client_slug: {client_slug}, file_type: {file_type}, period_date: {period_date}, is_processed: {is_processed})")
        return blob_name
    else:
        folder_type = "processed" if is_processed else "raw"
        blob_name = f"{client_slug}/{folder_type}/{file_type}"
        logger.info(f"Generated blob name: {blob_name} (client_slug: {client_slug}, file_type: {file_type}, is_processed: {is_processed})")
        return blob_name

def create_client_structure(client_slug: str) -> dict:
    """Create the folder structure for a client"""
    if USE_LOCAL_STORAGE:
        # Local storage
        client_path = BASE_DIR / client_slug
        raw_path = client_path / "raw"
        processed_path = client_path / "processed"
        
        # Create directories
        raw_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "client_path": str(client_path),
            "raw_path": str(raw_path),
            "processed_path": str(processed_path)
        }
    else:
        # Azure Blob Storage - virtual folders
        return {
            "client_path": f"azure://{CONTAINER_NAME}/{client_slug}",
            "raw_path": f"azure://{CONTAINER_NAME}/{client_slug}/raw",
            "processed_path": f"azure://{CONTAINER_NAME}/{client_slug}/processed"
        }

def create_period_structure(client_slug: str, period_date: str) -> dict:
    """Create period-specific folders"""
    if USE_LOCAL_STORAGE:
        # Local storage
        client_path = BASE_DIR / client_slug
        raw_period_path = client_path / "raw" / period_date
        processed_period_path = client_path / "processed" / period_date
        
        raw_period_path.mkdir(parents=True, exist_ok=True)
        processed_period_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "raw_period_path": str(raw_period_path),
            "processed_period_path": str(processed_period_path)
        }
    else:
        # Azure Blob Storage - virtual folders
        return {
            "raw_period_path": f"azure://{CONTAINER_NAME}/{client_slug}/raw/{period_date}",
            "processed_period_path": f"azure://{CONTAINER_NAME}/{client_slug}/processed/{period_date}"
        }

async def extract_tables_from_pdf(file_path: str) -> List[dict]:
    """Extract tables from PDF using Azure Document Intelligence"""
    if not AZURE_ENDPOINT or not AZURE_KEY:
        logger.warning("Azure credentials not configured, skipping PDF processing")
        return []
    
    try:
        client = DocumentAnalysisClient(
            endpoint=AZURE_ENDPOINT, 
            credential=AzureKeyCredential(AZURE_KEY)
        )
        
        # Handle both local files and Azure Blob Storage files
        if file_path.startswith("azure://"):
            # Download from Azure Blob Storage to temporary file
            blob_name = file_path.replace(f"azure://{CONTAINER_NAME}/", "")
            content = await download_from_blob_storage(blob_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, "rb") as document:
                    poller = client.begin_analyze_document("prebuilt-document", document.read())
                    result = poller.result()
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        else:
            # Local file
            with open(file_path, "rb") as document:
                poller = client.begin_analyze_document("prebuilt-document", document.read())
                result = poller.result()
        
        tables = []
        for table in result.tables:
            table_data = []
            for row in table.rows:
                row_data = {}
                for i, cell in enumerate(row.cells):
                    if cell.content:
                        row_data[f"column_{i}"] = cell.content
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return []

def extract_tables_from_excel(file_path: str) -> List[dict]:
    """Extract tables from Excel file"""
    try:
        # Handle both local files and Azure Blob Storage files
        if file_path.startswith("azure://"):
            # Download from Azure Blob Storage to temporary file
            import asyncio
            blob_name = file_path.replace(f"azure://{CONTAINER_NAME}/", "")
            content = asyncio.run(download_from_blob_storage(blob_name))
            
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
            
            return tables
    except Exception as e:
        logger.error(f"Error extracting tables from Excel: {e}")
        return []

def extract_tables_from_csv(file_path: str) -> List[dict]:
    """Extract tables from CSV file"""
    try:
        # Handle both local files and Azure Blob Storage files
        if file_path.startswith("azure://"):
            # Download from Azure Blob Storage to temporary file
            import asyncio
            blob_name = file_path.replace(f"azure://{CONTAINER_NAME}/", "")
            content = asyncio.run(download_from_blob_storage(blob_name))
            
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
            
            return [table_data] if table_data else []
    except Exception as e:
        logger.error(f"Error extracting tables from CSV: {e}")
        return []

def standardize_table_data(table_data: List[dict]) -> dict:
    """AI standardizer - converts table data to standardized format"""
    # This is a simplified AI standardizer
    # In a real implementation, this would use an AI model
    
    standardized_data = []
    
    for row in table_data:
        # Extract potential account information
        account_info = {}
        
        # Look for common patterns in the data
        for key, value in row.items():
            if any(keyword in str(value).lower() for keyword in ['konto', 'account', 'racun']):
                account_info['konto'] = str(value)
            elif any(keyword in str(value).lower() for keyword in ['naziv', 'name', 'partner', 'partnera']):
                account_info['naziv_partnera'] = str(value)
            elif any(keyword in str(value).lower() for keyword in ['duguje', 'debit']):
                account_info['duguje'] = str(value)
            elif any(keyword in str(value).lower() for keyword in ['potrazuje', 'credit']):
                account_info['potrazuje'] = str(value)
            elif any(keyword in str(value).lower() for keyword in ['saldo', 'balance']):
                account_info['saldo'] = str(value)
        
        # If we found some account information, add it
        if account_info:
            standardized_data.append(account_info)
    
    return {
        "standardized_records": standardized_data,
        "total_records": len(standardized_data)
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/clients")
async def create_client(client_name: str = Form(...)):
    """Create a new client and folder structure"""
    try:
        logger.info(f"Creating client with name: {client_name}")
        client_slug = normalize_client_name(client_name)
        logger.info(f"Normalized client slug: {client_slug}")
        
        if not client_slug:
            raise HTTPException(status_code=400, detail="Invalid client name")
        
        # Check if client already exists
        if USE_LOCAL_STORAGE:
            client_path = BASE_DIR / client_slug
            if client_path.exists():
                logger.info(f"Client already exists locally: {client_slug}")
                return {"message": "Client already exists", "client_slug": client_slug}
        else:
            # Check if client exists in Azure Blob Storage
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            blobs = container_client.list_blobs(name_starts_with=f"{client_slug}/")
            if any(blob.name.startswith(f"{client_slug}/") for blob in blobs):
                logger.info(f"Client already exists in Azure: {client_slug}")
                return {"message": "Client already exists", "client_slug": client_slug}
        
        # Create folder structure
        structure = create_client_structure(client_slug)
        logger.info(f"Created client structure: {structure}")
        
        return {
            "message": "Client created successfully",
            "client_slug": client_slug,
            "structure": structure
        }
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/fixed")
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
        
        logger.info(f"Period 1 - has_files: {period1_has_files}, date: {period1_date}")
        logger.info(f"Period 1 files - kupci: {kupci_prethodna_fiskalna_godina is not None}, dobavljaci: {dobavljaci_prethodna_fiskalna_godina is not None}")
        
        if period1_has_files and period1_date:
            if not all(period1_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 1 must be provided")
            
            period1_file_mappings = {
                "kupci-prethodna-fiskalna-godina": (kupci_prethodna_fiskalna_godina, file_contents.get('kupci_prethodna_fiskalna_godina')),
                "dobavljaci-prethodna-fiskalna-godina": (dobavljaci_prethodna_fiskalna_godina, file_contents.get('dobavljaci_prethodna_fiskalna_godina')),
                "kupci-prethodna-fiskalna-godina-iv": (kupci_prethodna_fiskalna_godina_iv, file_contents.get('kupci_prethodna_fiskalna_godina_iv'))
            }
            
            # Create period 1 structure
            period1_structure = create_period_structure(client_slug, period1_date)
            logger.info(f"Created period 1 structure: {period1_structure}")
            period1_results = await process_files_with_content(period1_file_mappings, period1_structure, "Period 1")
            all_results.extend(period1_results)
        
        # Process Period 2 if files are provided
        period2_required_files = [kupci_bilans_preseka, dobavljaci_bilans_preseka]
        period2_has_files = any(file is not None for file in period2_required_files)
        
        logger.info(f"Period 2 - has_files: {period2_has_files}, date: {period2_date}")
        logger.info(f"Period 2 files - kupci: {kupci_bilans_preseka is not None}, dobavljaci: {dobavljaci_bilans_preseka is not None}")
        
        if period2_has_files and period2_date:
            if not all(period2_required_files):
                raise HTTPException(status_code=400, detail="All required files for Period 2 must be provided")
            
            period2_file_mappings = {
                "kupci-bilans-preseka": (kupci_bilans_preseka, file_contents.get('kupci_bilans_preseka')),
                "dobavljaci-bilans-preseka": (dobavljaci_bilans_preseka, file_contents.get('dobavljaci_bilans_preseka')),
                "kupci-bilans-preseka-iv": (kupci_bilans_preseka_iv, file_contents.get('kupci_bilans_preseka_iv'))
            }
            
            # Create period 2 structure
            period2_structure = create_period_structure(client_slug, period2_date)
            logger.info(f"Created period 2 structure: {period2_structure}")
            period2_results = await process_files_with_content(period2_file_mappings, period2_structure, "Period 2")
            all_results.extend(period2_results)
        
        if not all_results:
            raise HTTPException(status_code=400, detail="No valid files provided for either period")
        
        return {
            "message": "Files processed successfully",
            "period1_date": period1_date,
            "period2_date": period2_date,
            "results": all_results
        }
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/fixed/single")
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
        period_structure = create_period_structure(client_slug, period_date)
        results = await process_files_with_content(file_mappings, period_structure, period_name)
        
        return {
            "message": "Files processed successfully",
            "period_date": period_date,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_files(file_mappings: dict, period_structure: dict, period_name: str) -> List[dict]:
    """Process files for a specific period"""
    results = []
    
    logger.info(f"Processing {period_name} with {len(file_mappings)} file mappings")
    
    for file_type, file in file_mappings.items():
        # Check if file was actually uploaded (has filename)
        file_uploaded = file is not None and file.filename
        logger.info(f"Checking {period_name} - {file_type}: {file_uploaded}")
        
        if file_uploaded:
            # Validate file type
            allowed_extensions = {'.xlsx', '.csv', '.doc', '.docx', '.pdf'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not allowed for {file_type}"
                )
            
            # Read file content once and store it in memory
            content = await file.read()
            
            # Save raw file to storage
            if USE_LOCAL_STORAGE:
                raw_file_path = Path(period_structure["raw_period_path"]) / f"{file_type}{file_extension}"
                async with aiofiles.open(raw_file_path, 'wb') as f:
                    await f.write(content)
                raw_file_path_str = str(raw_file_path)
            else:
                # Azure Blob Storage
                blob_name = get_blob_name(
                    client_slug=period_structure["raw_period_path"].split('/')[3],  # Extract client_slug
                    file_type=f"{file_type}{file_extension}",
                    period_date=period_structure["raw_period_path"].split('/')[-1]  # Extract period_date
                )
                raw_file_path_str = await upload_to_blob_storage(blob_name, content)
            
            # Process file based on type
            tables = []
            if file_extension == '.pdf':
                tables = await extract_tables_from_pdf(raw_file_path_str)
            elif file_extension in ['.xlsx']:
                tables = extract_tables_from_excel(raw_file_path_str)
            elif file_extension == '.csv':
                tables = extract_tables_from_csv(raw_file_path_str)
            
            # Standardize data
            if tables:
                standardized_data = standardize_table_data(tables[0])  # Use first table
                
                # Save processed data to storage
                if USE_LOCAL_STORAGE:
                    processed_file_path = Path(period_structure["processed_period_path"]) / f"{file_type}-processed.json"
                    async with aiofiles.open(processed_file_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(standardized_data, indent=2, ensure_ascii=False))
                    processed_file_path_str = str(processed_file_path)
                else:
                    # Azure Blob Storage
                    processed_blob_name = get_blob_name(
                        client_slug=period_structure["processed_period_path"].split('/')[3],  # Extract client_slug
                        file_type=f"{file_type}-processed.json",
                        period_date=period_structure["processed_period_path"].split('/')[-1],  # Extract period_date
                        is_processed=True
                    )
                    processed_content = json.dumps(standardized_data, indent=2, ensure_ascii=False).encode('utf-8')
                    processed_file_path_str = await upload_to_blob_storage(processed_blob_name, processed_content)
                
                results.append({
                    "period": period_name,
                    "file_type": file_type,
                    "original_filename": file.filename,
                    "raw_path": raw_file_path_str,
                    "processed_path": processed_file_path_str,
                    "records_processed": standardized_data.get("total_records", 0)
                })
    
    return results

async def process_files_with_content(file_mappings: dict, period_structure: dict, period_name: str) -> List[dict]:
    """Process files for a specific period using pre-read content"""
    results = []
    
    logger.info(f"Processing {period_name} with {len(file_mappings)} file mappings (using pre-read content)")
    
    for file_type, (file_obj, content) in file_mappings.items():
        # Check if file was actually uploaded (has filename and content)
        file_uploaded = file_obj is not None and file_obj.filename and content is not None
        logger.info(f"Checking {period_name} - {file_type}: {file_uploaded}")
        
        if file_uploaded:
            # Validate file type
            allowed_extensions = {'.xlsx', '.csv', '.doc', '.docx', '.pdf'}
            file_extension = Path(file_obj.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not allowed for {file_type}"
                )
            
            # Save raw file to storage
            if USE_LOCAL_STORAGE:
                raw_file_path = Path(period_structure["raw_period_path"]) / f"{file_type}{file_extension}"
                async with aiofiles.open(raw_file_path, 'wb') as f:
                    await f.write(content)
                raw_file_path_str = str(raw_file_path)
            else:
                # Azure Blob Storage
                blob_name = get_blob_name(
                    client_slug=period_structure["raw_period_path"].split('/')[3],  # Extract client_slug
                    file_type=f"{file_type}{file_extension}",
                    period_date=period_structure["raw_period_path"].split('/')[-1]  # Extract period_date
                )
                raw_file_path_str = await upload_to_blob_storage(blob_name, content)
            
            # Process file based on type
            tables = []
            if file_extension == '.pdf':
                tables = await extract_tables_from_pdf(raw_file_path_str)
            elif file_extension in ['.xlsx']:
                tables = extract_tables_from_excel(raw_file_path_str)
            elif file_extension == '.csv':
                tables = extract_tables_from_csv(raw_file_path_str)
            
            # Standardize data
            if tables:
                standardized_data = standardize_table_data(tables[0])  # Use first table
                
                # Save processed data to storage
                if USE_LOCAL_STORAGE:
                    processed_file_path = Path(period_structure["processed_period_path"]) / f"{file_type}-processed.json"
                    async with aiofiles.open(processed_file_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(standardized_data, indent=2, ensure_ascii=False))
                    processed_file_path_str = str(processed_file_path)
                else:
                    # Azure Blob Storage
                    processed_blob_name = get_blob_name(
                        client_slug=period_structure["processed_period_path"].split('/')[3],  # Extract client_slug
                        file_type=f"{file_type}-processed.json",
                        period_date=period_structure["processed_period_path"].split('/')[-1],  # Extract period_date
                        is_processed=True
                    )
                    processed_content = json.dumps(standardized_data, indent=2, ensure_ascii=False).encode('utf-8')
                    processed_file_path_str = await upload_to_blob_storage(processed_blob_name, processed_content)
                
                results.append({
                    "period": period_name,
                    "file_type": file_type,
                    "original_filename": file_obj.filename,
                    "raw_path": raw_file_path_str,
                    "processed_path": processed_file_path_str,
                    "records_processed": standardized_data.get("total_records", 0)
                })
    
    return results

@app.post("/api/upload/unfixed")
async def upload_unfixed_files(
    client_slug: str = Form(...),
    period_date: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload unfixed files (placeholder for future implementation)"""
    try:
        # Create period structure
        period_structure = create_period_structure(client_slug, period_date)
        
        results = []
        for file in files:
            content = await file.read()
            
            # Save raw file to storage
            if USE_LOCAL_STORAGE:
                raw_file_path = Path(period_structure["raw_period_path"]) / file.filename
                async with aiofiles.open(raw_file_path, 'wb') as f:
                    await f.write(content)
                raw_file_path_str = str(raw_file_path)
            else:
                # Azure Blob Storage
                blob_name = get_blob_name(
                    client_slug=client_slug,
                    file_type=file.filename,
                    period_date=period_date
                )
                raw_file_path_str = await upload_to_blob_storage(blob_name, content)
            
            results.append({
                "filename": file.filename,
                "raw_path": raw_file_path_str,
                "status": "saved (unfixed processing not implemented yet)"
            })
        
        return {
            "message": "Unfixed files uploaded (processing not implemented)",
            "period_date": period_date,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error uploading unfixed files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_slug}/structure")
async def get_client_structure(client_slug: str):
    """Get client folder structure"""
    try:
        if USE_LOCAL_STORAGE:
            client_path = BASE_DIR / client_slug
            if not client_path.exists():
                raise HTTPException(status_code=404, detail="Client not found")
            
            structure = {
                "client_slug": client_slug,
                "raw_path": str(client_path / "raw"),
                "processed_path": str(client_path / "processed")
            }
        else:
            # Check if client exists in Azure Blob Storage
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            blobs = container_client.list_blobs(name_starts_with=f"{client_slug}/")
            if not any(blob.name.startswith(f"{client_slug}/") for blob in blobs):
                raise HTTPException(status_code=404, detail="Client not found")
            
            structure = {
                "client_slug": client_slug,
                "raw_path": f"azure://{CONTAINER_NAME}/{client_slug}/raw",
                "processed_path": f"azure://{CONTAINER_NAME}/{client_slug}/processed"
            }
        
        return structure
    except Exception as e:
        logger.error(f"Error getting client structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 