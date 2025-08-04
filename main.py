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
import logging

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

# Create base directories
BASE_DIR = Path("clients")
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

def create_client_structure(client_slug: str) -> dict:
    """Create the folder structure for a client"""
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

def create_period_structure(client_slug: str, period_date: str) -> dict:
    """Create period-specific folders"""
    client_path = BASE_DIR / client_slug
    raw_period_path = client_path / "raw" / period_date
    processed_period_path = client_path / "processed" / period_date
    
    raw_period_path.mkdir(parents=True, exist_ok=True)
    processed_period_path.mkdir(parents=True, exist_ok=True)
    
    return {
        "raw_period_path": str(raw_period_path),
        "processed_period_path": str(processed_period_path)
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
        client_slug = normalize_client_name(client_name)
        
        if not client_slug:
            raise HTTPException(status_code=400, detail="Invalid client name")
        
        # Check if client already exists
        client_path = BASE_DIR / client_slug
        if client_path.exists():
            return {"message": "Client already exists", "client_slug": client_slug}
        
        # Create folder structure
        structure = create_client_structure(client_slug)
        
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
        all_results = []
        
        # Process Period 1 if files are provided
        period1_files = [kupci_prethodna_fiskalna_godina, dobavljaci_prethodna_fiskalna_godina]
        if any(period1_files) and period1_date:
            if not all(period1_files):
                raise HTTPException(status_code=400, detail="All required files for Period 1 must be provided")
            
            period1_file_mappings = {
                "kupci-prethodna-fiskalna-godina": kupci_prethodna_fiskalna_godina,
                "dobavljaci-prethodna-fiskalna-godina": dobavljaci_prethodna_fiskalna_godina,
                "kupci-prethodna-fiskalna-godina-iv": kupci_prethodna_fiskalna_godina_iv
            }
            
            # Create period 1 structure
            period1_structure = create_period_structure(client_slug, period1_date)
            period1_results = await process_files(period1_file_mappings, period1_structure, "Period 1")
            all_results.extend(period1_results)
        
        # Process Period 2 if files are provided
        period2_files = [kupci_bilans_preseka, dobavljaci_bilans_preseka]
        if any(period2_files) and period2_date:
            if not all(period2_files):
                raise HTTPException(status_code=400, detail="All required files for Period 2 must be provided")
            
            period2_file_mappings = {
                "kupci-bilans-preseka": kupci_bilans_preseka,
                "dobavljaci-bilans-preseka": dobavljaci_bilans_preseka,
                "kupci-bilans-preseka-iv": kupci_bilans_preseka_iv
            }
            
            # Create period 2 structure
            period2_structure = create_period_structure(client_slug, period2_date)
            period2_results = await process_files(period2_file_mappings, period2_structure, "Period 2")
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

async def process_files(file_mappings: dict, period_structure: dict, period_name: str) -> List[dict]:
    """Process files for a specific period"""
    results = []
    
    for file_type, file in file_mappings.items():
        if file:
            # Validate file type
            allowed_extensions = {'.xlsx', '.csv', '.doc', '.docx', '.pdf'}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not allowed for {file_type}"
                )
            
            # Save raw file
            raw_file_path = Path(period_structure["raw_period_path"]) / f"{file_type}{file_extension}"
            async with aiofiles.open(raw_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process file based on type
            tables = []
            if file_extension == '.pdf':
                tables = await extract_tables_from_pdf(str(raw_file_path))
            elif file_extension in ['.xlsx']:
                tables = extract_tables_from_excel(str(raw_file_path))
            elif file_extension == '.csv':
                tables = extract_tables_from_csv(str(raw_file_path))
            
            # Standardize data
            if tables:
                standardized_data = standardize_table_data(tables[0])  # Use first table
                
                # Save processed data
                processed_file_path = Path(period_structure["processed_period_path"]) / f"{file_type}-processed.json"
                async with aiofiles.open(processed_file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(standardized_data, indent=2, ensure_ascii=False))
                
                results.append({
                    "period": period_name,
                    "file_type": file_type,
                    "original_filename": file.filename,
                    "raw_path": str(raw_file_path),
                    "processed_path": str(processed_file_path),
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
            # Save raw file
            raw_file_path = Path(period_structure["raw_period_path"]) / file.filename
            async with aiofiles.open(raw_file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            results.append({
                "filename": file.filename,
                "raw_path": str(raw_file_path),
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
        client_path = BASE_DIR / client_slug
        if not client_path.exists():
            raise HTTPException(status_code=404, detail="Client not found")
        
        structure = {
            "client_slug": client_slug,
            "raw_path": str(client_path / "raw"),
            "processed_path": str(client_path / "processed")
        }
        
        return structure
    except Exception as e:
        logger.error(f"Error getting client structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 