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
from collections import defaultdict
from dotenv import load_dotenv
import aiofiles
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobClient, ContainerClient
import logging
import logging.config
import tempfile
from typing import Dict, List, Any, Optional

# Configure logging
logging.config.dictConfig({
    'version':1,
    'disable_existing_loggers':True,
    'loggers': { '': {'level':'INFO'}}
})
logger = logging.getLogger('azure')
logger.setLevel(logging.ERROR)

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

# Loading .env file
load_dotenv()

# Azure Document Intelligence configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")

# Azure Blob Storage configuration
AZURE_BLOB_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING", "")
AZURE_BLOB_STORAGE_CONTAINER_NAME = os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME", "")

# OpenAI configuration for AI column mapping
AZURE_AI_FOUNDRY_CONNECTION = os.getenv("AZURE_AI_FOUNDRY_CONNECTION", "")
AZURE_AI_FOUNDRY_KEY = os.getenv("AZURE_AI_FOUNDRY_KEY", "")
AZURE_AI_FOUNDRY_MODEL = os.getenv("AZURE_AI_FOUNDRY_MODEL", "")

# Initialize OpenAI client if configured
ai_foundry_client = None
if AZURE_AI_FOUNDRY_CONNECTION:
    try:
        ai_foundry_client = ChatCompletionsClient(
            endpoint=AZURE_AI_FOUNDRY_CONNECTION,
            credential=AzureKeyCredential(AZURE_AI_FOUNDRY_KEY)
        )
        logger.info("Azure AI Foundry client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure AI Foundry client: {e}")
        ai_foundry_client = None
else:
    logger.warning("Azure AI Foundry credentials not configured, AI column mapping will use fallback")

# Initialize Azure Blob Storage client
blob_service_client = None
if AZURE_BLOB_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_STORAGE_CONNECTION_STRING)
        # Ensure container exists
        container_client = blob_service_client.get_container_client(AZURE_BLOB_STORAGE_CONTAINER_NAME)
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
    
    logger.info(f"Uploading to Azure Blob Storage: container={AZURE_BLOB_STORAGE_CONTAINER_NAME}, blob={blob_name}")
    blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_STORAGE_CONTAINER_NAME, blob=blob_name)
    if not content_type:
        content_settings = None
    else:
        content_settings = ContentSettings(content_type=content_type)
    blob_client.upload_blob(content, overwrite=True, content_settings=content_settings)
    result_path = f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{blob_name}"
    logger.info(f"Successfully uploaded to: {result_path}")
    return result_path

async def download_from_blob_storage(blob_name: str) -> bytes:
    """Download content from Azure Blob Storage"""
    if not blob_service_client:
        raise Exception("Azure Blob Storage not configured")
    
    blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_STORAGE_CONTAINER_NAME, blob=blob_name)
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

def get_blob_name(client_slug: str, file_type: str, period_date: str = None, is_extracted: bool = False, is_processed: bool = False) -> str:
    """Generate blob name for Azure Blob Storage"""
    if is_extracted:
        folder_type = "extracted"
    elif is_processed:          
        folder_type = "processed"
    else:
        folder_type = "raw"
    
    if period_date:
        blob_name = f"{client_slug}/{folder_type}/{period_date}/{file_type}"
        logger.info(f"Generated blob name: {blob_name} (client_slug: {client_slug}, file_type: {file_type}, period_date: {period_date}, is_extracted: {is_extracted}, is_processed: {is_processed})")
        return blob_name
    else:
        blob_name = f"{client_slug}/{folder_type}/{file_type}"
        logger.info(f"Generated blob name: {blob_name} (client_slug: {client_slug}, file_type: {file_type}, is_extracted: {is_extracted}, is_processed: {is_processed})")
        return blob_name

def create_client_structure(client_slug: str) -> dict:
    """Create the folder structure for a client"""
    if USE_LOCAL_STORAGE:
        # Local storage
        client_path = BASE_DIR / client_slug
        raw_path = client_path / "raw"
        extracted_path = client_path / "extracted"
        processed_path = client_path / "processed"
        
        # Create directories
        raw_path.mkdir(parents=True, exist_ok=True)
        extracted_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "client_path": str(client_path),
            "raw_path": str(raw_path),
            "extracted_path": str(extracted_path),
            "processed_path": str(processed_path)
        }
    else:
        # Azure Blob Storage - virtual folders
        return {
            "client_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}",
            "raw_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/raw",
            "extracted_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/extracted",
            "processed_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/processed"
        }

def create_period_structure(client_slug: str, period_date: str) -> dict:
    """Create period-specific folders"""
    if USE_LOCAL_STORAGE:
        # Local storage
        client_path = BASE_DIR / client_slug
        raw_period_path = client_path / "raw" / period_date
        extracted_period_path = client_path / "extracted" / period_date
        processed_period_path = client_path / "processed" / period_date
        
        raw_period_path.mkdir(parents=True, exist_ok=True)
        extracted_period_path.mkdir(parents=True, exist_ok=True)
        processed_period_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "raw_period_path": str(raw_period_path),
            "extracted_period_path": str(extracted_period_path),
            "processed_period_path": str(processed_period_path)
        }
    else:
        # Azure Blob Storage - virtual folders
        return {
            "raw_period_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/raw/{period_date}",
            "extracted_period_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/extracted/{period_date}",
            "processed_period_path": f"azure://{AZURE_AI_FOUNDRY_CONNECTION}/{client_slug}/processed/{period_date}"
        }

async def extract_tables_from_pdf(file_path: str) -> List[dict]:
    """Extract tables from PDF using Azure Document Intelligence"""
    if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
        logger.warning("Azure credentials not configured, skipping PDF processing")
        return []
    
    try:
        client = DocumentAnalysisClient(
            endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, 
            credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY),
            logging_enable=False
        )
        
        # Handle both local files and Azure Blob Storage files
        if file_path.startswith("azure://"):
            # Download from Azure Blob Storage to temporary file
            blob_name = file_path.replace(f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/", "")
            content = await download_from_blob_storage(blob_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, "rb") as document:
                    poller = client.begin_analyze_document("prebuilt-document", document.read(), logger=logger)
                    result = poller.result()
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        else:
            # Local file
            with open(file_path, "rb") as document:
                poller = client.begin_analyze_document("prebuilt-document", document.read(), logger=logger)
                result = poller.result()
        
        tables = []
        for table in result.tables:
            temp_table = defaultdict(dict)
            for cell in table.cells:
                temp_table[cell.column_index][cell.row_index] = cell.content

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
            formatted_data = []
            headers = rows[0]
            normalized_headers = [re.sub(r'\W+', '', h).strip().lower() for h in headers]
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
                logger.warning(f"No data extracted from table {table.id}")

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
            blob_name = file_path.replace(f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/", "")
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
            blob_name = file_path.replace(f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/", "")
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

def determine_file_type(filename: str) -> str:
    """Determine if file is 'kupci' or 'dobavljaci' based on filename"""
    filename_lower = filename.lower()
    
    if 'kupci' in filename_lower:
        return 'kupci'
    elif 'dobavljaci' in filename_lower:
        return 'dobavljaci'
    else:
        raise Exception(f"Something went wrong with file type detection for {filename}")

def get_desired_keys(file_type: str) -> List[str]:
    """Get desired mapping keys based on file type"""
    if file_type == 'kupci':
        return ['konto', 'naziv_partnera', 'promet duguje', 'saldo']
    elif file_type == 'dobavljaci':
        return ['konto', 'naziv_partnera', 'promet potrazuje', 'saldo']
    else:
        return []

async def ai_column_mapping(tables: List[dict], filename: str) -> Dict[str, Any]:
    """
    AI column mapping function that analyzes table headers and creates mappings
    """
    try:
        file_type = determine_file_type(filename)
        desired_keys = get_desired_keys(file_type)
        
        logger.info(f"AI column mapping for file: {filename}, type: {file_type}, desired keys: {desired_keys}")
        
        # If OpenAI is not available, use fallback logic
        if not ai_foundry_client:
            return [] #await fallback_column_mapping(tables, desired_keys, file_type)
        
        # Prepare table headers for AI analysis
        table_headers = []
        seen_headers = []
        for i, table in enumerate(tables):
            if table:
                # Extract headers from first row
                headers = list(table[0].keys())
                if headers not in seen_headers:
                    seen_headers.append(headers)
                    table_headers.append({
                        "table_index": i,
                        "headers": headers,
                        "sample_data": table[:15] # First 15 rows for context
                    })
            else:
                logger.warning(f"Table {i} is empty")
        
        # Create AI prompt for column mapping
        prompt = create_column_mapping_prompt(table_headers, desired_keys, file_type)
        
        # Call Azure AI Foundry
        response = await call_azure_ai_foundry(prompt)
        
        # Parse AI response
        column_mapping = parse_ai_response(response, desired_keys)
        
        # Create mapping result
        mapping_result = {
            "filename": filename,
            "file_type": file_type,
            "desired_keys": desired_keys,
            "column_mapping": column_mapping,
            "tables_analyzed": len(table_headers),
            "mapping_confidence": "high" if column_mapping else "low"
        }
        
        logger.info(f"AI column mapping completed: {mapping_result}")
        return mapping_result
        
    except Exception as e:
        logger.error(f"Error in AI column mapping: {e}")
        # Fallback to basic mapping
        return [] #await fallback_column_mapping(tables, desired_keys, determine_file_type(filename))

def create_column_mapping_prompt(table_headers: List[dict], desired_keys: List[str], file_type: str) -> str:
    """Create prompt for AI column mapping"""
    prompt_tables = ""
    for table_info in table_headers:
        prompt_tables += f"\nTable {table_info['table_index']}:\n"
        prompt_tables += f"Headers: {', '.join(table_info['headers'])}\n"
        if table_info['sample_data']:
            prompt_tables += "Sample data (first 15 rows):\n"
            for i, row in enumerate(table_info['sample_data']):
                prompt_tables += f"Row {i+1}: {row}\n"
        prompt_tables += "\n"

    prompt = f"""
        You are an expert data analyst specializing in financial document processing. You need to map table columns to standardized keys for {file_type} files.

        DESIRED MAPPING KEYS:
        {', '.join(desired_keys)}

        TABLES TO ANALYZE: {prompt_tables}
            
        INSTRUCTIONS:
        1. Analyze each table's headers and sample data
        2. Map the headers to the desired keys: {', '.join(desired_keys)}
        3. Consider semantic similarity, abbreviations, and common variations
        4. For each desired key, provide the most likely matching header
        5. If no good match is found, use null
        6. Consider the context: this is a {file_type} file

        RESPONSE FORMAT (JSON):
        {{
            "table_mappings": [
                {{
                    "table_index": 0,
                    "mapping": {{
                        "{desired_keys[0]}": "matching_header_or_null",
                        "{desired_keys[1]}": "matching_header_or_null",
                        ...
                    }}
                }}
            ]
        }}

        Provide only the JSON response, no additional text.
        """
    
    return prompt

async def call_azure_ai_foundry(prompt: str) -> str:
    """Call Azure AI Foundry API"""
    try:
        response = ai_foundry_client.complete(
            stream=True,
            messages=[
                SystemMessage(content="You are a data mapping expert. Provide only JSON responses."),
                UserMessage(content=prompt)
            ],
            temperature=0.1,
            max_tokens=4096,
            model=AZURE_AI_FOUNDRY_MODEL
        )

        return_response_string = ""
        for update in response:
            if update.choices:
                return_response_string += update.choices[0].delta.content or ""
        
        return return_response_string
        
    except Exception as e:
        logger.error(f"Error calling Azure AI Foundry: {e}")
        raise

def parse_ai_response(response: str, desired_keys: List[str]) -> Dict[str, Any]:
    """Parse AI response and extract column mappings"""
    try:
        # Clean response and extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        
        # Parse JSON
        import json
        parsed = json.loads(response)
        
        # Extract mappings
        mappings = {}
        if 'table_mappings' in parsed:
            for table_mapping in parsed['table_mappings']:
                table_index = table_mapping.get('table_index', 0)
                mapping = table_mapping.get('mapping', {})
                mappings[f"table_{table_index}"] = mapping
        
        return mappings
        
    except Exception as e:
        logger.error(f"Error parsing AI response: {e}")
        return {}

'''async def fallback_column_mapping(tables: List[dict], desired_keys: List[str], file_type: str) -> Dict[str, Any]:
    """Fallback column mapping when AI is not available"""
    logger.info("Using fallback column mapping")
    
    mappings = {}
    
    for i, table in enumerate(tables):
        if not table:
            continue
            
        # Get headers from first row
        headers = list(table[0].keys()) if table else []
        
        # Simple keyword-based mapping
        mapping = {}
        for desired_key in desired_keys:
            best_match = None
            
            # Look for exact matches first
            for header in headers:
                if desired_key.lower() in header.lower():
                    best_match = header
                    break
            
            # Look for semantic matches
            if not best_match:
                semantic_matches = {
                    'konto': ['account', 'racun', 'konto', 'broj'],
                    'naziv_partnera': ['naziv', 'name', 'partner', 'partnera', 'klijent'],
                    'duguje': ['duguje', 'debit', 'dugovanje'],
                    'potrazuje': ['potrazuje', 'credit', 'potrazivanje'],
                    'saldo': ['saldo', 'balance', 'stanje']
                }
                
                if desired_key in semantic_matches:
                    for keyword in semantic_matches[desired_key]:
                        for header in headers:
                            if keyword.lower() in header.lower():
                                best_match = header
                                break
                        if best_match:
                            break
            
            mapping[desired_key] = best_match
        
        mappings[f"table_{i}"] = mapping
    
    return {
        "filename": "unknown",
        "file_type": file_type,
        "desired_keys": desired_keys,
        "column_mapping": mappings,
        "tables_analyzed": len(tables),
        "mapping_confidence": "low"
    }

async def save_column_mapping(mapping_result: Dict[str, Any], storage_path: str, filename: str) -> str:
    """Save column mapping to JSON file"""
    try:
        # Create filename for mapping
        mapping_filename = f"{filename}_column_mapping.json"
        
        if USE_LOCAL_STORAGE:
            # Local storage
            mapping_path = Path(storage_path) / mapping_filename
            async with aiofiles.open(mapping_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(mapping_result, indent=2, ensure_ascii=False))
            return str(mapping_path)
        else:
            # Azure Blob Storage
            blob_name = f"{storage_path.replace('azure://' + CONTAINER_NAME + '/', '')}/{mapping_filename}"
            content = json.dumps(mapping_result, indent=2, ensure_ascii=False).encode('utf-8')
            return await upload_to_blob_storage(blob_name, content)
            
    except Exception as e:
        logger.error(f"Error saving column mapping: {e}")
        raise

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
    }'''

def standardize_table_data_with_mapping(table_data: List[dict], mapping_result: Dict[str, Any]) -> dict:
    """Standardize table data using AI-generated column mapping"""
    try:
        file_type = mapping_result.get("file_type", "unknown")
        desired_keys = mapping_result.get("desired_keys", [])
        column_mapping = mapping_result.get("column_mapping", {})
        
        # Get the first table mapping (assuming single table for now)
        table_mapping = list(column_mapping.values())[0] if column_mapping else {}
        
        standardized_data = []
        
        for row in table_data:
            standardized_row = {}
            
            # Map each desired key to the corresponding column
            for desired_key in desired_keys:
                mapped_column = table_mapping.get(desired_key)
                if mapped_column and mapped_column in row:
                    standardized_row[desired_key] = str(row[mapped_column])
                else:
                    # Try to find a match using fallback logic
                    for _, col_value in row.items():
                        if any(keyword in str(col_value).lower() for keyword in [desired_key, desired_key.replace('_', '')]):
                            standardized_row[desired_key] = str(col_value)
                            break
                    else:
                        standardized_row[desired_key] = ""
            
            if any(standardized_row.values()):  # Only add if we have some data
                standardized_data.append(standardized_row)
        
        return {
            "standardized_records": standardized_data,
            "total_records": len(standardized_data),
            "mapping_used": table_mapping,
            "file_type": file_type
        }
        
    except Exception as e:
        logger.error(f"Error in standardize_table_data_with_mapping: {e}")
        # Fallback to original standardization
        return standardize_table_data(table_data)

def standardize_table_data(table_data: List[dict]) -> dict:
    standardized_data = []
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
            container_client = blob_service_client.get_container_client(AZURE_BLOB_STORAGE_CONTAINER_NAME)
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
                # Save extracted data to storage
                if USE_LOCAL_STORAGE:
                    extracted_file_path = Path(period_structure["extracted_period_path"]) / f"{file_type}-extracted.json"
                    async with aiofiles.open(extracted_file_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(tables, indent=2, ensure_ascii=False))
                    extracted_file_path_str = str(extracted_file_path)
                else:
                    # Azure Blob Storage
                    extracted_blob_name = get_blob_name(
                        client_slug=period_structure["extracted_period_path"].split('/')[3],  # Extract client_slug
                        file_type=f"{file_type}-extracted.json",
                        period_date=period_structure["extracted_period_path"].split('/')[-1],  # Extract period_date
                        is_extracted=True
                    )
                    extracted_content = json.dumps(tables, indent=2, ensure_ascii=False).encode('utf-8')
                    extracted_file_path_str = await upload_to_blob_storage(extracted_blob_name, extracted_content)
                
                # Standardize data
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
                    "extracted_path": extracted_file_path_str,
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
            
            # AI Column Mapping - NEW STEP
            if tables:
                # Save extracted data to storage
                if USE_LOCAL_STORAGE:
                    extracted_file_path = Path(period_structure["extracted_period_path"]) / f"{file_type}-extracted.json"
                    async with aiofiles.open(extracted_file_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(tables, indent=2, ensure_ascii=False))
                    extracted_file_path_str = str(extracted_file_path)
                else:
                    # Azure Blob Storage
                    extracted_blob_name = get_blob_name(
                        client_slug=period_structure["extracted_period_path"].split('/')[3],  # Extract client_slug
                        file_type=f"{file_type}-extracted.json",
                        period_date=period_structure["extracted_period_path"].split('/')[-1],  # Extract period_date
                        is_extracted=True
                    )
                    extracted_content = json.dumps(tables, indent=2, ensure_ascii=False).encode('utf-8')
                    extracted_file_path_str = await upload_to_blob_storage(extracted_blob_name, extracted_content)
                
                column_mapping_result = await ai_column_mapping(tables, file_obj.filename)

                if USE_LOCAL_STORAGE:
                    #Local storage
                    mapping_path = Path(period_structure["extracted_period_path"]) / f"{file_type}-mapping.json"
                    async with aiofiles.open(mapping_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(column_mapping_result, indent=2, ensure_ascii=False))
                    column_mapping_path_str = str(mapping_path)
                else:
                    # Azure Blob Storage
                    mapping_blob_name = get_blob_name(
                        client_slug=period_structure["extracted_period_path"].split('/')[3],  # Extract client_slug
                        file_type=f"{file_type}-mapping.json",
                        period_date=period_structure["extracted_period_path"].split('/')[-1],  # Extract period_date
                        is_extracted=True
                    )
                    column_mapping_content = json.dumps(column_mapping_result, indent=2, ensure_ascii=False).encode('utf-8')
                    column_mapping_path_str = await upload_to_blob_storage(mapping_blob_name, column_mapping_content)
    
                # Standardize data using the mapping
                standardized_data_list = {}
                for i, table in enumerate(tables):
                    standardized_data = standardize_table_data_with_mapping(table, column_mapping_result[i])
                    standardized_data_list[i] = standardized_data
                
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
                    "extracted_path": extracted_file_path_str,
                    "column_mapping_path": column_mapping_path_str,
                    "processed_path": processed_file_path_str,
                    "records_processed": standardized_data.get("total_records", 0),
                    "mapping_confidence": column_mapping_result.get("mapping_confidence", "unknown")
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
            container_client = blob_service_client.get_container_client(AZURE_BLOB_STORAGE_CONTAINER_NAME)
            blobs = container_client.list_blobs(name_starts_with=f"{client_slug}/")
            if not any(blob.name.startswith(f"{client_slug}/") for blob in blobs):
                raise HTTPException(status_code=404, detail="Client not found")
            
            structure = {
                "client_slug": client_slug,
                "raw_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/raw",
                "processed_path": f"azure://{AZURE_BLOB_STORAGE_CONTAINER_NAME}/{client_slug}/processed"
            }
        
        return structure
    except Exception as e:
        logger.error(f"Error getting client structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 