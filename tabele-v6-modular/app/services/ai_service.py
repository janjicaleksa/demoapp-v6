"""
AI service for column mapping and data standardization
"""

import json
from typing import List, Dict, Any
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.utils import determine_file_type, get_desired_keys
from ..models.schemas import AIColumnMappingResponse, StandardizedRecord

logger = get_logger(__name__)


class AIService:
    """Service for AI-powered column mapping and data standardization"""
    
    def __init__(self):
        self.ai_foundry_client = None
        self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize Azure AI Foundry client"""
        if settings.azure_ai_foundry_connection:
            try:
                self.ai_foundry_client = ChatCompletionsClient(
                    endpoint=settings.azure_ai_foundry_connection,
                    credential=AzureKeyCredential(settings.azure_ai_foundry_key)
                )
                logger.info("Azure AI Foundry client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure AI Foundry client: {e}")
                self.ai_foundry_client = None
        else:
            logger.warning("Azure AI Foundry credentials not configured, AI column mapping will use fallback")
    
    async def ai_column_mapping(self, tables: List[List[Dict[str, Any]]], filename: str) -> AIColumnMappingResponse:
        """
        AI column mapping function that analyzes table headers and creates mappings
        
        Args:
            tables: List of tables extracted from document
            filename: Original filename for context
            
        Returns:
            AIColumnMappingResponse with mapping results
        """
        try:
            file_type = determine_file_type(filename)
            desired_keys = get_desired_keys(file_type)
            
            logger.info(f"AI column mapping for file: {filename}, type: {file_type}, desired keys: {desired_keys}")
            
            # If AI client is not available, use fallback logic
            if not self.ai_foundry_client:
                return await self._fallback_column_mapping(tables, desired_keys, file_type, filename)
            
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
                            "sample_data": table[:15]  # First 15 rows for context
                        })
                else:
                    logger.warning(f"Table {i} is empty")
            
            # Create AI prompt for column mapping
            prompt = self._create_column_mapping_prompt(table_headers, desired_keys, file_type)
            
            # Call Azure AI Foundry
            response = await self._call_azure_ai_foundry(prompt)
            
            # Parse AI response
            column_mapping = self._parse_ai_response(response, desired_keys)
            
            # Create mapping result
            mapping_result = AIColumnMappingResponse(
                filename=filename,
                file_type=file_type,
                desired_keys=desired_keys,
                column_mapping=column_mapping,
                tables_analyzed=len(table_headers),
                mapping_confidence="high" if column_mapping else "low"
            )
            
            logger.info(f"AI column mapping completed: {mapping_result}")
            return mapping_result
            
        except Exception as e:
            logger.error(f"Error in AI column mapping: {e}")
            # Fallback to basic mapping
            return await self._fallback_column_mapping(tables, get_desired_keys(determine_file_type(filename)), 
                                                    determine_file_type(filename), filename)
    
    def _create_column_mapping_prompt(self, table_headers: List[Dict[str, Any]], 
                                    desired_keys: List[str], file_type: str) -> str:
        """Create prompt for AI column mapping"""
        prompt_tables = ""
        for table_info in table_headers:
            prompt_tables += f"\\nTable {table_info['table_index']}:\\n"
            prompt_tables += f"Headers: {', '.join(table_info['headers'])}\\n"
            if table_info['sample_data']:
                prompt_tables += "Sample data (first 15 rows):\\n"
                for i, row in enumerate(table_info['sample_data']):
                    prompt_tables += f"Row {i+1}: {row}\\n"
            prompt_tables += "\\n"

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
                "{desired_keys[0] if desired_keys else 'key1'}": "matching_header_or_null",
                "{desired_keys[1] if len(desired_keys) > 1 else 'key2'}": "matching_header_or_null"
            }}
        }}
    ]
}}

Provide only the JSON response, no additional text.
"""
        
        return prompt
    
    async def _call_azure_ai_foundry(self, prompt: str) -> str:
        """Call Azure AI Foundry API"""
        try:
            response = self.ai_foundry_client.complete(
                messages=[
                    SystemMessage(content="You are a data mapping expert. Provide only JSON responses."),
                    UserMessage(content=prompt)
                ],
                temperature=0.1,
                max_tokens=4096,
                model=settings.azure_ai_foundry_model
            )

            return_response_string = ""
            for update in response:
                if update.choices:
                    return_response_string += update.choices[0].delta.content or ""
            
            return return_response_string
            
        except Exception as e:
            logger.error(f"Error calling Azure AI Foundry: {e}")
            raise
    
    def _parse_ai_response(self, response: str, desired_keys: List[str]) -> Dict[str, Any]:
        """Parse AI response and extract column mappings"""
        try:
            # Clean response and extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parse JSON
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
    
    async def _fallback_column_mapping(self, tables: List[List[Dict[str, Any]]], desired_keys: List[str], 
                                     file_type: str, filename: str) -> AIColumnMappingResponse:
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
        
        return AIColumnMappingResponse(
            filename=filename,
            file_type=file_type,
            desired_keys=desired_keys,
            column_mapping=mappings,
            tables_analyzed=len(tables),
            mapping_confidence="low"
        )
    
    def standardize_table_data_with_mapping(self, table_data: List[Dict[str, Any]], 
                                          mapping_result: AIColumnMappingResponse) -> Dict[str, Any]:
        """Standardize table data using AI-generated column mapping"""
        try:
            file_type = mapping_result.file_type
            desired_keys = mapping_result.desired_keys
            column_mapping = mapping_result.column_mapping
            
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
                        for col_key, col_value in row.items():
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
            # Fallback to simple standardization
            return self._standardize_table_data_simple(table_data)
    
    def _standardize_table_data_simple(self, table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple fallback standardization"""
        return {
            "standardized_records": [],
            "total_records": 0
        }