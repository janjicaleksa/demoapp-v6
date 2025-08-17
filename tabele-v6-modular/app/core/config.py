"""
Configuration module for AI Processor Kupci Dobavljaci
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "AI Processor Kupci Dobavljaci"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Azure Document Intelligence configuration
    azure_document_intelligence_endpoint: Optional[str] = None
    azure_document_intelligence_key: Optional[str] = None
    
    # Azure Blob Storage configuration
    azure_blob_storage_connection_string: Optional[str] = None
    azure_blob_storage_container_name: str = "documents"
    
    # Azure AI Foundry configuration for AI column mapping
    azure_ai_foundry_connection: Optional[str] = None
    azure_ai_foundry_key: Optional[str] = None
    azure_ai_foundry_model: str = "gpt-4"
    
    # CORS settings
    cors_origins: list = ["*"]  # Should be restricted in production
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # File processing settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: set = {'.xlsx', '.csv', '.doc', '.docx', '.pdf'}
    
    # Storage settings
    local_storage_path: Path = Path("clients")
    
    # Logging settings
    log_level: str = "INFO"
    azure_log_level: str = "ERROR"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()