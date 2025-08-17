"""
Storage service for handling Azure Blob Storage and local storage operations
"""

from pathlib import Path
from typing import Dict, Optional
import aiofiles
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.utils import get_blob_name
from ..models.schemas import ClientStructure, PeriodStructure

logger = get_logger(__name__)


class StorageService:
    """Service for handling storage operations"""
    
    def __init__(self):
        self.blob_service_client = None
        self.use_local_storage = True
        self.base_dir = settings.local_storage_path
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage clients"""
        # Initialize Azure Blob Storage if configured
        if settings.azure_blob_storage_connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    settings.azure_blob_storage_connection_string
                )
                
                # Ensure container exists
                container_client = self.blob_service_client.get_container_client(
                    settings.azure_blob_storage_container_name
                )
                if not container_client.exists():
                    container_client.create_container()
                    logger.info(f"Created container: {settings.azure_blob_storage_container_name}")
                
                self.use_local_storage = False
                logger.info("Azure Blob Storage client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Storage client: {e}")
                self.blob_service_client = None
                self.use_local_storage = True
        else:
            logger.warning("Azure Blob Storage not configured, using local storage fallback")
        
        # Ensure local storage directory exists
        if self.use_local_storage:
            self.base_dir.mkdir(exist_ok=True)
            logger.info(f"Using local storage at: {self.base_dir}")
    
    async def upload_to_blob_storage(self, blob_name: str, content: bytes, 
                                   content_type: str = None) -> str:
        """
        Upload content to Azure Blob Storage
        
        Args:
            blob_name: Name of the blob
            content: Content to upload
            content_type: MIME type of the content
            
        Returns:
            Path to the uploaded blob
            
        Raises:
            Exception: If Azure Blob Storage is not configured
        """
        if not self.blob_service_client:
            raise Exception("Azure Blob Storage not configured")
        
        logger.info(f"Uploading to Azure Blob Storage: container={settings.azure_blob_storage_container_name}, blob={blob_name}")
        
        blob_client = self.blob_service_client.get_blob_client(
            container=settings.azure_blob_storage_container_name, 
            blob=blob_name
        )
        
        content_settings = ContentSettings(content_type=content_type) if content_type else None
        blob_client.upload_blob(content, overwrite=True, content_settings=content_settings)
        
        result_path = f"azure://{settings.azure_blob_storage_container_name}/{blob_name}"
        logger.info(f"Successfully uploaded to: {result_path}")
        return result_path
    
    async def download_from_blob_storage(self, blob_name: str) -> bytes:
        """
        Download content from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Content of the blob
            
        Raises:
            Exception: If Azure Blob Storage is not configured
        """
        if not self.blob_service_client:
            raise Exception("Azure Blob Storage not configured")
        
        blob_client = self.blob_service_client.get_blob_client(
            container=settings.azure_blob_storage_container_name, 
            blob=blob_name
        )
        download_stream = blob_client.download_blob()
        return download_stream.readall()
    
    async def save_file_to_storage(self, file_path: str, content: bytes) -> str:
        """
        Save file to storage (Azure Blob Storage or local fallback)
        
        Args:
            file_path: Path where to save the file
            content: File content
            
        Returns:
            Path to the saved file
        """
        if self.use_local_storage:
            # Local storage fallback
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, 'wb') as f:
                await f.write(content)
            logger.debug(f"Saved file locally: {path}")
            return str(path)
        else:
            # Azure Blob Storage
            return await self.upload_to_blob_storage(file_path, content)
    
    async def read_file_from_storage(self, file_path: str) -> bytes:
        """
        Read file from storage (Azure Blob Storage or local fallback)
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content
        """
        if self.use_local_storage:
            # Local storage fallback
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            logger.debug(f"Read file locally: {file_path}")
            return content
        else:
            # Azure Blob Storage
            return await self.download_from_blob_storage(file_path)
    
    def create_client_structure(self, client_slug: str) -> ClientStructure:
        """
        Create the folder structure for a client
        
        Args:
            client_slug: Normalized client name
            
        Returns:
            ClientStructure with paths
        """
        if self.use_local_storage:
            # Local storage
            client_path = self.base_dir / client_slug
            raw_path = client_path / "raw"
            extracted_path = client_path / "extracted"
            processed_path = client_path / "processed"
            
            # Create directories
            raw_path.mkdir(parents=True, exist_ok=True)
            extracted_path.mkdir(parents=True, exist_ok=True)
            processed_path.mkdir(parents=True, exist_ok=True)
            
            return ClientStructure(
                client_slug=client_slug,
                raw_path=str(raw_path),
                extracted_path=str(extracted_path),
                processed_path=str(processed_path)
            )
        else:
            # Azure Blob Storage - virtual folders
            container_name = settings.azure_blob_storage_container_name
            return ClientStructure(
                client_slug=client_slug,
                raw_path=f"azure://{container_name}/{client_slug}/raw",
                extracted_path=f"azure://{container_name}/{client_slug}/extracted",
                processed_path=f"azure://{container_name}/{client_slug}/processed"
            )
    
    def create_period_structure(self, client_slug: str, period_date: str) -> PeriodStructure:
        """
        Create period-specific folders
        
        Args:
            client_slug: Normalized client name
            period_date: Period date string
            
        Returns:
            PeriodStructure with paths
        """
        if self.use_local_storage:
            # Local storage
            client_path = self.base_dir / client_slug
            raw_period_path = client_path / "raw" / period_date
            extracted_period_path = client_path / "extracted" / period_date
            processed_period_path = client_path / "processed" / period_date
            
            raw_period_path.mkdir(parents=True, exist_ok=True)
            extracted_period_path.mkdir(parents=True, exist_ok=True)
            processed_period_path.mkdir(parents=True, exist_ok=True)
            
            return PeriodStructure(
                raw_period_path=str(raw_period_path),
                extracted_period_path=str(extracted_period_path),
                processed_period_path=str(processed_period_path)
            )
        else:
            # Azure Blob Storage - virtual folders
            container_name = settings.azure_blob_storage_container_name
            return PeriodStructure(
                raw_period_path=f"azure://{container_name}/{client_slug}/raw/{period_date}",
                extracted_period_path=f"azure://{container_name}/{client_slug}/extracted/{period_date}",
                processed_period_path=f"azure://{container_name}/{client_slug}/processed/{period_date}"
            )
    
    async def client_exists(self, client_slug: str) -> bool:
        """
        Check if client exists in storage
        
        Args:
            client_slug: Normalized client name
            
        Returns:
            True if client exists, False otherwise
        """
        if self.use_local_storage:
            client_path = self.base_dir / client_slug
            return client_path.exists()
        else:
            # Check if client exists in Azure Blob Storage
            container_client = self.blob_service_client.get_container_client(
                settings.azure_blob_storage_container_name
            )
            blobs = container_client.list_blobs(name_starts_with=f"{client_slug}/")
            return any(blob.name.startswith(f"{client_slug}/") for blob in blobs)
    
    def get_storage_path(self, client_slug: str, file_type: str, period_date: str = None,
                        is_extracted: bool = False, is_processed: bool = False) -> str:
        """
        Get appropriate storage path for a file
        
        Args:
            client_slug: Normalized client name
            file_type: Type/name of the file
            period_date: Period date (optional)
            is_extracted: Whether this is extracted data
            is_processed: Whether this is processed data
            
        Returns:
            Storage path for the file
        """
        if self.use_local_storage:
            # Local storage path
            folder_type = "processed" if is_processed else ("extracted" if is_extracted else "raw")
            if period_date:
                return str(self.base_dir / client_slug / folder_type / period_date / file_type)
            else:
                return str(self.base_dir / client_slug / folder_type / file_type)
        else:
            # Azure Blob Storage path
            return get_blob_name(client_slug, file_type, period_date, is_extracted, is_processed)