"""
Client management routes
"""

from fastapi import APIRouter, HTTPException, Form
from ..core.logging_config import get_logger
from ..core.utils import normalize_client_name
from ..models.schemas import ClientResponse, ErrorResponse
from ..services.storage_service import StorageService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/clients", tags=["clients"])

# Initialize storage service
storage_service = StorageService()


@router.post("", response_model=ClientResponse)
async def create_client(client_name: str = Form(...)):
    """Create a new client and folder structure"""
    try:
        logger.info(f"Creating client with name: {client_name}")
        client_slug = normalize_client_name(client_name)
        logger.info(f"Normalized client slug: {client_slug}")
        
        if not client_slug:
            raise HTTPException(status_code=400, detail="Invalid client name")
        
        # Check if client already exists
        if await storage_service.client_exists(client_slug):
            logger.info(f"Client already exists: {client_slug}")
            return ClientResponse(
                message="Client already exists", 
                client_slug=client_slug
            )
        
        # Create folder structure
        structure = storage_service.create_client_structure(client_slug)
        logger.info(f"Created client structure: {structure}")
        
        return ClientResponse(
            message="Client created successfully",
            client_slug=client_slug,
            structure=structure.dict()
        )
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{client_slug}/structure")
async def get_client_structure(client_slug: str):
    """Get client folder structure"""
    try:
        # Check if client exists
        if not await storage_service.client_exists(client_slug):
            raise HTTPException(status_code=404, detail="Client not found")
        
        structure = storage_service.create_client_structure(client_slug)
        
        return structure.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))