"""
Client management routes for Easy Tables
"""

import logging
from fastapi import APIRouter, Form, HTTPException
from ..core.utils import sanitize_client_name, create_directory_structure
from ..models.schemas import ClientResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clients", tags=["clients"])


@router.post("/", response_model=ClientResponse)
async def create_client(client_name: str = Form(...)):
    """Create a new client with parameter-based directory structure"""
    try:
        logger.info(f"Creating client: {client_name}")
        
        # Create safe client slug
        client_slug = sanitize_client_name(client_name)
        
        if not client_slug:
            raise HTTPException(status_code=400, detail="Invalid client name")
        
        # Create parameter-based directory structure
        structure = create_directory_structure(client_slug)
        
        logger.info(f"Created client structure for: {client_slug}")
        
        return ClientResponse(
            message=f"Client '{client_name}' created successfully",
            client_slug=client_slug,
            structure=structure
        )
        
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail=str(e))