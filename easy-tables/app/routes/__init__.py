"""Routes for Easy Tables"""

from .main import router as main_router
from .clients import router as clients_router
from .upload import router as upload_router

__all__ = ["main_router", "clients_router", "upload_router"]