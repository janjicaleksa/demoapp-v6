"""
Routes module for AI Processor Kupci Dobavljaci
"""

from .clients import router as clients_router
from .upload import router as upload_router
from .main import router as main_router

__all__ = ['clients_router', 'upload_router', 'main_router']