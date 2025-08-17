"""
Core module for AI Processor Kupci Dobavljaci
"""

from .config import settings, get_settings
from .logging_config import setup_logging, get_logger

__all__ = ['settings', 'get_settings', 'setup_logging', 'get_logger']