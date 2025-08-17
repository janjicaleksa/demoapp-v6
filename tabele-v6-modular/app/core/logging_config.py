"""
Logging configuration for AI Processor Kupci Dobavljaci
"""

import logging
import logging.config
from .config import settings


def setup_logging():
    """Configure application logging"""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level,
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'app.log',
                'level': settings.log_level,
                'formatter': 'detailed'
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': settings.log_level,
                'handlers': ['console', 'file']
            },
            'azure': {  # Azure SDK logger
                'level': settings.azure_log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Get logger for the application
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)