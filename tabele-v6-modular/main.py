"""
Main application file for AI Processor Kupci Dobavljaci (Modular Version)
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.routes import clients_router, upload_router, main_router

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered processor for financial documents (Kupci/Dobavljaci)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(main_router)
app.include_router(clients_router)
app.include_router(upload_router)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Azure Document Intelligence configured: {bool(settings.azure_document_intelligence_endpoint)}")
    logger.info(f"Azure Blob Storage configured: {bool(settings.azure_blob_storage_connection_string)}")
    logger.info(f"Azure AI Foundry configured: {bool(settings.azure_ai_foundry_connection)}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info(f"Shutting down {settings.app_name}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "azure_document_intelligence": bool(settings.azure_document_intelligence_endpoint),
        "azure_blob_storage": bool(settings.azure_blob_storage_connection_string),
        "azure_ai_foundry": bool(settings.azure_ai_foundry_connection)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )