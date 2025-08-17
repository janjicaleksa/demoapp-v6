"""
Main routes for the application
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    logger.info("Home page accessed")
    return templates.TemplateResponse("index.html", {"request": request})