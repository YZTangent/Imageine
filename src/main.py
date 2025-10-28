"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.routes import router, init_models
from src.utils.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load config
try:
    config = Config("config/default.yaml")
except FileNotFoundError:
    logger.error("Config file not found. Please ensure config/default.yaml exists.")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="Imageine API",
    description="General-purpose image composition API powered by Stable Diffusion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("=" * 60)
    logger.info("Starting Imageine API...")
    logger.info("=" * 60)

    try:
        init_models(config)
        logger.info("=" * 60)
        logger.info("✅ Imageine API is ready!")
        logger.info(f"   Listening on http://{config.api.host}:{config.api.port}")
        logger.info(f"   API docs: http://{config.api.host}:{config.api.port}/docs")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        logger.error("=" * 60)
        logger.error("Startup failed. Please check:")
        logger.error("  1. Models are downloaded (run: python scripts/download_models.py)")
        logger.error("  2. CUDA is available if using GPU")
        logger.error("  3. Sufficient disk space and memory")
        logger.error("=" * 60)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Imageine API...")


if __name__ == "__main__":
    import uvicorn

    logger.info("Running in development mode with auto-reload")

    uvicorn.run(
        "src.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level="info"
    )
