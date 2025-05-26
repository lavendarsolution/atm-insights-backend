import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
from contextlib import asynccontextmanager
import time

# Internal imports
from config import settings
from database import init_db
from services import TelemetryService, CacheService, BackgroundTaskService
from api.routes.v1 import telemetry, dashboard, atms, health, metrics

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
cache_service: CacheService = None
telemetry_service: TelemetryService = None
background_service: BackgroundTaskService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with service initialization"""
    global cache_service, telemetry_service, background_service
    
    # Startup
    logger.info(f"🚀 Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize database
        await init_db()
        
        # Initialize services
        cache_service = CacheService()
        await cache_service.connect()
        
        telemetry_service = TelemetryService(cache_service)
        background_service = BackgroundTaskService(cache_service)
        
        # Start background tasks
        await background_service.start()
        
        # Set services in route modules
        telemetry.set_services(cache_service, telemetry_service, background_service)
        dashboard.set_services(cache_service, telemetry_service, background_service)
        atms.set_services(cache_service, telemetry_service, background_service)
        health.set_services(cache_service, telemetry_service, background_service)
        metrics.set_services(cache_service, telemetry_service, background_service)
        
        logger.info("✅ All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    # Shutdown
    logger.info("🔄 Shutting down application")
    
    if background_service:
        await background_service.stop()
    
    if cache_service:
        await cache_service.disconnect()

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None
    )

    # Add middleware for performance and security
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["https://atm-backend.lavendarsolution.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Include routers
    app.include_router(telemetry.router, prefix="/api/v1", tags=["telemetry"])
    app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
    app.include_router(atms.router, prefix="/api/v1", tags=["atms"])
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

    return app

# Create app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
