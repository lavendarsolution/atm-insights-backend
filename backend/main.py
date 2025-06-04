import asyncio
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from api.routes.v1 import (
    alerts,
    analytics,
    atms,
    auth,
    dashboard,
    health,
    metrics,
    telemetry,
    websocket,
)

# Internal imports
from config import settings
from database import init_db
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from services import BackgroundTaskService, CacheService, TelemetryService
from services.metrics_service import metrics_middleware, metrics_service
from services.websocket_service import ConnectionManager, set_connection_manager

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instances
cache_service: CacheService = None
telemetry_service: TelemetryService = None
background_service: BackgroundTaskService = None
connection_manager: ConnectionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with service initialization"""
    global cache_service, telemetry_service, background_service, connection_manager

    # Startup
    logger.info(f"🚀 Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(
        f"Prometheus metrics: {'enabled' if settings.prometheus_enabled else 'disabled'}"
    )

    try:
        # Initialize database
        init_db()

        # Initialize services
        cache_service = CacheService()
        await cache_service.connect()

        telemetry_service = TelemetryService(cache_service)
        background_service = BackgroundTaskService(cache_service)

        # Initialize WebSocket connection manager
        connection_manager = ConnectionManager(cache_service)
        set_connection_manager(connection_manager)

        # Start Redis subscriber for real-time updates
        await connection_manager.start_redis_subscriber()

        # Start background tasks
        await background_service.start()

        # Set services in route modules
        telemetry.set_services(cache_service, telemetry_service, background_service)
        dashboard.set_services(cache_service, telemetry_service, background_service)
        atms.set_services(cache_service, telemetry_service, background_service)
        health.set_services(cache_service, telemetry_service, background_service)
        metrics.set_services(cache_service, telemetry_service, background_service)

        # Start metrics collection background task
        if settings.prometheus_enabled:
            asyncio.create_task(start_metrics_collection())

        logger.info("✅ All services initialized successfully")
        logger.info("📡 WebSocket real-time service ready")
        logger.info("📊 Prometheus metrics collection started")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

    # Shutdown
    logger.info("🔄 Shutting down application")

    if connection_manager:
        await connection_manager.stop_redis_subscriber()

    if background_service:
        await background_service.stop()

    if cache_service:
        await cache_service.disconnect()


async def start_metrics_collection():
    """Background task to collect custom metrics"""
    while True:
        try:
            # Collect ATM status counts
            if telemetry_service:
                # This would need to be implemented in your telemetry service
                # status_counts = await telemetry_service.get_atm_status_counts()
                # metrics_service.update_atm_status_counts(status_counts)
                pass

            # Collect database connection metrics
            if cache_service:
                # redis_connections = cache_service.get_connection_count()
                # metrics_service.update_redis_connections(redis_connections)
                pass

            # Collect WebSocket connection metrics
            if connection_manager:
                dashboard_count = len(connection_manager.dashboard_connections)
                atm_detail_count = sum(
                    len(conns)
                    for conns in connection_manager.atm_detail_connections.values()
                )
                alerts_count = len(connection_manager.alerts_connections)

                metrics_service.update_websocket_connections(
                    "dashboard", dashboard_count
                )
                metrics_service.update_websocket_connections(
                    "atm_detail", atm_detail_count
                )
                metrics_service.update_websocket_connections("alerts", alerts_count)

            await asyncio.sleep(30)  # Collect metrics every 30 seconds

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            await asyncio.sleep(60)  # Wait longer on error


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # Add middleware for performance and security
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add metrics middleware if Prometheus is enabled
    if getattr(settings, "prometheus_enabled", True):
        app.middleware("http")(metrics_middleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=(["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Prometheus metrics endpoint
    @app.get("/metrics")
    async def get_prometheus_metrics():
        """Prometheus metrics endpoint"""
        if not getattr(settings, "prometheus_enabled", True):
            return {"error": "Metrics disabled"}

        return Response(
            content=metrics_service.get_metrics(), media_type=CONTENT_TYPE_LATEST
        )

    # Include routers
    app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
    app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
    app.include_router(telemetry.router, prefix="/api/v1", tags=["telemetry"])
    app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
    app.include_router(atms.router, prefix="/api/v1", tags=["atms"])
    app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
    app.include_router(health.router, prefix="", tags=["health"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

    # Include WebSocket routes
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])

    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
