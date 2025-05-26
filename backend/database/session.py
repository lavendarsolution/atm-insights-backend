import logging

from config import settings
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Create database engine with optimized settings
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=20,  # Increased for scalability
    max_overflow=40,  # Allow more connections during peak
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,
    echo=settings.debug,
    future=True,  # Use SQLAlchemy 2.0 style
)

# Optimized session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Keep objects accessible after commit
)

# Base class for all models
Base = declarative_base()


# PostgreSQL-specific optimizations
@event.listens_for(Engine, "connect")
def set_postgresql_settings(dbapi_connection, connection_record):
    """Set PostgreSQL-specific settings for performance"""
    if settings.database_url.startswith("postgresql"):
        # Check if this is an asyncpg connection
        if hasattr(dbapi_connection, "_protocol"):
            # This is an asyncpg connection, skip sync cursor operations
            logger.debug("Skipping sync cursor operations for asyncpg connection")
            return

        try:
            # For psycopg2 connections, use the cursor normally
            with dbapi_connection.cursor() as cursor:
                # Optimize for time-series workloads
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("SET lock_timeout = '10s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")
                # Commit the settings
                dbapi_connection.commit()
        except Exception as e:
            logger.warning(f"Could not set PostgreSQL settings: {str(e)}")


def get_db() -> Session:
    """
    Dependency for getting database session
    Includes proper error handling and cleanup
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


async def init_db():
    """
    Initialize database with extensions and optimizations
    This runs only database setup, not table creation (handled by Alembic)
    """
    try:
        with engine.connect() as conn:
            conn.commit()

        logger.info("✅ Database initialization completed")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        # Don't raise the exception if it's just extension issues in development
        if settings.is_development:
            logger.warning(
                "⚠️ Continuing without TimescaleDB extensions in development mode"
            )
        else:
            raise
