import os
from typing import Optional, List
from pydantic import PostgresDsn, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings optimized for scalability"""
    
    # Environment
    env: str = Field("development", description="Environment: development, staging, production")
    debug: bool = Field(True, description="Enable debug mode")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")
    api_title: str = Field("ATM Insight API", description="API title")
    api_version: str = Field("1.0.0", description="API version")
    
    # Security
    secret_key: str = Field("your-super-secret-key-change-in-production", description="Secret key for JWT")
    access_token_expire_minutes: int = Field(30, description="JWT token expiration in minutes")
    
    # Database Configuration
    postgres_user: str = Field("postgres", description="PostgreSQL username")
    postgres_password: str = Field("postgres", description="PostgreSQL password")
    postgres_host: str = Field("localhost", description="PostgreSQL host")
    postgres_port: int = Field(5432, description="PostgreSQL port")
    postgres_db: str = Field("atm_insight", description="PostgreSQL database name")
    database_url: Optional[str] = Field(None, description="Complete database URL")
    
    # Database Pool Configuration (for scalability)
    db_pool_size: int = Field(20, description="Database connection pool size")
    db_max_overflow: int = Field(40, description="Database max overflow connections")
    db_pool_timeout: int = Field(30, description="Database pool timeout in seconds")
    db_pool_recycle: int = Field(3600, description="Database connection recycle time in seconds")
    
    # Redis Configuration
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: Optional[str] = Field("redis", description="Redis password")
    redis_db: int = Field(0, description="Redis database number")
    redis_url: Optional[str] = Field(None, description="Complete Redis URL")
    
    # Redis Pool Configuration
    redis_pool_size: int = Field(20, description="Redis connection pool size")
    redis_max_connections: int = Field(50, description="Redis max connections")
    
    # Application Performance Configuration
    max_connections_count: int = Field(100, description="Max concurrent connections")
    min_connections_count: int = Field(10, description="Min concurrent connections")
    
    # Background Task Configuration
    background_task_workers: int = Field(4, description="Number of background task workers")
    telemetry_batch_size: int = Field(1000, description="Telemetry batch processing size")
    telemetry_batch_timeout: int = Field(5, description="Telemetry batch timeout in seconds")
    
    # Caching Configuration
    cache_ttl_seconds: int = Field(300, description="Default cache TTL in seconds")
    dashboard_cache_ttl: int = Field(60, description="Dashboard data cache TTL in seconds")
    
    # Rate Limiting
    rate_limit_requests: int = Field(1000, description="Rate limit requests per minute")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    
    # Monitoring and Logging
    log_level: str = Field("INFO", description="Logging level")
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    metrics_port: int = Field(9090, description="Metrics server port")
    
    # TimescaleDB Configuration
    timescale_chunk_interval: str = Field("1 hour", description="TimescaleDB chunk interval")
    timescale_compression_after: str = Field("1 day", description="Enable compression after")
    timescale_retention_period: str = Field("2 years", description="Data retention period")
    
    # Alerting Configuration
    alert_cooldown_default: int = Field(10, description="Default alert cooldown in minutes")
    max_alerts_per_atm_per_hour: int = Field(5, description="Max alerts per ATM per hour")
    
    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v
        
        values = info.data if hasattr(info, 'data') else {}
        return str(PostgresDsn.build(
            scheme="postgresql",
            username=values.get("postgres_user"),
            password=values.get("postgres_password"),
            host=values.get("postgres_host"),
            port=values.get("postgres_port"),
            path=values.get("postgres_db"),
        ))
    
    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v
        
        values = info.data if hasattr(info, 'data') else {}
        password_part = f":{values.get('redis_password')}@" if values.get('redis_password') else ""
        return f"redis://{password_part}{values.get('redis_host')}:{values.get('redis_port')}/{values.get('redis_db')}"
    
    @property
    def is_development(self) -> bool:
        return self.env == "development"
    
    @property
    def is_production(self) -> bool:
        return self.env == "production"
    
    @property
    def is_staging(self) -> bool:
        return self.env == "staging"
    
    @property
    def async_database_url(self) -> str:
        """Get async database URL for async operations"""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()