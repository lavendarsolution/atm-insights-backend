from functools import lru_cache
from typing import Optional

from pydantic import ConfigDict, Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    env: str = Field(
        "development", description="Environment: development, staging, production"
    )
    debug: bool = Field(True, description="Enable debug mode")

    # API Configuration
    api_title: str = Field("ATM Insight API", description="API title")
    api_version: str = Field("1.0.0", description="API version")

    # Monitoring Configuration
    prometheus_enabled: bool = Field(True, description="Enable Prometheus metrics")
    metrics_enabled: bool = Field(
        True, description="Enable application metrics collection"
    )
    enable_metrics: bool = Field(True, description="Enable metrics endpoint")

    # Grafana Configuration
    grafana_enabled: bool = Field(True, description="Enable Grafana integration")
    grafana_user: str = Field("admin", description="Grafana admin username")
    grafana_password: str = Field("admin", description="Grafana admin password")

    # Database Configuration
    postgres_user: str = Field("postgres", description="PostgreSQL username")
    postgres_password: str = Field("postgres", description="PostgreSQL password")
    postgres_host: str = Field("localhost", description="PostgreSQL host")
    postgres_port: int = Field(5432, description="PostgreSQL port")
    postgres_db: str = Field("atm_insights", description="PostgreSQL database name")
    database_url: Optional[str] = Field(None, description="Complete database URL")

    # Redis Configuration
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: Optional[str] = Field("redis", description="Redis password")
    redis_db: int = Field(0, description="Redis database number")
    redis_url: Optional[str] = Field(None, description="Complete Redis URL")
    redis_max_connections: int = Field(50, description="Redis max connections")

    # Logging
    log_level: str = Field("INFO", description="Logging level")

    # JWT settings
    jwt_secret_key: str = Field(
        "your-jwt-secret-key", description="Secret key for JWT encoding/decoding"
    )

    # Google OAuth settings
    google_client_id: str = Field(
        "your-google-client-id", description="Google OAuth client ID"
    )

    # Notification Configuration
    resend_api_key: str = Field(
        "your-resend-api-key", description="Resend.com API key for email notifications"
    )
    notification_from_email: str = Field(
        "alerts@yourdomain.com", description="From email address for notifications"
    )
    notification_target_email: str = Field(
        "lavendarsolution@gmail.com",
        description="Target email address for all notifications",
    )

    # Telegram Configuration
    telegram_bot_token: str = Field(
        "your-telegram-bot-token", description="Telegram bot token for notifications"
    )
    telegram_chat_id: str = Field(
        "your-telegram-chat-id",
        description="Default Telegram chat ID for notifications",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v

        values = info.data if hasattr(info, "data") else {}
        return str(
            PostgresDsn.build(
                scheme="postgresql",
                username=values.get("postgres_user"),
                password=values.get("postgres_password"),
                host=values.get("postgres_host"),
                port=values.get("postgres_port"),
                path=values.get("postgres_db"),
            )
        )

    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str):
            return v

        values = info.data if hasattr(info, "data") else {}
        password_part = (
            f":{values.get('redis_password')}@" if values.get("redis_password") else ""
        )
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

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
