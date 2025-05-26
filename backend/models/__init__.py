"""Database models package"""

# Import the base here to ensure all models are registered
from database.session import Base

from .alert import Alert, AlertRule
from .atm import ATM
from .atm_telemetry import ATMTelemetry
from .user import User

__all__ = ["ATM", "ATMTelemetry", "Alert", "AlertRule", "User", "Base"]
