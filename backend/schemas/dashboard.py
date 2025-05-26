from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class DashboardStats(BaseModel):
    """Dashboard statistics response model"""
    total_atms: int
    online_atms: int
    offline_atms: int
    error_atms: int
    total_transactions_today: int
    avg_cash_level: float
    critical_alerts: int
    last_updated: str