from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class ATMStatus(BaseModel):
    """ATM status response model"""
    atm_id: str
    name: str
    region: str
    status: str
    last_update: Optional[str]
    temperature: Optional[float]
    cash_level: Optional[float]
    transactions_today: int
    error_code: Optional[str]
    error_message: Optional[str]