import os
from typing import Optional

# Simulator configuration from environment variables
class SimulatorConfig:
    def __init__(self):
        self.num_atms = int(os.getenv("NUM_ATMS", "500"))
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.send_interval_seconds = int(os.getenv("SEND_INTERVAL", "30"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "50"))
        
        # Validation
        if self.num_atms <= 0:
            raise ValueError("NUM_ATMS must be positive")
        if self.send_interval_seconds <= 0:
            raise ValueError("SEND_INTERVAL must be positive")
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")

# Create global config instance
config = SimulatorConfig()

# Rest of your existing config...
REGIONS = ["NYC", "LA", "CHI", "MIA", "SEA", "DAL", "BOS", "SF", "LV", "ATL"]
ATM_MODELS = ["SecureMax Pro", "CashFlow Elite", "MoneyTech X1", "SafeBank 2000"]

ERROR_CODES = {
    "E001": "Card reader malfunction",
    "E002": "Cash dispenser jam", 
    "E003": "Network connectivity issue",
    "E004": "Low cash warning",
    "E005": "Temperature sensor error",
    "E006": "Receipt printer error",
    "E007": "Keypad malfunction"
}

def generate_atm_config(atm_id: str) -> dict:
    import random
    from datetime import datetime, timedelta
    
    return {
        "atm_id": atm_id,
        "region": atm_id.split('-')[1] if '-' in atm_id else random.choice(REGIONS),
        "model": random.choice(ATM_MODELS),
        "health_factor": random.uniform(0.75, 1.0),
        "cash_capacity": random.randint(50000, 200000),
        "last_maintenance": datetime.now() - timedelta(days=random.randint(1, 90)),
        "location_type": random.choice(["mall", "bank_branch", "airport", "convenience_store", "hospital"])
    }