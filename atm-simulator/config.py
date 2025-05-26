import os
from typing import Optional

# Create global config instance
SIMULATOR_CONFIG = {
    "num_atms": 500,
    "api_base_url": "http://localhost:8000",
    "send_interval_seconds": 30,
    "batch_size": 50,
}

# Rest of your existing config...
REGIONS = ["LEGION-1", "LEGION-2", "LEGION-3", "LEGION-4", "LEGION-5"]
ATM_MODELS = ["SecureMax Pro", "CashFlow Elite", "MoneyTech X1", "SafeBank 2000"]

ERROR_CODES = {
    "E001": "Card reader malfunction",
    "E002": "Cash dispenser jam",
    "E003": "Network connectivity issue",
    "E004": "Low cash warning",
    "E005": "Temperature sensor error",
    "E006": "Receipt printer error",
    "E007": "Keypad malfunction",
}


def generate_atm_config(atm_id: str) -> dict:
    import random
    from datetime import datetime, timedelta

    return {
        "atm_id": atm_id,
        "region": atm_id.split("-")[1] if "-" in atm_id else random.choice(REGIONS),
        "model": random.choice(ATM_MODELS),
        "health_factor": random.uniform(0.75, 1.0),
        "cash_capacity": random.randint(50000, 200000),
        "last_maintenance": datetime.now() - timedelta(days=random.randint(1, 90)),
        "location_type": random.choice(
            ["mall", "bank_branch", "airport", "convenience_store", "hospital"]
        ),
    }
