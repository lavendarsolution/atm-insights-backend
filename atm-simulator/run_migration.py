"""
Simple ATM registration script
Creates 612 random ATMs via API
"""

import asyncio
import random

import aiohttp
from config import ATM_MANUFACTURERS, ATM_MODELS, REGIONS


async def create_atms():
    """Create 612 random ATMs"""
    api_url = "http://localhost:8000"

    async with aiohttp.ClientSession() as session:
        for i in range(1, 613):
            region = random.choice(REGIONS)
            atm_id = f"ATM-{i:03d}"

            atm_data = {
                "atm_id": atm_id,
                "name": f"ATM-{region}-{i:03d}",
                "location_address": f"Address {i}, {region} Region",
                "model": random.choice(ATM_MODELS),
                "manufacturer": random.choice(ATM_MANUFACTURERS),
                "status": "active",
            }

            try:
                async with session.post(
                    f"{api_url}/api/v1/atms", json=atm_data
                ) as response:
                    if response.status in [200, 201]:
                        print(f"✅ Created {atm_id}")
                    else:
                        print(f"❌ Failed {atm_id}")
            except Exception as e:
                print(f"❌ Error {atm_id}: {e}")


if __name__ == "__main__":
    print("Creating 612 ATMs...")
    asyncio.run(create_atms())
    print("Done!")
