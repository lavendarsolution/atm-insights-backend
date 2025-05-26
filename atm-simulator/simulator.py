import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
from config import ERROR_CODES, REGIONS, SIMULATOR_CONFIG, generate_atm_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ATMSimulator:
    """Simple but realistic ATM telemetry simulator"""

    def __init__(self):
        self.config = SIMULATOR_CONFIG
        self.atms = self._generate_atm_fleet()
        self.running = False
        self.stats = {"total_sent": 0, "successful": 0, "failed": 0, "start_time": None}

    def _generate_atm_fleet(self) -> List[Dict]:
        """Generate fleet of ATMs with realistic distribution"""
        atms = []

        for i in range(1, self.config["num_atms"] + 1):
            region = random.choice(REGIONS)
            atm_id = f"ATM-{region}-{i:03d}"
            atm_config = generate_atm_config(atm_id)
            atms.append(atm_config)

        logger.info(f"🏪 Generated {len(atms)} ATM configurations")
        return atms

    def _generate_realistic_telemetry(self, atm: Dict) -> Dict:
        """Generate realistic telemetry based on ATM characteristics"""
        now = datetime.utcnow()
        health = atm["health_factor"]

        # Determine status based on health and time
        if health > 0.95:
            status = "online"
        elif health > 0.85:
            status = "online" if random.random() > 0.05 else "error"
        elif health > 0.75:
            status = random.choices(
                ["online", "error", "offline"], weights=[0.7, 0.25, 0.05]
            )[0]
        else:
            status = random.choices(
                ["online", "error", "offline", "maintenance"],
                weights=[0.5, 0.3, 0.15, 0.05],
            )[0]

        # Temperature simulation (affected by health and location type)
        base_temp = 22.0
        if atm["location_type"] == "mall":
            base_temp = 20.0  # Air conditioned
        elif atm["location_type"] == "convenience_store":
            base_temp = 24.0  # Less controlled

        temperature = (
            base_temp + random.uniform(-2, 3) + (1 - health) * random.uniform(0, 8)
        )

        # Cash level simulation
        days_since_maintenance = (now - atm["last_maintenance"]).days
        daily_usage = random.uniform(5, 15)  # % per day
        cash_level = max(
            5, 100 - (days_since_maintenance * daily_usage) + random.uniform(-10, 5)
        )

        # Transaction simulation based on time and location
        hour = now.hour
        transaction_multiplier = 1.0

        if atm["location_type"] == "mall":
            if 10 <= hour <= 22:
                transaction_multiplier = 2.5
        elif atm["location_type"] == "bank_branch":
            if 9 <= hour <= 17:
                transaction_multiplier = 3.0
        elif atm["location_type"] == "airport":
            transaction_multiplier = 1.8  # Steady traffic

        # Approximate Poisson distribution using exponential distribution
        lambda_param = 8 * transaction_multiplier * health
        transactions = int(
            random.expovariate(1.0 / lambda_param) if lambda_param > 0 else 0
        )
        failed_transactions = int(transactions * random.uniform(0, 0.08) * (1 - health))

        # System metrics
        cpu_usage = random.uniform(15, 35) + (1 - health) * random.uniform(0, 40)
        memory_usage = random.uniform(45, 75) + (1 - health) * random.uniform(0, 20)

        # Build telemetry
        telemetry = {
            "atm_id": atm["atm_id"],
            "timestamp": now.isoformat(),
            "status": status,
            "temperature": round(temperature, 1),
            "cash_level": round(cash_level, 1),
            "transactions_count": transactions,
            "failed_transactions": failed_transactions,
            "cpu_usage": round(cpu_usage, 1),
            "memory_usage": round(memory_usage, 1),
        }

        # Add error details for problematic ATMs
        if status == "error" or (health < 0.9 and random.random() < 0.3):
            error_code = random.choice(list(ERROR_CODES.keys()))
            telemetry.update(
                {"error_code": error_code, "error_message": ERROR_CODES[error_code]}
            )

        return telemetry

    async def _send_telemetry_batch(
        self, session: aiohttp.ClientSession, atm_batch: List[Dict]
    ):
        """Send telemetry for a batch of ATMs"""
        tasks = []

        for atm in atm_batch:
            telemetry = self._generate_realistic_telemetry(atm)
            task = self._send_single_telemetry(session, telemetry)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful

        self.stats["total_sent"] += len(results)
        self.stats["successful"] += successful
        self.stats["failed"] += failed

        return successful, failed

    async def _send_single_telemetry(
        self, session: aiohttp.ClientSession, telemetry: Dict
    ) -> bool:
        """Send telemetry for a single ATM"""
        try:
            async with session.post(
                f"{self.config['api_base_url']}/api/v1/telemetry",
                json=telemetry,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(f"❌ {telemetry['atm_id']}: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.debug(f"❌ {telemetry['atm_id']}: {str(e)}")
            return False

    async def _simulation_cycle(self):
        """Run one complete simulation cycle"""
        batch_size = self.config["batch_size"]
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            # Process ATMs in batches
            for i in range(0, len(self.atms), batch_size):
                batch = self.atms[i : i + batch_size]
                successful, failed = await self._send_telemetry_batch(session, batch)

                # Small delay between batches to avoid overwhelming the server
                if i + batch_size < len(self.atms):
                    await asyncio.sleep(0.1)

        duration = time.time() - start_time
        success_rate = (
            (self.stats["successful"] / self.stats["total_sent"]) * 100
            if self.stats["total_sent"] > 0
            else 0
        )

        logger.info(
            f"📊 Cycle completed: {len(self.atms)} ATMs in {duration:.2f}s | Success rate: {success_rate:.1f}%"
        )

    def _print_stats(self):
        """Print simulation statistics"""
        if self.stats["start_time"] and self.stats["total_sent"] > 0:
            runtime = time.time() - self.stats["start_time"]
            rate = self.stats["total_sent"] / runtime if runtime > 0 else 0

            print(f"\n📈 SIMULATION STATS:")
            print(f"   Runtime: {runtime:.0f}s")
            print(f"   Total sent: {self.stats['total_sent']}")
            print(f"   Successful: {self.stats['successful']}")
            print(f"   Failed: {self.stats['failed']}")
            print(f"   Rate: {rate:.1f} msg/sec")
            print(
                f"   Success rate: {(self.stats['successful']/self.stats['total_sent']*100):.1f}%"
            )

    async def start_simulation(self):
        """Start the continuous ATM simulation"""
        self.running = True
        self.stats["start_time"] = time.time()
        interval = self.config["send_interval_seconds"]

        logger.info(f"🚀 Starting ATM simulation:")
        logger.info(f"   ATMs: {len(self.atms)}")
        logger.info(f"   Interval: {interval}s")
        logger.info(f"   Target API: {self.config['api_base_url']}")

        try:
            while self.running:
                await self._simulation_cycle()

                # Print stats every 10 cycles
                if (self.stats["total_sent"] // len(self.atms)) % 10 == 0:
                    self._print_stats()

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("🛑 Simulation stopped by user")
        finally:
            self.running = False
            self._print_stats()

    def stop(self):
        """Stop the simulation"""
        self.running = False
