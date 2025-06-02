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
    """Optimized ATM telemetry simulator for essential data only"""

    def __init__(self):
        self.config = SIMULATOR_CONFIG
        self.atms = self._generate_atm_fleet()
        self.running = False
        self.stats = {"total_sent": 0, "successful": 0, "failed": 0, "start_time": None}

    def _generate_atm_fleet(self) -> List[Dict]:
        """Generate fleet of ATMs with realistic distribution"""
        atms = []

        for i in range(1, self.config["num_atms"] + 1):
            atm_id = f"ATM-{i:03d}"
            atm_config = generate_atm_config(atm_id)
            atms.append(atm_config)

        logger.info(f"🏪 Generated {len(atms)} ATM configurations")
        return atms

    def _generate_optimized_telemetry(self, atm: Dict) -> Dict:
        """Generate optimized telemetry with essential fields only"""
        now = datetime.now()
        health = atm["health_factor"]

        # Calculate error probability: 1 error per hour per ATM
        # With 5 cycles per minute (12s interval), that's 300 cycles per hour
        # So error probability = 1/300 = 0.00333 per cycle per ATM
        error_probability = 1.0 / 300.0  # 1 error per hour per ATM

        # Most ATMs should be online with very low error rates
        if random.random() < error_probability:
            # This ATM will have an error this cycle
            status = "error"
        else:
            # Normal operation - mostly online
            if health > 0.90:
                status = "online"
            elif health > 0.80:
                status = (
                    "online" if random.random() > 0.005 else "maintenance"
                )  # Very low maintenance
            elif health > 0.70:
                status = random.choices(
                    ["online", "maintenance"], weights=[0.98, 0.02]  # Mostly online
                )[0]
            else:
                status = random.choices(
                    ["online", "maintenance", "offline"],
                    weights=[0.95, 0.04, 0.01],  # Still mostly online
                )[0]

        # Cash level simulation (most critical metric)
        days_since_maintenance = (now - atm["last_maintenance"]).days
        daily_usage = random.uniform(5, 15)  # % per day
        cash_level = max(
            5, 100 - (days_since_maintenance * daily_usage) + random.uniform(-10, 5)
        )

        # Temperature simulation (affected by health and location type)
        base_temp = 22.0
        if atm["location_type"] == "mall":
            base_temp = 20.0  # Air conditioned
        elif atm["location_type"] == "convenience_store":
            base_temp = 24.0  # Less controlled

        temperature = (
            base_temp
            + random.uniform(-2, 3)
            + (1 - health) * random.uniform(0, 2)  # Reduced temperature variance
        )

        # System performance metrics - normal ranges
        cpu_usage = random.uniform(15, 30) + (1 - health) * random.uniform(
            0, 10
        )  # Normal CPU usage
        memory_usage = random.uniform(45, 65) + (1 - health) * random.uniform(
            0, 10
        )  # Normal memory usage
        disk_usage = random.uniform(20, 50) + (1 - health) * random.uniform(
            0, 10
        )  # Normal disk usage

        # Network status - mostly connected
        network_status = "connected"
        network_latency = random.randint(50, 200)  # Normal latency range

        # Only disconnect network if status is error
        if status == "error":
            if random.random() < 0.3:  # 30% of errors are network related
                network_status = "disconnected"
                network_latency = None

        # Uptime simulation
        uptime_base = 24 * 3600 * random.randint(1, 30)  # 1-30 days
        uptime = int(uptime_base * health + random.uniform(-3600, 3600))

        # Build optimized telemetry with essential fields only
        telemetry = {
            "atm_id": atm["atm_id"],
            "timestamp": now.isoformat() + "Z",
            "status": status,
            "uptime_seconds": max(0, uptime),
            "cash_level_percent": round(cash_level, 1),
            "temperature_celsius": round(temperature, 1),
            "cpu_usage_percent": round(min(100, max(0, cpu_usage)), 1),
            "memory_usage_percent": round(min(100, max(0, memory_usage)), 1),
            "disk_usage_percent": round(min(100, max(0, disk_usage)), 1),
            "network_status": network_status,
        }

        # Add network latency only if connected
        if network_latency is not None:
            telemetry["network_latency_ms"] = network_latency

        # Add error details only when status is error
        if status == "error":
            error_code = random.choice(list(ERROR_CODES.keys()))
            telemetry.update(
                {"error_code": error_code, "error_message": ERROR_CODES[error_code]}
            )

        # Very rarely add critical conditions (separate from errors)
        if random.random() < 0.001:  # Very rare critical conditions
            critical_condition = random.choice(["low_cash", "high_temp", "high_cpu"])

            if critical_condition == "low_cash":
                telemetry["cash_level_percent"] = random.uniform(5, 15)
            elif critical_condition == "high_temp":
                telemetry["temperature_celsius"] = random.uniform(35, 40)
            elif critical_condition == "high_cpu":
                telemetry["cpu_usage_percent"] = random.uniform(80, 95)

        return telemetry

    async def _send_single_telemetry(
        self, session: aiohttp.ClientSession, telemetry: Dict
    ) -> bool:
        """Send telemetry for a single ATM with improved error handling"""
        try:
            async with session.post(
                f"{self.config['api_base_url']}/api/v1/telemetry",
                json=telemetry,
                timeout=aiohttp.ClientTimeout(
                    total=15
                ),  # Increased timeout from 10 to 15 seconds
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(f"❌ {telemetry['atm_id']}: HTTP {response.status}")
                    return False

        except asyncio.TimeoutError:
            logger.debug(f"⏱️ {telemetry['atm_id']}: Timeout")
            return False
        except Exception as e:
            logger.debug(f"❌ {telemetry['atm_id']}: {str(e)}")
            return False

    async def _send_telemetry_batch(
        self, session: aiohttp.ClientSession, atm_batch: List[Dict]
    ):
        """Send telemetry for a batch of ATMs with improved concurrency"""
        tasks = []

        for atm in atm_batch:
            telemetry = self._generate_optimized_telemetry(atm)
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

    async def _simulation_cycle(self):
        """Run one complete simulation cycle with improved batch processing"""
        batch_size = min(
            self.config["batch_size"], 50
        )  # Limit batch size to avoid overwhelming
        start_time = time.time()

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100, limit_per_host=50
            )  # Improved connection pooling
        ) as session:
            # Process ATMs in batches using individual endpoint only
            for i in range(0, len(self.atms), batch_size):
                batch = self.atms[i : i + batch_size]
                successful, failed = await self._send_telemetry_batch(session, batch)

                # Smaller delay between batches to reduce bottlenecks
                if i + batch_size < len(self.atms):
                    await asyncio.sleep(0.5)  # Reduced from 1 second to 0.5 seconds

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
        logger.info(f"   Batch size: {self.config['batch_size']}")
        logger.info(f"   Target API: {self.config['api_base_url']}")

        try:
            cycle_count = 0
            while self.running:
                await self._simulation_cycle()
                cycle_count += 1

                # Print stats every 10 cycles
                if cycle_count % 10 == 0:
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
