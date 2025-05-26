import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import settings
from models import ATM, ATMTelemetry
from services.cache_service import CacheService
from sqlalchemy import func, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TelemetryService:
    """Optimized telemetry processing service for essential ATM data"""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.batch_buffer: List[Dict] = []
        self.last_flush = datetime.now()

    async def process_telemetry_batch(
        self, db: Session, telemetry_list: List[Dict]
    ) -> Dict:
        """Process a batch of optimized telemetry data efficiently"""
        try:
            processed = 0
            errors = []

            # Batch insert for performance
            telemetry_objects = []
            for telemetry_data in telemetry_list:
                try:
                    # Parse timestamp
                    timestamp_str = telemetry_data["timestamp"]
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str.replace("Z", "+00:00")
                    timestamp = datetime.fromisoformat(timestamp_str)

                    # Create optimized telemetry object with essential fields only
                    telemetry_obj = ATMTelemetry(
                        time=timestamp,
                        atm_id=telemetry_data["atm_id"],
                        status=telemetry_data["status"],
                        uptime_seconds=telemetry_data.get("uptime_seconds"),
                        cash_level_percent=telemetry_data.get("cash_level_percent"),
                        temperature_celsius=telemetry_data.get("temperature_celsius"),
                        cpu_usage_percent=telemetry_data.get("cpu_usage_percent"),
                        memory_usage_percent=telemetry_data.get("memory_usage_percent"),
                        disk_usage_percent=telemetry_data.get("disk_usage_percent"),
                        network_status=telemetry_data.get("network_status"),
                        network_latency_ms=telemetry_data.get("network_latency_ms"),
                        error_code=telemetry_data.get("error_code"),
                        error_message=telemetry_data.get("error_message"),
                    )

                    telemetry_objects.append(telemetry_obj)
                    processed += 1

                except Exception as e:
                    errors.append(
                        f"Error processing {telemetry_data.get('atm_id', 'unknown')}: {str(e)}"
                    )

            # Bulk insert
            if telemetry_objects:
                db.bulk_save_objects(telemetry_objects)
                db.commit()

                # Update cache for recent data
                await self._update_cache_for_batch(telemetry_objects)

                # Trigger alert checking for critical conditions
                await self._check_alerts_for_batch(telemetry_objects)

            return {"processed": processed, "errors": errors, "success": processed > 0}

        except Exception as e:
            db.rollback()
            logger.error(f"Batch processing error: {str(e)}")
            raise

    async def _update_cache_for_batch(self, telemetry_objects: List[ATMTelemetry]):
        """Update cache with latest telemetry for fast dashboard queries"""
        try:
            # Group by ATM ID and get latest for each
            atm_latest = {}
            for obj in telemetry_objects:
                if (
                    obj.atm_id not in atm_latest
                    or obj.time > atm_latest[obj.atm_id].time
                ):
                    atm_latest[obj.atm_id] = obj

            # Update cache for each ATM
            cache_tasks = []
            for atm_id, latest_telemetry in atm_latest.items():
                cache_key = f"latest_telemetry:{atm_id}"
                cache_data = {
                    "atm_id": latest_telemetry.atm_id,
                    "time": latest_telemetry.time.isoformat(),
                    "status": latest_telemetry.status,
                    "uptime_seconds": latest_telemetry.uptime_seconds,
                    "cash_level_percent": latest_telemetry.cash_level_percent,
                    "temperature_celsius": latest_telemetry.temperature_celsius,
                    "cpu_usage_percent": latest_telemetry.cpu_usage_percent,
                    "memory_usage_percent": latest_telemetry.memory_usage_percent,
                    "disk_usage_percent": latest_telemetry.disk_usage_percent,
                    "network_status": latest_telemetry.network_status,
                    "network_latency_ms": latest_telemetry.network_latency_ms,
                    "error_code": latest_telemetry.error_code,
                    "error_message": latest_telemetry.error_message,
                }

                cache_tasks.append(
                    self.cache_service.set(
                        cache_key, cache_data, ttl=settings.dashboard_cache_ttl
                    )
                )

            # Execute cache updates concurrently
            await asyncio.gather(*cache_tasks, return_exceptions=True)

        except Exception as e:
            logger.warning(f"Cache update failed: {str(e)}")

    async def _check_alerts_for_batch(self, telemetry_objects: List[ATMTelemetry]):
        """Check for alert conditions in the batch"""
        try:
            alerts = []

            for telemetry in telemetry_objects:
                # Check critical cash level
                if (
                    telemetry.cash_level_percent is not None
                    and telemetry.cash_level_percent <= 15
                ):
                    alerts.append(
                        {
                            "atm_id": telemetry.atm_id,
                            "severity": "critical",
                            "type": "cash_level_critical",
                            "message": f"Critical cash level: {telemetry.cash_level_percent}%",
                            "timestamp": telemetry.time.isoformat(),
                        }
                    )

                # Check temperature alerts
                if telemetry.temperature_celsius is not None:
                    if telemetry.temperature_celsius > 35:
                        alerts.append(
                            {
                                "atm_id": telemetry.atm_id,
                                "severity": "warning",
                                "type": "temperature_high",
                                "message": f"High temperature: {telemetry.temperature_celsius}°C",
                                "timestamp": telemetry.time.isoformat(),
                            }
                        )
                    elif telemetry.temperature_celsius < 5:
                        alerts.append(
                            {
                                "atm_id": telemetry.atm_id,
                                "severity": "warning",
                                "type": "temperature_low",
                                "message": f"Low temperature: {telemetry.temperature_celsius}°C",
                                "timestamp": telemetry.time.isoformat(),
                            }
                        )

                # Check CPU usage
                if (
                    telemetry.cpu_usage_percent is not None
                    and telemetry.cpu_usage_percent > 80
                ):
                    alerts.append(
                        {
                            "atm_id": telemetry.atm_id,
                            "severity": "warning",
                            "type": "cpu_high",
                            "message": f"High CPU usage: {telemetry.cpu_usage_percent}%",
                            "timestamp": telemetry.time.isoformat(),
                        }
                    )

                # Check memory usage
                if (
                    telemetry.memory_usage_percent is not None
                    and telemetry.memory_usage_percent > 85
                ):
                    alerts.append(
                        {
                            "atm_id": telemetry.atm_id,
                            "severity": "warning",
                            "type": "memory_high",
                            "message": f"High memory usage: {telemetry.memory_usage_percent}%",
                            "timestamp": telemetry.time.isoformat(),
                        }
                    )

                # Check error codes
                if telemetry.error_code:
                    alerts.append(
                        {
                            "atm_id": telemetry.atm_id,
                            "severity": "error",
                            "type": "error_reported",
                            "message": f"Error {telemetry.error_code}: {telemetry.error_message or 'Unknown error'}",
                            "timestamp": telemetry.time.isoformat(),
                        }
                    )

                # Check network connectivity
                if telemetry.network_status == "disconnected":
                    alerts.append(
                        {
                            "atm_id": telemetry.atm_id,
                            "severity": "critical",
                            "type": "network_disconnected",
                            "message": "ATM network disconnected",
                            "timestamp": telemetry.time.isoformat(),
                        }
                    )

            # Cache alerts for dashboard
            if alerts:
                await self.cache_service.set("recent_alerts", alerts, ttl=300)
                logger.info(f"Generated {len(alerts)} alerts from telemetry batch")

        except Exception as e:
            logger.warning(f"Alert checking failed: {str(e)}")

    async def get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts from cache"""
        try:
            alerts = await self.cache_service.get("recent_alerts")
            return alerts or []
        except Exception as e:
            logger.warning(f"Error getting recent alerts: {str(e)}")
            return []

    async def get_atm_telemetry_history(
        self, db: Session, atm_id: str, hours: int = 24
    ) -> List[Dict]:
        """Get telemetry history for specific ATM"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            history_query = text(
                """
                SELECT 
                    time,
                    status,
                    cash_level_percent,
                    temperature_celsius,
                    cpu_usage_percent,
                    memory_usage_percent,
                    network_status,
                    network_latency_ms,
                    error_code,
                    error_message
                FROM atm_telemetry 
                WHERE atm_id = :atm_id AND time >= :cutoff_time
                ORDER BY time DESC
                LIMIT 1000
            """
            )

            results = db.execute(
                history_query, {"atm_id": atm_id, "cutoff_time": cutoff_time}
            ).fetchall()

            history = []
            for row in results:
                history.append(
                    {
                        "time": row.time.isoformat(),
                        "status": row.status,
                        "cash_level_percent": row.cash_level_percent,
                        "temperature_celsius": row.temperature_celsius,
                        "cpu_usage_percent": row.cpu_usage_percent,
                        "memory_usage_percent": row.memory_usage_percent,
                        "network_status": row.network_status,
                        "network_latency_ms": row.network_latency_ms,
                        "error_code": row.error_code,
                        "error_message": row.error_message,
                    }
                )

            return history

        except Exception as e:
            logger.error(f"Error getting telemetry history for {atm_id}: {str(e)}")
            return []
