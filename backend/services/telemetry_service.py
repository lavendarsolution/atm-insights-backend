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
    """Enhanced telemetry processing service with selective real-time updates"""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.batch_buffer: List[Dict] = []
        self.last_flush = datetime.now()
        self.atm_status_cache = {}  # Cache previous statuses for change detection

    async def process_telemetry_batch(
        self, db: Session, telemetry_list: List[Dict]
    ) -> Dict:
        """Process telemetry batch with selective real-time updates and ATM status management"""
        try:
            processed = 0
            errors = []
            status_changes = []
            telemetry_updates = []
            atm_status_updates = []

            # Batch insert for performance
            telemetry_objects = []
            for telemetry_data in telemetry_list:
                try:
                    # Parse timestamp
                    timestamp_str = telemetry_data["timestamp"]
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str.replace("Z", "+00:00")
                    timestamp = datetime.fromisoformat(timestamp_str)

                    atm_id = telemetry_data["atm_id"]
                    current_status = telemetry_data["status"]

                    # Check for status changes ONLY
                    previous_status = self.atm_status_cache.get(atm_id)
                    if previous_status and previous_status != current_status:
                        status_changes.append(
                            {
                                "atm_id": atm_id,
                                "old_status": previous_status,
                                "new_status": current_status,
                                "timestamp": timestamp.isoformat(),
                            }
                        )
                        logger.info(
                            f"ATM {atm_id} status changed: {previous_status} -> {current_status}"
                        )

                    # Update status cache
                    self.atm_status_cache[atm_id] = current_status

                    # Update ATM table status if there's an error or status change
                    if current_status == "error" or (
                        previous_status and previous_status != current_status
                    ):
                        atm_status_updates.append(
                            {
                                "atm_id": atm_id,
                                "status": current_status,
                                "last_error_code": telemetry_data.get("error_code"),
                                "last_error_message": telemetry_data.get(
                                    "error_message"
                                ),
                                "updated_at": timestamp,
                            }
                        )

                    # Create telemetry object
                    telemetry_obj = ATMTelemetry(
                        time=timestamp,
                        atm_id=atm_id,
                        status=current_status,
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

                    # Always add telemetry for ATM detail pages
                    telemetry_updates.append(telemetry_data)
                    processed += 1

                except Exception as e:
                    errors.append(
                        f"Error processing {telemetry_data.get('atm_id', 'unknown')}: {str(e)}"
                    )

            # Bulk insert telemetry
            if telemetry_objects:
                db.bulk_save_objects(telemetry_objects)

                # Update ATM statuses in batch
                if atm_status_updates:
                    await self._update_atm_statuses(db, atm_status_updates)

                db.commit()

                # Handle real-time updates (status changes only)
                await self._handle_realtime_updates(
                    telemetry_objects, status_changes, telemetry_updates
                )

                # Check alerts for critical conditions - now using unified alert system
                await self._check_alerts_for_batch(telemetry_objects)

            return {"processed": processed, "errors": errors, "success": processed > 0}

        except Exception as e:
            db.rollback()
            logger.error(f"Batch processing error: {str(e)}")
            raise

    async def _handle_realtime_updates(
        self,
        telemetry_objects: List[ATMTelemetry],
        status_changes: List[Dict],
        telemetry_updates: List[Dict],
    ):
        """Handle selective real-time updates including error notifications"""
        try:
            # Update cache for each ATM (for API endpoints)
            await self._update_cache_for_batch(telemetry_objects)

            # Update telemetry history cache (for ATM detail pages)
            await self._update_telemetry_history_cache(telemetry_objects)

            # Publish telemetry updates to Redis (for ATM detail pages only)
            for update in telemetry_updates:
                await self.cache_service.publish("telemetry_updates", update)

            # Publish status changes to Redis ONLY when status actually changes
            for change in status_changes:
                await self.cache_service.publish("atm_status_changes", change)
                logger.info(
                    f"Published status change: ATM {change['atm_id']} {change['old_status']} -> {change['new_status']}"
                )

            # Publish error notifications for real-time alerts
            for telemetry_obj in telemetry_objects:
                if telemetry_obj.error_code:
                    error_notification = {
                        "type": "atm_error",
                        "atm_id": telemetry_obj.atm_id,
                        "error_code": telemetry_obj.error_code,
                        "error_message": telemetry_obj.error_message,
                        "timestamp": telemetry_obj.time.isoformat(),
                        "status": telemetry_obj.status,
                    }
                    await self.cache_service.publish("atm_errors", error_notification)
                    logger.info(
                        f"Published error notification for ATM {telemetry_obj.atm_id}: {telemetry_obj.error_code}"
                    )

            # NOTE: Dashboard stats are NOT published here anymore
            # Frontend will fetch them every 15 seconds via API

        except Exception as e:
            logger.warning(f"Real-time update handling failed: {str(e)}")

    async def _update_cache_for_batch(self, telemetry_objects: List[ATMTelemetry]):
        """Update cache with latest telemetry for API endpoints"""
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

    async def _update_telemetry_history_cache(
        self, telemetry_objects: List[ATMTelemetry]
    ):
        """Update telemetry history cache for ATM detail pages"""
        try:
            # Group by ATM ID
            atm_telemetry = {}
            for obj in telemetry_objects:
                if obj.atm_id not in atm_telemetry:
                    atm_telemetry[obj.atm_id] = []
                atm_telemetry[obj.atm_id].append(obj)

            # Update history for each ATM
            for atm_id, telemetries in atm_telemetry.items():
                # Get existing history
                history_key = f"telemetry_history:{atm_id}"
                existing_history = await self.cache_service.get(history_key) or []

                # Convert new telemetries to dict format
                new_entries = []
                for t in telemetries:
                    new_entries.append(
                        {
                            "time": t.time.isoformat(),
                            "status": t.status,
                            "uptime_seconds": t.uptime_seconds,
                            "cash_level_percent": t.cash_level_percent,
                            "temperature_celsius": t.temperature_celsius,
                            "cpu_usage_percent": t.cpu_usage_percent,
                            "memory_usage_percent": t.memory_usage_percent,
                            "disk_usage_percent": t.disk_usage_percent,
                            "network_status": t.network_status,
                            "network_latency_ms": t.network_latency_ms,
                            "error_code": t.error_code,
                            "error_message": t.error_message,
                        }
                    )

                # Combine and sort by time (most recent first)
                combined_history = new_entries + existing_history
                combined_history.sort(key=lambda x: x["time"], reverse=True)

                # Keep only last 100 entries
                updated_history = combined_history[:100]

                # Cache updated history
                await self.cache_service.set(history_key, updated_history, ttl=3600)

        except Exception as e:
            logger.warning(f"Telemetry history cache update failed: {str(e)}")

    async def _update_atm_statuses(self, db: Session, status_updates: List[Dict]):
        """Update ATM table statuses when errors occur or status changes"""
        try:
            # Import here to avoid circular imports
            from models.atm import ATM

            for update in status_updates:
                atm = db.query(ATM).filter(ATM.atm_id == update["atm_id"]).first()
                if atm:
                    atm.status = update["status"]
                    atm.updated_at = update["updated_at"]

                    # If this is an error, we could store additional error info
                    # For now we'll just update the status
                    logger.info(
                        f"Updated ATM {update['atm_id']} status to {update['status']}"
                    )

        except Exception as e:
            logger.warning(f"ATM status update failed: {str(e)}")

    async def _check_alerts_for_batch(self, telemetry_objects: List[ATMTelemetry]):
        """Check for alert conditions using unified alert service only"""
        try:
            # Import here to avoid circular imports
            from database.session import SessionLocal
            from services.alert_service import alert_service

            alerts = []

            # Use the unified alert service for all alert checking
            try:
                db = SessionLocal()
                try:
                    for telemetry in telemetry_objects:
                        # Convert telemetry to dict for condition evaluation
                        telemetry_data = {
                            "cash_level_percent": telemetry.cash_level_percent,
                            "temperature_celsius": telemetry.temperature_celsius,
                            "cpu_usage_percent": telemetry.cpu_usage_percent,
                            "memory_usage_percent": telemetry.memory_usage_percent,
                            "disk_usage_percent": telemetry.disk_usage_percent,
                            "network_latency_ms": telemetry.network_latency_ms,
                            "uptime_seconds": telemetry.uptime_seconds,
                            "status": telemetry.status,
                            "network_status": telemetry.network_status,
                            "error_code": telemetry.error_code,
                        }

                        # Check unified alert conditions - this handles all alert logic
                        new_alerts = alert_service.check_alert_conditions(
                            db, telemetry_data, telemetry.atm_id
                        )

                        # Convert alerts to cache-compatible format for backwards compatibility
                        for alert in new_alerts:
                            alerts.append(
                                {
                                    "id": str(alert.alert_id),
                                    "atm_id": alert.atm_id,
                                    "severity": alert.severity,
                                    "type": alert.rule_type,
                                    "message": alert.message,
                                    "timestamp": alert.triggered_at.isoformat(),
                                    "title": alert.title,
                                }
                            )
                except Exception as e:
                    logger.error(f"Database session error: {e}")
                    db.rollback()
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Alert checking failed: {e}")

            # Update cache with new alerts
            if alerts:
                existing_alerts = await self.cache_service.get("recent_alerts") or []
                combined_alerts = alerts + existing_alerts
                # Keep only last 50 alerts
                recent_alerts = combined_alerts[:50]
                await self.cache_service.set("recent_alerts", recent_alerts, ttl=300)

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
        self, db: Session, atm_id: str, hours: int = 24, limit: int = 100
    ) -> List[Dict]:
        """Get telemetry history for specific ATM with caching"""
        try:
            # Try cache first for recent data
            if hours <= 24 and limit <= 100:
                cached_history = await self.cache_service.get(
                    f"telemetry_history:{atm_id}"
                )
                if cached_history:
                    # Filter by time if needed
                    if hours < 24:
                        cutoff_time = datetime.now() - timedelta(hours=hours)
                        filtered_history = [
                            entry
                            for entry in cached_history
                            if datetime.fromisoformat(entry["time"]) >= cutoff_time
                        ]
                        return filtered_history[:limit]
                    return cached_history[:limit]

            # Fallback to database query
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
                LIMIT :limit
            """
            )

            results = db.execute(
                history_query,
                {"atm_id": atm_id, "cutoff_time": cutoff_time, "limit": limit},
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

    async def get_dashboard_stats(self, db: Session) -> Dict:
        """Get dashboard statistics - called by API endpoint every 15 seconds"""
        try:
            # Don't use cache here - always fetch fresh data for dashboard

            # Calculate stats from database
            stats_query = text(
                """
                WITH latest_telemetry AS (
                    SELECT DISTINCT ON (atm_id) 
                        atm_id, status, cash_level_percent, error_code, time
                    FROM atm_telemetry 
                    ORDER BY atm_id, time DESC
                ),
                atm_counts AS (
                    SELECT 
                        COUNT(*) as total_atms,
                        COUNT(*) FILTER (WHERE status = 'online') as online_atms,
                        COUNT(*) FILTER (WHERE status = 'offline') as offline_atms,
                        COUNT(*) FILTER (WHERE status = 'error') as error_atms,
                        AVG(cash_level_percent) as avg_cash_level,
                        COUNT(*) FILTER (WHERE error_code IS NOT NULL) as critical_alerts
                    FROM latest_telemetry
                ),
                transaction_counts AS (
                    SELECT COUNT(*) as total_transactions_today
                    FROM atm_telemetry 
                    WHERE time >= CURRENT_DATE
                )
                SELECT * FROM atm_counts, transaction_counts
            """
            )

            result = db.execute(stats_query).fetchone()

            stats = {
                "total_atms": result.total_atms or 0,
                "online_atms": result.online_atms or 0,
                "offline_atms": result.offline_atms or 0,
                "error_atms": result.error_atms or 0,
                "total_transactions_today": result.total_transactions_today or 0,
                "avg_cash_level": round(result.avg_cash_level or 0, 1),
                "critical_alerts": result.critical_alerts or 0,
                "last_updated": datetime.now().isoformat(),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting dashboard stats: {str(e)}")
            return {
                "total_atms": 0,
                "online_atms": 0,
                "offline_atms": 0,
                "error_atms": 0,
                "total_transactions_today": 0,
                "avg_cash_level": 0,
                "critical_alerts": 0,
                "last_updated": datetime.now().isoformat(),
            }
