import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List

from config import settings
from database.session import SessionLocal
from services.cache_service import CacheService
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """Background task service with ML prediction scheduling"""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.running_tasks = {}
        self.is_running = False
        self.prediction_scheduler: Optional[PredictionScheduler] = None
        self.ml_service = None

    async def start(self, ml_service=None):
        """Start all background tasks including ML prediction scheduler"""
        if self.is_running:
            logger.warning("Background task service is already running")
            return
            
        self.is_running = True
        self.ml_service = ml_service

        # Start existing tasks
        self.running_tasks["cleanup_old_cache"] = asyncio.create_task(
            self._periodic_task(self._cleanup_old_cache, 3600)  # Every hour
        )

        self.running_tasks["health_monitor"] = asyncio.create_task(
            self._periodic_task(self._monitor_system_health, 60)  # Every minute
        )
        
        self.running_tasks["prediction_stats"] = asyncio.create_task(
            self._periodic_task(self._update_prediction_stats, 300)  # Every 5 minutes
        )

        # Start ML prediction scheduler if ML service is available
        if ml_service and hasattr(ml_service, 'model_loaded') and ml_service.model_loaded:
            self.prediction_scheduler = PredictionScheduler(ml_service, self.cache_service)
            await self.prediction_scheduler.start_prediction_scheduler()
            logger.info("✅ Background task service started with ML prediction scheduler")
        else:
            logger.warning("⚠️ ML service not available, starting background tasks without predictions")
            logger.info("✅ Background task service started (without ML predictions)")

    async def stop(self):
        """Stop all background tasks"""
        self.is_running = False

        # Stop ML prediction scheduler first
        if self.prediction_scheduler:
            await self.prediction_scheduler.stop_prediction_scheduler()

        # Cancel all running tasks
        for task_name, task in self.running_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Background task {task_name} cancelled")

        self.running_tasks.clear()
        logger.info("🛑 Background task service stopped")

    async def _periodic_task(self, func: Callable, interval: int):
        """Execute a function periodically"""
        while self.is_running:
            try:
                await func()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic task {func.__name__}: {str(e)}")
                await asyncio.sleep(interval)

    def _table_exists(self, db: Session, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            check_query = text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """
            )
            result = db.execute(check_query, {"table_name": table_name}).scalar()
            return bool(result)
        except Exception as e:
            logger.debug(f"Error checking if table {table_name} exists: {str(e)}")
            return False

    async def _cleanup_old_cache(self):
        """Clean up old cache entries"""
        try:
            # Clean up expired prediction caches
            patterns_to_clean = [
                "ml_prediction_v2:*",
                "telemetry_history:*",
                "atm_details:*",
                "dashboard_stats_*"
            ]
            
            total_cleaned = 0
            for pattern in patterns_to_clean:
                cleaned = await self.cache_service.invalidate_pattern(pattern)
                total_cleaned += cleaned
            
            if total_cleaned > 0:
                logger.info(f"🧹 Cache cleanup: removed {total_cleaned} expired entries")
            else:
                logger.debug("🧹 Cache cleanup completed (no expired entries)")

        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")

    async def _monitor_system_health(self):
        """Monitor system health and log metrics"""
        try:
            db = SessionLocal()
            try:
                # Check if required tables exist before querying them
                if not self._table_exists(db, "atm_telemetry"):
                    logger.debug("Tables not yet created, skipping health monitoring")
                    return

                if not self._table_exists(db, "atms"):
                    logger.debug("ATMs table not yet created, skipping health monitoring")
                    return

                # Check recent telemetry data flow
                recent_count_query = text(
                    """
                    SELECT COUNT(*) FROM atm_telemetry 
                    WHERE time >= NOW() - INTERVAL '5 minutes'
                """
                )
                recent_count = db.execute(recent_count_query).scalar()

                # Check for any stuck ATMs (no data in 10 minutes)
                stuck_atms_query = text(
                    """
                    SELECT COUNT(DISTINCT atm_id) FROM atms a
                    WHERE a.status = 'active' 
                    AND NOT EXISTS (
                        SELECT 1 FROM atm_telemetry t 
                        WHERE t.atm_id = a.atm_id 
                        AND t.time >= NOW() - INTERVAL '10 minutes'
                    )
                """
                )
                stuck_atms = db.execute(stuck_atms_query).scalar()

                # Check prediction system health
                prediction_health = await self._check_prediction_health()

                # Log health metrics
                health_metrics = {
                    "recent_telemetry_count": recent_count or 0,
                    "stuck_atms_count": stuck_atms or 0,
                    "ml_predictions_enabled": prediction_health["enabled"],
                    "ml_predictions_healthy": prediction_health["healthy"],
                    "ml_last_run": prediction_health["last_run"],
                    "cache_connected": bool(self.cache_service._connection),
                    "timestamp": datetime.now().isoformat(),
                }

                # Cache health metrics
                await self.cache_service.set("system_health", health_metrics, ttl=300)

                # Alert thresholds
                if stuck_atms and stuck_atms > 10:  # Alert threshold
                    logger.warning(f"⚠️ High number of stuck ATMs detected: {stuck_atms}")
                
                if not prediction_health["healthy"] and prediction_health["enabled"]:
                    logger.warning("⚠️ ML prediction system appears unhealthy")

                logger.debug(f"💓 System health: recent_data={recent_count}, stuck_atms={stuck_atms}, ml_healthy={prediction_health['healthy']}")

            except Exception as e:
                logger.debug(f"Could not get system health metrics: {str(e)}")
                db.rollback()
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error monitoring system health: {str(e)}")
    
    async def _check_prediction_health(self) -> dict:
        """Check ML prediction system health"""
        try:
            # Check if ML service is available and loaded
            ml_enabled = (
                self.ml_service is not None and 
                hasattr(self.ml_service, 'model_loaded') and 
                self.ml_service.model_loaded
            )
            
            if not ml_enabled:
                return {"enabled": False, "healthy": False, "last_run": None}
            
            # Check last prediction run
            summary = await self.cache_service.get("ml_prediction_summary")
            last_run = None
            healthy = False
            
            if summary and summary.get("last_updated"):
                last_run = summary["last_updated"]
                # Consider healthy if last run was within last 30 minutes
                last_run_time = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                time_diff = datetime.now() - last_run_time.replace(tzinfo=None)
                healthy = time_diff.total_seconds() < 1800  # 30 minutes
            
            return {
                "enabled": ml_enabled,
                "healthy": healthy,
                "last_run": last_run
            }
            
        except Exception as e:
            logger.debug(f"Error checking prediction health: {e}")
            return {"enabled": False, "healthy": False, "last_run": None}
    
    async def _update_prediction_stats(self):
        """Update prediction system statistics"""
        try:
            if not self.ml_service or not hasattr(self.ml_service, 'prediction_stats'):
                return
            
            # Get ML service stats
            ml_stats = getattr(self.ml_service, 'prediction_stats', {})
            
            # Get cached summaries
            prediction_summary = await self.cache_service.get("ml_prediction_summary")
            alert_summary = await self.cache_service.get("ml_alert_summary")
            
            # Combine stats
            combined_stats = {
                "ml_service_stats": ml_stats,
                "prediction_summary": prediction_summary,
                "alert_summary": alert_summary,
                "system_stats": {
                    "cache_connected": bool(self.cache_service._connection),
                    "scheduler_running": bool(self.prediction_scheduler and self.prediction_scheduler.is_running),
                    "background_tasks_count": len(self.running_tasks),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            # Cache combined stats
            await self.cache_service.set("ml_system_stats", combined_stats, ttl=600)
            
            logger.debug("📊 Updated prediction system statistics")
            
        except Exception as e:
            logger.error(f"Error updating prediction stats: {e}")

    # Public methods for manual operations
    # async def trigger_prediction_run(self):
    #     """Manually trigger a prediction run"""
    #     try:
    #         if self.prediction_scheduler and self.prediction_scheduler.is_running:
    #             logger.info("🔮 Manually triggering prediction run...")
    #             await self.prediction_scheduler._run_bulk_predictions()
    #             return True
    #         else:
    #             logger.warning("Prediction scheduler is not running")
    #             return False
    #     except Exception as e:
    #         logger.error(f"Error triggering manual prediction run: {e}")
    #         return False
    
    # async def get_system_status(self) -> dict:
    #     """Get comprehensive system status"""
    #     try:
    #         health_metrics = await self.cache_service.get("system_health") or {}
    #         ml_stats = await self.cache_service.get("ml_system_stats") or {}
            
    #         return {
    #             "background_service": {
    #                 "running": self.is_running,
    #                 "tasks_count": len(self.running_tasks),
    #                 "active_tasks": list(self.running_tasks.keys())
    #             },
    #             "prediction_scheduler": {
    #                 "enabled": bool(self.prediction_scheduler),
    #                 "running": bool(self.prediction_scheduler and self.prediction_scheduler.is_running),
    #                 "ml_service_loaded": bool(self.ml_service and getattr(self.ml_service, 'model_loaded', False))
    #             },
    #             "health_metrics": health_metrics,
    #             "ml_stats": ml_stats,
    #             "timestamp": datetime.now().isoformat()
    #         }
    #     except Exception as e:
    #         logger.error(f"Error getting system status: {e}")
    #         return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # async def restart_prediction_scheduler(self):
    #     """Restart the prediction scheduler"""
    #     try:
    #         if self.prediction_scheduler:
    #             await self.prediction_scheduler.stop_prediction_scheduler()
            
    #         if self.ml_service and getattr(self.ml_service, 'model_loaded', False):
    #             self.prediction_scheduler = PredictionScheduler(self.ml_service, self.cache_service)
    #             await self.prediction_scheduler.start_prediction_scheduler()
    #             logger.info("🔄 Prediction scheduler restarted successfully")
    #             return True
    #         else:
    #             logger.warning("Cannot restart prediction scheduler: ML service not available")
    #             return False
    #     except Exception as e:
    #         logger.error(f"Error restarting prediction scheduler: {e}")
    #         return False







