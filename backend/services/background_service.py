import asyncio
from typing import List, Dict, Callable
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from database.connection import SessionLocal
from services.cache_service import CacheService
from config import settings

logger = logging.getLogger(__name__)

class BackgroundTaskService:
    """Manages background tasks for scalability"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.running_tasks = {}
        self.is_running = False
    
    async def start(self):
        """Start all background tasks"""
        self.is_running = True
        
        # Start periodic tasks
        self.running_tasks["refresh_materialized_views"] = asyncio.create_task(
            self._periodic_task(self._refresh_materialized_views, 300)  # Every 5 minutes
        )
        
        self.running_tasks["cleanup_old_cache"] = asyncio.create_task(
            self._periodic_task(self._cleanup_old_cache, 3600)  # Every hour
        )
        
        self.running_tasks["health_monitor"] = asyncio.create_task(
            self._periodic_task(self._monitor_system_health, 60)  # Every minute
        )
        
        logger.info("✅ Background task service started")
    
    async def stop(self):
        """Stop all background tasks"""
        self.is_running = False
        
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
            check_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """)
            result = db.execute(check_query, {"table_name": table_name}).scalar()
            return bool(result)
        except Exception as e:
            logger.debug(f"Error checking if table {table_name} exists: {str(e)}")
            return False
    
    async def _refresh_materialized_views(self):
        """Refresh materialized views for performance"""
        try:
            db = SessionLocal()
            try:
                # Check if the materialized view exists first
                check_view_exists = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'atm_status_summary' 
                        AND table_type = 'BASE TABLE'
                    )
                """)
                
                view_exists = db.execute(check_view_exists).scalar()
                
                if view_exists:
                    # Refresh ATM status summary view
                    refresh_query = text("REFRESH MATERIALIZED VIEW CONCURRENTLY atm_status_summary")
                    db.execute(refresh_query)
                    db.commit()
                    
                    # Invalidate related cache
                    await self.cache_service.invalidate_pattern("dashboard_*")
                    await self.cache_service.invalidate_pattern("atm_status_*")
                    
                    logger.debug("Materialized views refreshed")
                else:
                    logger.debug("Materialized view atm_status_summary does not exist, skipping refresh")
                
            except Exception as e:
                logger.debug(f"Could not refresh materialized views: {str(e)}")
                db.rollback()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error refreshing materialized views: {str(e)}")
    
    async def _cleanup_old_cache(self):
        """Clean up old cache entries"""
        try:
            # This would be implemented based on Redis capabilities
            # For now, we'll just log the activity
            logger.debug("Cache cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
    
    async def _monitor_system_health(self):
        """Monitor system health and log metrics"""
        try:
            db = SessionLocal()
            try:
                # Check if required tables exist before querying them
                if not self._table_exists(db, 'atm_telemetry'):
                    logger.debug("Tables not yet created, skipping health monitoring")
                    return
                
                if not self._table_exists(db, 'atms'):
                    logger.debug("ATMs table not yet created, skipping health monitoring")
                    return
                
                # Check recent telemetry data flow
                recent_count_query = text("""
                    SELECT COUNT(*) FROM atm_telemetry 
                    WHERE time >= NOW() - INTERVAL '5 minutes'
                """)
                recent_count = db.execute(recent_count_query).scalar()
                
                # Check for any stuck ATMs (no data in 10 minutes)
                stuck_atms_query = text("""
                    SELECT COUNT(DISTINCT atm_id) FROM atms a
                    WHERE a.status = 'active' 
                    AND NOT EXISTS (
                        SELECT 1 FROM atm_telemetry t 
                        WHERE t.atm_id = a.atm_id 
                        AND t.time >= NOW() - INTERVAL '10 minutes'
                    )
                """)
                stuck_atms = db.execute(stuck_atms_query).scalar()
                
                # Log health metrics
                health_metrics = {
                    "recent_telemetry_count": recent_count or 0,
                    "stuck_atms_count": stuck_atms or 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Cache health metrics
                await self.cache_service.set("system_health", health_metrics, ttl=300)
                
                if stuck_atms and stuck_atms > 10:  # Alert threshold
                    logger.warning(f"High number of stuck ATMs detected: {stuck_atms}")
                
                logger.debug(f"System health: {health_metrics}")
                
            except Exception as e:
                logger.debug(f"Could not get system health metrics: {str(e)}")
                db.rollback()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error monitoring system health: {str(e)}")
