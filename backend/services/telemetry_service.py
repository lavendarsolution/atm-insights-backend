from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text, func
import asyncio
import logging
from database.models import ATMTelemetry, ATM
from services.cache_service import CacheService
from config import settings

logger = logging.getLogger(__name__)

class TelemetryService:
    """High-performance telemetry processing service"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.batch_buffer: List[Dict] = []
        self.last_flush = datetime.utcnow()
    
    async def process_telemetry_batch(self, db: Session, telemetry_list: List[Dict]) -> Dict:
        """Process a batch of telemetry data efficiently"""
        try:
            processed = 0
            errors = []
            
            # Batch insert for performance
            telemetry_objects = []
            for telemetry_data in telemetry_list:
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(
                        telemetry_data["timestamp"].replace('Z', '+00:00')
                    )
                    
                    # Create telemetry object
                    telemetry_obj = ATMTelemetry(
                        time=timestamp,
                        atm_id=telemetry_data["atm_id"],
                        status=telemetry_data["status"],
                        temperature_celsius=telemetry_data.get("temperature"),
                        cash_level_percent=telemetry_data.get("cash_level"),
                        transactions_count=telemetry_data.get("transactions_count", 0),
                        failed_transactions_count=telemetry_data.get("failed_transactions", 0),
                        cpu_usage_percent=telemetry_data.get("cpu_usage"),
                        memory_usage_percent=telemetry_data.get("memory_usage"),
                        disk_usage_percent=telemetry_data.get("disk_usage"),
                        network_latency_ms=telemetry_data.get("network_latency_ms"),
                        uptime_seconds=telemetry_data.get("uptime_seconds"),
                        error_code=telemetry_data.get("error_code"),
                        error_message=telemetry_data.get("error_message")
                    )
                    
                    telemetry_objects.append(telemetry_obj)
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Error processing {telemetry_data.get('atm_id', 'unknown')}: {str(e)}")
            
            # Bulk insert
            if telemetry_objects:
                db.bulk_save_objects(telemetry_objects)
                db.commit()
                
                # Update cache for recent data
                await self._update_cache_for_batch(telemetry_objects)
            
            return {
                "processed": processed,
                "errors": errors,
                "success": processed > 0
            }
            
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
                if obj.atm_id not in atm_latest or obj.time > atm_latest[obj.atm_id].time:
                    atm_latest[obj.atm_id] = obj
            
            # Update cache for each ATM
            cache_tasks = []
            for atm_id, latest_telemetry in atm_latest.items():
                cache_key = f"latest_telemetry:{atm_id}"
                cache_data = {
                    "atm_id": latest_telemetry.atm_id,
                    "time": latest_telemetry.time.isoformat(),
                    "status": latest_telemetry.status,
                    "temperature": latest_telemetry.temperature_celsius,
                    "cash_level": latest_telemetry.cash_level_percent,
                    "transactions_count": latest_telemetry.transactions_count,
                    "failed_transactions": latest_telemetry.failed_transactions_count,
                    "error_code": latest_telemetry.error_code,
                    "error_message": latest_telemetry.error_message
                }
                
                cache_tasks.append(
                    self.cache_service.set(cache_key, cache_data, ttl=settings.dashboard_cache_ttl)
                )
            
            # Execute cache updates concurrently
            await asyncio.gather(*cache_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.warning(f"Cache update failed: {str(e)}")
    
    async def get_dashboard_stats(self, db: Session) -> Dict:
        """Get optimized dashboard statistics"""
        cache_key = "dashboard_stats"
        
        # Try cache first
        cached_stats = await self.cache_service.get(cache_key)
        if cached_stats:
            return cached_stats
        
        try:
            # Use optimized query with materialized view
            stats_query = text("""
                SELECT 
                    COUNT(*) as total_atms,
                    COUNT(*) FILTER (WHERE effective_status = 'online') as online_atms,
                    COUNT(*) FILTER (WHERE effective_status = 'offline') as offline_atms,
                    COUNT(*) FILTER (WHERE effective_status = 'error') as error_atms,
                    COALESCE(AVG(cash_level_percent), 0) as avg_cash_level,
                    COALESCE(SUM(transactions_count), 0) as total_transactions,
                    COUNT(*) FILTER (WHERE cash_level_percent < 20 OR error_code IS NOT NULL) as critical_alerts
                FROM atm_status_summary
            """)
            
            result = db.execute(stats_query).fetchone()
            
            stats = {
                "total_atms": result.total_atms or 0,
                "online_atms": result.online_atms or 0,
                "offline_atms": result.offline_atms or 0,
                "error_atms": result.error_atms or 0,
                "total_transactions_today": result.total_transactions or 0,
                "avg_cash_level": round(result.avg_cash_level or 0, 1),
                "critical_alerts": result.critical_alerts or 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Cache for fast access
            await self.cache_service.set(cache_key, stats, ttl=settings.dashboard_cache_ttl)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dashboard stats: {str(e)}")
            raise
    
    async def get_atm_status_list(self, db: Session) -> List[Dict]:
        """Get optimized ATM status list"""
        cache_key = "atm_status_list"
        
        # Try cache first
        cached_list = await self.cache_service.get(cache_key)
        if cached_list:
            return cached_list
        
        try:
            # Use materialized view for performance
            status_query = text("""
                SELECT 
                    atm_id,
                    name,
                    region,
                    effective_status as status,
                    last_update,
                    temperature_celsius,
                    cash_level_percent,
                    transactions_count,
                    error_code,
                    error_message
                FROM atm_status_summary
                ORDER BY atm_id
            """)
            
            results = db.execute(status_query).fetchall()
            
            atm_statuses = []
            for row in results:
                atm_statuses.append({
                    "atm_id": row.atm_id,
                    "name": row.name,
                    "region": row.region,
                    "status": row.status,
                    "last_update": row.last_update.isoformat() if row.last_update else None,
                    "temperature": row.temperature_celsius,
                    "cash_level": row.cash_level_percent,
                    "transactions_today": row.transactions_count or 0,
                    "error_code": row.error_code,
                    "error_message": row.error_message
                })
            
            # Cache the results
            await self.cache_service.set(cache_key, atm_statuses, ttl=settings.dashboard_cache_ttl)
            
            return atm_statuses
            
        except Exception as e:
            logger.error(f"Error getting ATM status list: {str(e)}")
            raise