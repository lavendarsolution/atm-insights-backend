import logging
from datetime import datetime, timedelta
from typing import Dict, List

from models import ATM, ATMTelemetry
from services.cache_service import CacheService
from sqlalchemy import func, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DashboardService:
    """Enhanced dashboard service with real-time capabilities"""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service

    async def get_dashboard_stats(self, db: Session) -> Dict:
        """Get comprehensive dashboard statistics with caching"""
        try:
            # Try cache first
            cache_key = "dashboard_stats_detailed"
            cached_stats = await self.cache_service.get(cache_key)
            if cached_stats:
                return cached_stats

            # Calculate comprehensive stats from database
            stats_query = text(
                """
                WITH latest_telemetry AS (
                    SELECT DISTINCT ON (atm_id) 
                        atm_id, status, cash_level_percent, error_code, time,
                        temperature_celsius, cpu_usage_percent, memory_usage_percent,
                        network_status
                    FROM atm_telemetry 
                    ORDER BY atm_id, time DESC
                ),
                atm_counts AS (
                    SELECT 
                        COUNT(*) as total_atms,
                        COUNT(*) FILTER (WHERE status = 'online') as online_atms,
                        COUNT(*) FILTER (WHERE status = 'offline') as offline_atms,
                        COUNT(*) FILTER (WHERE status = 'error') as error_atms,
                        COUNT(*) FILTER (WHERE status = 'maintenance') as maintenance_atms,
                        AVG(cash_level_percent) as avg_cash_level,
                        COUNT(*) FILTER (WHERE error_code IS NOT NULL) as critical_alerts,
                        COUNT(*) FILTER (WHERE cash_level_percent < 20) as low_cash_atms,
                        COUNT(*) FILTER (WHERE temperature_celsius > 35) as high_temp_atms,
                        COUNT(*) FILTER (WHERE cpu_usage_percent > 80) as high_cpu_atms,
                        COUNT(*) FILTER (WHERE network_status = 'disconnected') as disconnected_atms
                    FROM latest_telemetry
                ),
                transaction_counts AS (
                    SELECT 
                        COUNT(*) as total_transactions_today,
                        COUNT(*) FILTER (WHERE error_code IS NOT NULL) as failed_transactions_today
                    FROM atm_telemetry 
                    WHERE time >= CURRENT_DATE
                ),
                regional_stats AS (
                    SELECT 
                        CASE 
                            WHEN position('-' in a.atm_id) > 0 
                            THEN split_part(a.atm_id, '-', 2)
                            ELSE 'UNKNOWN'
                        END as region,
                        COUNT(*) as atms_in_region,
                        COUNT(*) FILTER (WHERE lt.status = 'online') as online_in_region
                    FROM atms a
                    LEFT JOIN latest_telemetry lt ON a.atm_id = lt.atm_id
                    GROUP BY region
                )
                SELECT 
                    ac.*,
                    tc.*,
                    json_agg(
                        json_build_object(
                            'region', rs.region,
                            'total', rs.atms_in_region,
                            'online', rs.online_in_region,
                            'availability', 
                            CASE 
                                WHEN rs.atms_in_region > 0 
                                THEN ROUND((rs.online_in_region::float / rs.atms_in_region * 100), 1)
                                ELSE 0 
                            END
                        )
                    ) as regional_breakdown
                FROM atm_counts ac, transaction_counts tc, regional_stats rs
                GROUP BY ac.total_atms, ac.online_atms, ac.offline_atms, ac.error_atms, 
                         ac.maintenance_atms, ac.avg_cash_level, ac.critical_alerts, 
                         ac.low_cash_atms, ac.high_temp_atms, ac.high_cpu_atms, 
                         ac.disconnected_atms, tc.total_transactions_today, tc.failed_transactions_today
            """
            )

            result = db.execute(stats_query).fetchone()

            if not result:
                # Return empty stats if no data
                return self._get_empty_stats()

            stats = {
                "total_atms": result.total_atms or 0,
                "online_atms": result.online_atms or 0,
                "offline_atms": result.offline_atms or 0,
                "error_atms": result.error_atms or 0,
                "maintenance_atms": result.maintenance_atms or 0,
                "total_transactions_today": result.total_transactions_today or 0,
                "failed_transactions_today": result.failed_transactions_today or 0,
                "avg_cash_level": round(result.avg_cash_level or 0, 1),
                "critical_alerts": result.critical_alerts or 0,
                "low_cash_atms": result.low_cash_atms or 0,
                "high_temp_atms": result.high_temp_atms or 0,
                "high_cpu_atms": result.high_cpu_atms or 0,
                "disconnected_atms": result.disconnected_atms or 0,
                "regional_breakdown": result.regional_breakdown or [],
                "overall_availability": round(
                    (
                        (result.online_atms / result.total_atms * 100)
                        if result.total_atms > 0
                        else 0
                    ),
                    1,
                ),
                "transaction_success_rate": round(
                    (
                        (
                            (
                                result.total_transactions_today
                                - result.failed_transactions_today
                            )
                            / result.total_transactions_today
                            * 100
                        )
                        if result.total_transactions_today > 0
                        else 100
                    ),
                    1,
                ),
                "last_updated": datetime.now().isoformat(),
            }

            # Cache the stats for 1 minute
            await self.cache_service.set(cache_key, stats, ttl=60)

            return stats

        except Exception as e:
            logger.error(f"Error getting dashboard stats: {str(e)}")
            return self._get_empty_stats()

    def _get_empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            "total_atms": 0,
            "online_atms": 0,
            "offline_atms": 0,
            "error_atms": 0,
            "maintenance_atms": 0,
            "total_transactions_today": 0,
            "failed_transactions_today": 0,
            "avg_cash_level": 0,
            "critical_alerts": 0,
            "low_cash_atms": 0,
            "high_temp_atms": 0,
            "high_cpu_atms": 0,
            "disconnected_atms": 0,
            "regional_breakdown": [],
            "overall_availability": 0,
            "transaction_success_rate": 100,
            "last_updated": datetime.now().isoformat(),
        }

    async def get_recent_activities(self, db: Session, limit: int = 20) -> List[Dict]:
        """Get recent activities for dashboard"""
        try:
            cache_key = f"recent_activities_{limit}"
            cached_activities = await self.cache_service.get(cache_key)
            if cached_activities:
                return cached_activities

            # Query recent activities from database
            activities_query = text(
                """
                WITH recent_changes AS (
                    SELECT 
                        atm_id,
                        time,
                        status,
                        error_code,
                        error_message,
                        cash_level_percent,
                        LAG(status) OVER (PARTITION BY atm_id ORDER BY time) as prev_status,
                        LAG(error_code) OVER (PARTITION BY atm_id ORDER BY time) as prev_error
                    FROM atm_telemetry 
                    WHERE time >= NOW() - INTERVAL '24 hours'
                    ORDER BY time DESC
                ),
                significant_events AS (
                    SELECT 
                        atm_id,
                        time,
                        status,
                        error_code,
                        error_message,
                        cash_level_percent,
                        CASE 
                            WHEN prev_status IS NULL THEN 'atm_first_report'
                            WHEN status != prev_status THEN 'status_change'
                            WHEN error_code IS NOT NULL AND prev_error IS NULL THEN 'new_error'
                            WHEN error_code IS NULL AND prev_error IS NOT NULL THEN 'error_resolved'
                            WHEN cash_level_percent < 20 THEN 'low_cash'
                            ELSE 'regular_update'
                        END as event_type
                    FROM recent_changes
                    WHERE (
                        prev_status IS NULL OR 
                        status != prev_status OR 
                        (error_code IS NOT NULL AND prev_error IS NULL) OR
                        (error_code IS NULL AND prev_error IS NOT NULL) OR
                        cash_level_percent < 20
                    )
                )
                SELECT * FROM significant_events 
                ORDER BY time DESC 
                LIMIT :limit
            """
            )

            results = db.execute(activities_query, {"limit": limit}).fetchall()

            activities = []
            for row in results:
                activity = {
                    "id": f"{row.atm_id}_{row.time.isoformat()}",
                    "atm_id": row.atm_id,
                    "timestamp": row.time.isoformat(),
                    "event_type": row.event_type,
                    "status": row.status,
                    "description": self._generate_activity_description(row),
                    "severity": self._get_activity_severity(
                        row.event_type, row.status, row.error_code
                    ),
                    "error_code": row.error_code,
                    "cash_level": row.cash_level_percent,
                }
                activities.append(activity)

            # Cache for 2 minutes
            await self.cache_service.set(cache_key, activities, ttl=120)

            return activities

        except Exception as e:
            logger.error(f"Error getting recent activities: {str(e)}")
            return []

    def _generate_activity_description(self, row) -> str:
        """Generate human-readable description for activity"""
        event_type = row.event_type
        atm_id = row.atm_id

        if event_type == "status_change":
            return f"ATM {atm_id} changed status to {row.status}"
        elif event_type == "new_error":
            error_msg = row.error_message or "Unknown error"
            return f"ATM {atm_id} reported error: {error_msg}"
        elif event_type == "error_resolved":
            return f"ATM {atm_id} error resolved, status now {row.status}"
        elif event_type == "low_cash":
            return f"ATM {atm_id} has low cash level: {row.cash_level_percent}%"
        elif event_type == "atm_first_report":
            return f"ATM {atm_id} came online"
        else:
            return f"ATM {atm_id} regular update"

    def _get_activity_severity(
        self, event_type: str, status: str, error_code: str
    ) -> str:
        """Determine activity severity"""
        if event_type == "new_error" or status == "error":
            return "high"
        elif event_type == "low_cash" or status == "offline":
            return "medium"
        elif event_type == "error_resolved":
            return "low"
        else:
            return "info"

    async def get_performance_metrics(self, db: Session, hours: int = 24) -> Dict:
        """Get performance metrics for dashboard charts"""
        try:
            cache_key = f"performance_metrics_{hours}h"
            cached_metrics = await self.cache_service.get(cache_key)
            if cached_metrics:
                return cached_metrics

            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Query performance metrics
            metrics_query = text(
                """
                WITH hourly_stats AS (
                    SELECT 
                        date_trunc('hour', time) as hour,
                        COUNT(*) as total_reports,
                        COUNT(DISTINCT atm_id) as active_atms,
                        AVG(cash_level_percent) as avg_cash_level,
                        AVG(cpu_usage_percent) as avg_cpu_usage,
                        AVG(memory_usage_percent) as avg_memory_usage,
                        AVG(temperature_celsius) as avg_temperature,
                        COUNT(*) FILTER (WHERE error_code IS NOT NULL) as error_count,
                        COUNT(*) FILTER (WHERE network_status = 'disconnected') as network_issues
                    FROM atm_telemetry 
                    WHERE time >= :cutoff_time
                    GROUP BY date_trunc('hour', time)
                    ORDER BY hour
                )
                SELECT 
                    hour,
                    total_reports,
                    active_atms,
                    ROUND(avg_cash_level, 1) as avg_cash_level,
                    ROUND(avg_cpu_usage, 1) as avg_cpu_usage,
                    ROUND(avg_memory_usage, 1) as avg_memory_usage,
                    ROUND(avg_temperature, 1) as avg_temperature,
                    error_count,
                    network_issues
                FROM hourly_stats
            """
            )

            results = db.execute(metrics_query, {"cutoff_time": cutoff_time}).fetchall()

            metrics = {
                "time_series": [],
                "summary": {
                    "total_data_points": 0,
                    "avg_active_atms": 0,
                    "avg_cash_level": 0,
                    "avg_cpu_usage": 0,
                    "avg_memory_usage": 0,
                    "avg_temperature": 0,
                    "total_errors": 0,
                    "total_network_issues": 0,
                },
            }

            total_active_atms = 0
            total_cash = 0
            total_cpu = 0
            total_memory = 0
            total_temp = 0
            total_errors = 0
            total_network_issues = 0
            data_points = len(results)

            for row in results:
                metrics["time_series"].append(
                    {
                        "hour": row.hour.isoformat(),
                        "total_reports": row.total_reports,
                        "active_atms": row.active_atms,
                        "avg_cash_level": row.avg_cash_level,
                        "avg_cpu_usage": row.avg_cpu_usage,
                        "avg_memory_usage": row.avg_memory_usage,
                        "avg_temperature": row.avg_temperature,
                        "error_count": row.error_count,
                        "network_issues": row.network_issues,
                    }
                )

                total_active_atms += row.active_atms
                total_cash += row.avg_cash_level or 0
                total_cpu += row.avg_cpu_usage or 0
                total_memory += row.avg_memory_usage or 0
                total_temp += row.avg_temperature or 0
                total_errors += row.error_count
                total_network_issues += row.network_issues

            if data_points > 0:
                metrics["summary"] = {
                    "total_data_points": data_points,
                    "avg_active_atms": round(total_active_atms / data_points, 1),
                    "avg_cash_level": round(total_cash / data_points, 1),
                    "avg_cpu_usage": round(total_cpu / data_points, 1),
                    "avg_memory_usage": round(total_memory / data_points, 1),
                    "avg_temperature": round(total_temp / data_points, 1),
                    "total_errors": total_errors,
                    "total_network_issues": total_network_issues,
                }

            # Cache for 5 minutes
            await self.cache_service.set(cache_key, metrics, ttl=300)

            return metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"time_series": [], "summary": {}}
