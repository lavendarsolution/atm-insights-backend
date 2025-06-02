import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from models import ATM, Alert, ATMTelemetry
from schemas.analytics import (
    AnalyticsData,
    AnalyticsOverview,
    CashLevelDistribution,
    LocationAnalytics,
    StatusDistribution,
    TrendData,
)
from sqlalchemy import func, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analytics data processing"""

    def __init__(self):
        self.status_colors = {
            "active": "#10B981",
            "warning": "#F59E0B",
            "error": "#EF4444",
            "inactive": "#6B7280",
            "maintenance": "#8B5CF6",
            "decommissioned": "#6B7280",
        }

        self.cash_level_colors = {
            "< 20%": "#EF4444",
            "20% - 50%": "#F59E0B",
            "50% - 80%": "#3B82F6",
            "> 80%": "#10B981",
        }

    async def get_analytics_data(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive analytics data"""
        try:
            # Get all ATMs
            atms = db.query(ATM).all()
            total_atms = len(atms)

            if total_atms == 0:
                return self._get_empty_analytics()

            # Get latest telemetry for each ATM
            latest_telemetry = self._get_latest_telemetry_per_atm(db)

            # Calculate overview metrics
            overview = self._calculate_overview(atms, latest_telemetry)

            # Calculate distributions
            cash_levels = self._calculate_cash_level_distribution(latest_telemetry)
            status_data = self._calculate_status_distribution(atms)
            region_data = self._calculate_location_analytics(atms, latest_telemetry)

            # Get weekly trends (mock data for now - can be enhanced with real transaction data)
            weekly_trends = self._get_weekly_trends(db)

            return {
                "overview": overview,
                "cash_levels": cash_levels,
                "status_data": status_data,
                "region_data": region_data,
                "weekly_trends": weekly_trends,
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            raise

    def _get_latest_telemetry_per_atm(self, db: Session) -> Dict[str, ATMTelemetry]:
        """Get the latest telemetry record for each ATM"""
        try:
            # Get latest telemetry for each ATM using a subquery
            subquery = (
                db.query(
                    ATMTelemetry.atm_id, func.max(ATMTelemetry.time).label("max_time")
                )
                .group_by(ATMTelemetry.atm_id)
                .subquery()
            )

            latest_telemetry_records = (
                db.query(ATMTelemetry)
                .join(
                    subquery,
                    (ATMTelemetry.atm_id == subquery.c.atm_id)
                    & (ATMTelemetry.time == subquery.c.max_time),
                )
                .all()
            )

            return {record.atm_id: record for record in latest_telemetry_records}

        except Exception as e:
            logger.error(f"Error getting latest telemetry: {str(e)}")
            return {}

    def _calculate_overview(
        self, atms: List[ATM], latest_telemetry: Dict[str, ATMTelemetry]
    ) -> AnalyticsOverview:
        """Calculate overview metrics"""
        total_atms = len(atms)
        active_atms = len([atm for atm in atms if atm.status == "active"])
        issue_atms = len(
            [atm for atm in atms if atm.status in ["error", "warning", "maintenance"]]
        )

        # Get unique regions
        regions = set(atm.region for atm in atms if atm.region)

        operational_rate = (active_atms / total_atms * 100) if total_atms > 0 else 0
        issue_rate = (issue_atms / total_atms * 100) if total_atms > 0 else 0

        return AnalyticsOverview(
            total_atms=total_atms,
            operational_rate=round(operational_rate, 1),
            issue_rate=round(issue_rate, 1),
            regions_count=len(regions),
            active_atms=active_atms,
            atms_with_issues=issue_atms,
        )

    def _calculate_cash_level_distribution(
        self, latest_telemetry: Dict[str, ATMTelemetry]
    ) -> List[CashLevelDistribution]:
        """Calculate cash level distribution"""
        levels = {"< 20%": 0, "20% - 50%": 0, "50% - 80%": 0, "> 80%": 0}

        for telemetry in latest_telemetry.values():
            if telemetry.cash_level_percent is not None:
                cash_level = telemetry.cash_level_percent
                if cash_level < 20:
                    levels["< 20%"] += 1
                elif cash_level < 50:
                    levels["20% - 50%"] += 1
                elif cash_level < 80:
                    levels["50% - 80%"] += 1
                else:
                    levels["> 80%"] += 1

        return [
            CashLevelDistribution(
                name=name, value=value, color=self.cash_level_colors[name]
            )
            for name, value in levels.items()
        ]

    def _calculate_status_distribution(
        self, atms: List[ATM]
    ) -> List[StatusDistribution]:
        """Calculate status distribution"""
        status_counts = {}
        for atm in atms:
            status = atm.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return [
            StatusDistribution(
                name=status.title(),
                value=count,
                color=self.status_colors.get(status, "#6B7280"),
            )
            for status, count in status_counts.items()
        ]

    def _calculate_location_analytics(
        self, atms: List[ATM], latest_telemetry: Dict[str, ATMTelemetry]
    ) -> List[LocationAnalytics]:
        """Calculate region-based analytics"""
        region_data = {}

        for atm in atms:
            region = atm.region or "Unknown"

            if region not in region_data:
                region_data[region] = {
                    "region": region,
                    "count": 0,
                    "active": 0,
                    "warning": 0,
                    "error": 0,
                    "inactive": 0,
                    "maintenance": 0,
                    "decommissioned": 0,
                }

            region_data[region]["count"] += 1
            status = atm.status
            if status in region_data[region]:
                region_data[region][status] += 1

        # Convert to list and sort by count
        result = []
        for data in region_data.values():
            result.append(
                LocationAnalytics(
                    region=data["region"],
                    count=data["count"],
                    active=data.get("active", 0),
                    warning=data.get("warning", 0),
                    error=data.get("error", 0),
                    inactive=data.get("inactive", 0),
                )
            )

        # Return all regions (should be 5: AIRPORT, SUPERMARKET, MALL, HOSPITAL, UNIVERSITY)
        return sorted(result, key=lambda x: x.count, reverse=True)

    def _get_weekly_trends(self, db: Session) -> List[TrendData]:
        """Get weekly trends data"""
        # For now, return mock data - this can be enhanced with real transaction data
        # when transaction models are available

        # Get error counts from alerts for the past week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Mock weekly data with some real error data if available
        weekly_data = [
            TrendData(name="Mon", transactions=420, errors=8),
            TrendData(name="Tue", transactions=380, errors=12),
            TrendData(name="Wed", transactions=510, errors=5),
            TrendData(name="Thu", transactions=470, errors=10),
            TrendData(name="Fri", transactions=590, errors=15),
            TrendData(name="Sat", transactions=750, errors=7),
            TrendData(name="Sun", transactions=400, errors=4),
        ]

        return weekly_data

    def _get_empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics data when no ATMs exist"""
        return {
            "overview": AnalyticsOverview(
                total_atms=0,
                operational_rate=0.0,
                issue_rate=0.0,
                regions_count=0,
                active_atms=0,
                atms_with_issues=0,
            ),
            "cash_levels": [],
            "status_data": [],
            "region_data": [],
            "weekly_trends": [],
            "last_updated": datetime.now().isoformat(),
        }
