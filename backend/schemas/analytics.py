from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CashLevelDistribution(BaseModel):
    """Cash level distribution data"""

    name: str
    value: int
    color: str


class StatusDistribution(BaseModel):
    """Status distribution data"""

    name: str
    value: int
    color: str


class LocationAnalytics(BaseModel):
    """Region-based analytics data"""

    region: str
    count: int
    active: int
    warning: int
    error: int
    inactive: int


class TrendData(BaseModel):
    """Trend data for analytics"""

    name: str
    transactions: int
    errors: int


class AnalyticsOverview(BaseModel):
    """Analytics overview response model"""

    total_atms: int
    operational_rate: float
    issue_rate: float
    regions_count: int
    active_atms: int
    atms_with_issues: int


class AnalyticsData(BaseModel):
    """Complete analytics data response"""

    overview: AnalyticsOverview
    cash_levels: List[CashLevelDistribution]
    status_data: List[StatusDistribution]
    region_data: List[LocationAnalytics]
    weekly_trends: List[TrendData]
    last_updated: str
