import logging

from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from schemas.analytics import AnalyticsData
from services.analytics_service import AnalyticsService
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Analytics service instance
analytics_service = AnalyticsService()


@router.get("/analytics/data", response_model=AnalyticsData)
async def get_analytics_data(db: Session = Depends(get_db)):
    """Get comprehensive analytics data for the dashboard"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return AnalyticsData(**data)

    except Exception as e:
        logger.error(f"Error getting analytics data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/overview")
async def get_analytics_overview(db: Session = Depends(get_db)):
    """Get analytics overview data only"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return data["overview"]

    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/cash-levels")
async def get_cash_level_distribution(db: Session = Depends(get_db)):
    """Get cash level distribution data"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return data["cash_levels"]

    except Exception as e:
        logger.error(f"Error getting cash level distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/status-distribution")
async def get_status_distribution(db: Session = Depends(get_db)):
    """Get status distribution data"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return data["status_data"]

    except Exception as e:
        logger.error(f"Error getting status distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/regions")
async def get_region_analytics(db: Session = Depends(get_db)):
    """Get region-based analytics data"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return data["region_data"]

    except Exception as e:
        logger.error(f"Error getting region analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/trends")
async def get_weekly_trends(db: Session = Depends(get_db)):
    """Get weekly trends data"""
    try:
        data = await analytics_service.get_analytics_data(db)
        return data["weekly_trends"]

    except Exception as e:
        logger.error(f"Error getting weekly trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
