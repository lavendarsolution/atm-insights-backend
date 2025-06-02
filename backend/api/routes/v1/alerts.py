from typing import List, Optional
from uuid import UUID

from const.alert_rules import get_rule_config
from database.session import get_db
from dependencies.auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.user import User
from schemas.alert import (
    AlertCreate,
    AlertResponse,
    AlertRuleResponse,
    AlertRuleTestRequest,
    AlertRuleTestResponse,
    AlertStats,
    AlertUpdate,
    NotificationTestRequest,
    NotificationTestResponse,
)
from services.alert_service import alert_service
from services.notification_service import notification_service
from sqlalchemy.orm import Session

router = APIRouter()


# Alert Rules endpoints
@router.get("/alerts/rules", response_model=List[AlertRuleResponse])
async def get_alert_rules(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: Optional[bool] = Query(None),
    current_user: User = Depends(get_current_user),
):
    """Get pre-defined alert rule configurations"""
    try:
        rules = alert_service.get_alert_rules(skip, limit, is_active)
        return rules
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alert rules: {str(e)}",
        )


@router.get("/rules/{rule_type}", response_model=AlertRuleResponse)
async def get_alert_rule(
    rule_type: str,
    current_user: User = Depends(get_current_user),
):
    """Get a specific pre-defined alert rule configuration"""
    try:
        rule = alert_service.get_alert_rule(rule_type)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert rule not found"
            )
        return rule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alert rule: {str(e)}",
        )


@router.post("/alerts/rules/test", response_model=AlertRuleTestResponse)
def test_alert_rule(
    test_request: AlertRuleTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test if a pre-defined alert rule condition would trigger"""
    try:
        # Use test data or provide default test data
        test_data = test_request.test_data or {}

        would_trigger = alert_service._evaluate_condition(
            test_request.rule_type, test_data, test_request.custom_threshold
        )

        # Get rule config for details
        config = get_rule_config(test_request.rule_type)

        evaluation_details = {
            "rule_type": test_request.rule_type,
            "rule_name": config.get("name", "Unknown Rule"),
            "test_data": test_data,
            "threshold": test_request.custom_threshold
            or config.get("default_threshold", 0),
            "result": would_trigger,
        }

        simulated_alert = None
        if would_trigger:
            simulated_alert = {
                "atm_id": test_request.atm_id,
                "title": f"Test Alert: {config.get('name', 'Unknown Rule')}",
                "message": f"This alert would be triggered: {config.get('description', 'No description')}",
                "severity": config.get("default_severity", "medium"),
                "rule_type": test_request.rule_type,
                "trigger_data": test_data,
            }

        return AlertRuleTestResponse(
            would_trigger=would_trigger,
            evaluation_details=evaluation_details,
            simulated_alert=simulated_alert,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to test alert rule: {str(e)}",
        )


# Alerts endpoints
@router.post(
    "/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED
)
def create_alert(
    alert_data: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new alert (usually called by system)"""
    try:
        alert = alert_service.create_alert(db, alert_data)
        return alert
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create alert: {str(e)}",
        )


@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    atm_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get alerts with optional filtering"""
    try:
        alerts = alert_service.get_alerts(db, skip, limit, status, severity, atm_id)
        return alerts
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alerts: {str(e)}",
        )


@router.get("/alerts/stats", response_model=AlertStats)
def get_alert_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get alert statistics"""
    try:
        stats = alert_service.get_alert_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alert stats: {str(e)}",
        )


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific alert"""
    try:
        alert = alert_service.get_alert(db, alert_id)
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
            )
        return alert
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alert: {str(e)}",
        )


@router.put("/alerts/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: UUID,
    alert_data: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an alert (acknowledge, resolve, etc.)"""
    try:
        alert = alert_service.update_alert(db, alert_id, alert_data, current_user.id)
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
            )
        return alert
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update alert: {str(e)}",
        )


# Notification testing endpoints
@router.post("/alerts/notifications/test", response_model=NotificationTestResponse)
async def test_notifications(
    test_request: NotificationTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test notification channels"""
    try:
        results = {}
        details = {}

        test_alert_data = {
            "title": "Test Notification",
            "message": test_request.test_message,
            "atm_id": "TEST-001",
            "severity": "medium",
            "triggered_at": "2025-06-02T12:00:00Z",
        }

        notification_results = await notification_service.send_alert_notification(
            alert_data=test_alert_data,
            notification_channels=test_request.channels,
            user_emails=(
                [current_user.email]
                if current_user.email and "email" in test_request.channels
                else []
            ),
        )

        for channel in test_request.channels:
            success = notification_results.get(channel, False)
            results[channel] = success
            details[channel] = "Sent successfully" if success else "Failed to send"

        return NotificationTestResponse(results=results, details=details)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test notifications: {str(e)}",
        )
