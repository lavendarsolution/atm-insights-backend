import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from const.alert_rules import get_all_rule_configs, get_rule_config, get_rule_types
from models.alert import Alert
from models.atm import ATM
from models.user import User
from schemas.alert import (
    AlertCreate,
    AlertRuleConfig,
    AlertRuleResponse,
    AlertRuleType,
    AlertSeverity,
    AlertStats,
    AlertUpdate,
)
from services.notification_service import notification_service
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AlertService:
    """Service for managing alerts and pre-defined alert rules"""

    @staticmethod
    def get_alert_rules(
        skip: int = 0, limit: int = 100, is_active: Optional[bool] = None
    ) -> List[AlertRuleResponse]:
        """Get pre-defined alert rule configurations"""
        try:
            all_configs = get_all_rule_configs()
            rules = []

            for rule_type, config in all_configs.items():
                rule_response = AlertRuleResponse(
                    rule_type=rule_type,
                    name=config["name"],
                    description=config["description"],
                    severity=config["default_severity"],
                    threshold=config["default_threshold"],
                    condition_description=config["condition_description"],
                    is_active=True,  # Pre-defined rules are always available
                    notification_channels=config["notification_channels"],
                    cooldown_minutes=config["cooldown_minutes"],
                    target_atms=None,
                )
                rules.append(rule_response)

            # Apply pagination
            return rules[skip : skip + limit]

        except Exception as e:
            logger.error(f"Failed to get alert rules: {str(e)}")
            raise e

    @staticmethod
    def get_alert_rule(rule_type: str) -> Optional[AlertRuleResponse]:
        """Get a specific pre-defined alert rule configuration"""
        try:
            # Convert string to AlertRuleType if needed
            if isinstance(rule_type, str):
                try:
                    rule_type_enum = AlertRuleType(rule_type)
                except ValueError:
                    return None
            else:
                rule_type_enum = rule_type

            config = get_rule_config(rule_type_enum)
            if not config:
                return None

            return AlertRuleResponse(
                rule_type=rule_type_enum,
                name=config["name"],
                description=config["description"],
                severity=config["default_severity"],
                threshold=config["default_threshold"],
                condition_description=config["condition_description"],
                is_active=True,
                notification_channels=config["notification_channels"],
                cooldown_minutes=config["cooldown_minutes"],
                target_atms=None,
            )

        except Exception as e:
            logger.error(f"Failed to get alert rule {rule_type}: {str(e)}")
            return None

    @staticmethod
    def _evaluate_condition(
        rule_type: AlertRuleType,
        test_data: Dict,
        custom_threshold: Optional[float] = None,
    ) -> bool:
        """Evaluate if an alert condition would trigger based on test data"""
        try:
            config = get_rule_config(rule_type)
            if not config:
                return False

            threshold = (
                custom_threshold
                if custom_threshold is not None
                else config["default_threshold"]
            )
            metric = config.get("metric", "")

            # Simple evaluation logic based on rule type
            if rule_type == AlertRuleType.LOW_CASH:
                cash_level = test_data.get("cash_level_percentage", 100)
                return cash_level < threshold

            elif rule_type == AlertRuleType.HIGH_TRANSACTION_FAILURES:
                failure_rate = test_data.get("failure_rate_percentage", 0)
                return failure_rate > threshold

            elif rule_type == AlertRuleType.NETWORK_ISSUES:
                network_failures = test_data.get("network_failures", 0)
                return network_failures >= threshold

            elif rule_type == AlertRuleType.HARDWARE_MALFUNCTION:
                hardware_errors = test_data.get("hardware_errors", 0)
                return hardware_errors >= threshold

            elif rule_type == AlertRuleType.MAINTENANCE_DUE:
                days_since_maintenance = test_data.get("days_since_maintenance", 0)
                return days_since_maintenance >= threshold

            elif rule_type == AlertRuleType.UNUSUAL_ACTIVITY:
                anomaly_score = test_data.get("activity_anomaly_score", 0)
                return anomaly_score > threshold

            return False

        except Exception as e:
            logger.error(f"Failed to evaluate condition for {rule_type}: {str(e)}")
            return False

    @staticmethod
    def create_alert(
        db: Session, alert_data: AlertCreate, send_notifications: bool = True
    ) -> Alert:
        """Create a new alert"""
        try:
            alert = Alert(
                rule_type=alert_data.rule_type,
                atm_id=alert_data.atm_id,
                severity=alert_data.severity,
                title=alert_data.title,
                message=alert_data.message,
                status="active",
                trigger_data=alert_data.trigger_data,
            )

            db.add(alert)
            db.commit()
            db.refresh(alert)

            logger.info(f"Created alert {alert.alert_id} for ATM {alert.atm_id}")

            # Send notifications if enabled
            if send_notifications:
                AlertService._send_alert_notifications(db, alert)

            return alert

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise

    @staticmethod
    def get_alerts(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        atm_id: Optional[str] = None,
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        try:
            query = db.query(Alert)

            if status:
                query = query.filter(Alert.status == status)
            if severity:
                query = query.filter(Alert.severity == severity)
            if atm_id:
                query = query.filter(Alert.atm_id == atm_id)

            query = query.order_by(desc(Alert.triggered_at)).offset(skip).limit(limit)
            result = query.all()
            return result

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            raise

    @staticmethod
    def get_alert(db: Session, alert_id: UUID) -> Optional[Alert]:
        """Get a specific alert by ID"""
        try:
            result = db.get(Alert, alert_id)
            return result
        except Exception as e:
            logger.error(f"Failed to get alert {alert_id}: {e}")
            raise

    @staticmethod
    def update_alert(
        db: Session,
        alert_id: UUID,
        alert_data: AlertUpdate,
        user_id: Optional[UUID] = None,
    ) -> Optional[Alert]:
        """Update an alert"""
        try:
            alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
            if not alert:
                return None

            if alert_data.status:
                alert.status = alert_data.status

                if alert_data.status == "acknowledged" and user_id:
                    alert.acknowledged_by = user_id
                    alert.acknowledged_at = datetime.utcnow()
                elif alert_data.status == "resolved":
                    alert.resolved_at = datetime.utcnow()

            if alert_data.resolution_notes:
                alert.resolution_notes = alert_data.resolution_notes

            db.commit()
            db.refresh(alert)

            logger.info(f"Updated alert {alert_id}")
            return alert

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update alert {alert_id}: {e}")
            raise

    @staticmethod
    def get_alert_stats(db: Session) -> AlertStats:
        """Get alert statistics"""
        try:
            # Get total counts by status
            total_alerts = db.query(func.count(Alert.alert_id)).scalar() or 0

            active_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.status == "active")
                .scalar()
                or 0
            )

            acknowledged_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.status == "acknowledged")
                .scalar()
                or 0
            )

            resolved_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.status == "resolved")
                .scalar()
                or 0
            )

            # Get counts by severity
            critical_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.severity == "critical")
                .scalar()
                or 0
            )

            high_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.severity == "high")
                .scalar()
                or 0
            )

            medium_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.severity == "medium")
                .scalar()
                or 0
            )

            low_alerts = (
                db.query(func.count(Alert.alert_id))
                .filter(Alert.severity == "low")
                .scalar()
                or 0
            )

            return AlertStats(
                total_alerts=total_alerts,
                active_alerts=active_alerts,
                acknowledged_alerts=acknowledged_alerts,
                resolved_alerts=resolved_alerts,
                critical_alerts=critical_alerts,
                high_alerts=high_alerts,
                medium_alerts=medium_alerts,
                low_alerts=low_alerts,
            )

        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            raise

    @staticmethod
    def check_alert_conditions(db: Session, atm_data: Dict, atm_id: str) -> List[Alert]:
        """Check if any pre-defined alert conditions are met for given ATM data"""
        try:
            alerts_created = []
            all_rule_configs = get_all_rule_configs()

            for rule_type, config in all_rule_configs.items():
                # Check cooldown (you might want to implement this in a separate cache/tracking system)
                # For now, skip cooldown check as it would require storing last trigger times

                # Evaluate condition
                if AlertService._evaluate_condition(rule_type, atm_data):
                    # Create alert
                    alert_data = AlertCreate(
                        rule_type=rule_type,
                        atm_id=atm_id,
                        severity=config["default_severity"],
                        title=f"Alert: {config['name']}",
                        message=config["description"],
                        trigger_data=atm_data,
                    )

                    alert = AlertService.create_alert(
                        db, alert_data, send_notifications=True
                    )
                    alerts_created.append(alert)

            return alerts_created

        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
            raise

    @staticmethod
    def _send_alert_notifications(db: Session, alert: Alert):
        """Send notifications for an alert (sync wrapper for async notification service)"""
        try:
            # Get rule config for notification channels
            config = get_rule_config(AlertRuleType(alert.rule_type))
            if not config:
                return

            channels = config.get("notification_channels", [])
            if not channels:
                return

            # Prepare alert data for notification
            alert_data = {
                "alert_id": str(alert.alert_id),
                "title": alert.title,
                "message": alert.message,
                "atm_id": alert.atm_id,
                "severity": alert.severity,
                "triggered_at": alert.triggered_at.isoformat(),
                "rule_name": config["name"],
            }

            # Get user emails if email notifications are enabled
            user_emails = []
            if "email" in channels:
                # Get users who should receive notifications (you may want to customize this)
                users_query = db.query(User).filter(User.is_active == True)
                users = users_query.all()
                user_emails = [user.email for user in users if user.email]

            # Since notification_service is async, we need to handle this properly
            # For now, we'll use a basic approach - in production you might want to use a task queue
            import asyncio

            try:
                # Create new event loop if none exists (for sync context)
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async notification in the event loop
            results = loop.run_until_complete(
                notification_service.send_alert_notification(
                    alert_data=alert_data,
                    notification_channels=channels,
                    user_emails=user_emails,
                )
            )

            logger.info(f"Sent notifications for alert {alert.alert_id}: {results}")

        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")


# Global instance
alert_service = AlertService()
