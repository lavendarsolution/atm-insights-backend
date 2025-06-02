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

            # Simple evaluation logic based on rule type
            if rule_type == AlertRuleType.LOW_CASH:
                cash_level = test_data.get("cash_level_percent", 100)
                return cash_level is not None and cash_level < threshold

            elif rule_type == AlertRuleType.HIGH_TRANSACTION_FAILURES:
                failure_rate = test_data.get("failure_rate_percentage", 0)
                return failure_rate > threshold

            elif rule_type == AlertRuleType.NETWORK_ISSUES:
                network_status = test_data.get("network_status", "connected")
                network_latency = test_data.get("network_latency_ms", 0)
                return network_status == "disconnected" or (
                    network_latency is not None and network_latency > threshold
                )

            elif rule_type == AlertRuleType.HARDWARE_MALFUNCTION:
                # Check for hardware errors via error codes or status
                error_code = test_data.get("error_code")
                status = test_data.get("status")
                return error_code is not None or status == "error"

            elif rule_type == AlertRuleType.MAINTENANCE_DUE:
                days_since_maintenance = test_data.get("days_since_maintenance", 0)
                return days_since_maintenance >= threshold

            elif rule_type == AlertRuleType.UNUSUAL_ACTIVITY:
                anomaly_score = test_data.get("activity_anomaly_score", 0)
                return anomaly_score > threshold

            # Additional condition checks for CPU, temperature, etc.
            cpu_usage = test_data.get("cpu_usage_percent")
            if cpu_usage is not None and cpu_usage > 80:
                return True

            temperature = test_data.get("temperature_celsius")
            if temperature is not None and (temperature > 35 or temperature < 5):
                return True

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

            # Broadcast alert to WebSocket connections
            AlertService._broadcast_alert_update(alert, "new_alert")

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

            # Broadcast alert update to WebSocket connections
            AlertService._broadcast_alert_update(alert, "alert_updated")

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

            # Define specific alert conditions with their rules
            alert_checks = [
                {
                    "rule_type": AlertRuleType.LOW_CASH,
                    "condition": lambda data: data.get("cash_level_percent", 100)
                    < 10,  # Further reduced from 12% to 10%
                    "title": "Low Cash Level",
                    "message_template": "Cash level critically low: {cash_level_percent}%",
                },
                {
                    "rule_type": AlertRuleType.HARDWARE_MALFUNCTION,
                    "condition": lambda data: data.get("error_code") is not None
                    or data.get("status") == "error",
                    "title": "Hardware Error",
                    "message_template": "Hardware error detected: {error_code} - {error_message}",
                },
                {
                    "rule_type": AlertRuleType.NETWORK_ISSUES,
                    "condition": lambda data: data.get("network_status")
                    == "disconnected",
                    "title": "Network Disconnected",
                    "message_template": "ATM network connection lost",
                },
                {
                    "rule_type": "HIGH_TEMPERATURE",
                    "condition": lambda data: data.get("temperature_celsius", 0)
                    > 38,  # Increased from 35 to 38
                    "title": "High Temperature Alert",
                    "message_template": "Temperature critically high: {temperature_celsius}°C",
                },
                {
                    "rule_type": "LOW_TEMPERATURE",
                    "condition": lambda data: data.get("temperature_celsius", 25)
                    < 2,  # Decreased from 5 to 2
                    "title": "Low Temperature Alert",
                    "message_template": "Temperature critically low: {temperature_celsius}°C",
                },
                {
                    "rule_type": "HIGH_CPU",
                    "condition": lambda data: data.get("cpu_usage_percent", 0)
                    > 85,  # Increased from 80 to 85
                    "title": "High CPU Usage",
                    "message_template": "CPU usage critically high: {cpu_usage_percent}%",
                },
            ]

            for check in alert_checks:
                if check["condition"](atm_data):
                    # Check for cooldown to prevent duplicate alerts
                    # Use extremely long cooldown periods to drastically reduce frequency
                    if check["rule_type"] == AlertRuleType.LOW_CASH:
                        cooldown_minutes = (
                            720  # 12 hours for cash level alerts (increased from 8h)
                        )
                    elif check["rule_type"] == AlertRuleType.HARDWARE_MALFUNCTION:
                        cooldown_minutes = (
                            720  # 12 hours for hardware errors (doubled from 6h)
                        )
                    elif check["rule_type"] == AlertRuleType.NETWORK_ISSUES:
                        cooldown_minutes = (
                            240  # 4 hours for network issues (increased from 3h)
                        )
                    elif check["rule_type"] in ["HIGH_TEMPERATURE", "LOW_TEMPERATURE"]:
                        cooldown_minutes = (
                            360  # 6 hours for temperature alerts (increased from 4h)
                        )
                    elif check["rule_type"] == "HIGH_CPU":
                        cooldown_minutes = (
                            240  # 4 hours for CPU alerts (increased from 3h)
                        )
                    else:
                        cooldown_minutes = 360  # 6 hours default (increased from 4h)

                    recent_alert = (
                        db.query(Alert)
                        .filter(
                            Alert.atm_id == atm_id,
                            Alert.rule_type == check["rule_type"],
                            Alert.status == "active",
                            Alert.triggered_at
                            > datetime.utcnow()
                            - timedelta(minutes=cooldown_minutes),  # Variable cooldown
                        )
                        .first()
                    )

                    if not recent_alert:
                        # Determine severity based on rule type and severity of the condition
                        if check["rule_type"] == AlertRuleType.LOW_CASH:
                            cash_level = atm_data.get("cash_level_percent", 100)
                            if cash_level < 10:
                                severity = "critical"
                            elif cash_level < 15:
                                severity = "high"
                            else:
                                severity = "medium"
                        elif check["rule_type"] == AlertRuleType.HARDWARE_MALFUNCTION:
                            # Different severity based on error type
                            error_code = atm_data.get("error_code", "")
                            if error_code in [
                                "E002",
                                "E007",
                            ]:  # Critical hardware issues
                                severity = "critical"
                            elif error_code in ["E001", "E003"]:  # High priority issues
                                severity = "high"
                            else:  # Medium priority issues
                                severity = "medium"
                        elif check["rule_type"] == AlertRuleType.NETWORK_ISSUES:
                            severity = "high"  # Reduced from critical
                        elif check["rule_type"] == "HIGH_TEMPERATURE":
                            temp = atm_data.get("temperature_celsius", 0)
                            if temp > 40:
                                severity = "critical"
                            elif temp > 37:
                                severity = "high"
                            else:
                                severity = "medium"
                        elif check["rule_type"] == "LOW_TEMPERATURE":
                            temp = atm_data.get("temperature_celsius", 25)
                            if temp < 0:
                                severity = "critical"
                            elif temp < 3:
                                severity = "high"
                            else:
                                severity = "medium"
                        elif check["rule_type"] == "HIGH_CPU":
                            cpu = atm_data.get("cpu_usage_percent", 0)
                            if cpu > 95:
                                severity = "critical"
                            elif cpu > 85:
                                severity = "high"
                            else:
                                severity = "medium"
                        else:
                            severity = "medium"  # Default to medium instead of critical

                        # Format message with actual values
                        try:
                            message = check["message_template"].format(**atm_data)
                        except (KeyError, ValueError) as e:
                            # Handle missing fields gracefully for hardware errors
                            if check["rule_type"] == AlertRuleType.HARDWARE_MALFUNCTION:
                                error_code = atm_data.get("error_code", "UNKNOWN")
                                error_message = atm_data.get(
                                    "error_message", "Hardware malfunction detected"
                                )
                                status = atm_data.get("status", "unknown")
                                message = f"Hardware error detected: {error_code} - {error_message}"
                                logger.debug(
                                    f"Hardware error alert formatting for {atm_id}: error_code={error_code}, error_message={error_message}, status={status}"
                                )
                            else:
                                message = check[
                                    "title"
                                ]  # Use title as fallback instead of template with placeholders
                                logger.warning(
                                    f"Alert message formatting failed for {check['rule_type']}: {e}"
                                )

                        # Create alert
                        alert_data = AlertCreate(
                            rule_type=check["rule_type"],
                            atm_id=atm_id,
                            severity=severity,
                            title=check["title"],
                            message=message,
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

    @staticmethod
    def _broadcast_alert_update(alert: Alert, update_type: str):
        """Broadcast alert update via Redis for WebSocket connections"""
        try:
            import asyncio
            import json

            import redis.asyncio as redis
            from config import get_settings

            settings = get_settings()

            # Prepare alert data for broadcast
            alert_data = {
                "alert_id": str(alert.alert_id),
                "rule_type": alert.rule_type,
                "atm_id": alert.atm_id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "status": alert.status,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
                "acknowledged_by": (
                    str(alert.acknowledged_by) if alert.acknowledged_by else None
                ),
                "trigger_data": alert.trigger_data,
                "resolution_notes": alert.resolution_notes,
            }

            # Publish to Redis for WebSocket broadcast
            async def publish_alert():
                redis_client = redis.Redis.from_url(settings.redis_url)
                try:
                    await redis_client.publish(
                        "alerts_updates",
                        json.dumps(
                            {
                                "type": update_type,
                                "data": alert_data,
                            }
                        ),
                    )
                    logger.info(
                        f"Published alert update {update_type} for alert {alert.alert_id}"
                    )
                finally:
                    await redis_client.close()

            # Run the async function in the current event loop or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, schedule it
                    asyncio.create_task(publish_alert())
                else:
                    # If we're in a sync context, run it
                    loop.run_until_complete(publish_alert())
            except RuntimeError:
                # No event loop exists, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(publish_alert())

        except Exception as e:
            logger.error(f"Failed to broadcast alert update: {e}")


# Global instance
alert_service = AlertService()
