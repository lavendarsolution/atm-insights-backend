from typing import Any, Dict

from schemas.alert import AlertRuleType, AlertSeverity

# Pre-defined alert rule configurations
ALERT_RULE_CONFIGS: Dict[AlertRuleType, Dict[str, Any]] = {
    AlertRuleType.LOW_CASH: {
        "name": "Low Cash Level",
        "description": "Alert when ATM cash level falls below threshold",
        "default_severity": AlertSeverity.HIGH,
        "default_threshold": 20.0,  # 20% of capacity
        "condition_description": "Cash level below 20% of capacity",
        "notification_channels": ["email", "telegram"],
        "cooldown_minutes": 60,
        "check_function": "check_low_cash",
        "metric": "cash_level_percentage",
    },
    AlertRuleType.HIGH_TRANSACTION_FAILURES: {
        "name": "High Transaction Failure Rate",
        "description": "Alert when transaction failure rate exceeds threshold",
        "default_severity": AlertSeverity.MEDIUM,
        "default_threshold": 10.0,  # 10% failure rate
        "condition_description": "Transaction failure rate above 10%",
        "notification_channels": ["email"],
        "cooldown_minutes": 30,
        "check_function": "check_transaction_failures",
        "metric": "failure_rate_percentage",
    },
    AlertRuleType.NETWORK_ISSUES: {
        "name": "Network Connectivity Issues",
        "description": "Alert when ATM experiences network connectivity problems",
        "default_severity": AlertSeverity.HIGH,
        "default_threshold": 3.0,  # 3 consecutive failures
        "condition_description": "Network connection failures detected",
        "notification_channels": ["email", "telegram"],
        "cooldown_minutes": 15,
        "check_function": "check_network_issues",
        "metric": "network_failures",
    },
    AlertRuleType.HARDWARE_MALFUNCTION: {
        "name": "Hardware Malfunction",
        "description": "Alert when hardware components are malfunctioning",
        "default_severity": AlertSeverity.CRITICAL,
        "default_threshold": 1.0,  # Any hardware error
        "condition_description": "Hardware component malfunction detected",
        "notification_channels": ["email", "telegram"],
        "cooldown_minutes": 5,
        "check_function": "check_hardware_issues",
        "metric": "hardware_errors",
    },
    AlertRuleType.MAINTENANCE_DUE: {
        "name": "Maintenance Due",
        "description": "Alert when ATM maintenance is due or overdue",
        "default_severity": AlertSeverity.MEDIUM,
        "default_threshold": 0.0,  # When maintenance is due
        "condition_description": "Scheduled maintenance is due",
        "notification_channels": ["email"],
        "cooldown_minutes": 1440,  # Once per day
        "check_function": "check_maintenance_due",
        "metric": "days_since_maintenance",
    },
    AlertRuleType.UNUSUAL_ACTIVITY: {
        "name": "Unusual Activity Detected",
        "description": "Alert when unusual transaction patterns are detected",
        "default_severity": AlertSeverity.MEDIUM,
        "default_threshold": 2.0,  # 2 standard deviations from normal
        "condition_description": "Transaction patterns deviate significantly from normal",
        "notification_channels": ["email"],
        "cooldown_minutes": 120,
        "check_function": "check_unusual_activity",
        "metric": "activity_anomaly_score",
    },
}


def get_rule_config(rule_type: AlertRuleType) -> Dict[str, Any]:
    """Get configuration for a specific rule type"""
    return ALERT_RULE_CONFIGS.get(rule_type, {})


def get_all_rule_configs() -> Dict[AlertRuleType, Dict[str, Any]]:
    """Get all rule configurations"""
    return ALERT_RULE_CONFIGS


def get_rule_types() -> list[AlertRuleType]:
    """Get list of all available rule types"""
    return list(ALERT_RULE_CONFIGS.keys())
