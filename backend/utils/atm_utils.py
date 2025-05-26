"""
ATM Management Utilities
Helper functions for ATM operations, validation, and data processing
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from models import ATM, ATMTelemetry
from sqlalchemy import func, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ATMValidator:
    """ATM data validation utilities"""

    # Valid status values
    VALID_STATUSES = ["active", "inactive", "maintenance", "decommissioned"]

    # ATM ID pattern (can be customized based on your naming convention)
    ATM_ID_PATTERN = re.compile(r"^ATM-[A-Z0-9]+-[0-9]{3,4}$")

    @classmethod
    def validate_atm_id(cls, atm_id: str) -> Tuple[bool, str]:
        """
        Validate ATM ID format
        Returns: (is_valid, error_message)
        """
        if not atm_id:
            return False, "ATM ID cannot be empty"

        atm_id = atm_id.strip().upper()

        if len(atm_id) > 20:
            return False, "ATM ID cannot exceed 20 characters"

        # Check if it matches the expected pattern
        if not cls.ATM_ID_PATTERN.match(atm_id):
            return (
                False,
                "ATM ID must follow format: ATM-REGION-XXX (e.g., ATM-NYC-001)",
            )

        return True, ""

    @classmethod
    def validate_status(cls, status: str) -> Tuple[bool, str]:
        """
        Validate ATM status
        Returns: (is_valid, error_message)
        """
        if not status:
            return True, ""  # Status is optional

        if status.lower() not in cls.VALID_STATUSES:
            return False, f"Status must be one of: {', '.join(cls.VALID_STATUSES)}"

        return True, ""

    @classmethod
    def extract_region_from_id(cls, atm_id: str) -> Optional[str]:
        """Extract region from ATM ID"""
        try:
            parts = atm_id.split("-")
            if len(parts) >= 2:
                return parts[1]
        except Exception:
            pass
        return None


class ATMStatsCalculator:
    """Calculate various ATM statistics and metrics"""

    @staticmethod
    def calculate_atm_health_score(atm_id: str, db: Session, hours: int = 24) -> float:
        """
        Calculate health score for an ATM based on recent telemetry
        Score: 0-100 (100 = perfect health)
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Get recent telemetry data
            telemetry_query = text(
                """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(*) FILTER (WHERE status = 'online') as online_count,
                    COUNT(*) FILTER (WHERE error_code IS NOT NULL) as error_count,
                    AVG(cpu_usage_percent) as avg_cpu,
                    AVG(memory_usage_percent) as avg_memory,
                    AVG(cash_level_percent) as avg_cash_level,
                    COUNT(*) FILTER (WHERE failed_transactions_count > 0) as failed_tx_count
                FROM atm_telemetry 
                WHERE atm_id = :atm_id AND time >= :cutoff_time
            """
            )

            result = db.execute(
                telemetry_query, {"atm_id": atm_id, "cutoff_time": cutoff_time}
            ).fetchone()

            if not result or result.total_records == 0:
                return 50.0  # Default score for no data

            # Calculate health components
            uptime_score = (
                result.online_count / result.total_records
            ) * 40  # 40% weight
            error_score = max(
                0, 30 - (result.error_count / result.total_records) * 30
            )  # 30% weight

            # Performance score (CPU + Memory)
            cpu_score = max(
                0, 15 - (max(0, (result.avg_cpu or 0) - 80) / 20) * 15
            )  # 15% weight
            memory_score = max(
                0, 15 - (max(0, (result.avg_memory or 0) - 80) / 20) * 15
            )  # 15% weight

            total_score = uptime_score + error_score + cpu_score + memory_score
            return min(100.0, max(0.0, total_score))

        except Exception as e:
            logger.warning(f"Error calculating health score for {atm_id}: {str(e)}")
            return 50.0

    @staticmethod
    def get_regional_stats(db: Session) -> List[Dict]:
        """Get statistics grouped by region"""
        try:
            stats_query = text(
                """
                SELECT 
                    CASE 
                        WHEN position('-' in atm_id) > 0 
                        THEN split_part(atm_id, '-', 2)
                        ELSE 'UNKNOWN'
                    END as region,
                    COUNT(*) as total_atms,
                    COUNT(*) FILTER (WHERE status = 'active') as active_atms,
                    COUNT(*) FILTER (WHERE status = 'inactive') as inactive_atms,
                    COUNT(*) FILTER (WHERE status = 'maintenance') as maintenance_atms,
                    COUNT(*) FILTER (WHERE status = 'decommissioned') as decommissioned_atms
                FROM atms
                GROUP BY region
                ORDER BY total_atms DESC
            """
            )

            results = db.execute(stats_query).fetchall()

            regional_stats = []
            for row in results:
                regional_stats.append(
                    {
                        "region": row.region,
                        "total_atms": row.total_atms,
                        "active_atms": row.active_atms,
                        "inactive_atms": row.inactive_atms,
                        "maintenance_atms": row.maintenance_atms,
                        "decommissioned_atms": row.decommissioned_atms,
                        "active_percentage": (
                            round((row.active_atms / row.total_atms) * 100, 1)
                            if row.total_atms > 0
                            else 0
                        ),
                    }
                )

            return regional_stats

        except Exception as e:
            logger.error(f"Error getting regional stats: {str(e)}")
            return []

    @staticmethod
    def get_manufacturer_distribution(db: Session) -> List[Dict]:
        """Get ATM distribution by manufacturer"""
        try:
            dist_query = text(
                """
                SELECT 
                    COALESCE(manufacturer, 'Unknown') as manufacturer,
                    COUNT(*) as count,
                    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM atms)), 2) as percentage
                FROM atms
                WHERE status != 'decommissioned'
                GROUP BY manufacturer
                ORDER BY count DESC
            """
            )

            results = db.execute(dist_query).fetchall()

            return [
                {
                    "manufacturer": row.manufacturer,
                    "count": row.count,
                    "percentage": float(row.percentage),
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error getting manufacturer distribution: {str(e)}")
            return []


class ATMMaintenanceHelper:
    """Helper functions for ATM maintenance operations"""

    @staticmethod
    def get_atms_needing_maintenance(
        db: Session, days_threshold: int = 30
    ) -> List[Dict]:
        """Get ATMs that may need maintenance based on various criteria"""
        try:
            # ATMs with recent errors or poor performance
            maintenance_query = text(
                """
                WITH recent_telemetry AS (
                    SELECT 
                        atm_id,
                        COUNT(*) as total_records,
                        COUNT(*) FILTER (WHERE error_code IS NOT NULL) as error_count,
                        AVG(cpu_usage_percent) as avg_cpu,
                        AVG(memory_usage_percent) as avg_memory,
                        AVG(cash_level_percent) as avg_cash_level,
                        MAX(time) as last_seen
                    FROM atm_telemetry 
                    WHERE time >= NOW() - INTERVAL ':days days'
                    GROUP BY atm_id
                ),
                atm_with_stats AS (
                    SELECT 
                        a.atm_id,
                        a.name,
                        a.status,
                        a.location_address,
                        rt.error_count,
                        rt.avg_cpu,
                        rt.avg_memory,
                        rt.avg_cash_level,
                        rt.last_seen,
                        CASE 
                            WHEN rt.error_count > 5 THEN 'High Error Rate'
                            WHEN rt.avg_cpu > 80 THEN 'High CPU Usage'
                            WHEN rt.avg_memory > 85 THEN 'High Memory Usage'
                            WHEN rt.avg_cash_level < 20 THEN 'Low Cash Level'
                            WHEN rt.last_seen < NOW() - INTERVAL '2 hours' THEN 'No Recent Data'
                            ELSE 'OK'
                        END as maintenance_reason
                    FROM atms a
                    LEFT JOIN recent_telemetry rt ON a.atm_id = rt.atm_id
                    WHERE a.status = 'active'
                )
                SELECT *
                FROM atm_with_stats
                WHERE maintenance_reason != 'OK'
                ORDER BY 
                    CASE maintenance_reason
                        WHEN 'High Error Rate' THEN 1
                        WHEN 'No Recent Data' THEN 2
                        WHEN 'High CPU Usage' THEN 3
                        WHEN 'High Memory Usage' THEN 4
                        WHEN 'Low Cash Level' THEN 5
                        ELSE 6
                    END
            """
            )

            results = db.execute(maintenance_query, {"days": days_threshold}).fetchall()

            maintenance_list = []
            for row in results:
                maintenance_list.append(
                    {
                        "atm_id": row.atm_id,
                        "name": row.name,
                        "status": row.status,
                        "location_address": row.location_address,
                        "maintenance_reason": row.maintenance_reason,
                        "error_count": row.error_count or 0,
                        "avg_cpu": round(row.avg_cpu or 0, 1),
                        "avg_memory": round(row.avg_memory or 0, 1),
                        "avg_cash_level": round(row.avg_cash_level or 0, 1),
                        "last_seen": (
                            row.last_seen.isoformat() if row.last_seen else None
                        ),
                        "priority": self._calculate_maintenance_priority(row),
                    }
                )

            return maintenance_list

        except Exception as e:
            logger.error(f"Error getting ATMs needing maintenance: {str(e)}")
            return []

    @staticmethod
    def _calculate_maintenance_priority(row) -> str:
        """Calculate maintenance priority based on various factors"""
        if row.maintenance_reason == "High Error Rate" and (row.error_count or 0) > 10:
            return "URGENT"
        elif row.maintenance_reason == "No Recent Data":
            return "HIGH"
        elif row.maintenance_reason in ["High CPU Usage", "High Memory Usage"]:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def schedule_maintenance(atm_id: str, reason: str, db: Session) -> bool:
        """Schedule an ATM for maintenance"""
        try:
            atm = db.query(ATM).filter(ATM.atm_id == atm_id).first()
            if not atm:
                return False

            atm.status = "maintenance"
            atm.updated_at = datetime.now()
            db.commit()

            logger.info(f"Scheduled ATM {atm_id} for maintenance. Reason: {reason}")
            return True

        except Exception as e:
            logger.error(f"Error scheduling maintenance for {atm_id}: {str(e)}")
            db.rollback()
            return False


class ATMDataImporter:
    """Utility for importing ATM data from various sources"""

    @staticmethod
    def import_from_csv_data(csv_data: List[Dict], db: Session) -> Dict:
        """Import ATMs from CSV data"""
        imported = 0
        errors = []
        skipped = 0

        for row in csv_data:
            try:
                # Validate required fields
                if not row.get("atm_id") or not row.get("name"):
                    errors.append(f"Missing required fields in row: {row}")
                    continue

                # Check if ATM already exists
                atm_id = row["atm_id"].strip().upper()
                existing = db.query(ATM).filter(ATM.atm_id == atm_id).first()
                if existing:
                    skipped += 1
                    continue

                # Create new ATM
                new_atm = ATM(
                    atm_id=atm_id,
                    name=row["name"].strip(),
                    location_address=row.get("location_address", "").strip(),
                    model=row.get("model", "").strip(),
                    manufacturer=row.get("manufacturer", "").strip(),
                    status=row.get("status", "active").lower(),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

                db.add(new_atm)
                imported += 1

            except Exception as e:
                errors.append(f"Error importing row {row}: {str(e)}")

        try:
            if imported > 0:
                db.commit()
        except Exception as e:
            db.rollback()
            errors.append(f"Database commit error: {str(e)}")
            imported = 0

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": errors,
        }

    @staticmethod
    def generate_sample_atms(count: int, regions: List[str] = None) -> List[Dict]:
        """Generate sample ATM data for testing"""
        if not regions:
            regions = ["NYC", "LA", "CHI", "MIA", "SEA"]

        manufacturers = ["NCR", "Diebold", "Hyosung", "Triton", "Hantle"]
        models = ["SecureMax Pro", "CashFlow Elite", "MoneyTech X1", "SafeBank 2000"]

        sample_atms = []
        for i in range(1, count + 1):
            region = regions[i % len(regions)]
            atm_id = f"ATM-{region}-{i:03d}"

            sample_atms.append(
                {
                    "atm_id": atm_id,
                    "name": f"ATM {region} {i:03d}",
                    "location_address": f"{i} Main Street, {region} City",
                    "model": models[i % len(models)],
                    "manufacturer": manufacturers[i % len(manufacturers)],
                    "status": "active",
                }
            )

        return sample_atms


# Utility functions for API endpoints
def validate_atm_bulk_data(atms_data: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Validate bulk ATM data and return valid data and errors"""
    valid_data = []
    errors = []

    for i, atm_data in enumerate(atms_data):
        try:
            # Validate ATM ID
            atm_id = atm_data.get("atm_id", "").strip().upper()
            is_valid, error_msg = ATMValidator.validate_atm_id(atm_id)
            if not is_valid:
                errors.append(f"Row {i+1}: {error_msg}")
                continue

            # Validate name
            name = atm_data.get("name", "").strip()
            if not name:
                errors.append(f"Row {i+1}: Name is required")
                continue

            # Validate status
            status = atm_data.get("status", "active").lower()
            is_valid, error_msg = ATMValidator.validate_status(status)
            if not is_valid:
                errors.append(f"Row {i+1}: {error_msg}")
                continue

            # Add validated data
            validated_atm = {
                "atm_id": atm_id,
                "name": name,
                "location_address": atm_data.get("location_address", "").strip(),
                "model": atm_data.get("model", "").strip(),
                "manufacturer": atm_data.get("manufacturer", "").strip(),
                "status": status,
            }
            valid_data.append(validated_atm)

        except Exception as e:
            errors.append(f"Row {i+1}: Validation error - {str(e)}")

    return valid_data, errors
