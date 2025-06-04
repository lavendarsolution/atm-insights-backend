import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from config import settings
from database.session import SessionLocal
from services.cache_service import CacheService
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Scheduler for running ML predictions periodically"""

    def __init__(self, ml_service, cache_service: CacheService):
        self.ml_service = ml_service
        self.cache_service = cache_service
        self.is_running = False
        self.prediction_task = None

    async def start_prediction_scheduler(self):
        """Start the prediction scheduler"""
        if self.is_running:
            logger.warning("Prediction scheduler is already running")
            return

        self.is_running = True

        # Schedule predictions every 20 minutes
        self.prediction_task = asyncio.create_task(
            self._prediction_loop(interval=1200)
        )  # 20 minutes
        logger.info("🤖 ML Prediction scheduler started (20-minute intervals)")

    async def stop_prediction_scheduler(self):
        """Stop the prediction scheduler"""
        self.is_running = False

        if self.prediction_task:
            self.prediction_task.cancel()
            try:
                await self.prediction_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 ML Prediction scheduler stopped")

    async def _prediction_loop(self, interval: int):
        """Run predictions periodically"""
        while self.is_running:
            try:
                await self._run_bulk_predictions()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _run_bulk_predictions(self):
        """Run predictions for all active ATMs"""
        try:
            # Import here to avoid circular imports
            from models.atm import ATM

            db = SessionLocal()
            try:
                # Get all active ATMs
                active_atms = db.query(ATM).filter(ATM.status == "active").all()
                atm_ids = [atm.atm_id for atm in active_atms]

                if not atm_ids:
                    logger.info("No active ATMs found for predictions")
                    return

                logger.info(f"🔮 Running predictions for {len(atm_ids)} ATMs")
                start_time = datetime.now()

                # Run bulk predictions
                predictions = await self.ml_service.predict_failure_bulk_optimized(
                    db=db,
                    atm_ids=atm_ids,
                    use_cache=False,  # Force fresh predictions
                    cache_ttl=1800,  # Cache for 30 minutes
                )

                processing_time = (datetime.now() - start_time).total_seconds()

                # Store aggregated results for dashboard
                await self._store_prediction_summary(predictions)

                # Publish high-risk alerts
                await self._check_high_risk_atms(predictions)

                logger.info(
                    f"✅ Completed predictions for {len(predictions)} ATMs in {processing_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Database error in bulk predictions: {e}")
                db.rollback()
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error running bulk predictions: {e}")

    async def _store_prediction_summary(self, predictions):
        """Store prediction summary for dashboard"""
        try:
            high_risk_atms = [p for p in predictions if p.failure_probability > 0.7]
            critical_risk_atms = [p for p in predictions if p.failure_probability > 0.8]

            # Sort by failure probability (highest first)
            high_risk_atms.sort(key=lambda x: x.failure_probability, reverse=True)
            critical_risk_atms.sort(key=lambda x: x.failure_probability, reverse=True)

            summary = {
                "total_predictions": len(predictions),
                "high_risk_atms": len(high_risk_atms),
                "critical_risk_atms": len(critical_risk_atms),
                "last_updated": datetime.now().isoformat(),
                "high_risk_atms_list": [
                    {
                        "atm_id": p.atm_id,
                        "failure_probability": p.failure_probability,
                        "confidence": p.confidence,
                        "risk_level": p.risk_level,
                        "top_risk_factors": (
                            p.top_risk_factors[:3] if p.top_risk_factors else []
                        ),  # Top 3 factors
                    }
                    for p in high_risk_atms[:20]  # Limit to top 20 high-risk ATMs
                ],
                "critical_risk_atms_list": [
                    {
                        "atm_id": p.atm_id,
                        "failure_probability": p.failure_probability,
                        "confidence": p.confidence,
                        "risk_level": p.risk_level,
                        "top_risk_factors": (
                            p.top_risk_factors[:3] if p.top_risk_factors else []
                        ),
                    }
                    for p in critical_risk_atms[
                        :10
                    ]  # Limit to top 10 critical-risk ATMs
                ],
            }

            # Cache summary for frontend
            await self.cache_service.set("ml_prediction_summary", summary, ttl=1800)

            # Also cache individual high-risk and critical lists for quick access
            await self.cache_service.set(
                "ml_high_risk_atms", high_risk_atms[:20], ttl=1800
            )
            await self.cache_service.set(
                "ml_critical_risk_atms", critical_risk_atms[:10], ttl=1800
            )

            logger.info(
                f"📊 Stored prediction summary: {len(high_risk_atms)} high-risk, {len(critical_risk_atms)} critical-risk ATMs"
            )

        except Exception as e:
            logger.error(f"Error storing prediction summary: {e}")

    async def _check_high_risk_atms(self, predictions):
        """Check for high-risk ATMs and publish alerts"""
        try:
            # Define thresholds
            critical_threshold = 0.8
            high_threshold = 0.7

            critical_risk_atms = [
                p for p in predictions if p.failure_probability > critical_threshold
            ]
            high_risk_atms = [
                p
                for p in predictions
                if high_threshold < p.failure_probability <= critical_threshold
            ]

            # Publish critical risk alerts if any
            if critical_risk_atms:
                critical_alert_data = {
                    "type": "critical_risk_prediction",
                    "severity": "critical",
                    "count": len(critical_risk_atms),
                    "threshold": critical_threshold,
                    "atms": [
                        {
                            "atm_id": p.atm_id,
                            "failure_probability": p.failure_probability,
                            "risk_level": p.risk_level,
                            "confidence": p.confidence,
                        }
                        for p in critical_risk_atms[:5]  # Top 5 only for alerts
                    ],
                    "timestamp": datetime.now().isoformat(),
                }

                # Publish to WebSocket for real-time updates
                # await self.cache_service.publish("ml_alerts", critical_alert_data)
                logger.warning(
                    f"🚨 Published CRITICAL risk alert for {len(critical_risk_atms)} ATMs"
                )

            # Publish high risk alerts if significant number
            if len(high_risk_atms) >= 5:  # Only alert if 5+ ATMs are high risk
                high_alert_data = {
                    "type": "high_risk_prediction",
                    "severity": "high",
                    "count": len(high_risk_atms),
                    "threshold": high_threshold,
                    "atms": [
                        {
                            "atm_id": p.atm_id,
                            "failure_probability": p.failure_probability,
                            "risk_level": p.risk_level,
                            "confidence": p.confidence,
                        }
                        for p in high_risk_atms[:5]  # Top 5 only for alerts
                    ],
                    "timestamp": datetime.now().isoformat(),
                }

                # Publish to WebSocket for real-time updates
                # await self.cache_service.publish("ml_alerts", high_alert_data)
                logger.info(
                    f"⚠️ Published HIGH risk alert for {len(high_risk_atms)} ATMs"
                )

            # Store latest alert counts for dashboard
            alert_summary = {
                "critical_count": len(critical_risk_atms),
                "high_count": len(high_risk_atms),
                "last_check": datetime.now().isoformat(),
                "next_check": (datetime.now() + timedelta(minutes=20)).isoformat(),
            }
            # await self.cache_service.set("ml_alert_summary", alert_summary, ttl=1800)

        except Exception as e:
            logger.error(f"Error checking high-risk ATMs: {e}")
