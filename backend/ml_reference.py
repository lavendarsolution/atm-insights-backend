# backend/ml/feature_engineering.py
"""
Feature Engineering with 28 features (including maintenance ratio)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ATMFeatureEngineer:
    """Extract 28 features from ATM telemetry data for failure prediction"""
    
    # Critical error codes that indicate hardware failures
    CRITICAL_ERROR_CODES = ['E002', 'E004', 'E005', 'E007', 'E010']
    
    @staticmethod
    def extract_features_from_telemetry(telemetry_history: List[Dict]) -> np.ndarray:
        """Extract 21 historical features from telemetry history"""
        logger.debug(f"Extracting features from {len(telemetry_history)} telemetry records")
        
        if not telemetry_history:
            logger.warning("No telemetry history provided, returning zero features")
            return np.zeros(21)
        
        # Convert to DataFrame
        df = pd.DataFrame(telemetry_history)
        logger.debug(f"DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Ensure time column is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
        
        features = []
        
        # Get last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        if 'time' in df.columns:
            recent_24h = df[df['time'] >= cutoff_time]
            logger.debug(f"Records in last 24h: {len(recent_24h)}")
        else:
            recent_24h = df
            logger.debug("No time column, using all data")
        
        if len(recent_24h) == 0:
            recent_24h = df
            logger.warning("No data in last 24h, using all available data")
        
        # Features 1-4: Status Distribution (online, error, offline, maintenance ratios)
        if 'status' in recent_24h.columns and len(recent_24h) > 0:
            status_counts = recent_24h['status'].value_counts()
            total_records = len(recent_24h)
            
            online_ratio = status_counts.get('online', 0) / total_records
            error_ratio = status_counts.get('error', 0) / total_records
            offline_ratio = status_counts.get('offline', 0) / total_records
            maintenance_ratio = status_counts.get('maintenance', 0) / total_records
            
            features.extend([
                online_ratio,      # Feature 1
                error_ratio,       # Feature 2
                offline_ratio,     # Feature 3
                maintenance_ratio  # Feature 4 (NEW)
            ])
            
            logger.debug(f"Status ratios - Online: {online_ratio:.3f}, Error: {error_ratio:.3f}, "
                        f"Offline: {offline_ratio:.3f}, Maintenance: {maintenance_ratio:.3f}")
        else:
            features.extend([1.0, 0.0, 0.0, 0.0])
            logger.warning("No status column, using default values")
        
        # Features 5-8: Cash Level Statistics
        if 'cash_level_percent' in df.columns:
            cash_levels = df['cash_level_percent'].dropna()
            if len(cash_levels) > 0:
                cash_mean = cash_levels.mean()
                cash_min = cash_levels.min()
                cash_std = cash_levels.std() if len(cash_levels) > 1 else 0
                low_cash_freq = (cash_levels < 20).sum() / len(cash_levels)
                
                features.extend([
                    cash_mean,      # Feature 5
                    cash_min,       # Feature 6
                    cash_std,       # Feature 7
                    low_cash_freq   # Feature 8
                ])
                
                logger.debug(f"Cash stats - Mean: {cash_mean:.1f}, Min: {cash_min:.1f}, "
                            f"Std: {cash_std:.1f}, Low freq: {low_cash_freq:.3f}")
            else:
                features.extend([50.0, 50.0, 0.0, 0.0])
        else:
            features.extend([50.0, 50.0, 0.0, 0.0])
        
        # Features 9-12: Temperature Statistics
        if 'temperature_celsius' in df.columns:
            temps = df['temperature_celsius'].dropna()
            if len(temps) > 0:
                temp_mean = temps.mean()
                temp_max = temps.max()
                temp_std = temps.std() if len(temps) > 1 else 0
                extreme_temp_freq = ((temps > 35) | (temps < 5)).sum() / len(temps)
                
                features.extend([
                    temp_mean,         # Feature 9
                    temp_max,          # Feature 10
                    temp_std,          # Feature 11
                    extreme_temp_freq  # Feature 12
                ])
                
                logger.debug(f"Temp stats - Mean: {temp_mean:.1f}, Max: {temp_max:.1f}, "
                            f"Std: {temp_std:.1f}, Extreme freq: {extreme_temp_freq:.3f}")
            else:
                features.extend([22.0, 22.0, 0.0, 0.0])
        else:
            features.extend([22.0, 22.0, 0.0, 0.0])
        
        # Features 13-15: CPU Usage Statistics
        if 'cpu_usage_percent' in df.columns:
            cpu_usage = df['cpu_usage_percent'].dropna()
            if len(cpu_usage) > 0:
                cpu_mean = cpu_usage.mean()
                cpu_max = cpu_usage.max()
                high_cpu_freq = (cpu_usage > 80).sum() / len(cpu_usage)
                
                features.extend([
                    cpu_mean,      # Feature 13
                    cpu_max,       # Feature 14
                    high_cpu_freq  # Feature 15
                ])
                
                logger.debug(f"CPU stats - Mean: {cpu_mean:.1f}, Max: {cpu_max:.1f}, "
                            f"High freq: {high_cpu_freq:.3f}")
            else:
                features.extend([30.0, 30.0, 0.0])
        else:
            features.extend([30.0, 30.0, 0.0])
        
        # Features 16-18: Memory Usage Statistics
        if 'memory_usage_percent' in df.columns:
            memory_usage = df['memory_usage_percent'].dropna()
            if len(memory_usage) > 0:
                mem_mean = memory_usage.mean()
                mem_max = memory_usage.max()
                high_mem_freq = (memory_usage > 85).sum() / len(memory_usage)
                
                features.extend([
                    mem_mean,      # Feature 16
                    mem_max,       # Feature 17
                    high_mem_freq  # Feature 18
                ])
                
                logger.debug(f"Memory stats - Mean: {mem_mean:.1f}, Max: {mem_max:.1f}, "
                            f"High freq: {high_mem_freq:.3f}")
            else:
                features.extend([50.0, 50.0, 0.0])
        else:
            features.extend([50.0, 50.0, 0.0])
        
        # Feature 19: Network Disconnect Frequency
        if 'network_status' in df.columns:
            network_issues = df['network_status'].dropna()
            if len(network_issues) > 0:
                disconnect_freq = (network_issues == 'disconnected').sum() / len(network_issues)
                features.append(disconnect_freq)  # Feature 19
                logger.debug(f"Network disconnect freq: {disconnect_freq:.3f}")
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Feature 20: Error Frequency
        if 'error_code' in df.columns:
            error_count = df['error_code'].notna().sum()
            error_freq = error_count / len(df) if len(df) > 0 else 0.0
            features.append(error_freq)  # Feature 20
            logger.debug(f"Error frequency: {error_freq:.3f} ({error_count} errors)")
        else:
            features.append(0.0)
        
        # Feature 21: Critical Error Frequency (NEW - specific to critical codes)
        if 'error_code' in df.columns:
            critical_errors = df[df['error_code'].isin(ATMFeatureEngineer.CRITICAL_ERROR_CODES)]
            critical_error_freq = len(critical_errors) / len(df) if len(df) > 0 else 0.0
            features.append(critical_error_freq)  # Feature 21
            logger.debug(f"Critical error frequency: {critical_error_freq:.3f}")
        else:
            features.append(0.0)
        
        # Ensure exactly 21 features
        features = features[:21]
        while len(features) < 21:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_realtime_features(current_telemetry: Dict, history: List[Dict]) -> np.ndarray:
        """Extract all 28 features combining current state and history"""
        logger.debug(f"Extracting realtime features for ATM with current status: {current_telemetry.get('status')}")
        
        # Get 21 historical features
        hist_features = ATMFeatureEngineer.extract_features_from_telemetry(history)
        
        # Extract 7 current state features
        current_features = []
        
        # Feature 22: Current status encoding
        status_map = {'online': 0, 'offline': 1, 'error': 2, 'maintenance': 3}
        current_status = current_telemetry.get('status', 'online')
        status_encoded = float(status_map.get(current_status, 0))
        current_features.append(status_encoded)
        
        # Features 23-28: Current metrics
        current_features.extend([
            float(current_telemetry.get('cash_level_percent', 50.0)),      # Feature 23
            float(current_telemetry.get('temperature_celsius', 22.0)),     # Feature 24
            float(current_telemetry.get('cpu_usage_percent', 30.0)),       # Feature 25
            float(current_telemetry.get('memory_usage_percent', 50.0)),    # Feature 26
            float(current_telemetry.get('network_latency_ms', 100) / 1000.0), # Feature 27
            1.0 if current_telemetry.get('error_code') in ATMFeatureEngineer.CRITICAL_ERROR_CODES else 0.0, # Feature 28
        ])
        
        logger.debug(f"Current features - Status: {current_status}({status_encoded}), "
                    f"Cash: {current_features[1]:.1f}, Temp: {current_features[2]:.1f}, "
                    f"Critical error: {current_features[6]}")
        
        # Combine all 28 features
        all_features = np.concatenate([hist_features, current_features])
        
        return all_features.astype(np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all 28 features"""
        return [
            # Historical features (1-21)
            "online_ratio_24h",              # 1
            "error_ratio_24h",               # 2
            "offline_ratio_24h",             # 3
            "maintenance_ratio_24h",         # 4 (NEW)
            "cash_level_mean",               # 5
            "cash_level_min",                # 6
            "cash_level_std",                # 7
            "low_cash_frequency",            # 8
            "temperature_mean",              # 9
            "temperature_max",               # 10
            "temperature_std",               # 11
            "extreme_temp_frequency",        # 12
            "cpu_usage_mean",                # 13
            "cpu_usage_max",                 # 14
            "high_cpu_frequency",            # 15
            "memory_usage_mean",             # 16
            "memory_usage_max",              # 17
            "high_memory_frequency",         # 18
            "network_disconnect_frequency",   # 19
            "error_frequency",               # 20
            "critical_error_frequency",      # 21 (NEW)
            # Current state features (22-28)
            "current_status_encoded",        # 22
            "current_cash_level",            # 23
            "current_temperature",           # 24
            "current_cpu_usage",             # 25
            "current_memory_usage",          # 26
            "current_network_latency_sec",   # 27
            "current_has_critical_error",    # 28 (UPDATED)
        ]


# backend/ml/model_training.py
"""
Model training with proper labeling for failure prediction
"""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from sqlalchemy import text
import logging
import os
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class ATMFailureModelTrainer:
    """Train models for ATM failure prediction"""
    
    CRITICAL_ERROR_CODES = ['E002', 'E004', 'E005', 'E007', 'E010']
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.failure_model = None
        self.model_dir = "ml/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_labeled_dataset(self, db_session) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create labeled dataset from telemetry data"""
        logger.info("Creating labeled dataset from telemetry data...")
        
        # SQL query to create labels based on our agreed approach
        query = text("""
        WITH maintenance_events AS (
            SELECT atm_id, time as maint_time 
            FROM atm_telemetry 
            WHERE status = 'maintenance'
        ),
        labeled_data AS (
            SELECT 
                t.*,
                CASE 
                    WHEN (t.status = 'error' AND t.error_code IN ('E002','E004','E005','E007','E010'))
                      OR EXISTS (
                          SELECT 1 FROM maintenance_events m 
                          WHERE m.atm_id = t.atm_id 
                          AND m.maint_time BETWEEN t.time AND t.time + INTERVAL '4 hours'
                      )
                    THEN 1
                    ELSE 0
                END as failure_label
            FROM atm_telemetry t
            WHERE t.status != 'maintenance'  -- Don't train on maintenance records
            ORDER BY t.atm_id, t.time
        )
        SELECT * FROM labeled_data
        """)
        
        # Execute query and get results
        result = db_session.execute(query)
        df = pd.DataFrame(result.fetchall())
        
        if df.empty:
            logger.error("No telemetry data found!")
            return pd.DataFrame(), np.array([])
        
        logger.info(f"Loaded {len(df)} telemetry records")
        logger.info(f"Failure labels - 0: {(df['failure_label']==0).sum()}, 1: {(df['failure_label']==1).sum()}")
        logger.info(f"Class balance: {(df['failure_label']==1).sum() / len(df) * 100:.2f}% positive")
        
        # Debug: Show some examples of labeled data
        positive_samples = df[df['failure_label'] == 1].head(5)
        if not positive_samples.empty:
            logger.debug("Sample positive (failure) records:")
            for _, row in positive_samples.iterrows():
                logger.debug(f"  ATM: {row['atm_id']}, Status: {row['status']}, "
                           f"Error: {row['error_code']}, Time: {row['time']}")
        
        return df, df['failure_label'].values
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from telemetry dataframe"""
        from ml.feature_engineering import ATMFeatureEngineer
        
        logger.info("Extracting features from telemetry data...")
        
        # Group by ATM to get history
        feature_engineer = ATMFeatureEngineer()
        all_features = []
        
        # Get unique ATMs
        atm_ids = df['atm_id'].unique()
        logger.info(f"Processing {len(atm_ids)} unique ATMs")
        
        for atm_id in atm_ids:
            atm_data = df[df['atm_id'] == atm_id].sort_values('time')
            
            # For each record, use previous records as history
            for i in range(1, len(atm_data)):
                # Current record
                current = atm_data.iloc[i].to_dict()
                
                # History (all previous records, max 24 hours)
                current_time = pd.to_datetime(current['time'])
                history_start = current_time - timedelta(hours=24)
                history_mask = (atm_data['time'] < current_time) & (atm_data['time'] >= history_start)
                history = atm_data[history_mask].to_dict('records')
                
                if len(history) > 0:  # Only if we have history
                    features = feature_engineer.extract_realtime_features(current, history)
                    all_features.append(features)
        
        logger.info(f"Extracted features for {len(all_features)} samples")
        
        if not all_features:
            logger.error("No features extracted!")
            return np.array([])
        
        return np.array(all_features)
    
    def train_failure_prediction_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost model for failure prediction"""
        logger.info("Training failure prediction model...")
        
        if len(X) == 0:
            logger.error("No training data available!")
            return {}
        
        # Handle class imbalance
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        logger.info(f"Class weight for positive class: {pos_weight:.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with class weight
        self.failure_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training XGBoost model...")
        self.failure_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        y_pred = self.failure_model.predict(X_test_scaled)
        y_pred_proba = self.failure_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"Model AUC: {auc_score:.3f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance
        from ml.feature_engineering import ATMFeatureEngineer
        feature_names = ATMFeatureEngineer.get_feature_names()
        feature_importance = dict(zip(feature_names, self.failure_model.feature_importances_))
        
        # Log top 10 features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 Important Features:")
        for i, (feat, imp) in enumerate(top_features, 1):
            logger.info(f"{i}. {feat}: {imp:.4f}")
        
        # Save models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f"failure_model_{timestamp}.pkl")
        joblib.dump(self.failure_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save as 'latest' for easy loading
        joblib.dump(self.failure_model, os.path.join(self.model_dir, "failure_model_latest.pkl"))
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            "auc_score": auc_score,
            "model_path": model_path,
            "test_accuracy": (y_pred == y_test).mean(),
            "feature_importance": feature_importance,
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum())
        }


# backend/services/ml_service.py
"""
ML Service for failure prediction only (no anomaly detection)
"""
import asyncio
import joblib
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
from sqlalchemy.orm import Session

from ml.feature_engineering import ATMFeatureEngineer
from services.cache_service import CacheService

logger = logging.getLogger(__name__)

class MLPredictionService:
    """ML prediction service for ATM failure prediction"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.feature_engineer = ATMFeatureEngineer()
        self.failure_model = None
        self.scaler = None
        self.model_loaded = False
        
    async def initialize(self):
        """Load ML models"""
        try:
            model_dir = "ml/models"
            
            # Load failure prediction model
            failure_model_path = os.path.join(model_dir, "failure_model_latest.pkl")
            if os.path.exists(failure_model_path):
                self.failure_model = joblib.load(failure_model_path)
                logger.info("✅ Loaded failure prediction model")
            else:
                logger.warning("⚠️ No failure prediction model found at " + failure_model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded feature scaler")
            else:
                logger.warning("⚠️ No scaler found at " + scaler_path)
            
            self.model_loaded = bool(self.failure_model and self.scaler)
            
            if self.model_loaded:
                logger.info("✅ ML service initialized successfully")
            else:
                logger.warning("⚠️ ML service initialized without models")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
            self.model_loaded = False
    
    async def predict_failure(
        self, 
        atm_id: str, 
        current_telemetry: Dict,
        telemetry_history: List[Dict]
    ) -> Dict[str, any]:
        """Predict failure probability for an ATM"""
        logger.debug(f"Predicting failure for ATM {atm_id}")
        
        try:
            if not self.failure_model or not self.scaler:
                logger.warning(f"Model not loaded for prediction of {atm_id}")
                return {
                    "atm_id": atm_id,
                    "failure_probability": 0.5,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "prediction_available": False,
                    "reason": "Model not loaded"
                }
            
            # Extract features
            features = self.feature_engineer.extract_realtime_features(
                current_telemetry, telemetry_history
            )
            
            logger.debug(f"Extracted {len(features)} features for {atm_id}")
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            failure_prob = self.failure_model.predict_proba(features_scaled)[0, 1]
            
            # Calculate confidence (based on prediction certainty)
            confidence = abs(failure_prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
            
            # Determine risk level
            if failure_prob >= 0.8:
                risk_level = "critical"
            elif failure_prob >= 0.6:
                risk_level = "high"
            elif failure_prob >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            logger.info(f"Prediction for {atm_id}: {failure_prob:.2%} ({risk_level})")
            
            # Get feature importance for this prediction
            feature_contributions = self._get_feature_contributions(features_scaled)
            
            result = {
                "atm_id": atm_id,
                "failure_probability": float(failure_prob),
                "confidence": float(confidence),
                "risk_level": risk_level,
                "prediction_available": True,
                "timestamp": datetime.now().isoformat(),
                "top_risk_factors": feature_contributions[:5] if feature_contributions else []
            }
            
            # Cache result
            cache_key = f"ml_prediction:{atm_id}"
            await self.cache_service.set(cache_key, result, ttl=300)
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting failure for {atm_id}: {e}", exc_info=True)
            return {
                "atm_id": atm_id,
                "failure_probability": 0.5,
                "confidence": 0.0,
                "risk_level": "unknown",
                "prediction_available": False,
                "reason": str(e)
            }
    
    def _get_feature_contributions(self, features_scaled: np.ndarray) -> List[Dict]:
        """Get top contributing features for this prediction"""
        try:
            if hasattr(self.failure_model, 'feature_importances_'):
                feature_names = self.feature_engineer.get_feature_names()
                importances = self.failure_model.feature_importances_
                
                # Get feature values
                feature_values = features_scaled[0]
                
                # Calculate contribution (importance * value)
                contributions = []
                for i, (name, importance) in enumerate(zip(feature_names, importances)):
                    if importance > 0.01:  # Only significant features
                        contributions.append({
                            "feature": name,
                            "importance": float(importance),
                            "value": float(feature_values[i]),
                            "contribution": float(importance * abs(feature_values[i]))
                        })
                
                # Sort by contribution
                contributions.sort(key=lambda x: x['contribution'], reverse=True)
                return contributions
        except Exception as e:
            logger.debug(f"Could not get feature contributions: {e}")
            return []
    
    async def batch_predict_failures(
        self,
        atm_predictions: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Batch predict failures for multiple ATMs"""
        logger.info(f"Batch predicting for {len(atm_predictions)} ATMs")
        
        tasks = []
        for pred_request in atm_predictions:
            task = self.predict_failure(
                pred_request["atm_id"],
                pred_request["current_telemetry"],
                pred_request["telemetry_history"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch prediction error for {atm_predictions[i]['atm_id']}: {result}")
                processed_results.append({
                    "atm_id": atm_predictions[i]["atm_id"],
                    "failure_probability": 0.5,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "prediction_available": False,
                    "reason": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results


# backend/api/routes/v1/predictions.py
"""
API endpoints for ML predictions (no training endpoint)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database import get_db
from schemas.predictions import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
from services.ml_service import MLPredictionService
from services.telemetry_service import TelemetryService
from dependencies.auth import get_current_user
from models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instances (will be set during startup)
ml_service: MLPredictionService = None
telemetry_service: TelemetryService = None

def set_services(ml_svc: MLPredictionService, telemetry_svc: TelemetryService):
    """Set service instances"""
    global ml_service, telemetry_service
    ml_service = ml_svc
    telemetry_service = telemetry_svc

@router.post("/predictions/failure", response_model=PredictionResponse)
async def predict_atm_failure(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Predict failure probability for a specific ATM"""
    logger.info(f"Prediction requested for ATM {request.atm_id} by user {current_user.email}")
    
    try:
        # Get telemetry history
        history = await telemetry_service.get_atm_telemetry_history(
            db, 
            request.atm_id, 
            hours=request.history_hours or 24
        )
        
        if not history:
            logger.warning(f"No telemetry history found for ATM {request.atm_id}")
            return PredictionResponse(
                atm_id=request.atm_id,
                failure_probability=0.5,
                confidence=0.0,
                risk_level="unknown",
                prediction_available=False,
                timestamp=datetime.now().isoformat(),
                reason="No telemetry data available"
            )
        
        # Get current telemetry (most recent)
        current_telemetry = history[0]
        
        logger.debug(f"Found {len(history)} telemetry records for {request.atm_id}")
        
        # Make prediction
        prediction = await ml_service.predict_failure(
            request.atm_id,
            current_telemetry,
            history
        )
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error predicting failure for {request.atm_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/batch", response_model=BatchPredictionResponse)
async def batch_predict_failures(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Batch predict failures for multiple ATMs"""
    logger.info(f"Batch prediction requested for {len(request.atm_ids)} ATMs by user {current_user.email}")
    
    try:
        start_time = datetime.now()
        
        # Prepare batch prediction data
        prediction_requests = []
        for atm_id in request.atm_ids:
            history = await telemetry_service.get_atm_telemetry_history(
                db, atm_id, hours=24
            )
            
            if history:
                current_telemetry = history[0]
                prediction_requests.append({
                    "atm_id": atm_id,
                    "current_telemetry": current_telemetry,
                    "telemetry_history": history
                })
            else:
                logger.warning(f"No telemetry data for ATM {atm_id}")
        
        # Make batch predictions
        if prediction_requests:
            predictions = await ml_service.batch_predict_failures(prediction_requests)
        else:
            predictions = []
        
        # Add missing ATMs with default values
        predicted_atm_ids = {p["atm_id"] for p in predictions}
        for atm_id in request.atm_ids:
            if atm_id not in predicted_atm_ids:
                predictions.append({
                    "atm_id": atm_id,
                    "failure_probability": 0.5,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "prediction_available": False,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "No telemetry data"
                })
        
        # Filter high-risk ATMs
        high_risk_atms = [
            p for p in predictions 
            if p.get("failure_probability", 0) > 0.7
        ]
        
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction completed: {len(predictions)} ATMs, "
                   f"{len(high_risk_atms)} high-risk, {processing_time_ms:.0f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            high_risk_count=len(high_risk_atms),
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    current_user: User = Depends(get_current_user)
):
    """Get information about loaded ML models"""
    try:
        model_info = {
            "failure_model_loaded": ml_service.failure_model is not None,
            "anomaly_model_loaded": False,  # We don't use anomaly detection
            "models_available": ml_service.model_loaded,
            "last_updated": datetime.now().isoformat()
        }
        
        if ml_service.failure_model:
            model_info["failure_model_type"] = type(ml_service.failure_model).__name__
            model_info["n_features"] = 28  # We know we use 28 features
            
        logger.debug(f"Model info requested by {current_user.email}: {model_info}")
        
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/high-risk")
async def get_high_risk_atms(
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Risk threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of ATMs to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get ATMs with high failure risk"""
    logger.info(f"High-risk ATMs requested by {current_user.email} (threshold={threshold})")
    
    try:
        from models import ATM
        
        # Get all active ATMs
        active_atms = db.query(ATM).filter(ATM.status == 'active').limit(100).all()
        
        if not active_atms:
            return {
                "high_risk_atms": [],
                "threshold": threshold,
                "total_checked": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get predictions for each ATM
        predictions = []
        for atm in active_atms:
            history = await telemetry_service.get_atm_telemetry_history(
                db, atm.atm_id, hours=24
            )
            
            if history:
                current_telemetry = history[0]
                prediction = await ml_service.predict_failure(
                    atm.atm_id,
                    current_telemetry,
                    history
                )
                
                if prediction["failure_probability"] >= threshold:
                    predictions.append({
                        "atm_id": atm.atm_id,
                        "name": atm.name,
                        "location": atm.location_address,
                        "failure_probability": prediction["failure_probability"],
                        "risk_level": prediction["risk_level"],
                        "top_risk_factors": prediction.get("top_risk_factors", [])
                    })
        
        # Sort by risk and limit
        predictions.sort(key=lambda x: x["failure_probability"], reverse=True)
        high_risk_atms = predictions[:limit]
        
        logger.info(f"Found {len(high_risk_atms)} high-risk ATMs out of {len(active_atms)} checked")
        
        return {
            "high_risk_atms": high_risk_atms,
            "threshold": threshold,
            "total_checked": len(active_atms),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting high-risk ATMs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# backend/schemas/predictions.py
"""
Schemas for ML predictions
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request for single ATM failure prediction"""
    atm_id: str = Field(..., description="ATM identifier")
    history_hours: Optional[int] = Field(24, ge=1, le=168, description="Hours of history to consider")

class PredictionResponse(BaseModel):
    """Response for failure prediction"""
    atm_id: str
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of failure (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical, unknown")
    prediction_available: bool
    timestamp: str
    reason: Optional[str] = Field(None, description="Reason if prediction unavailable")
    top_risk_factors: Optional[List[Dict[str, float]]] = Field(None, description="Top contributing features")

class BatchPredictionRequest(BaseModel):
    """Request for batch ATM predictions"""
    atm_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of ATM IDs")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    high_risk_count: int = Field(..., description="Number of high-risk ATMs")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ModelInfoResponse(BaseModel):
    """Information about loaded ML models"""
    failure_model_loaded: bool
    anomaly_model_loaded: bool  # Always False since we don't use it
    models_available: bool
    failure_model_type: Optional[str] = None
    n_features: Optional[int] = None
    last_updated: str


# backend/ml/train_models.py
#!/usr/bin/env python3
"""
Training script for ATM failure prediction models
Run this to train initial models or retrain with new data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_training import ATMFailureModelTrainer
from database.session import SessionLocal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Train ML models for failure prediction"""
    logger.info("="*60)
    logger.info("ATM Failure Prediction Model Training")
    logger.info("="*60)
    
    db = SessionLocal()
    trainer = ATMFailureModelTrainer()
    
    try:
        # Step 1: Create labeled dataset
        logger.info("\nStep 1: Creating labeled dataset...")
        df, labels = trainer.create_labeled_dataset(db)
        
        if df.empty:
            logger.error("No data available for training!")
            return
        
        logger.info(f"Dataset size: {len(df)} records")
        
        # Step 2: Extract features
        logger.info("\nStep 2: Extracting features (this may take a while)...")
        X = trainer.prepare_features(df)
        
        if len(X) == 0:
            logger.error("No features extracted!")
            return
        
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Step 3: Train model
        logger.info("\nStep 3: Training failure prediction model...")
        results = trainer.train_failure_prediction_model(X, labels[:len(X)])
        
        if results:
            logger.info("\n" + "="*60)
            logger.info("✅ Model training completed successfully!")
            logger.info(f"   - AUC Score: {results['auc_score']:.3f}")
            logger.info(f"   - Test Accuracy: {results['test_accuracy']:.3f}")
            logger.info(f"   - Model saved to: {results['model_path']}")
            logger.info(f"   - Training samples: {results['positive_samples']} positive, "
                       f"{results['negative_samples']} negative")
            logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()