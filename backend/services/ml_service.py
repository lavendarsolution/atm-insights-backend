import asyncio
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import os
from sqlalchemy.orm import Session
import time
from dataclasses import dataclass
from ml.feature_engineering import OptimizedATMFeatureEngineer
from services.cache_service import CacheService

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    atm_id: str
    failure_probability: float
    confidence: float
    risk_level: str
    prediction_available: bool
    timestamp: str
    top_risk_factors: Optional[List[Dict]] = None
    reason: Optional[str] = None
    processing_time_ms: Optional[float] = None

class VectorizedMLPredictionService:
    """ML prediction service with vectorized batch processing"""
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'critical': 0.8,
        'high': 0.6,
        'medium': 0.4,
        'low': 0.0
    }
    
    def __init__(self, cache_service: CacheService, max_workers: int = None):
        self.cache_service = cache_service
        self.feature_engineer = OptimizedATMFeatureEngineer()
        self.failure_model = None
        self.scaler = None
        self.model_loaded = False
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
        # Performance monitoring
        self.prediction_stats = {
            'total_predictions': 0,
            'total_processing_time': 0.0,
            'batch_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def initialize(self):
        """Load ML models with error handling"""
        try:
            model_dir = "ml/models"
            
            # Load failure prediction model
            failure_model_path = os.path.join(model_dir, "failure_model_latest.pkl")
            if os.path.exists(failure_model_path):
                self.failure_model = joblib.load(failure_model_path)
                logger.info("✅ Loaded optimized failure prediction model")
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
                # Warm up the model with dummy prediction
                await self._warmup_model()
            else:
                logger.warning("⚠️ ML service initialized without models")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {e}")
            self.model_loaded = False
    
    async def _warmup_model(self):
        """Warm up the model to avoid cold start latency"""
        try:
            dummy_features = np.random.random((1, 28)).astype(np.float32)
            dummy_features_scaled = self.scaler.transform(dummy_features)
            _ = self.failure_model.predict_proba(dummy_features_scaled)
            logger.debug("🔥 Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def predict_failure_bulk_optimized(
        self, 
        db: Session,
        atm_ids: List[str],
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> List[PredictionResult]:
        """
        Vectorized bulk prediction with massive performance optimizations
        
        Args:
            db: Database session
            atm_ids: List of ATM IDs to predict
            use_cache: Whether to use caching
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            List of PredictionResult objects
        """
        start_time = time.time()
        logger.info(f"🚀 Starting bulk prediction for {len(atm_ids)} ATMs")
        
        if not self.model_loaded:
            logger.warning("Model not loaded, returning default predictions")
            return [self._create_default_prediction(atm_id, "Model not loaded") for atm_id in atm_ids]
        
        try:
            # Step 1: Check cache for existing predictions
            cached_results, uncached_atm_ids = await self._get_cached_predictions(
                atm_ids, use_cache
            )
            
            # Step 2: Process uncached ATMs if any
            new_predictions = []
            if uncached_atm_ids:
                new_predictions = await self._process_uncached_predictions(
                    db, uncached_atm_ids, cache_ttl
                )
            
            # Step 3: Combine cached and new predictions
            all_predictions = cached_results + new_predictions
            
            # Step 4: Sort predictions to match input order
            atm_id_to_prediction = {pred.atm_id: pred for pred in all_predictions}
            ordered_predictions = [
                atm_id_to_prediction.get(atm_id, self._create_default_prediction(atm_id, "Processing failed"))
                for atm_id in atm_ids
            ]
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.prediction_stats['total_predictions'] += len(atm_ids)
            self.prediction_stats['total_processing_time'] += processing_time
            self.prediction_stats['batch_predictions'] += 1
            
            logger.info(f"✅ Bulk prediction completed in {processing_time:.2f}ms for {len(atm_ids)} ATMs")
            logger.info(f"📊 Cache hits: {len(cached_results)}, Cache misses: {len(uncached_atm_ids)}")
            
            return ordered_predictions
            
        except Exception as e:
            logger.error(f"❌ Bulk prediction failed: {str(e)}")
            return [self._create_default_prediction(atm_id, str(e)) for atm_id in atm_ids]
    
    async def _get_cached_predictions(
        self, 
        atm_ids: List[str], 
        use_cache: bool
    ) -> Tuple[List[PredictionResult], List[str]]:
        """Get cached predictions and return uncached ATM IDs"""
        cached_results = []
        uncached_atm_ids = []
        
        if not use_cache:
            return cached_results, atm_ids
        
        # Batch cache lookup
        cache_keys = [f"ml_prediction_v2:{atm_id}" for atm_id in atm_ids]
        
        # Use asyncio.gather for concurrent cache lookups
        cache_tasks = [self.cache_service.get(key) for key in cache_keys]
        cache_responses = await asyncio.gather(*cache_tasks, return_exceptions=True)
        
        for atm_id, cache_response in zip(atm_ids, cache_responses):
            if isinstance(cache_response, Exception) or cache_response is None:
                uncached_atm_ids.append(atm_id)
                self.prediction_stats['cache_misses'] += 1
            else:
                try:
                    cached_results.append(PredictionResult(**cache_response))
                    self.prediction_stats['cache_hits'] += 1
                except Exception as e:
                    logger.debug(f"Invalid cache data for {atm_id}: {e}")
                    uncached_atm_ids.append(atm_id)
                    self.prediction_stats['cache_misses'] += 1
        
        return cached_results, uncached_atm_ids
    
    async def _process_uncached_predictions(
        self, 
        db: Session, 
        atm_ids: List[str],
        cache_ttl: int
    ) -> List[PredictionResult]:
        """Process predictions for uncached ATMs with vectorized operations"""
        
        # Step 1: Bulk extract features using optimized database queries
        feature_results = self.feature_engineer.bulk_extract_features_from_database(
            db, atm_ids, hours=24
        )
        
        # Step 2: Prepare feature matrix for vectorized prediction
        feature_matrix, valid_atm_ids = self._prepare_feature_matrix(feature_results, atm_ids)
        
        # Step 3: Vectorized prediction
        if len(feature_matrix) > 0:
            predictions_array = self._vectorized_predict(feature_matrix)
            confidence_array = self._calculate_confidence_vectorized(predictions_array)
            risk_levels = self._calculate_risk_levels_vectorized(predictions_array)
        else:
            predictions_array = np.array([])
            confidence_array = np.array([])
            risk_levels = []
        
        # Step 4: Create prediction results
        prediction_results = []
        timestamp = datetime.now().isoformat()
        
        for i, atm_id in enumerate(valid_atm_ids):
            if i < len(predictions_array):
                prediction_result = PredictionResult(
                    atm_id=atm_id,
                    failure_probability=float(predictions_array[i]),
                    confidence=float(confidence_array[i]),
                    risk_level=risk_levels[i],
                    prediction_available=True,
                    timestamp=timestamp,
                    top_risk_factors=self._get_feature_contributions_single(
                        feature_results.get(atm_id, np.zeros(28))
                    )
                )
            else:
                prediction_result = self._create_default_prediction(atm_id, "Feature extraction failed")
            
            prediction_results.append(prediction_result)
        
        # Step 5: Cache results asynchronously
        await self._cache_predictions_batch(prediction_results, cache_ttl)
        
        return prediction_results
    
    def _prepare_feature_matrix(
        self, 
        feature_results: Dict[str, np.ndarray], 
        atm_ids: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for vectorized prediction"""
        features_list = []
        valid_atm_ids = []
        
        for atm_id in atm_ids:
            if atm_id in feature_results:
                features = feature_results[atm_id]
                if len(features) == 28:  # Ensure correct feature count
                    features_list.append(features)
                    valid_atm_ids.append(atm_id)
                else:
                    logger.warning(f"Invalid feature count for ATM {atm_id}: {len(features)}")
            else:
                logger.warning(f"No features found for ATM {atm_id}")
        
        if features_list:
            feature_matrix = np.vstack(features_list)
        else:
            feature_matrix = np.empty((0, 28))
        
        return feature_matrix, valid_atm_ids
    
    def _vectorized_predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Vectorized prediction for batch processing"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(feature_matrix)
            
            # Vectorized prediction
            predictions_proba = self.failure_model.predict_proba(features_scaled)[:, 1]
            
            return predictions_proba
            
        except Exception as e:
            logger.error(f"Vectorized prediction failed: {e}")
            return np.full(len(feature_matrix), 0.5)
    
    def _calculate_confidence_vectorized(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores using vectorized operations"""
        # Confidence based on distance from 0.5 (uncertainty)
        confidence = np.abs(predictions - 0.5) * 2
        return np.clip(confidence, 0, 1)
    
    def _calculate_risk_levels_vectorized(self, predictions: np.ndarray) -> List[str]:
        """Calculate risk levels using vectorized operations"""
        risk_levels = []
        for prob in predictions:
            if prob >= self.RISK_THRESHOLDS['critical']:
                risk_levels.append('critical')
            elif prob >= self.RISK_THRESHOLDS['high']:
                risk_levels.append('high')
            elif prob >= self.RISK_THRESHOLDS['medium']:
                risk_levels.append('medium')
            else:
                risk_levels.append('low')
        
        return risk_levels
    
    async def _cache_predictions_batch(
        self, 
        predictions: List[PredictionResult], 
        cache_ttl: int
    ):
        """Cache prediction results in batch"""
        cache_tasks = []
        
        for prediction in predictions:
            cache_key = f"ml_prediction_v2:{prediction.atm_id}"
            cache_data = {
                'atm_id': prediction.atm_id,
                'failure_probability': prediction.failure_probability,
                'confidence': prediction.confidence,
                'risk_level': prediction.risk_level,
                'prediction_available': prediction.prediction_available,
                'timestamp': prediction.timestamp,
                'top_risk_factors': prediction.top_risk_factors,
                'reason': prediction.reason
            }
            
            cache_tasks.append(
                self.cache_service.set(cache_key, cache_data, ttl=cache_ttl)
            )
        
        # Execute cache operations concurrently
        await asyncio.gather(*cache_tasks, return_exceptions=True)
    
    def _get_feature_contributions_single(self, features: np.ndarray) -> List[Dict]:
        """Get top contributing features for a single prediction"""
        try:
            if hasattr(self.failure_model, 'feature_importances_') and len(features) == 28:
                feature_names = self.feature_engineer.get_feature_names()
                importances = self.failure_model.feature_importances_
                
                contributions = []
                for i, (name, importance) in enumerate(zip(feature_names, importances)):
                    if importance > 0.01:  # Only significant features
                        contributions.append({
                            "feature": name,
                            "importance": float(importance),
                            "value": float(features[i]),
                            "contribution": float(importance * abs(features[i]))
                        })
                
                # Sort by contribution and return top 5
                contributions.sort(key=lambda x: x['contribution'], reverse=True)
                return contributions[:5]
        except Exception as e:
            logger.debug(f"Could not get feature contributions: {e}")
        
        return []
    
    def _create_default_prediction(self, atm_id: str, reason: str) -> PredictionResult:
        """Create default prediction when processing fails"""
        return PredictionResult(
            atm_id=atm_id,
            failure_probability=0.5,
            confidence=0.0,
            risk_level="unknown",
            prediction_available=False,
            timestamp=datetime.now().isoformat(),
            reason=reason
        )
    
