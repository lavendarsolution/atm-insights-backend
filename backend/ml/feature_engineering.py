import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import asyncio
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class OptimizedATMFeatureEngineer:
    """Enhanced ATM Feature Engineer with parallel processing and bulk operations"""
    
    # Critical error codes that indicate hardware failures
    CRITICAL_ERROR_CODES = ['E002', 'E004', 'E005', 'E007', 'E010']
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize with configurable worker pool"""
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        logger.info(f"Initialized OptimizedATMFeatureEngineer with {self.max_workers} workers")
    
    def bulk_extract_features_from_database(
        self, 
        db: Session, 
        atm_ids: List[str], 
        hours: int = 24
    ) -> Dict[str, np.ndarray]:
        """
        Bulk extract features for multiple ATMs with optimized database queries
        Returns: Dict[atm_id -> features_array]
        """
        logger.info(f"Starting bulk feature extraction for {len(atm_ids)} ATMs")
        start_time = datetime.now()
        
        # Single optimized query for all ATMs
        telemetry_data = self._bulk_fetch_telemetry(db, atm_ids, hours)
        
        # Group telemetry by ATM
        atm_telemetry = self._group_telemetry_by_atm(telemetry_data)
        
        # Parallel feature extraction
        feature_results = self._parallel_extract_features(atm_telemetry)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Bulk feature extraction completed in {processing_time:.2f}s for {len(feature_results)} ATMs")
        
        return feature_results
    
    def _bulk_fetch_telemetry(
        self, 
        db: Session, 
        atm_ids: List[str], 
        hours: int
    ) -> List[Dict]:
        """Optimized bulk query for multiple ATMs"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Convert ATM IDs to a format suitable for SQL IN clause
        atm_ids_str = "','".join(atm_ids)
        
        # Single optimized query with window functions for efficiency
        bulk_query = text(f"""
            WITH ranked_telemetry AS (
                SELECT 
                    atm_id,
                    time,
                    status,
                    cash_level_percent,
                    temperature_celsius,
                    cpu_usage_percent,
                    memory_usage_percent,
                    network_status,
                    error_code,
                    network_latency_ms,
                    uptime_seconds,
                    ROW_NUMBER() OVER (
                        PARTITION BY atm_id 
                        ORDER BY time DESC
                    ) as rn
                FROM atm_telemetry 
                WHERE atm_id IN ('{atm_ids_str}')
                AND time >= :cutoff_time
            )
            SELECT 
                atm_id,
                time,
                status,
                cash_level_percent,
                temperature_celsius,
                cpu_usage_percent,
                memory_usage_percent,
                network_status,
                error_code,
                network_latency_ms,
                uptime_seconds
            FROM ranked_telemetry
            WHERE rn <= 1000  -- Limit records per ATM for performance
            ORDER BY atm_id, time DESC
        """)
        
        result = db.execute(bulk_query, {"cutoff_time": cutoff_time})
        
        telemetry_data = []
        for row in result:
            telemetry_data.append({
                'atm_id': row.atm_id,
                'time': row.time,
                'status': row.status,
                'cash_level_percent': row.cash_level_percent,
                'temperature_celsius': row.temperature_celsius,
                'cpu_usage_percent': row.cpu_usage_percent,
                'memory_usage_percent': row.memory_usage_percent,
                'network_status': row.network_status,
                'error_code': row.error_code,
                'network_latency_ms': row.network_latency_ms,
                'uptime_seconds': row.uptime_seconds
            })
        
        logger.debug(f"Fetched {len(telemetry_data)} telemetry records for bulk processing")
        return telemetry_data
    
    def _group_telemetry_by_atm(self, telemetry_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group telemetry data by ATM ID efficiently"""
        atm_groups = {}
        for record in telemetry_data:
            atm_id = record['atm_id']
            if atm_id not in atm_groups:
                atm_groups[atm_id] = []
            atm_groups[atm_id].append(record)
        
        # Sort each group by time (most recent first)
        for atm_id in atm_groups:
            atm_groups[atm_id].sort(key=lambda x: x['time'], reverse=True)
        
        return atm_groups
    
    def _parallel_extract_features(
        self, 
        atm_telemetry: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """Extract features in parallel using ProcessPoolExecutor"""
        
        # Prepare data for parallel processing
        processing_tasks = []
        for atm_id, telemetry_list in atm_telemetry.items():
            if len(telemetry_list) > 0:
                current_telemetry = telemetry_list[0]  # Most recent
                history = telemetry_list[1:] if len(telemetry_list) > 1 else []
                processing_tasks.append((atm_id, current_telemetry, history))
        
        # Process in parallel batches for optimal performance
        feature_results = {}
        batch_size = max(1, len(processing_tasks) // self.max_workers)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks in batches
            future_to_atm = {}
            
            for i in range(0, len(processing_tasks), batch_size):
                batch = processing_tasks[i:i + batch_size]
                future = executor.submit(self._process_feature_batch, batch)
                future_to_atm[future] = [task[0] for task in batch]
            
            # Collect results as they complete
            for future in as_completed(future_to_atm):
                try:
                    batch_results = future.result()
                    feature_results.update(batch_results)
                except Exception as e:
                    atm_ids = future_to_atm[future]
                    logger.error(f"Error processing batch for ATMs {atm_ids}: {str(e)}")
                    # Add default features for failed ATMs
                    for atm_id in atm_ids:
                        feature_results[atm_id] = np.zeros(28, dtype=np.float32)
        
        logger.info(f"Parallel feature extraction completed for {len(feature_results)} ATMs")
        return feature_results
    
    @staticmethod
    def _process_feature_batch(batch_tasks: List[Tuple]) -> Dict[str, np.ndarray]:
        """Process a batch of feature extraction tasks (runs in separate process)"""
        batch_results = {}
        
        for atm_id, current_telemetry, history in batch_tasks:
            try:
                # Extract features for this ATM
                features = OptimizedATMFeatureEngineer._extract_single_atm_features(
                    current_telemetry, history
                )
                batch_results[atm_id] = features
            except Exception as e:
                logger.error(f"Error extracting features for ATM {atm_id}: {str(e)}")
                batch_results[atm_id] = np.zeros(28, dtype=np.float32)
        
        return batch_results
    
    @staticmethod
    def _extract_single_atm_features(
        current_telemetry: Dict, 
        history: List[Dict]
    ) -> np.ndarray:
        """Extract features for a single ATM (optimized for parallel execution)"""
        # Convert to DataFrame for efficient processing
        if history:
            df = pd.DataFrame(history)
            # Ensure time column is datetime
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
        else:
            df = pd.DataFrame()
        
        features = []
        
        # Get last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        if not df.empty and 'time' in df.columns:
            recent_24h = df[df['time'] >= cutoff_time]
        else:
            recent_24h = df
        
        if recent_24h.empty:
            recent_24h = df
        
        # Features 1-4: Status Distribution (vectorized operations)
        if 'status' in recent_24h.columns and len(recent_24h) > 0:
            status_counts = recent_24h['status'].value_counts()
            total_records = len(recent_24h)
            
            features.extend([
                status_counts.get('online', 0) / total_records,      # Feature 1
                status_counts.get('error', 0) / total_records,       # Feature 2
                status_counts.get('offline', 0) / total_records,     # Feature 3
                status_counts.get('maintenance', 0) / total_records  # Feature 4
            ])
        else:
            features.extend([1.0, 0.0, 0.0, 0.0])
        
        # Features 5-8: Cash Level Statistics (vectorized)
        if 'cash_level_percent' in df.columns and not df.empty:
            cash_levels = df['cash_level_percent'].dropna()
            if len(cash_levels) > 0:
                features.extend([
                    cash_levels.mean(),                        # Feature 5
                    cash_levels.min(),                         # Feature 6
                    cash_levels.std() if len(cash_levels) > 1 else 0,  # Feature 7
                    (cash_levels < 20).mean()                  # Feature 8
                ])
            else:
                features.extend([50.0, 50.0, 0.0, 0.0])
        else:
            features.extend([50.0, 50.0, 0.0, 0.0])
        
        # Features 9-12: Temperature Statistics (vectorized)
        if 'temperature_celsius' in df.columns and not df.empty:
            temps = df['temperature_celsius'].dropna()
            if len(temps) > 0:
                features.extend([
                    temps.mean(),                              # Feature 9
                    temps.max(),                               # Feature 10
                    temps.std() if len(temps) > 1 else 0,     # Feature 11
                    ((temps > 35) | (temps < 5)).mean()       # Feature 12
                ])
            else:
                features.extend([22.0, 22.0, 0.0, 0.0])
        else:
            features.extend([22.0, 22.0, 0.0, 0.0])
        
        # Features 13-15: CPU Usage Statistics (vectorized)
        if 'cpu_usage_percent' in df.columns and not df.empty:
            cpu_usage = df['cpu_usage_percent'].dropna()
            if len(cpu_usage) > 0:
                features.extend([
                    cpu_usage.mean(),                          # Feature 13
                    cpu_usage.max(),                           # Feature 14
                    (cpu_usage > 80).mean()                    # Feature 15
                ])
            else:
                features.extend([30.0, 30.0, 0.0])
        else:
            features.extend([30.0, 30.0, 0.0])
        
        # Features 16-18: Memory Usage Statistics (vectorized)
        if 'memory_usage_percent' in df.columns and not df.empty:
            memory_usage = df['memory_usage_percent'].dropna()
            if len(memory_usage) > 0:
                features.extend([
                    memory_usage.mean(),                       # Feature 16
                    memory_usage.max(),                        # Feature 17
                    (memory_usage > 85).mean()                 # Feature 18
                ])
            else:
                features.extend([50.0, 50.0, 0.0])
        else:
            features.extend([50.0, 50.0, 0.0])
        
        # Feature 19: Network Disconnect Frequency (vectorized)
        if 'network_status' in df.columns and not df.empty:
            network_issues = df['network_status'].dropna()
            if len(network_issues) > 0:
                features.append((network_issues == 'disconnected').mean())  # Feature 19
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Feature 20: Error Frequency (vectorized)
        if 'error_code' in df.columns and not df.empty:
            features.append(df['error_code'].notna().mean())  # Feature 20
        else:
            features.append(0.0)
        
        # Feature 21: Critical Error Frequency (vectorized)
        if 'error_code' in df.columns and not df.empty:
            critical_errors = df['error_code'].isin(OptimizedATMFeatureEngineer.CRITICAL_ERROR_CODES)
            features.append(critical_errors.mean())  # Feature 21
        else:
            features.append(0.0)
        
        # Current state features (22-28)
        status_map = {'online': 0, 'offline': 1, 'error': 2, 'maintenance': 3}
        current_status = current_telemetry.get('status', 'online')
        features.extend([
            float(status_map.get(current_status, 0)),                           # Feature 22
            float(current_telemetry.get('cash_level_percent', 50.0)),           # Feature 23
            float(current_telemetry.get('temperature_celsius', 22.0)),          # Feature 24
            float(current_telemetry.get('cpu_usage_percent', 30.0)),            # Feature 25
            float(current_telemetry.get('memory_usage_percent', 50.0)),         # Feature 26
            float(current_telemetry.get('network_latency_ms', 100) / 1000.0),   # Feature 27
            1.0 if current_telemetry.get('error_code') in OptimizedATMFeatureEngineer.CRITICAL_ERROR_CODES else 0.0  # Feature 28
        ])
        
        # Ensure exactly 28 features
        features = features[:28]
        while len(features) < 28:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all 28 features"""
        return [
            # Historical features (1-21)
            "online_ratio_24h",              # 1
            "error_ratio_24h",               # 2
            "offline_ratio_24h",             # 3
            "maintenance_ratio_24h",         # 4
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
            "critical_error_frequency",      # 21
            # Current state features (22-28)
            "current_status_encoded",        # 22
            "current_cash_level",            # 23
            "current_temperature",           # 24
            "current_cpu_usage",             # 25
            "current_memory_usage",          # 26
            "current_network_latency_sec",   # 27
            "current_has_critical_error",    # 28
        ]

# Backward compatibility wrapper
class ATMFeatureEngineer(OptimizedATMFeatureEngineer):
    """Backward compatibility wrapper for existing code"""
    
    @staticmethod
    def extract_features_from_telemetry(telemetry_history: List[Dict]) -> np.ndarray:
        """Extract 21 historical features from telemetry history (legacy method)"""
        if not telemetry_history:
            return np.zeros(21)
        
        current_telemetry = telemetry_history[0] if telemetry_history else {}
        history = telemetry_history[1:] if len(telemetry_history) > 1 else []
        
        full_features = OptimizedATMFeatureEngineer._extract_single_atm_features(
            current_telemetry, history
        )
        
        # Return only first 21 features for backward compatibility
        return full_features[:21]
    
    @staticmethod
    def extract_realtime_features(current_telemetry: Dict, history: List[Dict]) -> np.ndarray:
        """Extract all 28 features combining current state and history (legacy method)"""
        return OptimizedATMFeatureEngineer._extract_single_atm_features(
            current_telemetry, history
        )