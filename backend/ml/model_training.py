import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sqlalchemy import text
import logging
import os
from typing import Tuple, Dict, List
import warnings
import pickle

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

class ATMFailureModelTrainer:
    """Model trainer with Optuna optimization and time-series validation"""
    
    CRITICAL_ERROR_CODES = ['E002', 'E004', 'E005', 'E007', 'E010']
    
    def __init__(self, 
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 optimization_timeout: int = 3600,  # 1 hour
                 n_jobs: int = -1):
        """
        Initialize trainer
        
        Args:
            n_trials: Number of Optuna trials for hyperparameter optimization
            cv_folds: Number of time-series cross-validation folds
            optimization_timeout: Timeout for optimization in seconds
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.optimization_timeout = optimization_timeout
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.model_dir = "ml/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup Optuna study
        self.study = None
        self._setup_study()
    
    def _setup_study(self):
        """Setup Optuna study with optimal configuration"""
        sampler = TPESampler(
            n_startup_trials=20,
            n_ei_candidates=50,
            seed=42
        )
        
        pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f'atm_failure_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
    
    def create_labeled_dataset_optimized(self, db_session) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create labeled dataset with optimized SQL queries"""
        logger.info("Creating labeled dataset with optimized queries...")
        
        # Optimized query with better indexing hints and reduced complexity
        query = text("""
        WITH maintenance_windows AS (
            -- Pre-compute maintenance time windows
            SELECT 
                atm_id, 
                time as maint_start,
                time + INTERVAL '4 hours' as maint_end
            FROM atm_telemetry 
            WHERE status = 'maintenance'
        ),
        telemetry_with_labels AS (
            SELECT 
                t.*,
                CASE 
                    WHEN (t.status = 'error' AND t.error_code = ANY(:critical_codes))
                      OR EXISTS (
                          SELECT 1 FROM maintenance_windows m 
                          WHERE m.atm_id = t.atm_id 
                          AND t.time BETWEEN m.maint_start AND m.maint_end
                      )
                    THEN 1
                    ELSE 0
                END as failure_label,
                -- Add time-based features for sorting
                EXTRACT(EPOCH FROM t.time) as time_epoch
            FROM atm_telemetry t
            WHERE t.status != 'maintenance'  -- Exclude maintenance records from training
        )
        SELECT * FROM telemetry_with_labels
        ORDER BY atm_id, time_epoch  -- Crucial for time-series ordering
        """)
        
        # Execute with proper parameter binding
        result = db_session.execute(query, {
            "critical_codes": self.CRITICAL_ERROR_CODES
        })
        
        # Convert to DataFrame efficiently
        columns = result.keys()
        data = result.fetchall()
        df = pd.DataFrame(data, columns=columns)
        
        if df.empty:
            logger.error("No telemetry data found!")
            return pd.DataFrame(), np.array([])
        
        # Convert time column to datetime for proper time-series handling
        df['time'] = pd.to_datetime(df['time'])
        
        # Sort by ATM and time to ensure proper time-series order
        df = df.sort_values(['atm_id', 'time']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} telemetry records")
        logger.info(f"Failure labels - 0: {(df['failure_label']==0).sum()}, 1: {(df['failure_label']==1).sum()}")
        logger.info(f"Class balance: {(df['failure_label']==1).sum() / len(df) * 100:.2f}% positive")
        
        return df, df['failure_label'].values
    
    def prepare_features_vectorized(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Vectorized feature preparation with massive performance improvements"""
        from ml.feature_engineering import OptimizedATMFeatureEngineer
        
        logger.info("Extracting features with vectorized processing...")
        start_time = datetime.now()
        
        # Get unique ATMs and prepare for bulk processing
        atm_ids = df['atm_id'].unique().tolist()
        feature_engineer = OptimizedATMFeatureEngineer()
        
        # Group data by ATM for efficient processing
        atm_groups = df.groupby('atm_id')
        atm_telemetry_dict = {}
        
        for atm_id, group in atm_groups:
            # Convert to list of dictionaries efficiently
            records = group.sort_values('time').to_dict('records')
            atm_telemetry_dict[atm_id] = records
        
        # Use parallel feature extraction
        feature_results = feature_engineer._parallel_extract_features(atm_telemetry_dict)
        
        # Prepare training samples with proper time-series ordering
        all_features = []
        sample_indices = []
        
        for atm_id, group in atm_groups:
            if atm_id not in feature_results:
                continue
            
            records = group.sort_values('time').to_dict('records')
            features = feature_results[atm_id]
            
            # Create samples with time-series structure
            # Use sliding window approach for better temporal representation
            for i in range(1, len(records)):  # Skip first record (no history)
                current = records[i]
                # Use the extracted features for this ATM
                all_features.append(features)
                sample_indices.append(group.index[i])  # Track original indices
        
        if not all_features:
            logger.error("No features extracted!")
            return np.array([]), []
        
        feature_matrix = np.vstack(all_features)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Vectorized feature extraction completed in {processing_time:.2f}s")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix, sample_indices
    
    def time_series_cross_validate(self, X: np.ndarray, y: np.ndarray, 
                                 df: pd.DataFrame, sample_indices: List[int]) -> Dict:
        """Perform time-series cross-validation with proper temporal splits"""
        logger.info("Starting time-series cross-validation...")
        
        # Sort by time to ensure proper temporal order
        time_ordered_indices = np.argsort([df.loc[idx, 'time'] for idx in sample_indices])
        X_ordered = X[time_ordered_indices]
        y_ordered = y[time_ordered_indices]
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds, test_size=None)
        
        cv_scores = []
        fold_results = []
        
        logger.info(f"Performing {self.cv_folds}-fold time-series cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_ordered)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X_ordered[train_idx], X_ordered[val_idx]
            y_train_fold, y_val_fold = y_ordered[train_idx], y_ordered[val_idx]
            
            # Calculate class weights for this fold
            pos_weight = (y_train_fold == 0).sum() / (y_train_fold == 1).sum() if (y_train_fold == 1).sum() > 0 else 1
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Train model with proper XGBoost parameters
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=self.n_jobs,
                eval_metric='auc'
            )            
      
            model.fit(
                X_train_scaled, y_train_fold,
                eval_set=[(X_val_scaled, y_val_fold)],
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc_score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(auc_score)
            
            fold_results.append({
                'fold': fold + 1,
                'auc_score': auc_score,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'pos_weight': pos_weight
            })
            
            logger.info(f"Fold {fold + 1} AUC: {auc_score:.4f}")
        
        cv_results = {
            'mean_auc': np.mean(cv_scores),
            'std_auc': np.std(cv_scores),
            'fold_scores': cv_scores,
            'fold_details': fold_results
        }
        
        logger.info(f"Cross-validation completed: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        
        return cv_results
    
    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization"""
        
        # Sample hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        }
        
        # Calculate class weight
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        params['scale_pos_weight'] = pos_weight
        params['random_state'] = 42
        params['n_jobs'] = self.n_jobs
        params['eval_metric'] = 'auc'
        
        # Time-series split for validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=None)  # Reduced for faster optimization
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Train model
            model = xgb.XGBClassifier(**params)            
          
            model.fit(
                X_train_scaled, y_train_fold,
                eval_set=[(X_val_scaled, y_val_fold)],
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc_score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(auc_score)
            
            # Report intermediate value for pruning
            trial.report(auc_score, len(cv_scores))
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create objective function with data
        objective_with_data = lambda trial: self.objective(trial, X, y)
        
        # Optimize
        self.study.optimize(
            objective_with_data,
            n_trials=self.n_trials,
            timeout=self.optimization_timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        optimization_results = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number
        }
        
        logger.info(f"Optimization completed. Best AUC: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return optimization_results
    
    def train_final_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train final model with optimized parameters"""
        logger.info("Training final model with optimized parameters...")
        
        # Use optimized parameters or defaults
        if self.best_params:
            model_params = self.best_params.copy()
        else:
            logger.warning("No optimized parameters found, using defaults")
            model_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }        
       
        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        model_params.update({
            'scale_pos_weight': pos_weight,
            'random_state': 42,
            'n_jobs': self.n_jobs,
            'eval_metric': 'auc'
        })
        
        # Time-based train/test split (last 20% for testing)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train final model
        self.best_model = xgb.XGBClassifier(**model_params)        
  
        self.best_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=True
        )
        
        # Evaluate final model
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate comprehensive metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall curve for imbalanced dataset
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        logger.info(f"Final model AUC: {auc_score:.4f}")
        logger.info(f"Final model PR-AUC: {pr_auc:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance
        from ml.feature_engineering import OptimizedATMFeatureEngineer
        feature_names = OptimizedATMFeatureEngineer.get_feature_names()
        feature_importance = dict(zip(feature_names, self.best_model.feature_importances_))
        
        # Log top 10 features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 Important Features:")
        for i, (feat, imp) in enumerate(top_features, 1):
            logger.info(f"{i}. {feat}: {imp:.4f}")
        
        # Save models with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f"failure_model_optimized_{timestamp}.pkl")
        scaler_path = os.path.join(self.model_dir, f"scaler_optimized_{timestamp}.pkl")
        
        # Save with joblib for better performance
        joblib.dump(self.best_model, model_path, compress=3)
        joblib.dump(self.scaler, scaler_path, compress=3)
        
        # Save as 'latest' for easy loading
        joblib.dump(self.best_model, os.path.join(self.model_dir, "failure_model_latest.pkl"), compress=3)
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"), compress=3)
        
        # Save optimization study for analysis
        study_path = os.path.join(self.model_dir, f"optimization_study_{timestamp}.pkl")
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Optimized model saved to {model_path}")
        logger.info(f"Optimization study saved to {study_path}")
        
        training_results = {
            "auc_score": auc_score,
            "pr_auc_score": pr_auc,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "test_accuracy": (y_pred == y_test).mean(),
            "feature_importance": feature_importance,
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "optimization_trials": len(self.study.trials) if self.study else 0,
            "best_params": self.best_params,
            "cv_results": self.cv_results
        }
        
        return training_results
    
    def full_training_pipeline(self, db_session) -> Dict:
        """Complete training pipeline"""
        logger.info("="*80)
        logger.info("ATM FAILURE PREDICTION MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Create optimized labeled dataset
            logger.info("\n🔄 Step 1: Creating optimized labeled dataset...")
            df, labels = self.create_labeled_dataset_optimized(db_session)
            
            if df.empty:
                raise ValueError("No data available for training!")
            
            # Step 2: Vectorized feature extraction
            logger.info("\n🔄 Step 2: Extracting features with parallel processing...")
            X, sample_indices = self.prepare_features_vectorized(df)
            
            if len(X) == 0:
                raise ValueError("No features extracted!")
            
            # Align labels with extracted features
            y_aligned = labels[[df.index.get_loc(idx) for idx in sample_indices]]
            
            # Step 3: Time-series cross-validation
            logger.info("\n🔄 Step 3: Performing time-series cross-validation...")
            self.cv_results = self.time_series_cross_validate(X, y_aligned, df, sample_indices)
            
            # Step 4: Hyperparameter optimization
            logger.info("\n🔄 Step 4: Optimizing hyperparameters with Optuna...")
            optimization_results = self.optimize_hyperparameters(X, y_aligned)
            
            # Step 5: Train final model
            logger.info("\n🔄 Step 5: Training final optimized model...")
            training_results = self.train_final_model(X, y_aligned)
            
            # Combine all results
            final_results = {
                **training_results,
                **optimization_results,
                "cv_mean_auc": self.cv_results['mean_auc'],
                "cv_std_auc": self.cv_results['std_auc'],
                "pipeline_duration": (datetime.now() - pipeline_start).total_seconds()
            }
            
            logger.info("\n" + "="*80)
            logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"   📊 Final Model AUC: {final_results['auc_score']:.4f}")
            logger.info(f"   📊 Cross-Validation AUC: {final_results['cv_mean_auc']:.4f} ± {final_results['cv_std_auc']:.4f}")
            logger.info(f"   🎯 Best Hyperparameters Found: {self.best_params}")
            logger.info(f"   ⏱️  Total Pipeline Duration: {final_results['pipeline_duration']:.2f} seconds")
            logger.info(f"   💾 Model saved to: {final_results['model_path']}")
            logger.info("="*80)
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise

