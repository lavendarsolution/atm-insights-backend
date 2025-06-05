import logging
import os
import pickle
import time  
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


class ATMFailureModelTrainer:
    """Model trainer with Optuna optimization and time-series validation"""

    CRITICAL_ERROR_CODES = ["E002", "E004", "E005", "E007", "E010"]

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        optimization_timeout: int = 3600,  # 1 hour
        n_jobs: int = -1,
    ):
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
        sampler = TPESampler(n_startup_trials=20, n_ei_candidates=50, seed=42)

        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f'atm_failure_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        )

    def create_labeled_dataset_optimized(
        self, db_session
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create labeled dataset with optimized SQL queries"""
        logger.info("Creating labeled dataset with optimized queries...")

        # Optimized query with better indexing hints and reduced complexity
        query = text(
            """
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
        """
        )

        # Execute with proper parameter binding
        result = db_session.execute(
            query, {"critical_codes": self.CRITICAL_ERROR_CODES}
        )

        # Convert to DataFrame efficiently
        columns = result.keys()
        data = result.fetchall()
        df = pd.DataFrame(data, columns=columns)

        if df.empty:
            logger.error("No telemetry data found!")
            return pd.DataFrame(), np.array([])

        # Convert time column to datetime for proper time-series handling
        df["time"] = pd.to_datetime(df["time"])

        # Sort by ATM and time to ensure proper time-series order
        df = df.sort_values(["atm_id", "time"]).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} telemetry records")
        logger.info(
            f"Failure labels - 0: {(df['failure_label']==0).sum()}, 1: {(df['failure_label']==1).sum()}"
        )
        logger.info(
            f"Class balance: {(df['failure_label']==1).sum() / len(df) * 100:.2f}% positive"
        )

        return df, df["failure_label"].values

    def prepare_features_vectorized(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Vectorized feature preparation with massive performance improvements"""
        from ml.feature_engineering import OptimizedATMFeatureEngineer

        logger.info("Extracting features with vectorized processing...")
        start_time = datetime.now()

        # Get unique ATMs and prepare for bulk processing
        atm_ids = df["atm_id"].unique().tolist()
        feature_engineer = OptimizedATMFeatureEngineer()

        # Group data by ATM for efficient processing
        atm_groups = df.groupby("atm_id")
        atm_telemetry_dict = {}

        for atm_id, group in atm_groups:
            # Convert to list of dictionaries efficiently
            records = group.sort_values("time").to_dict("records")
            atm_telemetry_dict[atm_id] = records

        # Use parallel feature extraction
        feature_results = feature_engineer._parallel_extract_features(
            atm_telemetry_dict
        )

        # Prepare training samples with proper time-series ordering
        all_features = []
        sample_indices = []

        for atm_id, group in atm_groups:
            if atm_id not in feature_results:
                continue

            records = group.sort_values("time").to_dict("records")
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

        logger.info(
            f"Vectorized feature extraction completed in {processing_time:.2f}s"
        )
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")

        return feature_matrix, sample_indices

    def time_series_cross_validate(self, X, y, df, sample_indices):
        """Time-series cross-validation with proper parameter handling"""
        logger.info("Starting time-series cross-validation...")

        n_folds = self.cv_folds
        n_samples = len(X)
        fold_size = n_samples // n_folds

        cv_scores = []

        logger.info(f"Performing {n_folds}-fold time-series cross-validation")

        for fold in range(n_folds):
            logger.info(f"Processing fold {fold + 1}/{n_folds}")

            # Time-series split: use earlier data for training, later for validation
            train_end = (fold + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                break

            X_train_fold = X[:train_end]
            y_train_fold = y[:train_end]
            X_val_fold = X[val_start:val_end]
            y_val_fold = y[val_start:val_end]

            # Skip fold if insufficient data
            if len(X_train_fold) < 10 or len(X_val_fold) < 5:
                logger.warning(f"Skipping fold {fold + 1} due to insufficient data")
                continue

            # Check class distribution in training fold
            unique_classes = np.unique(y_train_fold)
            if len(unique_classes) < 2:
                logger.warning(
                    f"Skipping fold {fold + 1} due to single class in training data"
                )
                continue

            try:
                # Calculate proper base_score for this fold
                pos_rate = np.mean(y_train_fold)
                # Ensure base_score is in valid range (0, 1)
                base_score = max(0.001, min(0.999, pos_rate))

                # Create model with proper base_score
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    base_score=base_score,  # Set proper base_score
                    n_jobs=1,  # Use single thread for CV to avoid conflicts
                )

                # Fit model
                model.fit(X_train_fold, y_train_fold, verbose=False)

                # Predict and score
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                fold_auc = roc_auc_score(y_val_fold, y_pred_proba)
                cv_scores.append(fold_auc)

                logger.info(
                    f"Fold {fold + 1} AUC: {fold_auc:.4f} (base_score: {base_score:.3f})"
                )

            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue

        if not cv_scores:
            logger.error("No valid CV folds completed")
            return {"mean_auc": 0.0, "std_auc": 0.0, "fold_scores": []}

        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)

        logger.info(f"Cross-validation completed: {mean_auc:.4f} ± {std_auc:.4f}")

        return {"mean_auc": mean_auc, "std_auc": std_auc, "fold_scores": cv_scores}

    def objective(self, trial):
        """Optuna objective function with proper base_score handling"""
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": 1,
        }

        # Calculate proper base_score from training data
        if hasattr(self, "y_train") and self.y_train is not None:
            pos_rate = np.mean(self.y_train)
            base_score = max(0.001, min(0.999, pos_rate))
            params["base_score"] = base_score

        try:
            # Create and evaluate model
            model = xgb.XGBClassifier(**params)

            # Use cross-validation
            cv_results = self.time_series_cross_validate(
                self.X_train, self.y_train, self.df, self.sample_indices
            )

            return cv_results["mean_auc"]

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return 0.0

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logger.info(
            f"Starting hyperparameter optimization with {self.n_trials} trials..."
        )

        # Create objective function with data
        objective_with_data = lambda trial: self.objective(trial, X, y)

        # Optimize
        self.study.optimize(
            objective_with_data,
            n_trials=self.n_trials,
            timeout=self.optimization_timeout,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params

        optimization_results = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "best_trial": self.study.best_trial.number,
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
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
        model_params.update(
            {
                "scale_pos_weight": pos_weight,
                "random_state": 42,
                "n_jobs": self.n_jobs,
                "eval_metric": "auc",
            }
        )

        # Time-based train/test split (last 20% for testing)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples"
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train final model
        self.best_model = xgb.XGBClassifier(**model_params)

        self.best_model.fit(
            X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=True
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
        feature_importance = dict(
            zip(feature_names, self.best_model.feature_importances_)
        )

        # Log top 10 features
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        logger.info("\nTop 10 Important Features:")
        for i, (feat, imp) in enumerate(top_features, 1):
            logger.info(f"{i}. {feat}: {imp:.4f}")

        # Save models with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.model_dir, f"failure_model_optimized_{timestamp}.pkl"
        )
        scaler_path = os.path.join(self.model_dir, f"scaler_optimized_{timestamp}.pkl")

        # Save with joblib for better performance
        joblib.dump(self.best_model, model_path, compress=3)
        joblib.dump(self.scaler, scaler_path, compress=3)

        # Save as 'latest' for easy loading
        joblib.dump(
            self.best_model,
            os.path.join(self.model_dir, "failure_model_latest.pkl"),
            compress=3,
        )
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"), compress=3)

        # Save optimization study for analysis
        study_path = os.path.join(self.model_dir, f"optimization_study_{timestamp}.pkl")
        with open(study_path, "wb") as f:
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
            "cv_results": self.cv_results,
        }

        return training_results

    def prepare_training_data(self, db_session):
        """Prepare complete training dataset by combining data loading and feature extraction"""
        logger.info("Preparing training data...")

        try:
            # Step 1: Create labeled dataset
            df, y = self.create_labeled_dataset_optimized(db_session)

            if df.empty or len(y) == 0:
                logger.error("No training data available")
                return None

            # Step 2: Extract features
            X, sample_indices = self.prepare_features_vectorized(df)

            if len(X) == 0:
                logger.error("No features extracted")
                return None

            # Align labels with extracted features
            if len(sample_indices) > 0:
                y_aligned = y[sample_indices]
            else:
                # If no sample indices, assume direct alignment
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y_aligned = y[:min_len]
                sample_indices = list(range(min_len))

            logger.info(
                f"Training data prepared: {len(X)} samples with {X.shape[1]} features"
            )
            logger.info(f"Class distribution: {np.bincount(y_aligned.astype(int))}")

            return X, y_aligned, df, sample_indices

        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            return None

    def evaluate_and_save_model(self, X, y, df):
        """Evaluate and save the final trained model"""
        logger.info("Evaluating and saving final model...")

        try:
            # Time-based train/test split (last 20% for testing)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            logger.info(
                f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples"
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Evaluate final model
            y_pred = self.final_model.predict(X_test_scaled)
            y_pred_proba = self.final_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate comprehensive metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)

            # Precision-Recall curve for imbalanced dataset
            precision, recall, pr_thresholds = precision_recall_curve(
                y_test, y_pred_proba
            )
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
            feature_importance = dict(
                zip(feature_names, self.final_model.feature_importances_)
            )

            # Log top 10 features
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            logger.info("\nTop 10 Important Features:")
            for i, (feat, imp) in enumerate(top_features, 1):
                logger.info(f"{i}. {feat}: {imp:.4f}")

            # Save models with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                self.model_dir, f"failure_model_optimized_{timestamp}.pkl"
            )
            scaler_path = os.path.join(
                self.model_dir, f"scaler_optimized_{timestamp}.pkl"
            )

            # Save with joblib for better performance
            joblib.dump(self.final_model, model_path, compress=3)
            joblib.dump(self.scaler, scaler_path, compress=3)

            # Save as 'latest' for easy loading
            joblib.dump(
                self.final_model,
                os.path.join(self.model_dir, "failure_model_latest.pkl"),
                compress=3,
            )
            joblib.dump(
                self.scaler, os.path.join(self.model_dir, "scaler.pkl"), compress=3
            )

            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")

            return {
                "auc_score": auc_score,
                "pr_auc_score": pr_auc,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "test_accuracy": (y_pred == y_test).mean(),
                "feature_importance": feature_importance,
                "positive_samples": int((y == 1).sum()),
                "negative_samples": int((y == 0).sum()),
            }

        except Exception as e:
            logger.error(f"Failed to evaluate and save model: {str(e)}")
            raise

    def full_training_pipeline(self, db_session):
        """Training pipeline with proper error handling"""
        logger.info("🚀 Starting training pipeline...")

        try:
            # Step 1: Data preparation
            logger.info("📊 Step 1: Preparing training data...")
            data_result = self.prepare_training_data(db_session)

            if not data_result:
                raise ValueError("Failed to prepare training data")

            X, y, df, sample_indices = data_result

            # Store for Optuna objective
            self.X_train = X
            self.y_train = y
            self.df = df
            self.sample_indices = sample_indices

            # Check data quality
            pos_rate = np.mean(y)
            logger.info(
                f"Training data: {len(X)} samples, {pos_rate:.1%} positive rate"
            )

            if pos_rate == 0.0 or pos_rate == 1.0:
                raise ValueError(
                    f"Invalid class distribution: {pos_rate:.1%} positive rate"
                )

            # Step 2: Cross-validation
            logger.info("🔄 Step 2: Time-series cross-validation...")
            cv_results = self.time_series_cross_validate(X, y, df, sample_indices)
            self.cv_results = cv_results

            # Step 3: Hyperparameter optimization
            logger.info("🎯 Step 3: Hyperparameter optimization...")
            study = optuna.create_study(
                direction="maximize",
                study_name=f"atm_failure_prediction_{int(time.time())}",
            )

            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.optimization_timeout,
                n_jobs=1,  # Use single process for stability
            )

            self.best_params = study.best_params
            self.best_value = study.best_value

            # Step 4: Final model training
            logger.info("🏆 Step 4: Training final model...")

            # Calculate base_score for final model
            pos_rate = np.mean(y)
            base_score = max(0.001, min(0.999, pos_rate))

            final_params = self.best_params.copy()
            final_params.update(
                {
                    "base_score": base_score,
                    "random_state": 42,
                    "eval_metric": "auc",  
                    "n_jobs": self.n_jobs,
                }
            )

            # Scale features for final model
            X_scaled = self.scaler.fit_transform(X)

            self.final_model = xgb.XGBClassifier(**final_params)
            self.final_model.fit(X_scaled, y)

            # Step 5: Model evaluation and saving
            results = self.evaluate_and_save_model(X, y, df)

            # Add additional results
            results.update(
                {
                    "cv_mean_auc": cv_results["mean_auc"],
                    "cv_std_auc": cv_results["std_auc"],
                    "best_value": self.best_value,
                    "best_params": self.best_params,
                    "optimization_trials": len(study.trials),
                    "best_trial": study.best_trial.number if study.best_trial else None,
                }
            )

            return results

        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise
