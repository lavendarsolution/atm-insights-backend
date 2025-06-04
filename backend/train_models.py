"""
Training Script for ATM Failure Prediction
Production-grade training with all optimizations:
1. Hyperparameter optimization with Optuna
2. Time-series cross-validation
3. Parallel feature extraction
4. Bulk database queries
5. Vectorized predictions

Usage:
    python train_models.py [--trials 100] [--cv-folds 5] [--timeout 3600] [--workers 8]

Example:
    python train_models.py --trials 100 --cv-folds 5 --timeout 3600 --workers 8
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import warnings

import numpy as np
import optuna
from database.session import SessionLocal
from ml.feature_engineering import OptimizedATMFeatureEngineer
from ml.model_training import ATMFailureModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ATM Failure Prediction Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--trials", type=int, default=100, help="Number of Optuna optimization trials"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of time-series cross-validation folds",
    )

    parser.add_argument(
        "--timeout", type=int, default=3600, help="Optimization timeout in seconds"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (reduced trials and CV folds)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing model instead of training",
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )

    return parser.parse_args()


def validate_environment():
    """Validate that all required components are available"""
    logger.info("🔍 Validating environment...")

    try:
        # Test database connection
        from sqlalchemy import text

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("✅ Database connection: OK")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

    try:
        # Test feature engineering
        feature_engineer = OptimizedATMFeatureEngineer()
        logger.info("✅ Feature engineering: OK")
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return False

    try:
        # Test model imports
        import optuna
        import xgboost as xgb

        logger.info("✅ ML libraries: OK")
    except Exception as e:
        logger.error(f"❌ ML libraries failed: {e}")
        return False

    return True


def run_benchmarks():
    """Run performance benchmarks for all components"""
    logger.info("🏃 Running performance benchmarks...")

    db = SessionLocal()

    try:
        # Benchmark 1: Database query performance
        logger.info("📊 Benchmark 1: Database Query Performance")
        start_time = time.time()

        from sqlalchemy import text

        result = db.execute(text("SELECT COUNT(*) FROM atm_telemetry")).scalar()
        query_time = time.time() - start_time

        logger.info(f"   Total telemetry records: {result:,}")
        logger.info(f"   Query time: {query_time:.3f}s")

        # Benchmark 2: Feature extraction performance
        logger.info("📊 Benchmark 2: Feature Extraction Performance")

        # Get sample ATM IDs
        sample_atms = db.execute(
            text("SELECT DISTINCT atm_id FROM atm_telemetry LIMIT 50")
        ).fetchall()

        if sample_atms:
            atm_ids = [row[0] for row in sample_atms]

            feature_engineer = OptimizedATMFeatureEngineer()

            # Benchmark parallel extraction
            start_time = time.time()
            features = feature_engineer.bulk_extract_features_from_database(
                db, atm_ids[:10], hours=24
            )
            extraction_time = time.time() - start_time

            logger.info(f"   ATMs processed: {len(features)}")
            logger.info(f"   Extraction time: {extraction_time:.3f}s")
            logger.info(f"   Throughput: {len(features)/extraction_time:.1f} ATMs/sec")

        # Benchmark 3: Memory usage
        logger.info("📊 Benchmark 3: Memory Usage")
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        logger.info(f"   RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
        logger.info(f"   VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        db.close()


def validate_existing_model():
    """Validate existing model performance"""
    logger.info("🔍 Validating existing model...")

    try:
        from services.cache_service import CacheService
        from services.ml_service import VectorizedMLPredictionService

        # Initialize services
        cache_service = CacheService()
        ml_service = VectorizedMLPredictionService(cache_service)

        # Check if models can be loaded(Demo)

        import os

        model_dir = "ml/models"

        failure_model_path = os.path.join(model_dir, "failure_model_latest.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if os.path.exists(failure_model_path) and os.path.exists(scaler_path):
            logger.info("✅ Model files found")

            import joblib

            model = joblib.load(failure_model_path)
            scaler = joblib.load(scaler_path)

            logger.info(f"✅ Model type: {type(model).__name__}")
            logger.info(f"✅ Model parameters: {len(model.get_params())} params")

            # Test prediction
            dummy_features = np.random.random((5, 28)).astype(np.float32)
            scaled_features = scaler.transform(dummy_features)
            predictions = model.predict_proba(scaled_features)[:, 1]

            logger.info(f"✅ Test predictions: {predictions}")

        else:
            logger.warning("⚠️ Model files not found")

    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")


def main():
    """Main training function"""
    args = parse_arguments()

    print("=" * 80)
    print("🚀 ATM FAILURE PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed!")
        sys.exit(1)

    # Run benchmarks if requested
    if args.benchmark:
        run_benchmarks()
        return

    # Validate existing model if requested
    if args.validate:
        validate_existing_model()
        return

    # Adjust parameters for quick mode
    if args.quick:
        args.trials = 20
        args.cv_folds = 3
        args.timeout = 600
        logger.info("⚡ Quick training mode enabled")

    # Initialize database
    db = SessionLocal()

    try:
        # Initialize trainer
        logger.info("🔧 Initializing trainer...")
        trainer = ATMFailureModelTrainer(
            n_trials=args.trials,
            cv_folds=args.cv_folds,
            optimization_timeout=args.timeout,
            n_jobs=args.workers or -1,
        )

        logger.info(f"📋 Training Configuration:")
        logger.info(f"   Optuna trials: {args.trials}")
        logger.info(f"   CV folds: {args.cv_folds}")
        logger.info(f"   Timeout: {args.timeout}s")
        logger.info(f"   Workers: {args.workers or 'auto-detect'}")

        # Run the complete training pipeline
        pipeline_start = datetime.now()

        logger.info("\n🎯 Starting enhanced training pipeline...")
        results = trainer.full_training_pipeline(db)

        pipeline_duration = datetime.now() - pipeline_start

        # Print comprehensive results
        print("\n" + "=" * 80)
        print("✅ TRAINING PIPELINE COMPLETED!")
        print("=" * 80)

        print(f"📊 PERFORMANCE METRICS:")
        print(f"   Final Model AUC: {results.get('auc_score', 'N/A'):.4f}")
        print(f"   PR-AUC Score: {results.get('pr_auc_score', 'N/A'):.4f}")
        print(f"   Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")

        print(f"\n🎯 OPTIMIZATION RESULTS:")
        print(f"   Best AUC from Optuna: {results.get('best_value', 'N/A'):.4f}")
        print(f"   Optimization trials: {results.get('optimization_trials', 'N/A')}")
        print(f"   Best trial number: {results.get('best_trial', 'N/A')}")

        print(f"\n📈 CROSS-VALIDATION:")
        print(f"   CV Mean AUC: {results.get('cv_mean_auc', 'N/A'):.4f}")
        print(f"   CV Std AUC: {results.get('cv_std_auc', 'N/A'):.4f}")

        print(f"\n📁 OUTPUT FILES:")
        print(f"   Model: {results.get('model_path', 'N/A')}")
        print(f"   Scaler: {results.get('scaler_path', 'N/A')}")

        print(f"\n⏱️  TIMING:")
        print(f"   Pipeline duration: {pipeline_duration}")
        print(
            f"   Training samples: {results.get('positive_samples', 0) + results.get('negative_samples', 0):,}"
        )

        # Best hyperparameters
        if results.get("best_params"):
            print(f"\n⚙️  BEST HYPERPARAMETERS:")
            for param, value in results["best_params"].items():
                print(f"   {param}: {value}")

        # Feature importance (top 10)
        if results.get("feature_importance"):
            print(f"\n🎯 TOP 10 FEATURES:")
            sorted_features = sorted(
                results["feature_importance"].items(), key=lambda x: x[1], reverse=True
            )[:10]

            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"   {i:2d}. {feature:<30} {importance:.4f}")

        print("=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Performance summary
        total_records = results.get("positive_samples", 0) + results.get(
            "negative_samples", 0
        )
        if total_records > 0:
            print(f"\n📋 TRAINING SUMMARY:")
            print(f"   ✅ Model successfully trained and saved")
            print(f"   ✅ {total_records:,} training samples processed")
            print(f"   ✅ {args.trials} hyperparameter optimization trials")
            print(f"   ✅ {args.cv_folds}-fold time-series cross-validation")
            print(
                f"   ✅ Production-ready model with AUC: {results.get('auc_score', 0):.4f}"
            )

            if results.get("auc_score", 0) > 0.8:
                print(f"   🎯 EXCELLENT performance (AUC > 0.8)")
            elif results.get("auc_score", 0) > 0.7:
                print(f"   ✅ GOOD performance (AUC > 0.7)")
            else:
                print(f"   ⚠️  Consider reviewing features or data quality")

    except KeyboardInterrupt:
        logger.info("\n🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            if db:
                db.close()
        except Exception as cleanup_error:
            logger.warning(f"Database cleanup warning (non-critical): {cleanup_error}")
            pass

        end_time = datetime.now()
        print(f"\n🕐 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
