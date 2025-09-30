#!/usr/bin/env python3
"""
Training Script for New Model Architecture

This script demonstrates how to train models using the new architecture:
- Uses TrainingPipeline for end-to-end training
- Uses ModelPredictor for production deployment
- Uses ModelMonitor for model health tracking

Usage:
    python train_new_models.py --model residual_predictor --symbols SPY QQQ IWM
    python train_new_models.py --model ff5_regression --test-mode
"""

import logging
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from trading_system.models.training.pipeline import TrainingPipeline
from trading_system.models.training.trainer import TrainingConfig
from trading_system.models.serving.predictor import ModelPredictor
from trading_system.data.yfinance_provider import YFinanceProvider
from trading_system.data.ff5_provider import FF5DataProvider
from trading_system.feature_engineering import FeatureConfig, FeatureType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(symbols: list, start_date: datetime, end_date: datetime):
    """Create sample data for training."""
    logger.info(f"Fetching data for {len(symbols)} symbols...")

    # Fetch equity data
    equity_provider = YFinanceProvider()
    equity_data = {}

    for symbol in symbols:
        try:
            data = equity_provider.get_price_data(symbol, start_date, end_date)
            if data is not None and len(data) > 0:
                equity_data[symbol] = data
                logger.info(f"✅ Fetched {len(data)} rows for {symbol}")
            else:
                logger.warning(f"⚠️  No data found for {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")

    # Fetch factor data
    try:
        factor_provider = FF5DataProvider()
        factor_data = factor_provider.get_factor_data(start_date, end_date)
        logger.info(f"✅ Fetched {len(factor_data)} factor observations")
    except Exception as e:
        logger.error(f"❌ Failed to fetch factor data: {e}")
        factor_data = None

    return equity_data, factor_data


def train_residual_predictor(equity_data: dict, factor_data, symbols: list):
    """Train the residual predictor model using new pipeline."""
    logger.info("🚀 Starting Residual Predictor Training...")

    # Create training configuration
    config = TrainingConfig(
        use_cv=True,
        cv_folds=5,
        test_size=0.2,
        random_state=42,
        early_stopping=True,
        validation_split=0.2
    )

    # Create training pipeline
    pipeline = TrainingPipeline(
        model_type="residual_predictor",
        config=config
    )

    # Prepare data for pipeline
    # Combine all equity data
    all_data = []
    for symbol, data in equity_data.items():
        df = data.copy()
        df['symbol'] = symbol
        all_data.append(df)

    if not all_data:
        logger.error("❌ No equity data available for training")
        return None

    combined_data = pd.concat(all_data, ignore_index=True)

    try:
        # Run training pipeline
        logger.info("📚 Running training pipeline...")
        result = pipeline.run_pipeline(
            equity_data=equity_data,
            factor_data=factor_data,
            symbols=symbols,
            model_name="residual_predictor_prod"
        )

        logger.info("✅ Training completed successfully!")
        logger.info(f"   Model ID: {result.get('model_id', 'Unknown')}")
        logger.info(f"   Training metrics: {result.get('metrics', {})}")

        return result

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None


def train_ff5_model(equity_data: dict, factor_data, symbols: list):
    """Train the FF5 regression model using new pipeline."""
    logger.info("🚀 Starting FF5 Model Training...")

    # Create training configuration
    config = TrainingConfig(
        use_cv=False,  # FF5 is simpler, no CV needed
        test_size=0.2,
        random_state=42
    )

    # Create training pipeline
    pipeline = TrainingPipeline(
        model_type="ff5_regression",
        config=config
    )

    try:
        # Run training pipeline
        logger.info("📚 Running training pipeline...")
        result = pipeline.run_pipeline(
            equity_data=equity_data,
            factor_data=factor_data,
            symbols=symbols,
            model_name="ff5_regression_prod"
        )

        logger.info("✅ Training completed successfully!")
        logger.info(f"   Model ID: {result.get('model_id', 'Unknown')}")
        logger.info(f"   Training metrics: {result.get('metrics', {})}")

        return result

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None


def test_model_inference(model_type: str, symbols: list):
    """Test model inference using ModelPredictor."""
    logger.info("🧪 Testing Model Inference...")

    try:
        # Create model predictor
        predictor = ModelPredictor(
            enable_monitoring=True,
            cache_predictions=True
        )

        # Load model
        model_id = predictor.load_model(model_type)
        logger.info(f"✅ Loaded model: {model_id}")

        # Check model health
        health = predictor.get_model_health()
        if health:
            logger.info(f"📊 Model health: {health.status}")

        # Test prediction for each symbol
        logger.info("🔮 Making test predictions...")
        for symbol in symbols[:3]:  # Test first 3 symbols
            try:
                # Create some sample market data
                sample_data = pd.DataFrame({
                    'close': [100.0, 101.0, 102.0, 103.0, 104.0],
                    'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
                    'high': [101.0, 102.0, 103.0, 104.0, 105.0],
                    'low': [99.0, 100.0, 101.0, 102.0, 103.0],
                    'open': [100.0, 101.0, 102.0, 103.0, 104.0]
                })

                prediction = predictor.predict(
                    market_data=sample_data,
                    symbol=symbol,
                    prediction_date=datetime.now()
                )

                logger.info(f"✅ {symbol}: Predicted return = {prediction['prediction']:.4f}")

            except Exception as e:
                logger.warning(f"⚠️  Prediction failed for {symbol}: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Model inference test failed: {e}")
        return False


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train models using new architecture")
    parser.add_argument("--model", choices=["residual_predictor", "ff5_regression", "all"],
                       default="residual_predictor", help="Model type to train")
    parser.add_argument("--symbols", nargs="+",
                       default=["SPY", "QQQ", "IWM", "EFA", "EEM"],
                       help="Symbols to train on")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use shorter date range for testing")
    parser.add_argument("--years", type=int, default=2,
                       help="Number of years of historical data to use")

    args = parser.parse_args()

    logger.info("🎯 New Architecture Training Script")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Test mode: {args.test_mode}")

    # Set date range
    end_date = datetime.now()
    if args.test_mode:
        start_date = end_date - timedelta(days=180)  # 6 months for testing
    else:
        start_date = end_date - timedelta(days=args.years * 365)  # N years

    logger.info(f"   Date range: {start_date.date()} to {end_date.date()}")

    # Import pandas here to avoid dependency issues during argument parsing
    try:
        import pandas as pd
    except ImportError:
        logger.error("❌ pandas is required. Install with: pip install pandas")
        return 1

    # Create sample data
    equity_data, factor_data = create_sample_data(args.symbols, start_date, end_date)

    if not equity_data:
        logger.error("❌ No equity data available. Cannot proceed with training.")
        return 1

    # Train models
    results = {}

    if args.model in ["residual_predictor", "all"]:
        result = train_residual_predictor(equity_data, factor_data, args.symbols)
        if result:
            results["residual_predictor"] = result

    if args.model in ["ff5_regression", "all"]:
        result = train_ff5_model(equity_data, factor_data, args.symbols)
        if result:
            results["ff5_regression"] = result

    # Test inference
    logger.info("\n🧪 Testing Model Inference...")
    for model_type in results.keys():
        logger.info(f"\n--- Testing {model_type} ---")
        test_model_inference(model_type, args.symbols)

    # Summary
    logger.info("\n📋 Training Summary")
    logger.info(f"   Models trained: {list(results.keys())}")
    for model_type, result in results.items():
        model_id = result.get('model_id', 'Unknown')
        logger.info(f"   {model_type}: {model_id}")

    logger.info("\n🎉 Training script completed!")
    logger.info("💡 Next steps:")
    logger.info("   1. Use the trained models in your trading strategy")
    logger.info("   2. Monitor model health with ModelPredictor")
    logger.info("   3. Retrain models periodically with new data")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)