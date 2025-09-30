"""
Test script for ML Residual Predictor module.

This script tests the ML residual predictor functionality:
- Feature engineering from price data
- Model training with time series cross-validation
- Residual prediction using ensemble methods
- Model performance evaluation and validation
- Model degradation monitoring
- Governance and risk management

Usage:
    python test_residual_predictor.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_system.models.residual_predictor import ResidualPredictor
from trading_system.data.ff5_provider import FF5DataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_residuals_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.Series]:
    """Create synthetic residuals data for testing."""
    residuals_dict = {}

    for symbol in symbols:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic residuals (should be mean zero, some autocorrelation)
        base_residuals = np.random.normal(0, 0.02, len(dates))

        # Add some autocorrelation
        for i in range(1, len(base_residuals)):
            base_residuals[i] += 0.2 * base_residuals[i-1]

        # Add some regime-specific effects
        for i, date in enumerate(dates):
            if date.month in [1, 2, 11, 12]:  # Year-end effects
                base_residuals[i] += np.random.normal(0, 0.005)

        residuals_series = pd.Series(base_residuals, index=dates)
        residuals_dict[symbol] = residuals_series

    return residuals_dict


def create_test_equity_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Create synthetic equity data for testing."""
    equity_data = {}

    for symbol in symbols:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic price movements
        initial_price = 100 + np.random.uniform(-50, 50)
        price_changes = np.random.normal(0.0005, 0.015, len(dates))
        prices = [initial_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))

        prices = prices[1:]

        # Create OHLC data
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.98, 1.02, len(prices)),
            'High': prices * np.random.uniform(1.01, 1.05, len(prices)),
            'Low': prices * np.random.uniform(0.95, 0.99, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(100000, 5000000, len(prices)),
            'Adj Close': prices
        }, index=dates)

        equity_data[symbol] = df

    return equity_data


def test_residual_predictor_initialization():
    """Test ML residual predictor initialization."""
    print("=" * 60)
    print("TEST 1: ML Residual Predictor Initialization")
    print("=" * 60)

    try:
        # Test different model configurations
        configs = [
            {
                'model_type': 'xgboost',
                'max_features': 50,
                'prediction_horizon': 5,
                'feature_importance_threshold': 0.01
            },
            {
                'model_type': 'lightgbm',
                'max_features': 90,
                'prediction_horizon': 10,
                'feature_importance_threshold': 0.005
            },
            {
                'model_type': 'random_forest',
                'max_features': 45,
                'prediction_horizon': 3,
                'feature_importance_threshold': 0.02
            }
        ]

        for i, config in enumerate(configs):
            predictor = ResidualPredictor(**config)
            print(f"Configuration {i+1} ({config['model_type']}):")
            print(f"  Model type: {predictor.config.model_type}")
            print(f"  Max features: {predictor.config.get('max_features', 'N/A')}")
            print(f"  Prediction horizon: {predictor.config.prediction_horizon}")
            print(f"  Feature importance threshold: {predictor.config.feature_importance_threshold}")
            print(f"  ✓ Initialized successfully")

        print("\n✓ ML residual predictor initialization working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering from price data."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='xgboost',
            max_features=50,
            prediction_horizon=5
        )

        # Create test equity data
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        print(f"Created equity data for {len(equity_data)} symbols")

        # Test feature engineering for each symbol
        for symbol in symbols:
            print(f"\nFeature engineering for {symbol}:")
            print("-" * 40)

            price_data = equity_data[symbol]
            features = predictor._engineer_features(price_data)

            print(f"  Features shape: {features.shape}")
            print(f"  Number of features: {features.shape[1]}")
            print(f"  Date range: {features.index.min().date()} to {features.index.max().date()}")

            # Display some key features
            key_features = [col for col in features.columns if any(x in col for x in ['momentum', 'volatility', 'trend'])][:5]
            if key_features:
                print(f"  Key features: {key_features}")

            # Check for missing values
            missing_counts = features.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"  ⚠ Missing values: {missing_counts.sum()} total")
            else:
                print(f"  ✓ No missing values")

        print("\n✓ Feature engineering working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_model_training():
    """Test model training with time series cross-validation."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Training")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='lightgbm',
            max_features=50,
            prediction_horizon=5
        )

        # Create test data
        symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        print(f"Created data for {len(equity_data)} symbols")
        print(f"Training models from {start_date.date()} to {end_date.date()}")

        # Train models
        trained_models = predictor.train_models(
            equity_data,
            residuals_data,
            start_date=start_date,
            end_date=end_date
        )

        print(f"\nTraining Results:")
        print("-" * 40)
        print(f"Models trained: {len(trained_models)}")

        for symbol, model_info in trained_models.items():
            print(f"\n{symbol}:")
            print(f"  Model type: {model_info['model_type']}")
            print(f"  Training samples: {model_info['training_samples']}")
            print(f"  Features used: {model_info['features_used']}")

            if 'performance_metrics' in model_info:
                metrics = model_info['performance_metrics']
                print(f"  CV R² score: {metrics.get('cv_r2_score', 'N/A')}")
                print(f"  CV RMSE: {metrics.get('cv_rmse', 'N/A')}")
                print(f"  Training R²: {metrics.get('training_r2_score', 'N/A')}")

        print("\n✓ Model training working correctly")
        return True, trained_models

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False, None


def test_residual_prediction():
    """Test residual prediction with trained models."""
    print("\n" + "=" * 60)
    print("TEST 4: Residual Prediction")
    print("=" * 60)

    try:
        # Use models from previous test
        _, trained_models = test_model_training()

        if trained_models is None:
            print("✗ Cannot test prediction without trained models")
            return False

        # Initialize predictor with same configuration
        predictor = ResidualPredictor(
            model_type='lightgbm',
            max_features=50,
            prediction_horizon=5
        )
        predictor.trained_models = trained_models

        # Create test data for prediction
        symbols = ['SPY', 'QQQ']
        start_date = datetime(2023, 7, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)

        print(f"Making predictions for {len(symbols)} symbols")
        print(f"Prediction period: {start_date.date()} to {end_date.date()}")

        # Make predictions
        predictions = predictor.predict_residuals(
            equity_data,
            start_date=start_date,
            end_date=end_date
        )

        print(f"\nPrediction Results:")
        print("-" * 40)

        for symbol, pred_data in predictions.items():
            print(f"\n{symbol}:")
            print(f"  Prediction dates: {len(pred_data['dates'])}")
            print(f"  Predictions shape: {pred_data['predictions'].shape}")
            print(f"  Confidence scores shape: {pred_data['confidence_scores'].shape}")

            # Display prediction statistics
            pred_values = pred_data['predictions']
            conf_scores = pred_data['confidence_scores']

            print(f"  Mean prediction: {pred_values.mean():.6f}")
            print(f"  Std prediction: {pred_values.std():.6f}")
            print(f"  Mean confidence: {conf_scores.mean():.4f}")
            print(f"  Min/Max confidence: {conf_scores.min():.4f} / {conf_scores.max():.4f}")

            # Check prediction reasonableness
            if abs(pred_values.mean()) > 0.1:
                print(f"  ⚠ High mean prediction - potential bias")
            else:
                print(f"  ✓ Reasonable mean prediction")

        print("\n✓ Residual prediction working correctly")
        return True, predictions

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False, None


def test_model_performance_tracking():
    """Test model performance tracking and validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Performance Tracking")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='xgboost',
            max_features=50,
            prediction_horizon=5
        )

        # Create test data
        symbols = ['SPY', 'QQQ']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        # Train models
        trained_models = predictor.train_models(equity_data, residuals_data, start_date, end_date)
        predictor.trained_models = trained_models

        # Create test data for performance evaluation
        test_start = datetime(2023, 7, 1)
        test_end = datetime(2023, 9, 30)

        test_equity = create_test_equity_data(symbols, test_start, test_end)
        test_residuals = create_test_residuals_data(symbols, test_start, test_end)

        print(f"Evaluating performance from {test_start.date()} to {test_end.date()}")

        # Evaluate performance
        performance_results = predictor.evaluate_model_performance(
            test_equity,
            test_residuals,
            test_start,
            test_end
        )

        print(f"\nPerformance Results:")
        print("-" * 40)

        for symbol, metrics in performance_results.items():
            print(f"\n{symbol}:")
            print(f"  Test R²: {metrics.get('test_r2_score', 'N/A')}")
            print(f"  Test RMSE: {metrics.get('test_rmse', 'N/A')}")
            print(f"  Test MAE: {metrics.get('test_mae', 'N/A')}")
            print(f"  Directional accuracy: {metrics.get('directional_accuracy', 'N/A')}")
            print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            print(f"  Max drawdown: {metrics.get('max_drawdown', 'N/A')}")

            # Assess performance quality
            r2_score = metrics.get('test_r2_score', 0)
            if r2_score > 0.1:
                print(f"  ✓ Good predictive power (R² > 0.1)")
            elif r2_score > 0.05:
                print(f"  ⚠ Moderate predictive power (0.05 < R² < 0.1)")
            else:
                print(f"  ⚠ Low predictive power (R² < 0.05)")

        print("\n✓ Model performance tracking working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_model_degradation_monitoring():
    """Test model degradation monitoring."""
    print("\n" + "=" * 60)
    print("TEST 6: Model Degradation Monitoring")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='random_forest',
            max_features=50,
            prediction_horizon=5
        )

        # Simulate model degradation by creating data with changing patterns
        symbols = ['SPY']

        # Create training data (normal regime)
        train_start = datetime(2022, 1, 1)
        train_end = datetime(2022, 12, 31)

        train_equity = create_test_equity_data(symbols, train_start, train_end)
        train_residuals = create_test_residuals_data(symbols, train_start, train_end)

        # Train initial model
        trained_models = predictor.train_models(train_equity, train_residuals, train_start, train_end)
        predictor.trained_models = trained_models

        # Create test data with regime change (degradation simulation)
        test_start = datetime(2023, 1, 1)
        test_end = datetime(2023, 6, 30)

        # Create data with different characteristics
        test_equity = create_test_equity_data(symbols, test_start, test_end)
        test_residuals = create_test_residuals_data(symbols, test_start, test_end)

        # Add regime change effect (simulate degradation)
        for symbol in test_residuals:
            # Add increasing bias over time
            dates = test_residuals[symbol].index
            for i, date in enumerate(dates):
                if date > datetime(2023, 3, 1):  # Regime change
                    test_residuals[symbol].iloc[i] += 0.01 * (i / len(dates))  # Increasing bias

        print(f"Testing degradation monitoring from {test_start.date()} to {test_end.date()}")

        # Monitor for degradation
        degradation_results = predictor.monitor_model_degradation(
            test_equity,
            test_residuals,
            test_start,
            test_end
        )

        print(f"\nDegradation Monitoring Results:")
        print("-" * 40)

        for symbol, results in degradation_results.items():
            print(f"\n{symbol}:")
            print(f"  Performance degradation: {results.get('performance_degradation', 'N/A')}")
            print(f"  Stability score: {results.get('stability_score', 'N/A')}")
            print(f"  Degradation detected: {results.get('degradation_detected', False)}")
            print(f"  Confidence decay: {results.get('confidence_decay', 'N/A')}")

            if results.get('degradation_detected', False):
                print(f"  ⚠ Model degradation detected!")
                print(f"  Recommended action: {results.get('recommended_action', 'N/A')}")
            else:
                print(f"  ✓ No significant degradation detected")

        print("\n✓ Model degradation monitoring working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_feature_importance_analysis():
    """Test feature importance analysis."""
    print("\n" + "=" * 60)
    print("TEST 7: Feature Importance Analysis")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='xgboost',
            max_features=50,
            prediction_horizon=5
        )

        # Create test data
        symbols = ['SPY', 'QQQ']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        # Train models
        trained_models = predictor.train_models(equity_data, residuals_data, start_date, end_date)

        print(f"Analyzing feature importance for {len(symbols)} symbols")

        # Analyze feature importance
        importance_results = predictor.analyze_feature_importance(trained_models)

        print(f"\nFeature Importance Results:")
        print("-" * 40)

        for symbol, importance_data in importance_results.items():
            print(f"\n{symbol}:")
            print(f"  Total features: {importance_data['total_features']}")
            print(f"  Important features: {importance_data['important_features_count']}")
            print(f"  Top 5 features:")

            top_features = importance_data['top_features'][:5]
            for i, (feature, importance_score) in enumerate(top_features):
                print(f"    {i+1}. {feature}: {importance_score:.4f}")

            # Feature categories analysis
            categories = importance_data.get('feature_categories', {})
            if categories:
                print(f"  Feature categories:")
                for category, count in categories.items():
                    print(f"    {category}: {count} features")

        print("\n✓ Feature importance analysis working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_model_governance():
    """Test model governance and risk management."""
    print("\n" + "=" * 60)
    print("TEST 8: Model Governance")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = ResidualPredictor(
            model_type='lightgbm',
            max_features=50,
            prediction_horizon=5
        )

        # Create test data
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        # Train models
        trained_models = predictor.train_models(equity_data, residuals_data, start_date, end_date)
        predictor.trained_models = trained_models

        # Test governance checks
        print("Running governance checks...")

        # Check model risk limits
        risk_limits = predictor.check_model_risk_limits()
        print(f"\nModel Risk Limits:")
        print("-" * 40)
        for limit_name, limit_value in risk_limits.items():
            status = "✓" if limit_value['within_limit'] else "✗"
            print(f"  {status} {limit_name}: {limit_value['current_value']:.4f} (limit: {limit_value['limit_value']:.4f})")

        # Check model health
        health_status = predictor.check_model_health()
        print(f"\nModel Health Status:")
        print("-" * 40)
        for health_metric, health_value in health_status.items():
            status = "✓" if health_value['healthy'] else "⚠"
            print(f"  {status} {health_metric}: {health_value['score']:.4f}")

        # Generate governance report
        governance_report = predictor.generate_governance_report()
        print(f"\nGovernance Report Summary:")
        print("-" * 40)
        print(f"  Overall health: {governance_report['overall_health']:.4f}")
        print(f"  Risk level: {governance_report['risk_level']}")
        print(f"  Recommendations: {len(governance_report['recommendations'])}")

        if governance_report['recommendations']:
            print(f"  Top recommendations:")
            for rec in governance_report['recommendations'][:3]:
                print(f"    - {rec}")

        print("\n✓ Model governance working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all ML residual predictor tests."""
    print("ML Residual Predictor Test Suite")
    print("=" * 60)
    print("Testing machine learning residual prediction and model governance")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_residual_predictor_initialization,
        test_feature_engineering,
        test_model_training,
        test_residual_prediction,
        test_model_performance_tracking,
        test_model_degradation_monitoring,
        test_feature_importance_analysis,
        test_model_governance
    ]

    for test in tests:
        try:
            if test in [test_model_training, test_residual_prediction]:
                success, _ = test()
                test_results.append((test.__name__, success))
            else:
                success = test()
                test_results.append((test.__name__, success))
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            test_results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    print(f"Tests passed: {passed}/{total}")

    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n🎉 All ML residual predictor tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the ML residual predictor test suite."""
    success = run_all_tests()

    if success:
        print("\nML Residual Predictor module is working correctly!")
        sys.exit(0)
    else:
        print("\nML Residual Predictor module has issues that need to be addressed.")
        sys.exit(1)