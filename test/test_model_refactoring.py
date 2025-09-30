"""
Validation Tests for Refactored Models

This script tests the refactored model architecture to ensure:
1. New models work correctly
2. Model factory integration works
3. Trainer integration works
4. Performance evaluation works
5. Model monitoring works

This validates that the refactoring was successful and the models
follow the clean architecture principles.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples=1000):
    """Create synthetic test data for model validation."""
    np.random.seed(42)  # For reproducible results

    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Create FF5 factor data
    factor_data = pd.DataFrame({
        'MKT': np.random.normal(0.05, 0.15, n_samples),  # Market factor
        'SMB': np.random.normal(0.02, 0.10, n_samples),  # Size factor
        'HML': np.random.normal(0.03, 0.12, n_samples),  # Value factor
        'RMW': np.random.normal(0.01, 0.08, n_samples),  # Profitability factor
        'CMA': np.random.normal(0.02, 0.09, n_samples),  # Investment factor
    }, index=dates)

    # Create technical features for residual model
    technical_features = pd.DataFrame({
        'RSI': np.random.uniform(20, 80, n_samples),
        'MACD': np.random.normal(0, 1, n_samples),
        'momentum_20': np.random.normal(0.02, 0.05, n_samples),
        'volatility_20': np.random.uniform(0.1, 0.4, n_samples),
        'volume_ratio': np.random.uniform(0.8, 1.5, n_samples),
    }, index=dates)

    # Create true returns (FF5 + residual + noise)
    true_betas = [1.2, 0.3, -0.2, 0.4, 0.1]  # True factor betas
    factor_returns = (factor_data * true_betas).sum(axis=1)

    # Create residual component (using technical features)
    residual_coefficients = [0.01, -0.005, 0.02, -0.01, 0.008]
    residual_returns = (technical_features * residual_coefficients).sum(axis=1)

    # Add noise
    noise = np.random.normal(0, 0.02, n_samples)

    # Final returns
    returns = factor_returns + residual_returns + noise

    # Combine all features
    all_features = pd.concat([factor_data, technical_features], axis=1)

    return all_features, returns, factor_data, technical_features

def test_ff5_model():
    """Test FF5 regression model."""
    logger.info("Testing FF5 Regression Model...")

    try:
        from src.trading_system.models.implementations.ff5_model import FF5RegressionModel

        # Create test data
        X_factors, returns, _, _ = create_test_data(500)

        # Initialize model
        model = FF5RegressionModel(config={'regularization': 'ridge', 'alpha': 1.0})

        # Test fit
        model.fit(X_factors, returns)

        assert model.status == "trained", "Model should be trained after fit"
        assert len(model.metadata.features) == 5, "Should have 5 factor features"

        # Test predict
        predictions = model.predict(X_factors)
        assert len(predictions) == len(X_factors), "Predictions length should match input"

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 5, "Should have importance for all 5 factors"

        # Test model info
        info = model.get_model_info()
        assert 'factor_betas' in info, "Model info should contain factor betas"

        logger.info("✅ FF5 Model test passed")
        return True

    except Exception as e:
        logger.error(f"❌ FF5 Model test failed: {e}")
        return False

def test_residual_model():
    """Test residual prediction model."""
    logger.info("Testing Residual Prediction Model...")

    try:
        from src.trading_system.models.implementations.ff5_model import FF5RegressionModel
        from src.trading_system.models.implementations.residual_model import ResidualPredictionModel

        # Create test data
        X_all, returns, X_factors, X_technical = create_test_data(500)

        # Initialize model
        model = ResidualPredictionModel(config={
            'residual_model_type': 'random_forest',
            'residual_params': {'n_estimators': 10, 'max_depth': 3}
        })

        # Test fit
        model.fit(X_all, returns)

        assert model.status == "trained", "Model should be trained after fit"
        assert len(model.metadata.features) == 10, "Should have 10 total features"

        # Test predict
        predictions = model.predict(X_all)
        assert len(predictions) == len(X_all), "Predictions length should match input"

        # Test sub-models
        ff5_model = model.get_ff5_model()
        residual_model = model.get_residual_model()

        assert ff5_model is not None, "FF5 model should be set"
        assert residual_model is not None, "Residual model should be set"

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0, "Should have feature importance"

        logger.info("✅ Residual Model test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Residual Model test failed: {e}")
        return False

def test_model_factory():
    """Test model factory integration."""
    logger.info("Testing Model Factory...")

    try:
        from src.trading_system.models.registry import ModelFactory, create_ff5_model, create_residual_predictor

        # Test factory registration
        available_models = list(ModelFactory._registry.keys())
        assert "ff5_regression" in available_models, "FF5 model should be registered"
        assert "residual_predictor" in available_models, "Residual predictor should be registered"

        # Test factory creation
        X_all, returns, _, _ = create_test_data(200)

        # Create FF5 model via factory
        ff5_model = ModelFactory.create("ff5_regression")
        ff5_model.fit(X_all[['MKT', 'SMB', 'HML', 'RMW', 'CMA']], returns)
        assert ff5_model.status == "trained", "Factory-created FF5 model should work"

        # Create residual model via factory
        residual_model = ModelFactory.create("residual_predictor")
        residual_model.fit(X_all, returns)
        assert residual_model.status == "trained", "Factory-created residual model should work"

        # Test convenience functions
        ff5_model_2 = create_ff5_model()
        residual_model_2 = create_residual_predictor()

        assert ff5_model_2.model_type == "ff5_regression", "Convenience function should work"
        assert residual_model_2.model_type == "residual_predictor", "Convenience function should work"

        logger.info("✅ Model Factory test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Model Factory test failed: {e}")
        return False

def test_trainer_integration():
    """Test trainer integration."""
    logger.info("Testing Trainer Integration...")

    try:
        from src.trading_system.models.training.trainer import ModelTrainer, TrainingConfig
        from src.trading_system.models.implementations.ff5_model import FF5RegressionModel

        # Create test data
        X_factors, returns, _, _ = create_test_data(300)

        # Initialize trainer
        config = TrainingConfig(
            use_cross_validation=False,  # Skip CV for speed
            validation_split=0.2
        )
        trainer = ModelTrainer(config)

        # Initialize model
        model = FF5RegressionModel()

        # Train using trainer
        result = trainer.train(model, X_factors, returns)

        assert result.model.status == "trained", "Model should be trained by trainer"
        assert result.training_time > 0, "Training time should be recorded"
        assert len(result.validation_metrics) > 0, "Should have validation metrics"

        logger.info("✅ Trainer Integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Trainer Integration test failed: {e}")
        return False

def test_performance_evaluator():
    """Test performance evaluator."""
    logger.info("Testing Performance Evaluator...")

    try:
        from src.trading_system.models.utils.performance_evaluator import PerformanceEvaluator
        from src.trading_system.models.implementations.ff5_model import FF5RegressionModel

        # Create test data
        X_factors, returns, _, _ = create_test_data(200)

        # Train model
        model = FF5RegressionModel()
        model.fit(X_factors, returns)

        # Evaluate model
        metrics = PerformanceEvaluator.evaluate_model(model, X_factors, returns)

        assert 'r2' in metrics, "Should have R² metric"
        assert 'mse' in metrics, "Should have MSE metric"
        assert 'model_type' in metrics, "Should have model type info"

        # Test financial evaluation
        financial_metrics = PerformanceEvaluator.evaluate_financial_model(
            model, X_factors, returns
        )

        assert 'information_coefficient' in financial_metrics, "Should have IC metric"
        assert 'directional_accuracy' in financial_metrics, "Should have directional accuracy"

        logger.info("✅ Performance Evaluator test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Performance Evaluator test failed: {e}")
        return False

def test_model_monitor():
    """Test model monitor."""
    logger.info("Testing Model Monitor...")

    try:
        from src.trading_system.models.serving.monitor import ModelMonitor
        from src.trading_system.models.implementations.ff5_model import FF5RegressionModel

        # Create test data
        X_factors, returns, _, _ = create_test_data(100)

        # Train model
        model = FF5RegressionModel()
        model.fit(X_factors, returns)

        # Initialize monitor
        monitor = ModelMonitor("test_model", config={'performance_window': 7})

        # Log some predictions
        for i in range(min(50, len(X_factors))):
            features = X_factors.iloc[i].to_dict()
            prediction = model.predict(X_factors.iloc[i:i+1])[0]
            actual = returns.iloc[i]

            pred_id = monitor.log_prediction(
                features=features,
                prediction=prediction,
                actual=actual
            )

        # Check health status
        health = monitor.get_health_status(model)
        assert health.model_id == "test_model", "Health status should have correct model ID"
        assert len(health.metrics) > 0, "Should have health metrics"

        # Generate report
        report = monitor.generate_report()
        assert 'model_id' in report, "Report should have model ID"
        assert 'total_predictions' in report, "Report should have prediction count"

        logger.info("✅ Model Monitor test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Model Monitor test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    logger.info("🚀 Starting Model Refactoring Validation Tests")
    logger.info("=" * 60)

    tests = [
        ("FF5 Model", test_ff5_model),
        ("Residual Model", test_residual_model),
        ("Model Factory", test_model_factory),
        ("Trainer Integration", test_trainer_integration),
        ("Performance Evaluator", test_performance_evaluator),
        ("Model Monitor", test_model_monitor)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Model refactoring successful!")
        logger.info("\nKey achievements:")
        logger.info("✅ Models follow Single Responsibility Principle")
        logger.info("✅ Models integrate with Trainer correctly")
        logger.info("✅ Model Factory pattern works")
        logger.info("✅ Performance evaluation is externalized")
        logger.info("✅ Model monitoring is separate from model logic")
        logger.info("✅ Clean architecture achieved")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)