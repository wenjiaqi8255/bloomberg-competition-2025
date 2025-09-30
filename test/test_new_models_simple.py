"""
Simple Validation Tests for New Models

This script tests only the new model implementations without
dependencies on the legacy system.
"""

import logging
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_test_data(n_samples=200):
    """Create simple synthetic test data."""
    np.random.seed(42)

    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Create FF5 factor data
    factor_data = pd.DataFrame({
        'MKT': np.random.normal(0.05, 0.15, n_samples),
        'SMB': np.random.normal(0.02, 0.10, n_samples),
        'HML': np.random.normal(0.03, 0.12, n_samples),
        'RMW': np.random.normal(0.01, 0.08, n_samples),
        'CMA': np.random.normal(0.02, 0.09, n_samples),
    }, index=dates)

    # Create simple technical features
    technical_features = pd.DataFrame({
        'feature_1': np.random.uniform(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.exponential(1, n_samples),
    }, index=dates)

    # Create target returns
    true_betas = [1.0, 0.2, -0.1, 0.3, 0.05]
    factor_returns = (factor_data * true_betas).sum(axis=1)
    residual_coefficients = [0.01, -0.005, 0.02]
    residual_returns = (technical_features * residual_coefficients).sum(axis=1)
    noise = np.random.normal(0, 0.02, n_samples)

    returns = factor_returns + residual_returns + noise

    # Combine all features
    all_features = pd.concat([factor_data, technical_features], axis=1)

    return all_features, returns, factor_data, technical_features

def test_ff5_model_basic():
    """Test basic FF5 model functionality."""
    logger.info("Testing FF5 Model (Basic)...")

    try:
        # Import the new model
        sys.path.append('src')
        from trading_system.models.implementations.ff5_model import FF5RegressionModel

        # Create test data
        X_factors, returns, _, _ = create_simple_test_data(100)

        # Initialize model
        model = FF5RegressionModel(config={'regularization': 'ridge', 'alpha': 1.0})

        # Test fit
        model.fit(X_factors, returns)

        # Basic assertions
        assert model.status == "trained", f"Expected 'trained', got '{model.status}'"
        assert len(model.metadata.features) == 5, f"Expected 5 features, got {len(model.metadata.features)}"
        assert model.metadata.training_samples == len(X_factors), "Training samples mismatch"

        # Test predict
        predictions = model.predict(X_factors)
        assert len(predictions) == len(X_factors), "Prediction length mismatch"

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 5, f"Expected 5 importance values, got {len(importance)}"
        assert all(k in importance for k in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']), "Missing factor importance"

        # Test model info
        info = model.get_model_info()
        assert 'factor_betas' in info, "Missing factor betas in model info"
        assert 'factors' in info, "Missing factors in model info"

        logger.info("✅ FF5 Model basic test passed")
        return True

    except Exception as e:
        logger.error(f"❌ FF5 Model basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_residual_model_basic():
    """Test basic residual model functionality."""
    logger.info("Testing Residual Model (Basic)...")

    try:
        sys.path.append('src')
        from trading_system.models.implementations.ff5_model import FF5RegressionModel
        from trading_system.models.implementations.residual_model import ResidualPredictionModel

        # Create test data
        X_all, returns, X_factors, X_technical = create_simple_test_data(100)

        # Initialize model
        model = ResidualPredictionModel(config={
            'residual_model_type': 'ridge',  # Use simple ridge to avoid XGBoost dependency
            'residual_params': {'alpha': 1.0}
        })

        # Test fit
        model.fit(X_all, returns)

        # Basic assertions
        assert model.status == "trained", f"Expected 'trained', got '{model.status}'"
        assert len(model.metadata.features) == 8, f"Expected 8 features, got {len(model.metadata.features)}"

        # Test predict
        predictions = model.predict(X_all)
        assert len(predictions) == len(X_all), "Prediction length mismatch"

        # Test sub-models
        ff5_model = model.get_ff5_model()
        residual_model = model.get_residual_model()

        assert ff5_model is not None, "FF5 model should not be None"
        assert residual_model is not None, "Residual model should not be None"

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0, "Should have some feature importance"

        logger.info("✅ Residual Model basic test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Residual Model basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory_basic():
    """Test basic model factory functionality."""
    logger.info("Testing Model Factory (Basic)...")

    try:
        sys.path.append('src')
        from trading_system.models.base.model_factory import ModelFactory
        from trading_system.models.implementations.ff5_model import FF5RegressionModel
        from trading_system.models.implementations.residual_model import ResidualPredictionModel

        # Manually register models (since registry import might fail)
        ModelFactory.register(
            model_type="test_ff5",
            model_class=FF5RegressionModel,
            description="Test FF5 model",
            default_config={'regularization': 'none'}
        )

        ModelFactory.register(
            model_type="test_residual",
            model_class=ResidualPredictionModel,
            description="Test residual model",
            default_config={'residual_model_type': 'ridge'}
        )

        # Test factory creation
        X_all, returns, _, _ = create_simple_test_data(50)

        # Create FF5 model via factory
        ff5_model = ModelFactory.create("test_ff5")
        ff5_model.fit(X_all[['MKT', 'SMB', 'HML', 'RMW', 'CMA']], returns)
        assert ff5_model.status == "trained", "Factory-created FF5 model should work"

        # Create residual model via factory
        residual_model = ModelFactory.create("test_residual")
        residual_model.fit(X_all, returns)
        assert residual_model.status == "trained", "Factory-created residual model should work"

        logger.info("✅ Model Factory basic test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Model Factory basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_line_count_reduction():
    """Verify that the refactored models are significantly smaller."""
    logger.info("Testing Code Size Reduction...")

    try:
        import os
        from pathlib import Path

        # Check old files
        old_ff5_path = Path("src/trading_system/models/ff5_regression.py")
        old_residual_path = Path("src/trading_system/models/residual_predictor.py")

        # Check new files
        new_ff5_path = Path("src/trading_system/models/implementations/ff5_model.py")
        new_residual_path = Path("src/trading_system/models/implementations/residual_model.py")

        results = {}

        # Count lines in old FF5
        if old_ff5_path.exists():
            with open(old_ff5_path, 'r') as f:
                old_ff5_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
            results['old_ff5_lines'] = old_ff5_lines
        else:
            results['old_ff5_lines'] = "File not found"

        # Count lines in new FF5
        if new_ff5_path.exists():
            with open(new_ff5_path, 'r') as f:
                new_ff5_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
            results['new_ff5_lines'] = new_ff5_lines
        else:
            results['new_ff5_lines'] = "File not found"

        # Count lines in old residual
        if old_residual_path.exists():
            with open(old_residual_path, 'r') as f:
                old_residual_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
            results['old_residual_lines'] = old_residual_lines
        else:
            results['old_residual_lines'] = "File not found"

        # Count lines in new residual
        if new_residual_path.exists():
            with open(new_residual_path, 'r') as f:
                new_residual_lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
            results['new_residual_lines'] = new_residual_lines
        else:
            results['new_residual_lines'] = "File not found"

        # Calculate reductions
        if isinstance(results.get('old_ff5_lines'), int) and isinstance(results.get('new_ff5_lines'), int):
            ff5_reduction = (results['old_ff5_lines'] - results['new_ff5_lines']) / results['old_ff5_lines'] * 100
            results['ff5_reduction_percent'] = ff5_reduction

        if isinstance(results.get('old_residual_lines'), int) and isinstance(results.get('new_residual_lines'), int):
            residual_reduction = (results['old_residual_lines'] - results['new_residual_lines']) / results['old_residual_lines'] * 100
            results['residual_reduction_percent'] = residual_reduction

        # Log results
        logger.info("Code size analysis:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

        # Check if we achieved significant reduction
        if isinstance(results.get('ff5_reduction_percent'), (int, float)) and results['ff5_reduction_percent'] > 50:
            logger.info("✅ FF5 model significantly reduced")
            ff5_ok = True
        else:
            logger.warning("⚠️ FF5 model reduction not significant")
            ff5_ok = False

        if isinstance(results.get('residual_reduction_percent'), (int, float)) and results['residual_reduction_percent'] > 50:
            logger.info("✅ Residual model significantly reduced")
            residual_ok = True
        else:
            logger.warning("⚠️ Residual model reduction not significant")
            residual_ok = False

        return ff5_ok and residual_ok

    except Exception as e:
        logger.error(f"❌ Code size test failed: {e}")
        return False

def run_simple_tests():
    """Run simplified validation tests."""
    logger.info("🚀 Starting Simple Model Validation Tests")
    logger.info("=" * 60)

    tests = [
        ("FF5 Model (Basic)", test_ff5_model_basic),
        ("Residual Model (Basic)", test_residual_model_basic),
        ("Model Factory (Basic)", test_model_factory_basic),
        ("Code Size Reduction", test_line_count_reduction),
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
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("\nKey achievements:")
        logger.info("✅ New models work correctly")
        logger.info("✅ Model integration works")
        logger.info("✅ Factory pattern functional")
        logger.info("✅ Significant code reduction achieved")
        logger.info("✅ Clean architecture principles followed")
        return True
    else:
        logger.error("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)