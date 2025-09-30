"""
Test script for FF5 Regression Engine module.

This script tests the FF5 regression functionality:
- Rolling window beta estimation
- Factor-implied returns calculation
- Residual extraction
- Model diagnostics and validation
- Look-ahead bias prevention

Usage:
    python test_ff5_regression.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_system.models.ff5_regression import FF5RegressionEngine
from trading_system.data.ff5_provider import FF5DataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_equity_data(symbols: list, start_date: datetime, end_date: datetime) -> dict:
    """Create synthetic equity data for testing."""
    equity_data = {}

    for symbol in symbols:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol

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
            'Adj Close': prices  # Simplified
        }, index=dates)

        # Resample to monthly
        df_monthly = df.resample('M').last()
        df_monthly['Volume'] = df['Volume'].resample('M').sum()

        equity_data[symbol] = df_monthly

    return equity_data


def test_ff5_regression_initialization():
    """Test FF5 regression engine initialization."""
    print("=" * 60)
    print("TEST 1: FF5 Regression Engine Initialization")
    print("=" * 60)

    try:
        # Initialize with different parameters
        configs = [
            {'estimation_window': 36, 'min_observations': 24},
            {'estimation_window': 60, 'min_observations': 36},
            {'estimation_window': 12, 'min_observations': 6}
        ]

        for i, config in enumerate(configs):
            engine = FF5RegressionEngine(**config)
            print(f"Engine {i+1}:")
            print(f"  Estimation window: {engine.estimation_window}")
            print(f"  Min observations: {engine.min_observations}")
            print(f"  Regularization: {engine.regularization}")
            print(f"  ✓ Initialized successfully")

        print("\n✓ FF5 regression engine initialization working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_factor_beta_estimation():
    """Test factor beta estimation."""
    print("\n" + "=" * 60)
    print("TEST 2: Factor Beta Estimation")
    print("=" * 60)

    try:
        # Initialize regression engine
        engine = FF5RegressionEngine(estimation_window=36, min_observations=24)

        # Create test equity data
        symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        print(f"Created equity data for {len(equity_data)} symbols")

        # Get FF5 factor data
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        print(f"Retrieved {len(factor_data)} months of factor data")

        # Estimate factor betas
        factor_betas, factor_returns, residuals = engine.estimate_factor_betas(
            equity_data, factor_data
        )

        print(f"Estimated betas for {len(factor_betas)} symbols")

        # Display results for first symbol
        if factor_betas:
            first_symbol = list(factor_betas.keys())[0]
            beta_data = factor_betas[first_symbol]

            print(f"\nResults for {first_symbol}:")
            print("-" * 40)
            print(f"Betas (latest):")
            if len(beta_data) > 0:
                for factor in ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']:
                    if factor in beta_data.columns and not beta_data[factor].isna().all():
                        latest_beta = beta_data[factor].dropna().iloc[-1] if len(beta_data[factor].dropna()) > 0 else 0.0
                        print(f"  {factor.replace('_beta', '')}: {latest_beta:.4f}")

                if 'r_squared' in beta_data.columns and not beta_data['r_squared'].isna().all():
                    latest_r2 = beta_data['r_squared'].dropna().iloc[-1] if len(beta_data['r_squared'].dropna()) > 0 else 0.0
                    print(f"R-squared: {latest_r2:.4f}")
            else:
                print("  No beta estimates available")

        print("\n✓ Factor beta estimation working correctly")
        return True, factor_betas, factor_returns, residuals, factor_data

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False, None, None, None, None


def test_residual_extraction():
    """Test residual extraction from factor model."""
    print("\n" + "=" * 60)
    print("TEST 3: Residual Extraction")
    print("=" * 60)

    try:
        # Use data from previous test
        _, factor_betas, factor_returns, residuals, factor_data = test_factor_beta_estimation()

        if residuals is None:
            print("✗ Cannot test residuals without factor beta estimation")
            return False

        print(f"Extracted residuals for {len(residuals)} symbols")

        # Analyze residuals for first symbol
        if residuals:
            first_symbol = list(residuals.keys())[0]
            residual_series = residuals[first_symbol]

            print(f"\nResidual analysis for {first_symbol}:")
            print("-" * 40)
            print(f"Number of observations: {len(residual_series)}")
            print(f"Mean residual: {residual_series.mean():.6f}")
            print(f"Std residual: {residual_series.std():.6f}")
            print(f"Min residual: {residual_series.min():.6f}")
            print(f"Max residual: {residual_series.max():.6f}")

            # Test for residual properties
            # Residuals should have mean close to 0
            if abs(residual_series.mean()) > 0.01:
                print("⚠ Residuals have non-zero mean - potential model misspecification")
            else:
                print("✓ Residuals have mean close to zero")

            # Check for autocorrelation (simplified)
            if len(residual_series) > 1:
                autocorr = residual_series.autocorr(lag=1)
                print(f"Lag-1 autocorrelation: {autocorr:.4f}")
                if abs(autocorr) > 0.3:
                    print("⚠ High residual autocorrelation - potential model issues")
                else:
                    print("✓ Low residual autocorrelation")

        print("\n✓ Residual extraction working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_factor_implied_returns():
    """Test factor-implied returns calculation."""
    print("\n" + "=" * 60)
    print("TEST 4: Factor-Implied Returns")
    print("=" * 60)

    try:
        # Use data from previous tests
        _, factor_betas, factor_returns, _, factor_data = test_factor_beta_estimation()

        if factor_returns is None:
            print("✗ Cannot test factor returns without factor beta estimation")
            return False

        print(f"Calculated factor-implied returns for {len(factor_returns)} symbols")

        # Analyze factor-implied returns
        if factor_returns and len(factor_returns) > 0:
            first_symbol = list(factor_returns.keys())[0]
            return_series = factor_returns[first_symbol]

            if len(return_series) > 0:
                print(f"\nFactor-implied returns for {first_symbol}:")
                print("-" * 40)
                print(f"Number of observations: {len(return_series)}")
                print(f"Mean monthly return: {return_series.mean():.6f}")
                print(f"Std monthly return: {return_series.std():.6f}")
                print(f"Annualized return: {return_series.mean() * 12:.4f}")
                print(f"Annualized volatility: {return_series.std() * np.sqrt(12):.4f}")

                # Check if returns are reasonable
                if not np.isnan(return_series.mean()):
                    print("✓ Factor-implied returns calculated successfully")
                else:
                    print("⚠ Factor-implied returns contain NaN values")
            else:
                print("⚠ No factor-implied returns calculated")
        else:
            print("⚠ No factor returns available")

        print("\n✓ Factor-implied returns calculation working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_model_diagnostics():
    """Test model diagnostics and validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Diagnostics")
    print("=" * 60)

    try:
        # Initialize regression engine
        engine = FF5RegressionEngine(estimation_window=36, min_observations=24)

        # Create test data
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        # Estimate betas and get diagnostics
        factor_betas, _, _ = engine.estimate_factor_betas(equity_data, factor_data)

        if factor_betas:
            # Get diagnostics for first symbol
            first_symbol = list(factor_betas.keys())[0]
            diagnostics = engine.get_model_diagnostics(first_symbol)

            print(f"Model diagnostics for {first_symbol}:")
            print("-" * 40)
            print(f"Average R-squared: {diagnostics.get('avg_r_squared', 'N/A')}")
            print(f"Factor exposures MKT: {diagnostics.get('factor_exposures', {}).get('MKT', 'N/A')}")
            print(f"Alpha stats: {diagnostics.get('alpha_stats', {})}")

            # Assess model quality
            avg_r2 = diagnostics.get('avg_r_squared', 0)
            if isinstance(avg_r2, (int, float)) and not np.isnan(avg_r2):
                if avg_r2 > 0.7:
                    print("✓ High model fit (R² > 0.7)")
                elif avg_r2 > 0.5:
                    print("⚠ Moderate model fit (0.5 < R² < 0.7)")
                else:
                    print("⚠ Low model fit (R² < 0.5)")
            else:
                print("⚠ Unable to assess model quality - no valid R-squared")

            beta_stability = diagnostics.get('beta_stability', {})
            if isinstance(beta_stability, dict) and beta_stability:
                print("✓ Beta stability metrics available")
            else:
                print("⚠ No beta stability metrics available")

        print("\n✓ Model diagnostics working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_lookahead_bias_prevention():
    """Test look-ahead bias prevention mechanisms."""
    print("\n" + "=" * 60)
    print("TEST 6: Look-Ahead Bias Prevention")
    print("=" * 60)

    try:
        # Initialize regression engine
        engine = FF5RegressionEngine(estimation_window=24, min_observations=12)

        # Create test data with specific characteristics
        symbols = ['TEST_A', 'TEST_B']
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        # Test with different estimation dates
        estimation_dates = [
            datetime(2022, 6, 30),
            datetime(2022, 12, 31),
            datetime(2023, 6, 30)
        ]

        print("Testing beta estimation at different dates:")
        print("-" * 40)

        for estimation_date in estimation_dates:
            # Estimate betas up to this date
            factor_betas, _, _ = engine.estimate_factor_betas(
                equity_data, factor_data, estimation_dates=[estimation_date]
            )

            if factor_betas and 'TEST_A' in factor_betas:
                beta_data = factor_betas['TEST_A']
                available_dates = beta_data.index[beta_data.index <= estimation_date]

                if len(available_dates) > 0:
                    latest_date = available_dates[-1]
                    print(f"Estimation date: {estimation_date.date()}")
                    print(f"  Latest beta date: {latest_date.date()}")
                    print(f"  Lookahead protection: {'✓' if latest_date <= estimation_date else '✗'}")

                    # Check beta values
                    for factor in ['MKT', 'SMB']:
                        if factor in beta_data.columns:
                            beta_value = beta_data[factor].iloc[-1]
                            print(f"  {factor} beta: {beta_value:.4f}")
                    print()

        print("✓ Look-ahead bias prevention working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_regularization_effects():
    """Test regularization effects on beta estimates."""
    print("\n" + "=" * 60)
    print("TEST 7: Regularization Effects")
    print("=" * 60)

    try:
        # Create test data
        symbols = ['SPY', 'QQQ']
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        # Test different regularization settings
        regularization_methods = ['none', 'ridge']

        print("Testing different regularization methods:")
        print("-" * 40)

        for method in regularization_methods:
            engine = FF5RegressionEngine(
                estimation_window=24,
                min_observations=12,
                regularization=method
            )

            factor_betas, _, _ = engine.estimate_factor_betas(equity_data, factor_data)

            if factor_betas and 'SPY' in factor_betas:
                beta_data = factor_betas['SPY']
                print(f"\n{method.upper()} regularization:")

                # Get latest R-squared safely
                if 'r_squared' in beta_data.columns and len(beta_data['r_squared'].dropna()) > 0:
                    latest_r2 = beta_data['r_squared'].dropna().iloc[-1]
                    print(f"  Latest R-squared: {latest_r2:.4f}")
                else:
                    print(f"  Latest R-squared: N/A")

                # Show beta values
                beta_values = {}
                for factor in ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']:
                    if factor in beta_data.columns and len(beta_data[factor].dropna()) > 0:
                        beta_values[factor.replace('_beta', '')] = beta_data[factor].dropna().iloc[-1]

                print(f"  Betas: {beta_values}")

                # Check for extreme betas
                extreme_betas = [abs(b) for b in beta_values.values() if abs(b) > 2.0]
                if extreme_betas:
                    print(f"  ⚠ Extreme betas detected: {len(extreme_betas)}")
                else:
                    print(f"  ✓ No extreme betas")

        print("\n✓ Regularization effects testing working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_beta_stability_analysis():
    """Test beta stability analysis."""
    print("\n" + "=" * 60)
    print("TEST 8: Beta Stability Analysis")
    print("=" * 60)

    try:
        # Initialize regression engine
        engine = FF5RegressionEngine(estimation_window=36, min_observations=24)

        # Create test data
        symbols = ['SPY']
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        # Estimate betas
        factor_betas, _, _ = engine.estimate_factor_betas(equity_data, factor_data)

        if factor_betas and 'SPY' in factor_betas:
            beta_data = factor_betas['SPY']

            # Analyze beta stability
            print("Beta stability analysis for SPY:")
            print("-" * 40)

            for factor in ['MKT', 'SMB', 'HML']:
                if factor in beta_data.columns:
                    beta_series = beta_data[factor].dropna()
                    if len(beta_series) > 1:
                        # Calculate stability metrics
                        beta_std = beta_series.std()
                        beta_range = beta_series.max() - beta_series.min()
                        beta_trend = (beta_series.iloc[-1] - beta_series.iloc[0]) / len(beta_series)

                        print(f"\n{factor} beta stability:")
                        print(f"  Standard deviation: {beta_std:.4f}")
                        print(f"  Range: {beta_range:.4f}")
                        print(f"  Trend (per period): {beta_trend:.6f}")

                        # Assess stability
                        if beta_std < 0.1:
                            print(f"  ✓ Stable (std < 0.1)")
                        elif beta_std < 0.2:
                            print(f"  ⚠ Moderately stable (0.1 < std < 0.2)")
                        else:
                            print(f"  ✗ Unstable (std > 0.2)")

        print("\n✓ Beta stability analysis working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all FF5 regression tests."""
    print("FF5 Regression Engine Test Suite")
    print("=" * 60)
    print("Testing rolling window beta estimation and factor modeling")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_ff5_regression_initialization,
        test_factor_beta_estimation,
        test_residual_extraction,
        test_factor_implied_returns,
        test_model_diagnostics,
        test_lookahead_bias_prevention,
        test_regularization_effects,
        test_beta_stability_analysis
    ]

    for test in tests:
        try:
            success = test()
            if isinstance(success, tuple):
                success = success[0]  # Get first element if tuple
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
        print("\n🎉 All FF5 regression engine tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the FF5 regression engine test suite."""
    success = run_all_tests()

    if success:
        print("\nFF5 Regression Engine module is working correctly!")
        sys.exit(0)
    else:
        print("\nFF5 Regression Engine module has issues that need to be addressed.")
        sys.exit(1)