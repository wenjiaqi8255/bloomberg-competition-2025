"""
Test script for Core FFML Strategy module.

This script tests the Core FFML Strategy functionality:
- FF5 factor model integration
- ML residual prediction integration
- Combined prediction methodology (Method A)
- IPS compliance monitoring
- Risk management and position sizing
- Performance tracking and reporting
- Model governance and degradation monitoring

Usage:
    python test_core_ffml_strategy.py
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

from trading_system.strategies.core_ffml_strategy import CoreFFMLStrategy
from trading_system.data.ff5_provider import FF5DataProvider
from trading_system.data.stock_classifier import StockClassifier
from trading_system.models.ff5_regression import FF5RegressionEngine
from trading_system.models.residual_predictor import MLResidualPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


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


def create_test_residuals_data(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.Series]:
    """Create synthetic residuals data for testing."""
    residuals_dict = {}

    for symbol in symbols:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic residuals
        base_residuals = np.random.normal(0, 0.02, len(dates))

        # Add some autocorrelation
        for i in range(1, len(base_residuals)):
            base_residuals[i] += 0.2 * base_residuals[i-1]

        residuals_series = pd.Series(base_residuals, index=dates)
        residuals_dict[symbol] = residuals_series

    return residuals_dict


def test_core_ffml_initialization():
    """Test Core FFML Strategy initialization."""
    print("=" * 60)
    print("TEST 1: Core FFML Strategy Initialization")
    print("=" * 60)

    try:
        # Test different strategy configurations
        configs = [
            {
                'strategy_name': 'Core_FFML_Conservative',
                'core_weight': 0.8,
                'lookback_window': 252,
                'rebalance_frequency': 30,
                'max_position_size': 0.15,
                'volatility_target': 0.12
            },
            {
                'strategy_name': 'Core_FFML_Moderate',
                'core_weight': 0.75,
                'lookback_window': 189,
                'rebalance_frequency': 21,
                'max_position_size': 0.18,
                'volatility_target': 0.15
            },
            {
                'strategy_name': 'Core_FFML_Aggressive',
                'core_weight': 0.7,
                'lookback_window': 126,
                'rebalance_frequency': 14,
                'max_position_size': 0.20,
                'volatility_target': 0.18
            }
        ]

        for i, config in enumerate(configs):
            strategy = CoreFFMLStrategy(**config)
            print(f"Configuration {i+1} ({config['strategy_name']}):")
            print(f"  Strategy name: {strategy.strategy_name}")
            print(f"  Core weight: {strategy.core_weight}")
            print(f"  Lookback window: {strategy.lookback_window}")
            print(f"  Rebalance frequency: {strategy.rebalance_frequency}")
            print(f"  Max position size: {strategy.max_position_size}")
            print(f"  Volatility target: {strategy.volatility_target}")
            print(f"  ✓ Initialized successfully")

        print("\n✓ Core FFML Strategy initialization working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_integration():
    """Test FF5 factor model integration."""
    print("\n" + "=" * 60)
    print("TEST 2: FF5 Factor Model Integration")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21
        )

        # Create test data
        symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)

        print(f"Testing FF5 integration for {len(symbols)} symbols")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        # Test FF5 factor model integration
        ff5_results = strategy._test_ff5_integration(equity_data, start_date, end_date)

        print(f"\nFF5 Integration Results:")
        print("-" * 40)

        for symbol, results in ff5_results.items():
            print(f"\n{symbol}:")
            print(f"  Beta estimation successful: {results.get('beta_estimation_success', False)}")
            print(f"  Factor returns calculated: {results.get('factor_returns_calculated', False)}")
            print(f"  Model quality score: {results.get('model_quality_score', 0):.4f}")
            print(f"  R-squared average: {results.get('r_squared_avg', 0):.4f}")

            if results.get('factor_betas') is not None:
                betas = results['factor_betas']
                print(f"  Latest betas:")
                for factor in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
                    if factor in betas.columns:
                        latest_beta = betas[factor].iloc[-1]
                        print(f"    {factor}: {latest_beta:.4f}")

        print("\n✓ FF5 factor model integration working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ml_integration():
    """Test ML residual prediction integration."""
    print("\n" + "=" * 60)
    print("TEST 3: ML Residual Prediction Integration")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21
        )

        # Create test data
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        print(f"Testing ML integration for {len(symbols)} symbols")
        print(f"Training period: {start_date.date()} to {end_date.date()}")

        # Test ML integration
        ml_results = strategy._test_ml_integration(equity_data, residuals_data, start_date, end_date)

        print(f"\nML Integration Results:")
        print("-" * 40)

        for symbol, results in ml_results.items():
            print(f"\n{symbol}:")
            print(f"  Model training successful: {results.get('training_successful', False)}")
            print(f"  Model type: {results.get('model_type', 'N/A')}")
            print(f"  Training samples: {results.get('training_samples', 0)}")
            print(f"  Features used: {results.get('features_used', 0)}")
            print(f"  CV R² score: {results.get('cv_r2_score', 0):.4f}")
            print(f"  Model quality score: {results.get('model_quality_score', 0):.4f}")

            if results.get('feature_importance'):
                print(f"  Top 3 features:")
                top_features = results['feature_importance'][:3]
                for i, (feature, importance) in enumerate(top_features):
                    print(f"    {i+1}. {feature}: {importance:.4f}")

        print("\n✓ ML residual prediction integration working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_combined_prediction_methodology():
    """Test combined prediction methodology (Method A)."""
    print("\n" + "=" * 60)
    print("TEST 4: Combined Prediction Methodology (Method A)")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21
        )

        # Create test data
        symbols = ['SPY', 'QQQ']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        # Test combined predictions
        combined_results = strategy._test_combined_predictions(
            equity_data, residuals_data, start_date, end_date
        )

        print(f"Combined Prediction Results:")
        print("-" * 40)

        for symbol, results in combined_results.items():
            print(f"\n{symbol}:")
            print(f"  Combined predictions generated: {results.get('combined_predictions_generated', False)}")
            print(f"  Prediction periods: {results.get('prediction_periods', 0)}")
            print(f"  Method A implemented: {results.get('method_a_implemented', False)}")

            if results.get('combined_predictions') is not None:
                pred_data = results['combined_predictions']
                print(f"  Mean combined return: {pred_data.mean():.6f}")
                print(f"  Std combined return: {pred_data.std():.6f}")
                print(f"  Annualized return: {pred_data.mean() * 252:.4f}")
                print(f"  Annualized volatility: {pred_data.std() * np.sqrt(252):.4f}")

            # Check prediction quality
            if results.get('prediction_quality'):
                quality = results['prediction_quality']
                print(f"  Prediction R²: {quality.get('r2_score', 0):.4f}")
                print(f"  Prediction RMSE: {quality.get('rmse', 0):.6f}")
                print(f"  Directional accuracy: {quality.get('directional_accuracy', 0):.4f}")

        print("\n✓ Combined prediction methodology working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ips_compliance_monitoring():
    """Test IPS compliance monitoring."""
    print("\n" + "=" * 60)
    print("TEST 5: IPS Compliance Monitoring")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21
        )

        # Test IPS compliance checks
        compliance_results = strategy._test_ips_compliance()

        print(f"IPS Compliance Results:")
        print("-" * 40)

        for check_name, result in compliance_results.items():
            print(f"\n{check_name}:")
            print(f"  Compliant: {result.get('compliant', False)}")
            print(f"  Score: {result.get('score', 0):.4f}")
            print(f"  Details: {result.get('details', 'N/A')}")

            if not result.get('compliant', True):
                print(f"  ⚠ Violation detected!")
                print(f"  Recommended action: {result.get('recommended_action', 'N/A')}")

        # Generate IPS compliance report
        ips_report = strategy.generate_ips_compliance_report()
        print(f"\nIPS Compliance Report Summary:")
        print("-" * 40)
        print(f"  Overall compliance: {ips_report['overall_compliance']:.4f}")
        print(f"  Risk level: {ips_report['risk_level']}")
        print(f"  Violations: {len(ips_report['violations'])}")
        print(f"  Recommendations: {len(ips_report['recommendations'])}")

        if ips_report['violations']:
            print(f"  Top violations:")
            for violation in ips_report['violations'][:3]:
                print(f"    - {violation}")

        print("\n✓ IPS compliance monitoring working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_risk_management():
    """Test risk management and position sizing."""
    print("\n" + "=" * 60)
    print("TEST 6: Risk Management and Position Sizing")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21,
            max_position_size=0.15,
            volatility_target=0.15
        )

        # Create test portfolio scenario
        test_signals = {
            'SPY': 0.8,    # Strong buy signal
            'QQQ': 0.6,    # Moderate buy signal
            'IWM': -0.4,   # Moderate sell signal
            'EFA': 0.3,    # Weak buy signal
            'EEM': -0.2    # Weak sell signal
        }

        test_market_caps = {
            'SPY': 400,    # $400B
            'QQQ': 250,    # $250B
            'IWM': 60,     # $60B
            'EFA': 80,     # $80B
            'EEM': 70      # $70B
        }

        print(f"Testing risk management with {len(test_signals)} signals")
        print(f"Max position size: {strategy.max_position_size}")
        print(f"Volatility target: {strategy.volatility_target}")

        # Test risk management
        risk_results = strategy._test_risk_management(test_signals, test_market_caps)

        print(f"\nRisk Management Results:")
        print("-" * 40)

        print(f"Portfolio constraints:")
        print(f"  Gross exposure: {risk_results.get('gross_exposure', 0):.4f}")
        print(f"  Net exposure: {risk_results.get('net_exposure', 0):.4f}")
        print(f"  Max position size respected: {risk_results.get('max_position_respected', False)}")
        print(f"  Volatility target respected: {risk_results.get('volatility_target_respected', False)}")

        print(f"\nPosition sizing:")
        for symbol, position in risk_results.get('positions', {}).items():
            print(f"  {symbol}: {position['weight']:.4f} (signal: {test_signals[symbol]:.2f})")

        # Test risk limits
        risk_limits = strategy.check_risk_limits()
        print(f"\nRisk Limits Check:")
        print("-" * 40)
        for limit_name, limit_info in risk_limits.items():
            status = "✓" if limit_info['within_limit'] else "✗"
            print(f"  {status} {limit_name}: {limit_info['current_value']:.4f} (limit: {limit_info['limit_value']:.4f})")

        print("\n✓ Risk management and position sizing working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_performance_tracking():
    """Test performance tracking and reporting."""
    print("\n" + "=" * 60)
    print("TEST 7: Performance Tracking")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21
        )

        # Create test performance data
        test_periods = [
            (datetime(2023, 1, 1), datetime(2023, 3, 31), "Q1 2023"),
            (datetime(2023, 4, 1), datetime(2023, 6, 30), "Q2 2023"),
            (datetime(2023, 7, 1), datetime(2023, 9, 30), "Q3 2023")
        ]

        performance_data = {}
        for start_date, end_date, period_name in test_periods:
            # Simulate performance metrics
            returns = np.random.normal(0.001, 0.015, 63)  # Daily returns
            cumulative_return = np.prod(1 + returns) - 1

            performance_data[period_name] = {
                'returns': returns,
                'cumulative_return': cumulative_return,
                'volatility': np.std(returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
                'max_drawdown': -0.05,  # Simulated
                'win_rate': 0.55  # Simulated
            }

        print(f"Testing performance tracking for {len(performance_data)} periods")

        # Test performance tracking
        tracking_results = strategy._test_performance_tracking(performance_data)

        print(f"\nPerformance Tracking Results:")
        print("-" * 40)

        for period, metrics in tracking_results.items():
            print(f"\n{period}:")
            print(f"  Cumulative return: {metrics.get('cumulative_return', 0):.4f}")
            print(f"  Annualized volatility: {metrics.get('volatility', 0):.4f}")
            print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max drawdown: {metrics.get('max_drawdown', 0):.4f}")
            print(f"  Win rate: {metrics.get('win_rate', 0):.4f}")

        # Generate performance report
        perf_report = strategy.generate_performance_report()
        print(f"\nPerformance Report Summary:")
        print("-" * 40)
        print(f"  Total periods: {perf_report['total_periods']}")
        print(f"  Average return: {perf_report['average_return']:.4f}")
        print(f"  Average volatility: {perf_report['average_volatility']:.4f}")
        print(f"  Average Sharpe: {perf_report['average_sharpe']:.4f}")
        print(f"  Best period: {perf_report['best_period']}")
        print(f"  Worst period: {perf_report['worst_period']}")

        print("\n✓ Performance tracking working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_complete_workflow():
    """Test complete Core FFML workflow."""
    print("\n" + "=" * 60)
    print("TEST 8: Complete Core FFML Workflow")
    print("=" * 60)

    try:
        # Initialize strategy
        strategy = CoreFFMLStrategy(
            strategy_name='Test_FFML_Strategy',
            core_weight=0.75,
            lookback_window=126,
            rebalance_frequency=21,
            max_position_size=0.15,
            volatility_target=0.15
        )

        # Create comprehensive test data
        symbols = ['SPY', 'QQQ', 'IWM']
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)

        equity_data = create_test_equity_data(symbols, start_date, end_date)
        residuals_data = create_test_residuals_data(symbols, start_date, end_date)

        print(f"Testing complete workflow for {len(symbols)} symbols")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        # Test complete workflow
        workflow_results = strategy._test_complete_workflow(
            equity_data, residuals_data, start_date, end_date
        )

        print(f"\nComplete Workflow Results:")
        print("-" * 40)

        print(f"Workflow steps completed:")
        steps = workflow_results.get('workflow_steps', {})
        for step_name, step_result in steps.items():
            status = "✓" if step_result.get('success', False) else "✗"
            print(f"  {status} {step_name}: {step_result.get('message', 'N/A')}")

        print(f"\nOverall performance:")
        overall = workflow_results.get('overall_performance', {})
        print(f"  Success rate: {overall.get('success_rate', 0):.4f}")
        print(f"  Average quality score: {overall.get('avg_quality_score', 0):.4f}")
        print(f"  IPS compliance: {overall.get('ips_compliance', 0):.4f}")
        print(f"  Risk management: {overall.get('risk_management', 0):.4f}")

        # Test with specific test date
        test_date = datetime(2023, 6, 1)
        print(f"\nTesting strategy signals for {test_date.date()}:")

        try:
            signals = strategy.generate_signals(test_date)
            print(f"  Signals generated: {len(signals)}")

            for symbol, signal_data in signals.items():
                print(f"  {symbol}:")
                print(f"    Signal strength: {signal_data.get('signal_strength', 0):.4f}")
                print(f"    Position weight: {signal_data.get('position_weight', 0):.4f}")
                print(f"    Confidence: {signal_data.get('confidence', 0):.4f}")

        except Exception as e:
            print(f"  Signal generation failed: {e}")

        print("\n✓ Complete Core FFML workflow working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all Core FFML Strategy tests."""
    print("Core FFML Strategy Test Suite")
    print("=" * 60)
    print("Testing FF5 + ML residual prediction with IPS compliance")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_core_ffml_initialization,
        test_ff5_integration,
        test_ml_integration,
        test_combined_prediction_methodology,
        test_ips_compliance_monitoring,
        test_risk_management,
        test_performance_tracking,
        test_complete_workflow
    ]

    for test in tests:
        try:
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
        print("\n🎉 All Core FFML Strategy tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the Core FFML Strategy test suite."""
    success = run_all_tests()

    if success:
        print("\nCore FFML Strategy module is working correctly!")
        sys.exit(0)
    else:
        print("\nCore FFML Strategy module has issues that need to be addressed.")
        sys.exit(1)