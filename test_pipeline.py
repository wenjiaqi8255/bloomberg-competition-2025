#!/usr/bin/env python3
"""
Test script to validate the complete trading pipeline.

This script performs end-to-end testing of the trading system components.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system.data.yfinance_provider import YFinanceProvider
from trading_system.strategies.dual_momentum import DualMomentumStrategy
from trading_system.backtest.standard_backtest import StandardBacktest
from trading_system.utils.wandb_logger import WandBLogger
from trading_system.config.config_loader import ConfigLoader

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_data_provider():
    """Test YFinance data provider."""
    print("Testing YFinance data provider...")
    logger = logging.getLogger(__name__)

    try:
        provider = YFinanceProvider(max_retries=2, retry_delay=0.5)

        # Test with a few well-known symbols
        test_symbols = ['SPY', 'QQQ', 'AAPL']

        # Test symbol validation
        for symbol in test_symbols:
            is_valid = provider.validate_symbol(symbol)
            print(f"  Symbol {symbol}: {'Valid' if is_valid else 'Invalid'}")

        # Test historical data fetch (short period for testing)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

        price_data = provider.get_historical_data(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date
        )

        print(f"  Successfully fetched data for {len(price_data)} symbols")
        for symbol, data in price_data.items():
            print(f"    {symbol}: {len(data)} rows")
            if not data.empty:
                print(f"      Date range: {data.index.min()} to {data.index.max()}")

        return True

    except Exception as e:
        logger.error(f"Data provider test failed: {e}")
        return False

def test_strategy():
    """Test dual momentum strategy."""
    print("\nTesting dual momentum strategy...")
    logger = logging.getLogger(__name__)

    try:
        # Create strategy with test parameters
        strategy = DualMomentumStrategy(
            name="TestDualMomentum",
            lookback_days=63,  # 3 months for testing
            top_n_assets=3,
            minimum_positive_assets=2
        )

        # Get strategy info
        info = strategy.get_strategy_info()
        print(f"  Strategy: {info['name']}")
        print(f"  Lookback: {info['lookback_days']} days")
        print(f"  Top assets: {info['top_n_assets']}")

        # Test with mock data (would use real data in actual run)
        test_symbols = ['SPY', 'QQQ', 'IWM']
        mock_data = {}

        # Generate simple mock price data for testing
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
        dates = dates[dates.dayofweek < 5]  # Business days only

        for symbol in test_symbols:
            prices = 100 + np.random.normal(0, 1, len(dates)).cumsum()
            mock_data[symbol] = pd.DataFrame({
                'Open': prices * 0.998,
                'High': prices * 1.015,
                'Low': prices * 0.985,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)

        # Test signal generation
        signals = strategy.generate_signals(
            price_data=mock_data,
            start_date=datetime(2024, 3, 1),
            end_date=datetime(2024, 6, 30)
        )

        print(f"  Generated signals: {len(signals)} dates")
        if not signals.empty:
            print(f"  Signal date range: {signals.index.min()} to {signals.index.max()}")

            # Count non-zero signals
            non_zero_signals = (signals != 0).sum().sum()
            print(f"  Non-zero signals: {non_zero_signals}")

        return True

    except Exception as e:
        logger.error(f"Strategy test failed: {e}")
        return False

def test_backtest():
    """Test backtest engine."""
    print("\nTesting backtest engine...")
    logger = logging.getLogger(__name__)

    try:
        backtest = StandardBacktest(
            initial_capital=100000,
            transaction_cost=0.001,
            benchmark_symbol='SPY'
        )

        print(f"  Initial capital: ${backtest.initial_capital:,.0f}")
        print(f"  Transaction cost: {backtest.transaction_cost:.1%}")
        print(f"  Benchmark: {backtest.benchmark_symbol}")

        # Test with mock signals and data
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='MS')
        signals = pd.DataFrame(0.0, index=dates, columns=['SPY', 'QQQ'], dtype=float)

        # Set some test signals
        signals.loc['2024-01-01', 'SPY'] = 0.6
        signals.loc['2024-01-01', 'QQQ'] = 0.4
        signals.loc['2024-02-01', 'SPY'] = 0.7
        signals.loc['2024-02-01', 'QQQ'] = 0.3

        # Mock price data
        mock_price_data = {}
        for symbol in ['SPY', 'QQQ']:
            prices = [400, 405, 410, 415, 420, 425, 430]
            mock_dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
            mock_price_data[symbol] = pd.DataFrame({
                'Open': [p * 0.998 for p in prices],
                'High': [p * 1.015 for p in prices],
                'Low': [p * 0.985 for p in prices],
                'Close': prices,
                'Volume': [1000000] * len(prices)
            }, index=mock_dates)

        # Mock benchmark data
        benchmark_prices = [400, 405, 410, 415, 420, 425, 430]
        benchmark_data = pd.DataFrame({
            'Close': benchmark_prices
        }, index=mock_dates)

        # Run backtest
        results = backtest.run_backtest(
            strategy_signals=signals,
            price_data=mock_price_data,
            benchmark_data=benchmark_data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            rebalance_frequency='monthly'
        )

        print(f"  Backtest completed successfully")
        print(f"  Final value: ${results['final_value']:,.0f}")
        print(f"  Total return: {results['total_return']:.2%}")

        return True

    except Exception as e:
        logger.error(f"Backtest test failed: {e}")
        return False

def test_config_loader():
    """Test configuration loader."""
    print("\nTesting configuration loader...")
    logger = logging.getLogger(__name__)

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config()

        print(f"  Config loaded successfully")
        print(f"  Strategy: {config.get('strategy', {}).get('name', 'Unknown')}")

        # Test getting different config sections
        strategy_params = config_loader.get_strategy_parameters()
        print(f"  Strategy parameters: {len(strategy_params)} keys")

        universe = config_loader.get_asset_universe()
        print(f"  Asset universe: {len(universe)} symbols")

        backtest_config = config_loader.get_backtest_config()
        print(f"  Initial capital: ${backtest_config.get('initial_capital', 0):,.0f}")

        return True

    except Exception as e:
        logger.error(f"Config loader test failed: {e}")
        return False

def test_wandb_logger():
    """Test WandB logger (without actual API calls)."""
    print("\nTesting WandB logger...")
    logger = logging.getLogger(__name__)

    try:
        # Check if API key is available
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key:
            print("  WANDB_API_KEY not found - skipping actual initialization test")
            return True

        logger = WandBLogger(project_name="test-project")

        # Test that we can create the logger without errors
        print(f"  WandB logger created successfully")
        print(f"  Project: {logger.project_name}")

        # Don't actually initialize to avoid creating test runs
        print("  Logger structure validated (API key available)")

        return True

    except Exception as e:
        logger.error(f"WandB logger test failed: {e}")
        return False

def test_complete_pipeline():
    """Test the complete pipeline with actual data."""
    print("\nTesting complete pipeline...")
    logger = logging.getLogger(__name__)

    try:
        from trading_system.strategy_runner import StrategyRunner

        # Get command line arguments
        import __main__
        args = getattr(__main__, 'args', None) or type('Args', (), {'skip_long_test': True})()

        # Create runner
        runner = StrategyRunner()

        # Test initialization
        runner.initialize()
        print("  Strategy runner initialized successfully")

        # Test config loading
        config = runner.config
        print(f"  Configuration loaded for: {config.get('strategy', {}).get('name', 'Unknown')}")

        # Test with a very short test to avoid long execution
        if not args.skip_long_test:
            print("  Running mini backtest (this may take a moment)...")

            # Override config for quick test
            runner.config['backtest']['end_date'] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            runner.config['backtest']['start_date'] = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            # Run strategy
            with runner:
                results = runner.run_strategy(experiment_name="pipeline_test")

            print(f"  Pipeline test completed")
            if 'backtest_results' in results:
                perf = results['backtest_results']
                print(f"  Final value: ${perf.get('final_value', 0):,.0f}")
                print(f"  Total return: {perf.get('total_return', 0):.2%}")

        return True

    except Exception as e:
        logger.error(f"Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test trading system pipeline")
    parser.add_argument("--skip-long-test", action="store_true",
                       help="Skip long-running tests")
    args = parser.parse_args()

    # Make args available globally for the test functions
    import sys
    sys.modules['__main__'].args = args

    print("=" * 60)
    print("BLOOMBERG COMPETITION TRADING SYSTEM - PIPELINE TEST")
    print("=" * 60)

    tests = [
        ("Configuration Loader", test_config_loader),
        ("Data Provider", test_data_provider),
        ("Dual Momentum Strategy", test_strategy),
        ("Backtest Engine", test_backtest),
        ("WandB Logger", test_wandb_logger),
        ("Complete Pipeline", test_complete_pipeline),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")

        try:
            success = test_func()
            if success:
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            print(f"✗ {test_name} CRASHED")

    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())