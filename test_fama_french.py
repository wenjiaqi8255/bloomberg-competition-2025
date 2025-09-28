#!/usr/bin/env python3
"""
Test script specifically for Fama/French 5-factor strategy.
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

from trading_system.strategies.fama_french_5 import FamaFrench5Strategy
from trading_system.config.config_loader import ConfigLoader
from trading_system.strategy_runner import StrategyRunner

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_fama_french_strategy():
    """Test Fama/French 5-factor strategy specifically."""
    print("Testing Fama/French 5-factor strategy...")
    logger = logging.getLogger(__name__)

    try:
        # Test strategy initialization with different parameters
        strategies_to_test = [
            {"name": "FF5_Conservative", "top_n_assets": 3, "min_factor_score": 0.15},
            {"name": "FF5_Moderate", "top_n_assets": 5, "min_factor_score": 0.1},
            {"name": "FF5_Aggressive", "top_n_assets": 7, "min_factor_score": 0.05}
        ]

        for strategy_config in strategies_to_test:
            print(f"\n  Testing {strategy_config['name']} configuration:")

            strategy = FamaFrench5Strategy(
                name=strategy_config['name'],
                lookback_days=126,  # 6 months for testing
                top_n_assets=strategy_config['top_n_assets'],
                min_factor_score=strategy_config['min_factor_score']
            )

            info = strategy.get_strategy_info()
            print(f"    ✓ Strategy created: {info['name']}")
            print(f"    ✓ Factors: {info['factors']}")
            print(f"    ✓ Top assets: {info['top_n_assets']}")
            print(f"    ✓ Min score: {info['min_factor_score']}")

        return True

    except Exception as e:
        logger.error(f"Fama/French strategy test failed: {e}")
        return False

def test_factor_calculations():
    """Test factor calculation methods."""
    print("\nTesting factor calculation methods...")
    logger = logging.getLogger(__name__)

    try:
        strategy = FamaFrench5Strategy(name="FactorTest", lookback_days=126)

        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
        dates = dates[dates.dayofweek < 5]  # Business days only

        # Test with different market scenarios
        test_scenarios = [
            {"name": "Bull Market", "trend": 0.001, "volatility": 0.01},
            {"name": "Bear Market", "trend": -0.001, "volatility": 0.02},
            {"name": "Sideways", "trend": 0.0, "volatility": 0.015}
        ]

        for scenario in test_scenarios:
            print(f"\n  Testing {scenario['name']} scenario:")

            # Generate price data
            prices = 100 + np.random.normal(0, scenario['volatility'], len(dates)).cumsum()
            # Add trend
            for i in range(1, len(prices)):
                prices[i] = prices[i-1] * (1 + scenario['trend']) + (prices[i] - prices[i-1])

            test_data = pd.DataFrame({
                'Open': prices * 0.998,
                'High': prices * 1.015,
                'Low': prices * 0.985,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)

            # Test factor calculations
            if len(test_data) >= 126:
                test_date = test_data.index[-1]
                factor_score = strategy._calculate_factor_score(test_data, test_date)

                if factor_score is not None:
                    print(f"    ✓ Factor score calculated: {factor_score:.4f}")
                else:
                    print(f"    ⚠ Factor score not available (insufficient data)")

        return True

    except Exception as e:
        logger.error(f"Factor calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_runner_integration():
    """Test integration with strategy runner."""
    print("\nTesting strategy runner integration...")
    logger = logging.getLogger(__name__)

    try:
        # Test with Fama/French configuration
        config_loader = ConfigLoader('/Users/wenjiaqi/Downloads/bloomberg-competition/configs/fama_french_config.yaml')
        config = config_loader.load_config()

        print(f"  ✓ Config loaded: {config.get('strategy', {}).get('name', 'Unknown')}")

        # Test strategy runner can be created
        runner = StrategyRunner('/Users/wenjiaqi/Downloads/bloomberg-competition/configs/fama_french_config.yaml')
        runner.initialize()

        print(f"  ✓ Strategy runner initialized for {runner.strategy.get_name()}")
        print(f"  ✓ Strategy type: {type(runner.strategy).__name__}")

        return True

    except Exception as e:
        logger.error(f"Strategy runner integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation with mock data."""
    print("\nTesting signal generation...")
    logger = logging.getLogger(__name__)

    try:
        strategy = FamaFrench5Strategy(
            name="SignalTest",
            lookback_days=63,  # 3 months for testing
            top_n_assets=3,
            min_factor_score=0.0  # Lower threshold for testing
        )

        # Create test data for multiple assets
        test_symbols = ['SPY', 'QQQ', 'IWM']
        test_data = {}

        for symbol in test_symbols:
            dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
            dates = dates[dates.dayofweek < 5]

            # Generate different patterns for each asset
            base_price = 100 + hash(symbol) % 50  # Different starting prices
            prices = base_price + np.random.normal(0, 0.01, len(dates)).cumsum()

            test_data[symbol] = pd.DataFrame({
                'Open': prices * 0.998,
                'High': prices * 1.015,
                'Low': prices * 0.985,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)

        # Test signal generation
        signals = strategy.generate_signals(
            price_data=test_data,
            start_date=datetime(2024, 4, 1),
            end_date=datetime(2024, 6, 30)
        )

        print(f"  ✓ Generated signals for {len(signals)} dates")
        print(f"  ✓ Signal shape: {signals.shape}")

        if not signals.empty:
            # Count non-zero signals
            non_zero_signals = (signals != 0).sum().sum()
            print(f"  ✓ Non-zero signals: {non_zero_signals}")

            # Show sample signals
            print(f"  ✓ Sample signals:")
            for date in signals.index[:2]:  # Show first 2 dates
                active_signals = signals.loc[date][signals.loc[date] > 0]
                if len(active_signals) > 0:
                    print(f"    {date.date()}: {dict(active_signals)}")

        return True

    except Exception as e:
        logger.error(f"Signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Fama/French 5-factor tests."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Fama/French 5-factor strategy")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip integration tests")
    args = parser.parse_args()

    print("=" * 60)
    print("BLOOMBERG COMPETITION - FAMA/FRENCH 5-FACTOR STRATEGY TEST")
    print("=" * 60)

    tests = [
        ("Fama/French Strategy Initialization", test_fama_french_strategy),
        ("Factor Calculations", test_factor_calculations),
        ("Signal Generation", test_signal_generation),
    ]

    if not args.skip_integration:
        tests.append(("Strategy Runner Integration", test_strategy_runner_integration))

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
    print("FAMA/FRENCH 5-FACTOR TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All Fama/French 5-factor tests passed! Strategy is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())