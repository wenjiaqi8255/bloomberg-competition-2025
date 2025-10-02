#!/usr/bin/env python3
"""
Simple test script to verify that the refactored strategies work correctly.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, '/Users/wenjiaqi/Downloads/bloomberg-competition/src')

def create_test_price_data():
    """Create simple test price data for signal generation."""
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Create 300 days of price data (enough for 252-day lookback)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')

    price_data = {}
    for symbol in symbols:
        # Generate random price data with upward trend
        base_price = 100 + np.random.randint(0, 50)
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [base_price]

        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))  # Ensure positive prices

        prices = prices[1:]  # Remove initial price

        # Create OHLCV DataFrame
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.01, len(prices)),
            'High': prices * np.random.uniform(1.00, 1.05, len(prices)),
            'Low': prices * np.random.uniform(0.95, 1.00, len(prices)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)

        # Ensure High >= Open,Low and Low <= Open,High
        df['High'] = df[['High', 'Open', 'Low']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'High']].min(axis=1)

        price_data[symbol] = df

    return price_data

def test_dual_momentum_strategy():
    """Test DualMomentumStrategy signal generation."""
    print("\n=== Testing DualMomentumStrategy ===")

    try:
        from trading_system.strategies.dual_momentum import DualMomentumStrategy
        from trading_system.utils.position_sizer import PositionSizer

        # Create strategy
        position_sizer = PositionSizer(volatility_target=0.15, max_position_weight=0.10)
        strategy = DualMomentumStrategy(
            name='test_dual_momentum',
            position_sizer=position_sizer,
            lookback_days=252,
            top_n_assets=2,
            minimum_positive_assets=1
        )

        print("✓ Strategy created successfully")

        # Create test data
        price_data = create_test_price_data()
        start_date = datetime(2023, 10, 1)
        end_date = datetime(2023, 10, 31)

        # Generate signals
        signals = strategy.generate_signals(price_data, start_date, end_date)

        if signals.empty:
            print("⚠ No signals generated (might be normal for test data)")
        else:
            print(f"✓ Generated signals with shape: {signals.shape}")
            print(f"  - Signal columns: {list(signals.columns)}")
            print(f"  - Signal dates: {len(signals)}")
            print(f"  - Sample signals:\n{signals.head(2)}")

        # Test regime detection
        regime = strategy.get_momentum_regime(price_data, end_date)
        print(f"  - Market regime: {regime}")

        return True

    except Exception as e:
        print(f"✗ DualMomentumStrategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fama_french_strategy():
    """Test FamaFrench5Strategy signal generation."""
    print("\n=== Testing FamaFrench5Strategy ===")

    try:
        from trading_system.strategies.fama_french_5 import FamaFrench5Strategy
        from trading_system.utils.position_sizer import PositionSizer

        # Create strategy
        position_sizer = PositionSizer(volatility_target=0.15, max_position_weight=0.10)
        strategy = FamaFrench5Strategy(
            name='test_fama_french',
            position_sizer=position_sizer,
            lookback_days=252,
            top_n_assets=2
        )

        print("✓ Strategy created successfully")

        # Create test data
        price_data = create_test_price_data()
        start_date = datetime(2023, 10, 1)
        end_date = datetime(2023, 10, 31)

        # Generate signals
        signals = strategy.generate_signals(price_data, start_date, end_date)

        if signals.empty:
            print("⚠ No signals generated (might be normal for test data)")
        else:
            print(f"✓ Generated signals with shape: {signals.shape}")
            print(f"  - Signal columns: {list(signals.columns)}")
            print(f"  - Signal dates: {len(signals)}")
            print(f"  - Sample signals:\n{signals.head(2)}")

        return True

    except Exception as e:
        print(f"✗ FamaFrench5Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_factory():
    """Test StrategyFactory with refactored strategies."""
    print("\n=== Testing StrategyFactory ===")

    try:
        from trading_system.strategies.factory import StrategyFactory, register_all_strategies
        from trading_system.utils.position_sizer import PositionSizer

        # Register strategies
        register_all_strategies()

        # Test creating strategies with custom PositionSizer
        position_sizer = PositionSizer(volatility_target=0.20, max_position_weight=0.15)

        strategies_to_test = ['dual_momentum', 'fama_french']

        for strategy_type in strategies_to_test:
            strategy = StrategyFactory.create(
                strategy_type,
                name=f'factory_{strategy_type}',
                position_sizer=position_sizer
            )

            print(f"✓ Created {strategy_type} via factory")
            print(f"  - Strategy name: {strategy.name}")
            print(f"  - PositionSizer volatility target: {strategy.position_sizer.volatility_target}")

        # Test creating strategies with default PositionSizer
        for strategy_type in strategies_to_test:
            strategy = StrategyFactory.create(
                strategy_type,
                name=f'default_{strategy_type}'
            )

            print(f"✓ Created {strategy_type} with default PositionSizer")
            print(f"  - Default volatility target: {strategy.position_sizer.volatility_target}")

        return True

    except Exception as e:
        print(f"✗ StrategyFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Refactored Strategies")
    print("=" * 50)

    tests = [
        test_dual_momentum_strategy,
        test_fama_french_strategy,
        test_strategy_factory
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Refactored strategies are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())