"""
Simple test for strategy evaluation features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
import pandas as pd
import numpy as np

print("Testing core evaluation功能...")

# Test PortfolioCalculator methods directly
print("\nTest 1: PortfolioCalculator methods")
try:
    from src.trading_system.strategies.utils.portfolio_calculator import PortfolioCalculator
    
    # Create mock signals
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    symbols = ['SPY', 'QQQ', 'IWM', 'SHY']
    
    # Random signals
    np.random.seed(42)
    signals = pd.DataFrame(
        np.random.random((len(dates), len(symbols))) * 0.5,
        index=dates,
        columns=symbols
    )
    
    # Normalize to sum to 1
    signals = signals.div(signals.sum(axis=1), axis=0)
    
    print(f"  Mock signals shape: {signals.shape}")
    
    # Test signal quality evaluation
    signal_quality = PortfolioCalculator.calculate_signal_quality(signals)
    print(f"  ✓ Signal quality: {signal_quality}")
    assert 'avg_signal_intensity' in signal_quality
    
    # Test position metrics
    position_metrics = PortfolioCalculator.calculate_position_metrics(signals)
    print(f"  ✓ Position metrics: {position_metrics}")
    assert 'avg_number_of_positions' in position_metrics
    
    # Test concentration risk
    concentration = PortfolioCalculator.calculate_concentration_risk(signals)
    print(f"  ✓ Concentration risk: {concentration:.3f}")
    assert 0 <= concentration <= 1
    
    # Test turnover
    turnover = PortfolioCalculator.calculate_turnover(signals)
    print(f"  ✓ Turnover: {turnover:.3f}")
    assert turnover >= 0
    
    print("✓ All PortfolioCalculator methods work!")
    
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test that BaseStrategy has the new methods
print("\nTest 2: BaseStrategy has evaluation methods")
try:
    from src.trading_system.strategies.base_strategy import BaseStrategy
    
    # Check class has the methods
    assert hasattr(BaseStrategy, 'evaluate_signal_quality')
    assert hasattr(BaseStrategy, 'analyze_positions')
    assert hasattr(BaseStrategy, 'calculate_concentration_risk')
    assert hasattr(BaseStrategy, '_evaluate_and_cache_signals')
    
    print("✓ BaseStrategy has all core evaluation methods!")
    
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nCore features are working:")
print("  ✓ PortfolioCalculator methods functional")
print("  ✓ BaseStrategy has evaluation methods")
print("\nYou can now:")
print("  1. Run backtests - signal evaluation happens automatically")
print("  2. View aggregated metrics in StrategyRunner results")
print("  3. Use PortfolioCalculator directly for custom analysis")

