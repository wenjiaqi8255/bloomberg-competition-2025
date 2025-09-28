#!/usr/bin/env python3
"""
Quick test to verify the feature engineering fix works.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from trading_system.feature_engineering.feature_engine import FeatureEngine

def test_feature_fix():
    """Test that feature engineering works with yfinance data format."""
    print("Testing feature engineering with yfinance data format...")

    # Create sample data with yfinance column names
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.normal(100, 5, len(dates)),
        'High': np.random.normal(102, 5, len(dates)),
        'Low': np.random.normal(98, 5, len(dates)),
        'Close': np.random.normal(100, 5, len(dates)),
        'Volume': np.random.normal(1000000, 100000, len(dates)),
        'Adj Close': np.random.normal(100, 5, len(dates)),
    }, index=dates)

    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")

    # Test feature engineering
    feature_engine = FeatureEngine(
        lookback_periods=[20, 50],
        momentum_periods=[1, 3],
        volatility_windows=[10, 20],
        include_technical=True,
        include_theoretical=True,
        benchmark_symbol='SPY'
    )

    try:
        features = feature_engine.compute_features({'TEST': sample_data})
        print("✅ Feature engineering successful!")

        if 'TEST' in features:
            print(f"✅ Generated {len(features['TEST'].columns)} features for TEST")
            print(f"✅ Feature columns sample: {list(features['TEST'].columns[:10])}")
        else:
            print("⚠ No features generated for TEST")

        return True

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_fix()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")