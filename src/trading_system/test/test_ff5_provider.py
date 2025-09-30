"""
Test script for FF5 Data Provider module.

This script tests the FF5 data provider functionality:
- Data fetching from Kenneth French Data Library
- Data validation and cleaning
- Date filtering and alignment
- Factor statistics calculation

Usage:
    python test_ff5_provider.py
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

from trading_system.data.ff5_provider import FF5DataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ff5_data_fetching():
    """Test basic FF5 data fetching functionality."""
    print("=" * 60)
    print("TEST 1: FF5 Data Fetching")
    print("=" * 60)

    try:
        # Initialize FF5 provider with monthly data
        ff5_provider = FF5DataProvider(data_frequency="monthly")
        print(f"✓ Initialized FF5 provider with monthly data")

        # Get factor returns for recent period
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)

        print(f"Fetching FF5 data from {start_date.date()} to {end_date.date()}...")
        factor_data = ff5_provider.get_factor_returns(start_date, end_date)

        print(f"✓ Successfully fetched {len(factor_data)} rows of FF5 data")
        print(f"  Date range: {factor_data.index.min().date()} to {factor_data.index.max().date()}")
        print(f"  Columns: {list(factor_data.columns)}")

        # Display first few rows
        print("\nFirst 5 rows of FF5 data:")
        print(factor_data.head())

        return True, factor_data

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False, None


def test_ff5_data_validation():
    """Test FF5 data validation."""
    print("\n" + "=" * 60)
    print("TEST 2: FF5 Data Validation")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get factor data
        factor_data = ff5_provider.get_factor_returns()

        # Test data validation
        required_columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        missing_columns = [col for col in required_columns if col not in factor_data.columns]

        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False

        print("✓ All required columns present")

        # Check for reasonable value ranges
        for col in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
            values = factor_data[col]
            if ((values < -1.0) | (values > 1.0)).any():
                print(f"✗ Factor {col} has values outside reasonable range")
                return False

        print("✓ All factor values within reasonable ranges")

        # Check risk-free rate
        rf_values = factor_data['RF']
        if ((rf_values < -0.1) | (rf_values > 0.1)).any():
            print("✗ Risk-free rate has values outside reasonable range")
            return False

        print("✓ Risk-free rate within reasonable range")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_statistics():
    """Test FF5 factor statistics calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: FF5 Factor Statistics")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get factor data
        factor_data = ff5_provider.get_factor_returns()

        # Calculate factor statistics
        stats = ff5_provider.get_factor_statistics(factor_data)

        print("Factor Statistics:")
        print("-" * 40)

        for factor, factor_stats in stats.items():
            print(f"\n{factor}:")
            print(f"  Mean: {factor_stats['mean']:.4f}")
            print(f"  Std:  {factor_stats['std']:.4f}")
            print(f"  Min:  {factor_stats['min']:.4f}")
            print(f"  Max:  {factor_stats['max']:.4f}")
            print(f"  Annualized Mean: {factor_stats['annualized_mean']:.4f}")
            print(f"  Annualized Std:  {factor_stats['annualized_std']:.4f}")
            print(f"  Sharpe Ratio: {factor_stats['sharpe_ratio']:.4f}")

        print("✓ Successfully calculated factor statistics")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_correlations():
    """Test FF5 factor correlation calculation."""
    print("\n" + "=" * 60)
    print("TEST 4: FF5 Factor Correlations")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get factor correlations
        correlations = ff5_provider.get_factor_correlations()

        print("Factor Correlation Matrix:")
        print("-" * 40)
        print(correlations.round(4))

        # Check for extreme correlations
        high_correlations = []
        for i, factor1 in enumerate(correlations.columns):
            for j, factor2 in enumerate(correlations.columns):
                if i < j:  # Avoid duplicates
                    corr = correlations.iloc[i, j]
                    if abs(corr) > 0.7:
                        high_correlations.append((factor1, factor2, corr))

        if high_correlations:
            print(f"\n⚠ High correlations found:")
            for factor1, factor2, corr in high_correlations:
                print(f"  {factor1} - {factor2}: {corr:.4f}")
        else:
            print("✓ No extremely high correlations found")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_date_filtering():
    """Test FF5 date filtering functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: FF5 Date Filtering")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Test different date ranges
        test_cases = [
            (datetime(2021, 1, 1), datetime(2021, 12, 31), "2021 data"),
            (datetime(2022, 1, 1), datetime(2022, 12, 31), "2022 data"),
            (datetime(2020, 1, 1), datetime(2023, 12, 31), "Full period"),
        ]

        for start_date, end_date, description in test_cases:
            filtered_data = ff5_provider.get_factor_returns(start_date, end_date)

            print(f"\n{description}:")
            print(f"  Rows: {len(filtered_data)}")
            print(f"  Date range: {filtered_data.index.min().date()} to {filtered_data.index.max().date()}")

            if len(filtered_data) > 0:
                expected_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                print(f"  Expected months: {expected_months}, Actual: {len(filtered_data)}")

        print("✓ Date filtering working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_risk_free_rate():
    """Test risk-free rate extraction."""
    print("\n" + "=" * 60)
    print("TEST 6: Risk-Free Rate Extraction")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get risk-free rate
        rf_rate = ff5_provider.get_risk_free_rate()

        print(f"✓ Successfully extracted risk-free rate series")
        print(f"  Length: {len(rf_rate)}")
        print(f"  Date range: {rf_rate.index.min().date()} to {rf_rate.index.max().date()}")
        print(f"  Latest RF rate: {rf_rate.iloc[-1]:.4f}")
        print(f"  Mean RF rate: {rf_rate.mean():.4f}")
        print(f"  Std RF rate: {rf_rate.std():.4f}")

        # Check for reasonable values
        if (rf_rate < -0.05).any() or (rf_rate > 0.05).any():
            print("⚠ Some risk-free rates outside typical range (-5% to +5%)")
        else:
            print("✓ All risk-free rates within typical range")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_cumulative_returns():
    """Test cumulative returns calculation."""
    print("\n" + "=" * 60)
    print("TEST 7: Cumulative Factor Returns")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get cumulative returns
        cumulative_returns = ff5_provider.get_cumulative_factor_returns()

        print("Cumulative Factor Returns:")
        print("-" * 40)
        print(cumulative_returns.tail())

        # Find best and worst performing factors
        latest_returns = cumulative_returns.iloc[-1]
        best_factor = latest_returns.idxmax()
        worst_factor = latest_returns.idxmin()

        print(f"\nBest performing factor: {best_factor} ({latest_returns[best_factor]:.2%})")
        print(f"Worst performing factor: {worst_factor} ({latest_returns[worst_factor]:.2%})")

        print("✓ Successfully calculated cumulative factor returns")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ff5_provider_info():
    """Test FF5 provider information."""
    print("\n" + "=" * 60)
    print("TEST 8: Provider Information")
    print("=" * 60)

    try:
        # Initialize FF5 provider
        ff5_provider = FF5DataProvider(data_frequency="monthly")

        # Get provider information
        info = ff5_provider.get_data_info()

        print("FF5 Provider Information:")
        print("-" * 40)
        for key, value in info.items():
            print(f"{key}: {value}")

        print("✓ Successfully retrieved provider information")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all FF5 provider tests."""
    print("FF5 Data Provider Test Suite")
    print("=" * 60)
    print("Testing Kenneth French Data Library integration")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_ff5_data_fetching,
        test_ff5_data_validation,
        test_ff5_statistics,
        test_ff5_correlations,
        test_ff5_date_filtering,
        test_ff5_risk_free_rate,
        test_ff5_cumulative_returns,
        test_ff5_provider_info
    ]

    for test in tests:
        try:
            if test == test_ff5_data_fetching:
                success, factor_data = test()
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
        print("\n🎉 All FF5 provider tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the FF5 provider test suite."""
    success = run_all_tests()

    if success:
        print("\nFF5 Data Provider module is working correctly!")
        sys.exit(0)
    else:
        print("\nFF5 Data Provider module has issues that need to be addressed.")
        sys.exit(1)