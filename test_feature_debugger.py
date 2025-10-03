#!/usr/bin/env python3
"""
Test script for Feature Engineering Debugger

This script demonstrates how to use the FeatureEngineeringDebugger
to isolate and test the feature engineering pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_system.feature_engineering.debugger import FeatureEngineeringDebugger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run feature engineering debugging."""

    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Use a subset for faster debugging
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)

    print(f"=== Feature Engineering Debugger Test ===")
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()

    # Initialize debugger
    debugger = FeatureEngineeringDebugger()

    # Run debugging
    debug_report = debugger.debug_feature_pipeline(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        save_results=True
    )

    # Display summary
    print("\n=== DEBUGGING SUMMARY ===")

    # Data loading summary
    data_quality = debug_report.get('data_loading', {}).get('data_quality', {})
    price_completeness = data_quality.get('price_data_completeness', 0)
    print(f"Price data completeness: {price_completeness:.1f}%")

    # Feature computation summary
    feature_results = debug_report.get('feature_computation', {})
    final_features = feature_results.get('final_features', {})
    if final_features:
        shape = final_features.get('shape', (0, 0))
        nan_pct = final_features.get('nan_percentage', 100)
        print(f"Final features: {shape[1]} features, {shape[0]} rows")
        print(f"Overall NaN percentage: {nan_pct:.1f}%")

    # Quality score
    quality_results = debug_report.get('data_quality', {})
    quality_score = quality_results.get('overall_quality_score', 0)
    print(f"Overall data quality score: {quality_score:.1f}/100")

    # Recommendations
    recommendations = debug_report.get('recommendations', [])
    print(f"\nFound {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['title']}")
        print(f"   {rec['description']}")

    # NaN analysis
    nan_analysis = debug_report.get('nan_analysis', {})
    problematic_features = nan_analysis.get('problematic_features', [])
    if problematic_features:
        print(f"\nTop 5 problematic features:")
        for feature in problematic_features[:5]:
            print(f"  - {feature['name']}: {feature['nan_percentage']:.1f}% NaN ({feature['recommendation']})")

    # Test NaN handling strategies
    if 'final_features' in feature_results and feature_results['final_features']:
        print(f"\n=== TESTING NaN HANDLING STRATEGIES ===")

        # Create sample data for testing
        # We'll use the computed features to test NaN handling
        import pandas as pd
        import numpy as np

        # Get the step-by-step analysis to create sample data
        step_analysis = feature_results.get('step_by_step_analysis', {})
        sample_data = None

        # Re-create some sample data for testing
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)  # For reproducible results

        sample_data = pd.DataFrame({
            'Close': np.cumprod(1 + np.random.randn(len(dates)) * 0.02) * 100,
            'High': np.cumprod(1 + np.random.randn(len(dates)) * 0.02) * 102,
            'Low': np.cumprod(1 + np.random.randn(len(dates)) * 0.02) * 98,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Open': np.cumprod(1 + np.random.randn(len(dates)) * 0.02) * 100
        }, index=dates)

        # Add some NaN values to simulate real data
        for col in ['Close', 'High', 'Low']:
            nan_indices = np.random.choice(len(sample_data), size=int(len(sample_data) * 0.1), replace=False)
            sample_data.iloc[nan_indices, sample_data.columns.get_loc(col)] = np.nan

        try:
            strategy_results = debugger.test_nan_handling_strategies(sample_data)

            print("NaN Handling Strategy Performance:")
            for strategy_name, results in sorted(strategy_results.items(), key=lambda x: x[1].get('effectiveness_score', 0), reverse=True):
                if 'error' not in results:
                    print(f"  {strategy_name:20s}: {results['nan_reduction_percentage']:5.1f}% NaN reduction, "
                          f"{results['data_preservation_rate']:5.1f}% data preserved, "
                          f"score: {results['effectiveness_score']:5.1f}")
                else:
                    print(f"  {strategy_name:20s}: FAILED - {results['error']}")

        except Exception as e:
            print(f"Error testing NaN handling strategies: {e}")

    print(f"\n=== DEBUG COMPLETE ===")

    if debug_report.get('error'):
        print(f"Debugging failed with error: {debug_report['error']}")
        return 1
    else:
        print("Debugging completed successfully!")
        print("Check the 'results' directory for detailed JSON reports.")
        return 0

if __name__ == "__main__":
    sys.exit(main())