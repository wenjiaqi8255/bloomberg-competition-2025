#!/usr/bin/env python3
"""
Example script to run the ML strategy.

This script demonstrates how to use the ML strategy with the
complete pipeline including feature engineering, model training,
and backtesting.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from trading_system.strategy_runner import StrategyRunner


def main():
    """Run the ML strategy."""
    print("=" * 60)
    print("Bloomberg Competition - ML Strategy")
    print("=" * 60)

    # Path to ML configuration
    config_path = Path(__file__).parent / "configs" / "ml_strategy_config.yaml"

    print(f"Using configuration: {config_path}")
    print(f"Strategy: ML Strategy with XGBoost")
    print()

    # Create strategy runner
    runner = StrategyRunner(str(config_path))

    # Run the strategy
    experiment_name = f"ml_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running experiment: {experiment_name}")
    print("This will:")
    print("1. Fetch market data")
    print("2. Engineer features using technical indicators")
    print("3. Train ML model with Optuna optimization")
    print("4. Generate trading signals")
    print("5. Run backtest")
    print("6. Log results to Weights & Biases")
    print()

    try:
        with runner:
            results = runner.run_strategy(experiment_name)

            print("✅ Strategy execution completed successfully!")
            print()
            print("Key Results:")
            print(f"  - Total Return: {results.get('total_return', 0):.2%}")
            print(f"  - Annualized Return: {results.get('annualized_return', 0):.2%}")
            print(f"  - Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  - Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  - Model Features: {results.get('model_metadata', {}).get('n_features', 0)}")
            print(f"  - Model R²: {results.get('model_metadata', {}).get('r2', 0):.4f}")
            print()
            print("Results saved to:")
            print(f"  - ./results/{experiment_name}_results.json")
            print(f"  - ./results/{experiment_name}_signals.csv")
            print(f"  - ./results/{experiment_name}_portfolio.csv")

    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())