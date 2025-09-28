#!/usr/bin/env python3
"""
Fast version of ML strategy for quick testing.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from trading_system.strategy_runner import StrategyRunner


def main():
    """Run the ML strategy with fast settings."""
    print("=" * 60)
    print("Bloomberg Competition - ML Strategy (Fast Test)")
    print("=" * 60)

    # Path to ML configuration
    config_path = Path(__file__).parent / "configs" / "ml_strategy_config.yaml"

    print(f"Using configuration: {config_path}")
    print("Fast test settings:")
    print("- Small asset universe: ['SPY', 'QQQ']")
    print("- Short date range: 3 months")
    print("- Optuna trials: 5 (instead of 50)")
    print("- Quick feature engineering")
    print()

    # Create strategy runner
    runner = StrategyRunner(str(config_path))

    # Override configuration for fast test
    runner.config_loader.config['universe']['all_assets'] = ['SPY', 'QQQ']
    runner.config_loader.config['backtest']['start_date'] = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    runner.config_loader.config['backtest']['end_date'] = datetime.now().strftime('%Y-%m-%d')
    runner.config_loader.config['strategy']['optimization']['optuna_trials'] = 5
    runner.config_loader.config['strategy']['optimization']['use_optuna'] = True

    experiment_name = f"ml_strategy_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running experiment: {experiment_name}")

    try:
        with runner:
            results = runner.run_strategy(experiment_name)

            print("✅ Fast strategy test completed successfully!")
            print()
            print("Key Results:")
            print(f"  - Total Return: {results.get('total_return', 0):.2%}")
            print(f"  - Annualized Return: {results.get('annualized_return', 0):.2%}")
            print(f"  - Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  - Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  - Model Features: {results.get('model_metadata', {}).get('n_features', 0)}")
            print(f"  - Model R²: {results.get('model_metadata', {}).get('r2', 0):.4f}")

            # Check if signals were generated
            if 'strategy_signals' in results and not results['strategy_signals'].empty:
                print(f"  - Signals generated: {results['strategy_signals'].shape}")
                print(f"  - Non-zero signals: {(results['strategy_signals'] > 0).sum().sum()}")
            else:
                print("  - No signals generated")

    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())