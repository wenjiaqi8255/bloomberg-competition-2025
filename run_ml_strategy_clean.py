#!/usr/bin/env python3
"""
Clean version of ML strategy with optimized settings for production use.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from trading_system.strategy_runner import StrategyRunner


def main():
    """Run the ML strategy with optimized settings."""
    print("=" * 60)
    print("Bloomberg Competition - ML Strategy (Clean)")
    print("=" * 60)

    # Path to ML configuration
    config_path = Path(__file__).parent / "configs" / "ml_strategy_config.yaml"

    print(f"Configuration: {config_path}")
    print("Strategy: ML with XGBoost + Optuna optimization")
    print()

    # Create strategy runner
    runner = StrategyRunner(str(config_path))

    # Optimize configuration for better performance
    config_updates = {
        'backtest': {
            'start_date': '2022-01-01',  # 3 years of data
            'end_date': datetime.now().strftime('%Y-%m-%d')
        },
        'strategy': {
            'prediction_horizon': 21,  # ~1 month
            'top_n_assets': 8,
            'min_prediction_score': 0.3,  # Lower threshold for more signals
            'optimization': {
                'optuna_trials': 20,  # Balance between speed and optimization
                'use_optuna': True
            },
            'training': {
                'cv_folds': 3,  # Faster cross-validation
                'retrain_frequency': 'monthly'
            }
        }
    }

    # Apply configuration updates
    for section, updates in config_updates.items():
        if section in runner.config_loader.config:
            runner.config_loader.config[section].update(updates)
        else:
            runner.config_loader.config[section] = updates

    experiment_name = f"ml_strategy_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running experiment: {experiment_name}")
    print("Optimized settings:")
    print("- Date range: 2022-01-01 to present (~3 years)")
    print("- Optuna trials: 20")
    print("- Lower prediction threshold for more signals")
    print("- 3-fold CV for faster training")
    print()

    try:
        with runner:
            results = runner.run_strategy(experiment_name)

            print("✅ ML Strategy execution completed!")
            print()
            print("Performance Summary:")
            print(f"  • Total Return: {results.get('total_return', 0):.2%}")
            print(f"  • Annualized Return: {results.get('annualized_return', 0):.2%}")
            print(f"  • Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  • Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  • Alpha: {results.get('alpha', 0):.3f}")
            print(f"  • Beta: {results.get('beta', 0):.3f}")
            print()

            # Model performance
            model_info = results.get('model_metadata', {})
            print("Model Performance:")
            print(f"  • Features: {model_info.get('n_features', 0)}")
            print(f"  • R² Score: {model_info.get('r2', 0):.4f}")
            print(f"  • MSE: {model_info.get('mse', 0):.6f}")
            print()

            # Strategy metrics
            strategy_metrics = {k: v for k, v in results.items() if k in ['avg_positions_held', 'max_positions_held', 'avg_cash_allocation']}
            if strategy_metrics:
                print("Strategy Metrics:")
                for key, value in strategy_metrics.items():
                    print(f"  • {key.replace('_', ' ').title()}: {value}")

            # Check signals
            if 'strategy_signals' in results and not results['strategy_signals'].empty:
                signals = results['strategy_signals']
                non_zero_signals = (signals > 0).sum().sum()
                total_signals = signals.size
                print(f"  • Signal Activity: {non_zero_signals}/{total_signals} ({non_zero_signals/total_signals*100:.1f}%)")

            print()
            print("Results saved to:")
            print(f"  • ./results/{experiment_name}_results.json")
            print(f"  • ./results/{experiment_name}_signals.csv")
            print(f"  • ./results/{experiment_name}_portfolio.csv")
            print()
            print("📊 View detailed results on Weights & Biases")

    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())