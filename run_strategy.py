#!/usr/bin/env python3
"""
Main entry point for running trading strategies.

This script provides a command-line interface for running strategies
with the Bloomberg competition trading system.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system.strategy_runner import StrategyRunner


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('strategy_run.log')
        ]
    )


def main():
    """Main entry point for strategy execution."""
    parser = argparse.ArgumentParser(description="Run trading strategies")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with shorter date range")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize strategy runner
        runner = StrategyRunner(config_path=args.config)

        # Test mode: use shorter date range
        if args.test_mode:
            logger.info("Running in test mode with shorter date range")
            # Update config for test mode
            config = runner.config_loader.load_config()
            config['backtest']['end_date'] = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            config['backtest']['start_date'] = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            runner.config = config

        # Run strategy
        with runner:
            results = runner.run_strategy(experiment_name=args.experiment_name)

        # Print summary
        print("\n" + "="*60)
        print("STRATEGY EXECUTION SUMMARY")
        print("="*60)

        if 'backtest_results' in results:
            perf = results['backtest_results']
            print(f"Initial Capital: ${perf.get('initial_capital', 0):,.0f}")
            print(f"Final Value: ${perf.get('final_value', 0):,.0f}")
            print(f"Total Return: {perf.get('total_return', 0):.2%}")
            print(f"Annualized Return: {perf.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")

        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\nRisk Metrics:")
            print(f"Volatility: {metrics.get('volatility', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
            if 'alpha' in metrics:
                print(f"Alpha: {metrics.get('alpha', 0):.2%}")
                print(f"Beta: {metrics.get('beta', 0):.3f}")

        if 'strategy_metrics' in results:
            strat_metrics = results['strategy_metrics']
            print(f"\nStrategy Metrics:")
            if 'avg_positions_held' in strat_metrics:
                print(f"Avg Positions Held: {strat_metrics['avg_positions_held']:.1f}")
            if 'avg_cash_allocation' in strat_metrics:
                print(f"Avg Cash Allocation: {strat_metrics['avg_cash_allocation']:.1%}")

        print(f"\nExperiment: {results.get('experiment_name', 'Unknown')}")
        print(f"Results saved to: ./results/")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("Strategy execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())