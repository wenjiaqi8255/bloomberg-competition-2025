#!/usr/bin/env python3
"""
FF5 + Box-Based Portfolio Construction Demo
==========================================

This script demonstrates the integration of Fama-French 5-factor model training
with Box-First portfolio construction methodology.

Usage:
    python run_ff5_box_experiment.py [--demo] [--config CONFIG_FILE]

Options:
    --demo              Use the quick demo configuration
    --config CONFIG     Use custom configuration file
    --dry-run           Validate configuration without running
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from trading_system.experiment_orchestrator import ExperimentOrchestrator
import yaml

def setup_logging(level=logging.INFO):
    """Setup logging for the experiment."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Silence verbose libraries
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def validate_config(config_path: str) -> bool:
    """Validate configuration file before running."""
    logger = logging.getLogger(__name__)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Basic validation
        required_sections = ['data_provider', 'training_setup', 'backtest', 'strategy']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False

        # Check if strategy has portfolio_construction
        strategy_config = config.get('strategy', {})
        if 'portfolio_construction' not in strategy_config.get('parameters', {}):
            logger.error("Strategy missing 'portfolio_construction' configuration")
            return False

        # Check portfolio construction method
        pc_config = strategy_config['parameters']['portfolio_construction']
        if pc_config.get('method') != 'box_based':
            logger.warning(f"Portfolio construction method is '{pc_config.get('method')}', not 'box_based'")

        # Validate box configuration
        if 'box_weights' not in pc_config:
            logger.error("Missing 'box_weights' configuration")
            return False

        box_weights = pc_config['box_weights']
        if 'dimensions' not in box_weights:
            logger.error("Missing box dimensions configuration")
            return False

        dimensions = box_weights['dimensions']
        required_dims = ['size', 'style', 'region', 'sector']
        for dim in required_dims:
            if dim not in dimensions or not dimensions[dim]:
                logger.error(f"Missing or empty dimension: {dim}")
                return False

        logger.info("✓ Configuration validation passed")
        logger.info(f"  Method: {pc_config.get('method')}")
        logger.info(f"  Stocks per box: {pc_config.get('stocks_per_box', 'N/A')}")
        logger.info(f"  Allocation method: {pc_config.get('allocation_method', 'N/A')}")
        logger.info(f"  Box dimensions: {list(dimensions.keys())}")

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def print_experiment_summary(config_path: str):
    """Print experiment configuration summary."""
    logger = logging.getLogger(__name__)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("\n" + "="*60)
        print("FF5 + BOX-BASED EXPERIMENT SUMMARY")
        print("="*60)

        # Training info
        training = config.get('training_setup', {})
        print(f"📊 Training Period: {training.get('parameters', {}).get('start_date', 'N/A')} to {training.get('parameters', {}).get('end_date', 'N/A')}")
        print(f"📈 Model Type: {training.get('model', {}).get('model_type', 'N/A')}")
        print(f"🎯 Symbols: {len(training.get('parameters', {}).get('symbols', []))} stocks")

        # Portfolio construction info
        strategy = config.get('strategy', {})
        pc_config = strategy.get('parameters', {}).get('portfolio_construction', {})
        box_weights = pc_config.get('box_weights', {})
        dimensions = box_weights.get('dimensions', {})

        print(f"\n📦 Box-Based Construction:")
        print(f"   Method: {pc_config.get('method', 'N/A')}")
        print(f"   Stocks per box: {pc_config.get('stocks_per_box', 'N/A')}")
        print(f"   Allocation method: {pc_config.get('allocation_method', 'N/A')}")
        print(f"   Box weight method: {box_weights.get('method', 'N/A')}")

        print(f"\n🏗️  Box Dimensions:")
        for dim, values in dimensions.items():
            print(f"   {dim.title()}: {values}")

        # Calculate total boxes
        total_boxes = 1
        for values in dimensions.values():
            total_boxes *= len(values)
        print(f"   Total target boxes: {total_boxes}")

        # Backtest info
        backtest = config.get('backtest', {})
        print(f"\n🔄 Backtest Period: {backtest.get('start_date', 'N/A')} to {backtest.get('end_date', 'N/A')}")
        print(f"💰 Initial Capital: ${backtest.get('initial_capital', 'N/A'):,.0f}")
        print(f"📅 Rebalance: {backtest.get('rebalance_frequency', 'N/A')}")
        print(f"⚡ Position Limit: {backtest.get('position_limit', 'N/A')*100:.1f}%")

        # Experiment info
        experiment = config.get('experiment', {})
        print(f"\n🧪 Experiment: {experiment.get('name', 'N/A')}")
        print(f"📝 Description: {experiment.get('description', 'N/A')}")

        print("="*60)

    except Exception as e:
        logger.error(f"Failed to print experiment summary: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run FF5 + Box-Based portfolio construction experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick demo
  python run_ff5_box_experiment.py --demo

  # Run with full configuration
  python run_ff5_box_experiment.py --config configs/ff5_box_based_experiment.yaml

  # Validate configuration only
  python run_ff5_box_experiment.py --config configs/ff5_box_demo.yaml --dry-run
        """
    )

    parser.add_argument('--demo', action='store_true',
                       help='Use quick demo configuration')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-start experiment without confirmation prompt')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Determine configuration file
    if args.demo:
        config_file = 'configs/ff5_box_demo.yaml'
        logger.info("Using demo configuration")
    elif args.config:
        config_file = args.config
        logger.info(f"Using custom configuration: {config_file}")
    else:
        config_file = 'configs/ff5_box_based_experiment.yaml'
        logger.info(f"Using default configuration: {config_file}")

    # Check if configuration file exists
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        logger.info("Available configurations:")
        for f in ['configs/ff5_box_demo.yaml', 'configs/ff5_box_based_experiment.yaml']:
            if os.path.exists(f):
                logger.info(f"  - {f}")
        sys.exit(1)

    # Validate configuration
    logger.info("Validating configuration...")
    if not validate_config(config_file):
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Print experiment summary
    print_experiment_summary(config_file)

    if args.dry_run:
        logger.info("Dry run completed - configuration is valid")
        return

    # Confirm before running (unless auto mode)
    if not args.auto:
        try:
            response = input("\n🚀 Start experiment? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info("Experiment cancelled by user")
                return
        except KeyboardInterrupt:
            logger.info("\nExperiment cancelled by user")
            return
    else:
        logger.info("🚀 Auto-starting experiment...")

    # Run experiment
    logger.info(f"Starting FF5 + Box-Based experiment with config: {config_file}")

    try:
        # Initialize orchestrator
        orchestrator = ExperimentOrchestrator(experiment_config_path=config_file)

        # Run experiment
        logger.info("🔄 Running experiment pipeline...")
        results = orchestrator.run_experiment()

        # Display results
        logger.info("\n" + "="*60)
        logger.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        # Key results
        if 'backtest_results' in results:
            backtest = results['backtest_results']
            logger.info(f"📊 Final Portfolio Value: ${backtest.get('final_value', 0):,.2f}")
            logger.info(f"📈 Total Return: {backtest.get('total_return', 0)*100:.2f}%")
            logger.info(f"🎯 Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.3f}")
            logger.info(f"📉 Max Drawdown: {backtest.get('max_drawdown', 0)*100:.2f}%")

        # Box-based specific results
        if 'portfolio_construction_results' in results:
            pc_results = results['portfolio_construction_results']
            logger.info(f"📦 Construction Method: {pc_results.get('method', 'N/A')}")
            logger.info(f"🏗️  Box Coverage: {pc_results.get('box_coverage_ratio', 0)*100:.1f}%")
            logger.info(f"📊 Boxes Covered: {pc_results.get('covered_boxes', 0)}/{pc_results.get('total_boxes', 0)}")

        # Model results
        if 'model_training_results' in results:
            model_results = results['model_training_results']
            logger.info(f"🤖 Model R²: {model_results.get('r2_score', 0):.3f}")
            logger.info(f"📋 Best Parameters: {model_results.get('best_params', {})}")

        logger.info(f"📁 Results saved to: {results.get('output_directory', './results')}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        logger.info("Check the logs above for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()