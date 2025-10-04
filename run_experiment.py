# -*- coding: utf-8 -*-
"""
End-to-End Experiment Starter Script
====================================

This script serves as the main entry point for running a complete
experiment, from model training to strategy backtesting, using the
`ExperimentOrchestrator`.

To run an experiment:
1. Ensure you have a unified experiment configuration file in the `configs`
   directory (e.g., `e2e_ff5_experiment.yaml`).
2. Run the script from the root of the project:
   python run_experiment.py                           # Uses default config
   python run_experiment.py --config configs/my_experiment.yaml  # Custom config
   python run_experiment.py -c configs/test.yaml     # Short form
"""

import argparse
import logging
import json
import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from trading_system.experiment_orchestrator import ExperimentOrchestrator
from trading_system.models.finetune.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    HyperparameterConfig,
    SearchSpace
)
from trading_system.models.training.metamodel_config import MetaModelTrainingConfig

# --- Configuration ---
# Default configuration file for the experiment
DEFAULT_EXPERIMENT_CONFIG_FILE = 'configs/e2e_ff5_experiment.yaml'

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Silence overly verbose libraries
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run trading experiments, MetaModel training, or hyperparameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard experiment
  python run_experiment.py                                    # Use default config
  python run_experiment.py --config configs/my_exp.yaml       # Custom config
  python run_experiment.py -c configs/test.yaml              # Short form

  # Run MetaModel experiment
  python run_experiment.py metamodel --config configs/metamodel_experiment_config.yaml
  python run_experiment.py metamodel --method ridge --alpha 0.5

  # Run hyperparameter optimization
  python run_experiment.py optimize --type metamodel --trials 50
  python run_experiment.py optimize --type strategy --config configs/hpo_strategy_config.yaml
        """
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Standard experiment command (default behavior)
    experiment_parser = subparsers.add_parser('experiment', help='Run standard end-to-end experiment')
    experiment_parser.add_argument(
        '-c', '--config',
        type=str,
        default=DEFAULT_EXPERIMENT_CONFIG_FILE,
        help=f'Path to experiment configuration file (default: {DEFAULT_EXPERIMENT_CONFIG_FILE})'
    )

    # MetaModel command
    metamodel_parser = subparsers.add_parser('metamodel', help='Run MetaModel training experiment')
    metamodel_parser.add_argument(
        '--config',
        type=str,
        default='configs/metamodel_experiment_config.yaml',
        help='Path to MetaModel experiment configuration file'
    )
    metamodel_parser.add_argument(
        '--method',
        type=str,
        choices=['equal', 'lasso', 'ridge', 'dynamic'],
        help='MetaModel method'
    )
    metamodel_parser.add_argument(
        '--alpha',
        type=float,
        help='Regularization strength for lasso/ridge'
    )
    metamodel_parser.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategy names'
    )
    metamodel_parser.add_argument(
        '--start-date',
        type=str,
        help='Training start date (YYYY-MM-DD)'
    )
    metamodel_parser.add_argument(
        '--end-date',
        type=str,
        help='Training end date (YYYY-MM-DD)'
    )

    # Optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Run hyperparameter optimization')
    optimize_parser.add_argument(
        '--type',
        type=str,
        choices=['metamodel', 'strategy', 'ml'],
        required=True,
        help='Type of optimization to run'
    )
    optimize_parser.add_argument(
        '--config',
        type=str,
        help='Path to optimization configuration file'
    )
    optimize_parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    optimize_parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout in seconds for optimization'
    )
    optimize_parser.add_argument(
        '--sampler',
        type=str,
        choices=['tpe', 'random', 'cmaes'],
        default='tpe',
        help='Optimization sampler'
    )
    optimize_parser.add_argument(
        '--metric',
        type=str,
        default='r2',
        help='Optimization metric'
    )

    # Default to experiment if no command specified
    if len(sys.argv) == 1:
        sys.argv.append('experiment')

    return parser.parse_args()


def main():
    """
    Main entry point that routes to appropriate command handlers.
    """
    # Parse command line arguments
    args = parse_arguments()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        if args.command == 'experiment' or args.command is None:
            run_standard_experiment(args, logger)
        elif args.command == 'metamodel':
            run_metamodel_experiment(args, logger)
        elif args.command == 'optimize':
            run_optimization(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


def run_standard_experiment(args, logger):
    """Run standard end-to-end experiment."""
    config_file = args.config
    logger.info(f"Starting standard experiment using config: {config_file}")

    # Validate config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found at '{config_file}'")

    # Initialize the orchestrator with the config file path
    orchestrator = ExperimentOrchestrator(experiment_config_path=config_file)

    # Run the entire experiment pipeline
    results = orchestrator.run_experiment()

    # Print the final consolidated report
    logger.info("\n" + "="*50 + "\n--- EXPERIMENT FINISHED ---\n" + "="*50)
    print(json.dumps(results, indent=2, default=str))
    logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)


def run_metamodel_experiment(args, logger):
    """Run MetaModel training experiment."""
    logger.info("Starting MetaModel training experiment")

    # Import MetaModel experiment runner
    try:
        from run_metamodel_experiment import run_metamodel_experiment as run_mm_exp
        import yaml
        from datetime import datetime
    except ImportError as e:
        logger.error(f"Failed to import MetaModel components: {e}")
        logger.info("Make sure run_metamodel_experiment.py exists in the project root")
        sys.exit(1)

    # Load config or create from command line args
    if args.config and os.path.exists(args.config):
        # Load from YAML config
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = create_metamodel_config_from_yaml(yaml_config)
    else:
        # Create config from command line arguments
        config = create_metamodel_config_from_args(args)

    # Create minimal YAML config for runner
    yaml_config = {
        'experiment': {
            'name': f'metamodel_{config.method}_cmdline',
            'log_to_wandb': False
        },
        'model_registry': {'base_path': './models'},
        'system_integration': {'enabled': False},
        'reporting': {'generate_report': True}
    }

    # Run MetaModel experiment
    results = run_mm_exp(config, yaml_config)

    logger.info("\n" + "="*50 + "\n--- METAMODEL EXPERIMENT FINISHED ---\n" + "="*50)
    print(json.dumps(results, indent=2, default=str))
    logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)


def run_optimization(args, logger):
    """Run hyperparameter optimization."""
    logger.info(f"Starting hyperparameter optimization for {args.type}")

    # Create optimization configuration
    config = HyperparameterConfig(
        n_trials=args.trials,
        timeout_seconds=args.timeout,
        study_name=f"{args.type}_optimization",
        sampler=args.sampler,
        metric_name=args.metric,
        direction="maximize" if args.metric in ['r2', 'sharpe_ratio', 'accuracy'] else "minimize",
        save_results=True,
        results_dir=f"./hpo_results/{args.type}"
    )

    # Create optimizer
    optimizer = HyperparameterOptimizer(config=config)

    # Create appropriate search spaces
    if args.type == 'metamodel':
        optimizer.create_default_search_spaces('metamodel')
    elif args.type == 'strategy':
        optimizer.create_default_search_spaces('strategy')
    else:  # ml
        optimizer.create_default_search_spaces('xgboost')  # Default to XGBoost

    # Run optimization
    logger.info(f"Running {args.trials} trials for {args.type} optimization...")
    results = optimizer.optimize()

    # Display results
    logger.info("\n" + "="*50 + "\n--- OPTIMIZATION FINISHED ---\n" + "="*50)
    logger.info(f"Best score: {results.best_score:.4f}")
    logger.info(f"Best parameters: {results.best_params}")
    logger.info(f"Total trials: {results.n_trials}")
    logger.info(f"Optimization time: {results.optimization_time:.2f} seconds")

    print("\nBest Parameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value}")

    print(f"\nOptimization Summary:")
    print(f"  Best Score: {results.best_score:.4f}")
    print(f"  Total Trials: {results.n_trials}")
    print(f"  Successful Trials: {len([t for t in results.all_trials if t['state'] == 'COMPLETE'])}")
    print(f"  Pruned Trials: {len(results.pruned_trials)}")
    print(f"  Failed Trials: {len(results.failed_trials)}")
    print(f"  Optimization Time: {results.optimization_time:.2f}s")
    print(f"  Results saved to: {config.results_dir}")

    logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)


def create_metamodel_config_from_args(args) -> MetaModelTrainingConfig:
    """Create MetaModelTrainingConfig from command line arguments."""
    from datetime import datetime

    strategies = args.strategies.split(',') if args.strategies else ['DualMomentumStrategy', 'MLStrategy', 'FF5Strategy']
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else datetime(2022, 1, 1)
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else datetime(2023, 12, 31)

    return MetaModelTrainingConfig(
        method=args.method or 'ridge',
        alpha=args.alpha or 0.5,
        strategies=strategies,
        start_date=start_date,
        end_date=end_date,
        data_source='synthetic',
        tags={'experiment_type': 'command_line', 'method': args.method or 'ridge'}
    )


def create_metamodel_config_from_yaml(yaml_config) -> MetaModelTrainingConfig:
    """Create MetaModelTrainingConfig from YAML configuration."""
    from datetime import datetime

    metamodel_config = yaml_config.get('metamodel_training', {})
    start_date = datetime.fromisoformat(metamodel_config.get('start_date', '2022-01-01'))
    end_date = datetime.fromisoformat(metamodel_config.get('end_date', '2023-12-31'))

    return MetaModelTrainingConfig(
        method=metamodel_config.get('method', 'ridge'),
        alpha=metamodel_config.get('alpha', 0.5),
        strategies=metamodel_config.get('strategies', []),
        data_source=metamodel_config.get('data_source', 'synthetic'),
        start_date=start_date,
        end_date=end_date,
        target_benchmark=metamodel_config.get('target_benchmark'),
        use_cross_validation=metamodel_config.get('use_cross_validation', True),
        cv_folds=metamodel_config.get('cv_folds', 5),
        experiment_name=metamodel_config.get('experiment_name', 'metamodel_experiment'),
        tags=yaml_config.get('experiment', {}).get('tags', {}),
        track_strategy_correlation=metamodel_config.get('track_strategy_correlation', True),
        track_contribution_analysis=metamodel_config.get('track_contribution_analysis', True)
    )


if __name__ == "__main__":
    main()
