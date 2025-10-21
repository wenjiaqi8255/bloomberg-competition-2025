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
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from .experiment_orchestrator import ExperimentOrchestrator
# 使用简化组件
from ...trading_system.models.finetune.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    create_xgboost_hpo,
    create_metamodel_hpo
)
from ...trading_system.metamodel.pipeline import MetaModelPipeline, MetaModelRunConfig
from ...trading_system.models.training.config import TrainingConfig, load_config

# --- Configuration ---
# Default configuration file for the experiment
DEFAULT_EXPERIMENT_CONFIG_FILE = '../../../configs/e2e_ff5_experiment.yaml'

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
    """Run MetaModel training experiment - 简化版本."""
    logger.info("Starting MetaModel training experiment")

    # 使用新Pipeline训练元模型
    pipe = MetaModelPipeline()
    model_name = f"metamodel_{args.method or 'ridge'}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_cfg = MetaModelRunConfig(
        strategies=(args.strategies.split(',') if args.strategies else ["ml_strategy", "ff5_regression"]),
        start_date=args.start_date or '2022-01-01',
        end_date=args.end_date or '2023-12-31',
        data_source='backtest_results',
        method=args.method or 'ridge',
        alpha=args.alpha or 0.5,
        model_name=model_name,
        use_full_backtest=False
    )
    R, y = pipe.collect(run_cfg)
    model, weights, combined = pipe.train_and_combine(R, y, run_cfg)
    metrics = pipe.evaluate_light(combined)
    model_id = pipe.save(model, model_name, {'weights': weights, 'metrics': metrics})
    results = {
        'model_id': model_id,
        'method': run_cfg.method,
        'strategy_weights': weights,
        'metrics': metrics,
        'training_samples': len(R),
        'strategies': run_cfg.strategies
    }

    # 显示结果 - 简化版本
    logger.info("\n" + "="*50 + "\n--- METAMODEL TRAINING FINISHED ---\n" + "="*50)
    logger.info(f"Model ID: {results['model_id']}")
    logger.info(f"Method: {results['method']}")
    logger.info(f"Strategy weights: {results['strategy_weights']}")
    logger.info(f"R²: {results['metrics']['r2']:.4f}")
    logger.info(f"Annualized return: {results['metrics']['annualized_return']:.4f}")

    print(f"\nMetaModel Results:")
    print(f"  Model ID: {results['model_id']}")
    print(f"  Method: {results['method']}")
    print(f"  Training samples: {results['training_samples']}")
    print(f"  Strategies: {', '.join(results['strategies'])}")
    print(f"\nStrategy weights:")
    for strategy, weight in results['strategy_weights'].items():
        print(f"  {strategy}: {weight:.4f}")
    print(f"\nPerformance metrics:")
    # metrics is a financial metrics dict in pipeline.flow; display key ones if present
    if 'sharpe_ratio' in results['metrics']:
        print(f"  Sharpe: {results['metrics']['sharpe_ratio']:.4f}")

    logger.info("\n" + "="*50 + "\n--- METAMODEL EXPERIMENT FINISHED ---\n" + "="*50)
    print(json.dumps(results, indent=2, default=str))
    logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)


def run_optimization(args, logger):
    """Run hyperparameter optimization."""
    logger.info(f"Starting hyperparameter optimization for {args.type}")

    # 使用简化优化器 - 一行创建
    if args.type == 'metamodel':
        optimizer = create_metamodel_hpo(args.trials)
    elif args.type == 'strategy':
        # 策略优化暂时不支持，使用XGBoost作为默认
        optimizer = create_xgboost_hpo(args.trials)
    else:  # ml
        optimizer = create_xgboost_hpo(args.trials)

    # 创建简单的评估函数
    def dummy_evaluator(params):
        """简化的评估函数 - 实际应用中替换为真实评估逻辑"""
        logger.info(f"Evaluating params: {params}")
        # 这里应该是真实的模型训练和评估
        # 目前返回模拟分数用于演示
        import numpy as np
        base_score = 0.5
        if 'n_estimators' in params:
            base_score += 0.1 * np.log(params['n_estimators'] / 100)
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if 0.01 <= lr <= 0.1:
                base_score += 0.2
        if 'method' in params and params['method'] == 'ridge':
            base_score += 0.1
        return np.random.normal(base_score, 0.05)

    # 运行优化 - 一行执行
    logger.info(f"Running {args.trials} trials for {args.type} optimization...")
    results = optimizer.optimize(dummy_evaluator)

    # Display results - 简化版本
    logger.info("\n" + "="*50 + "\n--- OPTIMIZATION FINISHED ---\n" + "="*50)
    logger.info(f"Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Total trials: {results['n_trials']}")

    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")

    print(f"\nOptimization Summary:")
    print(f"  Best Score: {results['best_score']:.4f}")
    print(f"  Total Trials: {results['n_trials']}")
    print(f"  Study Name: {results['study_name']}")
    print(f"  Results saved to: ./hpo_results")

    logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)

if __name__ == "__main__":
    main()
