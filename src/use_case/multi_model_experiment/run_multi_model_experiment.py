#!/usr/bin/env python3
"""
Multi-Model Experiment Runner
=============================

Entry point for running multi-model experiments with proper training, prediction,
and backtesting for multiple models, then combining them with a metamodel.

This script replaces the flawed optimal_experiment with a properly architected
system that uses real training pipelines and actual backtesting.

Usage:
    # Run with configuration file
    python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml
    
    # Quick test mode (fewer trials, smaller dataset)
    python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml --quick-test
    
    # Verbose logging
    python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml --verbose
"""

import argparse
import logging
from pathlib import Path

# Ensure repository root is on sys.path
import sys
from pathlib import Path as _PathFix
_repo_root = _PathFix(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Now import with absolute path
from src.use_case.multi_model_experiment.multi_model_orchestrator import MultiModelOrchestrator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence overly verbose libraries
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('optuna').setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run multi-model experiment with proper training, prediction, and backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml
  
  # Quick test mode
  python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml --quick-test
  
  # Verbose logging
  python run_multi_model_experiment.py --config configs/multi_model_experiment.yaml --verbose
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to multi-model experiment configuration file'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Enable quick test mode with reduced trials and smaller dataset'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        logger.info("="*60)
        logger.info("MULTI-MODEL EXPERIMENT RUNNER")
        logger.info("="*60)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Quick test mode: {args.quick_test}")
        logger.info(f"Verbose logging: {args.verbose}")
        
        # Create orchestrator
        orchestrator = MultiModelOrchestrator(config_path=str(config_path))
        
        # Apply quick test modifications if requested
        if args.quick_test:
            logger.info("Applying quick test modifications...")
            _apply_quick_test_modifications(orchestrator)
        
        # Run the complete experiment
        logger.info("Starting multi-model experiment...")
        results = orchestrator.run_complete_experiment()
        
        # Display results
        _display_results(results)
        
        logger.info("Multi-model experiment completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Multi-model experiment failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def _apply_quick_test_modifications(orchestrator):
    """Apply quick test modifications to reduce execution time."""
    config = orchestrator.config
    
    # Reduce HPO trials
    if 'base_models' in config:
        for model_config in config['base_models']:
            if 'hpo_trials' in model_config:
                original_trials = model_config['hpo_trials']
                model_config['hpo_trials'] = min(5, original_trials)
                logging.info(f"Reduced {model_config['model_type']} trials from {original_trials} to {model_config['hpo_trials']}")
    
    if 'metamodel' in config and 'hpo_trials' in config['metamodel']:
        original_trials = config['metamodel']['hpo_trials']
        config['metamodel']['hpo_trials'] = min(5, original_trials)
        logging.info(f"Reduced metamodel trials from {original_trials} to {config['metamodel']['hpo_trials']}")
    
    # Reduce universe size
    if 'universe' in config and len(config['universe']) > 3:
        original_size = len(config['universe'])
        config['universe'] = config['universe'][:3]
        logging.info(f"Reduced universe from {original_size} to {len(config['universe'])} symbols")
    
    # Use shorter time periods
    if 'periods' in config:
        periods = config['periods']
        if 'train' in periods:
            periods['train']['start'] = '2023-01-01'
            periods['train']['end'] = '2023-06-30'
        if 'test' in periods:
            periods['test']['start'] = '2023-07-01'
            periods['test']['end'] = '2023-09-30'
        logging.info("Applied shorter time periods for quick test")
    
    # Relax system requirements
    if 'system_requirements' in config:
        requirements = config['system_requirements']
        requirements['min_sharpe_ratio'] = 0.3
        requirements['max_drawdown_threshold'] = -0.4
        requirements['min_win_rate'] = 0.35
        logging.info("Relaxed system requirements for quick test")

    # Relax validation requirements
    if 'advanced' in config and 'validation' in config['advanced']:
        validation = config['advanced']['validation']
        validation['min_absolute_stocks'] = 3  # Allow fewer stocks for quick test
        logging.info("Relaxed validation requirements for quick test")


def _display_results(results):
    """Display experiment results in a readable format."""
    print("\n" + "="*60)
    print("MULTI-MODEL EXPERIMENT RESULTS")
    print("="*60)
    
    # Experiment info
    experiment_info = results.get('experiment_info', {})
    print(f"Experiment: {experiment_info.get('name', 'N/A')}")
    print(f"Status: {results.get('status', 'UNKNOWN')}")
    print(f"Execution time: {results.get('execution_time', 0.0):.2f} seconds")
    
    # Base models summary
    base_models = results.get('base_models', {})
    base_summary = base_models.get('summary', {})
    print(f"\nBase Models Trained: {base_summary.get('total_models', 0)}")
    
    base_results = base_models.get('results', [])
    for i, model_result in enumerate(base_results):
        model_type = model_result.get('model_type', 'Unknown')
        model_id = model_result.get('model_id', 'N/A')
        metrics = model_result.get('performance_metrics', {})
        sharpe = metrics.get('sharpe_ratio', 0.0)
        total_return = metrics.get('total_return', 0.0)
        
        print(f"  {i+1}. {model_type}")
        print(f"     Model ID: {model_id}")
        print(f"     Sharpe: {sharpe:.3f}, Return: {total_return:.2%}")
    
    # Metamodel summary
    metamodel = results.get('metamodel', {})
    meta_summary = metamodel.get('summary', {})
    
    print(f"\nMetaModel:")
    print(f"  Model ID: {meta_summary.get('model_id', 'N/A')}")
    print(f"  Method: {meta_summary.get('method', 'N/A')}")
    print(f"  Alpha: {meta_summary.get('alpha', 1.0):.3f}")
    print(f"  Sharpe: {meta_summary.get('sharpe_ratio', 0.0):.3f}")
    print(f"  Return: {meta_summary.get('total_return', 0.0):.2%}")
    print(f"  Max DD: {meta_summary.get('max_drawdown', 0.0):.2%}")
    
    # Strategy weights
    weights = meta_summary.get('weights', {})
    if weights:
        print(f"  Strategy Weights:")
        for strategy, weight in weights.items():
            print(f"    {strategy}: {weight:.3f}")
    
    # System summary
    system_summary = results.get('system_summary', {})
    print(f"\nSystem Summary:")
    print(f"  Overall Performance: {system_summary.get('overall_performance', 'N/A')}")
    print(f"  Validation Status: {system_summary.get('validation_status', 'N/A')}")
    
    validation = system_summary.get('meets_requirements', {})
    print(f"  Meets Requirements:")
    print(f"    Sharpe: {validation.get('sharpe_ratio', False)}")
    print(f"    Drawdown: {validation.get('max_drawdown', False)}")
    print(f"    Overall Valid: {validation.get('overall_valid', False)}")
    
    improvement = system_summary.get('system_improvement', {})
    print(f"  Improvements:")
    print(f"    vs Best Base Model: {improvement.get('vs_best_base_model', 0.0):+.3f}")
    print(f"    vs Avg Base Model: {improvement.get('vs_avg_base_model', 0.0):+.3f}")
    
    # Output directory
    experiment_info = results.get('experiment_info', {})
    output_dir = experiment_info.get('output_dir', 'N/A')
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    sys.exit(main())
