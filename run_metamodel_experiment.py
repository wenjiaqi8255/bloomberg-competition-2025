#!/usr/bin/env python3
"""
MetaModel End-to-End Experiment Runner
=====================================

This script provides a complete workflow for MetaModel training and evaluation,
integrating with the existing experiment orchestrator infrastructure.

Usage:
    python run_metamodel_experiment.py --config configs/metamodel_experiment_config.yaml
    python run_metamodel_experiment.py --method ridge --alpha 0.5 --strategies DualMomentum,MLStrategy,FF5Strategy
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np

from trading_system.models.training.metamodel_config import (
    MetaModelTrainingConfig,
    MetaModelExperimentConfig
)
from trading_system.models.training.metamodel_pipeline import MetaModelTrainingPipeline
from trading_system.data.strategy_data_collector import StrategyDataCollector
from trading_system.models.model_persistence import ModelRegistry
from trading_system.orchestration.meta_model import MetaModel


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_metamodel_config_from_yaml(yaml_config: Dict[str, Any]) -> MetaModelTrainingConfig:
    """Create MetaModelTrainingConfig from YAML configuration."""
    metamodel_config = yaml_config.get('metamodel_training', {})

    # Parse dates
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


def run_metamodel_experiment(config: MetaModelTrainingConfig,
                           yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete MetaModel experiment.

    Args:
        config: MetaModelTrainingConfig instance
        yaml_config: Full YAML configuration dictionary

    Returns:
        Dictionary with experiment results
    """
    print(f"=== Running MetaModel Experiment: {config.experiment_name} ===")
    print(f"Method: {config.method}, Alpha: {config.alpha}")
    print(f"Strategies: {config.strategies}")
    print(f"Data source: {config.data_source}")
    print(f"Period: {config.start_date.date()} to {config.end_date.date()}")

    # Step 1: Initialize training pipeline
    print("\n1. Initializing MetaModel training pipeline...")
    model_registry_path = yaml_config.get('model_registry', {}).get('base_path', './models')
    pipeline = MetaModelTrainingPipeline(config, registry_path=model_registry_path)

    # Step 2: Run training
    print("\n2. Training MetaModel...")
    model_name_template = yaml_config.get('model_registry', {}).get('model_name_template', 'metamodel_{method}_{date}')
    model_name = model_name_template.format(
        method=config.method,
        date=datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    training_result = pipeline.run_metamodel_pipeline(model_name=model_name)

    print(f"✅ MetaModel training completed!")
    print(f"   Model ID: {training_result['pipeline_info']['model_id']}")
    print(f"   Training time: {training_result['pipeline_info']['training_time']:.2f} seconds")

    # Step 3: Analyze results
    print("\n3. Analyzing training results...")
    model_weights = training_result['model_weights']
    weight_analysis = training_result['performance_analysis']['weight_distribution']

    print(f"Strategy weights: {model_weights}")
    print(f"Weight distribution:")
    print(f"   - Mean weight: {weight_analysis['mean']:.3f}")
    print(f"   - Max weight: {weight_analysis['max']:.3f}")
    print(f"   - Effective strategies: {weight_analysis['effective_n']:.1f}")

    # Step 4: Validation metrics
    if training_result['training_results']['validation_metrics']:
        metrics = training_result['training_results']['validation_metrics']
        print(f"\nValidation metrics:")
        for metric, value in metrics.items():
            print(f"   - {metric.upper()}: {value:.4f}")

    # Step 5: System integration test (if enabled)
    integration_results = {}
    if yaml_config.get('system_integration', {}).get('enabled', False):
        print("\n4. Running system integration test...")
        integration_results = run_system_integration_test(training_result, yaml_config)

    # Step 6: Generate report
    print("\n5. Generating experiment report...")
    report_path = generate_experiment_report(training_result, yaml_config, integration_results)

    # Step 7: Log to experiment tracking
    if yaml_config.get('experiment', {}).get('log_to_wandb', False):
        try:
            log_to_wandb(training_result, yaml_config)
        except ImportError:
            print("⚠️  WandB not available, skipping logging")
        except Exception as e:
            print(f"⚠️  Failed to log to WandB: {e}")

    print(f"\n✅ MetaModel experiment completed successfully!")
    print(f"   Model saved: {training_result['pipeline_info']['model_id']}")
    print(f"   Report: {report_path}")

    return {
        'training_result': training_result,
        'integration_results': integration_results,
        'report_path': report_path
    }


def run_system_integration_test(training_result: Dict[str, Any],
                              yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test the trained MetaModel with SystemOrchestrator."""

    integration_config = yaml_config.get('system_integration', {})

    # Load the trained model
    model_registry_path = yaml_config.get('model_registry', {}).get('base_path', './models')
    registry = ModelRegistry(model_registry_path)

    model_id = training_result['pipeline_info']['model_id']
    loaded_result = registry.load_model_with_artifacts(model_id)

    if not loaded_result:
        return {'error': 'Failed to load trained model'}

    metamodel, artifacts = loaded_result

    # Create test data
    test_period = integration_config.get('test_period', {})
    universe = integration_config.get('universe', {})

    test_signals = generate_test_signals(
        strategies=training_result['metamodel_config']['strategies'],
        symbols=universe.get('symbols', ['SPY', 'AAPL', 'MSFT']),
        n_days=30  # Short test period
    )

    # Test MetaModel combination
    combined_signals = metamodel.combine(test_signals)

    # Validate expected behaviors
    expected_behaviors = integration_config.get('expected_behaviors', {})
    validation_results = validate_expected_behaviors(
        metamodel.strategy_weights,
        combined_signals,
        expected_behaviors
    )

    return {
        'test_signals_shape': {k: v.shape for k, v in test_signals.items()},
        'combined_signals_shape': combined_signals.shape,
        'strategy_weights': metamodel.strategy_weights,
        'validation_results': validation_results,
        'test_passed': all(validation_results.values())
    }


def generate_test_signals(strategies: List[str], symbols: List[str], n_days: int) -> Dict[str, pd.DataFrame]:
    """Generate synthetic test signals for integration testing."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')

    signals = {}
    np.random.seed(42)  # For reproducible results

    for strategy in strategies:
        # Generate random signals for each strategy
        strategy_data = {}
        for symbol in symbols:
            # Generate realistic signal values (-0.05 to 0.05)
            returns = np.random.normal(0, 0.02, n_days)
            strategy_data[symbol] = returns

        signals[strategy] = pd.DataFrame(strategy_data, index=dates)

    return signals


def validate_expected_behaviors(weights: Dict[str, float],
                               signals: pd.DataFrame,
                               expected: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that MetaModel behaves as expected."""
    results = {}

    # Check weight stability (simplified test)
    if 'weight_stability' in expected:
        max_concentration = max(weights.values()) if weights else 0
        results['weight_stability'] = max_concentration <= expected['weight_stability']

    # Check minimum active strategies
    if 'min_active_strategies' in expected:
        active_strategies = sum(1 for w in weights.values() if w > 0.01)
        results['min_active_strategies'] = active_strategies >= expected['min_active_strategies']

    # Check maximum concentration
    if 'max_concentration' in expected:
        max_weight = max(weights.values()) if weights else 0
        results['max_concentration'] = max_weight <= expected['max_concentration']

    return results


def generate_experiment_report(training_result: Dict[str, Any],
                            yaml_config: Dict[str, Any],
                            integration_results: Dict[str, Any]) -> str:
    """Generate comprehensive experiment report."""

    reporting_config = yaml_config.get('reporting', {})
    if not reporting_config.get('generate_report', True):
        return "Report generation disabled"

    # Create output directory
    output_dir = Path("./results/metamodel_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report filename
    experiment_name = yaml_config.get('experiment', {}).get('name', 'metamodel_experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"{experiment_name}_{timestamp}.json"
    report_path = output_dir / report_filename

    # Compile report data
    report_data = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': timestamp,
            'config': yaml_config
        },
        'training_results': training_result,
        'integration_results': integration_results,
        'summary': {
            'model_id': training_result['pipeline_info']['model_id'],
            'method': training_result['metamodel_config']['method'],
            'strategies': training_result['metamodel_config']['strategies'],
            'training_time': training_result['pipeline_info']['training_time'],
            'integration_test_passed': integration_results.get('test_passed', False)
        }
    }

    # Save report
    import json
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    return str(report_path)


def log_to_wandb(training_result: Dict[str, Any], yaml_config: Dict[str, Any]):
    """Log experiment results to Weights & Biases."""
    import wandb

    experiment_config = yaml_config.get('experiment', {})
    wandb.init(
        project="metamodel_experiments",
        name=experiment_config.get('name', 'metamodel_experiment'),
        tags=experiment_config.get('tags', []),
        config=training_result['metamodel_config']
    )

    # Log training metrics
    if training_result['training_results']['validation_metrics']:
        wandb.log(training_result['training_results']['validation_metrics'])

    # Log model weights
    wandb.log({
        'model_weights': training_result['model_weights'],
        'effective_strategies': training_result['performance_analysis']['weight_distribution']['effective_n']
    })

    # Log integration test results
    if 'integration_results' in training_result:
        integration = training_result['integration_results']
        wandb.log({
            'integration_test_passed': integration.get('test_passed', False),
            'validation_results': integration.get('validation_results', {})
        })

    wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run MetaModel end-to-end experiment')

    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('--method', type=str, choices=['equal', 'lasso', 'ridge', 'dynamic'],
                       help='MetaModel method')
    parser.add_argument('--alpha', type=float,
                       help='Regularization strength for lasso/ridge')
    parser.add_argument('--strategies', type=str,
                       help='Comma-separated list of strategy names')
    parser.add_argument('--start-date', type=str,
                       help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='Training end date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        yaml_config = load_experiment_config(args.config)
        config = create_metamodel_config_from_yaml(yaml_config)
    else:
        # Create config from command line arguments
        strategies = args.strategies.split(',') if args.strategies else ['DualMomentumStrategy', 'MLStrategy', 'FF5Strategy']
        start_date = datetime.fromisoformat(args.start_date) if args.start_date else datetime(2022, 1, 1)
        end_date = datetime.fromisoformat(args.end_date) if args.end_date else datetime(2023, 12, 31)

        config = MetaModelTrainingConfig(
            method=args.method or 'ridge',
            alpha=args.alpha or 0.5,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            data_source='synthetic'
        )

        yaml_config = {
            'experiment': {'name': f'metamodel_{config.method}_cmdline'},
            'model_registry': {'base_path': './models'},
            'system_integration': {'enabled': False},
            'reporting': {'generate_report': True}
        }

    # Override config with command line arguments
    if args.method:
        config.method = args.method
    if args.alpha:
        config.alpha = args.alpha

    try:
        # Run experiment
        results = run_metamodel_experiment(config, yaml_config)
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📊 Report: {results['report_path']}")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()