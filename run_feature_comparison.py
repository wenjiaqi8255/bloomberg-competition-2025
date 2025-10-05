#!/usr/bin/env python3
"""
Feature Comparison Workflow Script

This script compares different feature sets using fixed optimal hyperparameters
to identify the best feature combination for model performance.

Usage:
    poetry run python run_feature_comparison.py --config configs/feature_comparison_config.yaml
    poetry run python run_feature_comparison.py --config configs/feature_comparison_config.yaml --test-mode
"""

import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import yaml

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from trading_system.config.feature import FeatureConfig
from trading_system.feature_engineering.pipeline import FeatureEngineeringPipeline
from trading_system.models.base.model_factory import ModelFactory
from trading_system.models.training.training_pipeline import TrainingPipeline
from trading_system.backtesting.engine import BacktestEngine
from trading_system.models.training.types import TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureComparisonWorkflow:
    """
    Workflow for comparing different feature sets using fixed optimal hyperparameters.
    """

    def __init__(self, config_path: str, test_mode: bool = False):
        """
        Initialize the feature comparison workflow.

        Args:
            config_path: Path to the feature comparison configuration file
            test_mode: Whether to run in test mode (shorter periods, fewer trials)
        """
        self.config_path = config_path
        self.test_mode = test_mode
        self.config = self._load_config()
        self.results = []
        self.best_result = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _create_feature_configs(self) -> List[Tuple[str, FeatureConfig]]:
        """
        Create feature configuration objects for comparison.

        Returns:
            List of (name, FeatureConfig) tuples
        """
        feature_configs = []

        base_config = self.config['base_feature_config']
        feature_variations = self.config['feature_variations']

        for variation in feature_variations:
            name = variation['name']
            description = variation.get('description', '')

            # Create feature config by merging base config with variation
            config_dict = base_config.copy()
            config_dict.update(variation['parameters'])

            try:
                feature_config = FeatureConfig(**config_dict)
                feature_configs.append((name, feature_config))
                logger.info(f"Created feature config: {name} - {description}")

            except Exception as e:
                logger.error(f"Failed to create feature config for {name}: {e}")
                continue

        return feature_configs

    def _get_optimal_hyperparameters(self) -> Dict[str, Any]:
        """
        Get optimal hyperparameters for model training.
        In real usage, these would come from previous optimization results.

        Returns:
            Dictionary of optimal hyperparameters
        """
        if 'optimal_hyperparameters' in self.config:
            return self.config['optimal_hyperparameters']

        # Default optimal hyperparameters based on model type
        model_type = self.config['model_config']['model_type']

        if model_type == 'xgboost':
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        elif model_type == 'lstm':
            return {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'sequence_length': 30,
                'num_epochs': 100,
                'random_state': 42
            }
        elif model_type == 'ff5_regression':
            return {
                'regularization': 'ridge',
                'alpha': 1.0,
                'standardize': True,
                'fit_intercept': True
            }
        else:
            logger.warning(f"Unknown model type {model_type}, using empty hyperparameters")
            return {}

    def _run_single_experiment(self, feature_name: str, feature_config: FeatureConfig,
                              optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with given feature configuration.

        Args:
            feature_name: Name of the feature configuration
            feature_config: Feature configuration object
            optimal_params: Optimal hyperparameters to use

        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Running experiment for feature set: {feature_name}")

        try:
            # Initialize components
            data_provider = self._create_data_provider()
            training_config = self._create_training_config(feature_config)

            # Run feature engineering
            logger.info(f"Computing features for {feature_name}")
            feature_pipeline = FeatureEngineeringPipeline(feature_config)

            # Get symbols and date range
            symbols = self.config['data_config']['symbols'][:5] if self.test_mode else self.config['data_config']['symbols']
            start_date = "2023-01-01" if self.test_mode else self.config['data_config']['start_date']
            end_date = "2023-06-30" if self.test_mode else self.config['data_config']['end_date']

            # Compute features for each symbol
            all_features = []
            for symbol in symbols:
                try:
                    # Get price data
                    price_data_dict = data_provider.get_historical_data(symbol, start_date, end_date)

                    if price_data_dict is None or symbol not in price_data_dict:
                        logger.warning(f"No data returned for {symbol}, skipping")
                        continue

                    price_data = price_data_dict[symbol]

                    if price_data is None or len(price_data) < 30:
                        logger.warning(f"Insufficient data for {symbol} ({len(price_data)} rows), skipping")
                        continue

                    # Create data dict for pipeline (expects dict mapping symbols to DataFrames)
                    data_dict = {'price_data': {symbol: price_data}}

                    # Fit pipeline on this data and transform to get features
                    feature_pipeline.fit(data_dict)
                    features = feature_pipeline.transform(data_dict)

                    # Store features
                    features['symbol'] = symbol
                    all_features.append(features)

                except Exception as e:
                    logger.warning(f"Failed to compute features for {symbol}: {e}")
                    continue

            if not all_features:
                raise ValueError("No features computed successfully")

            # Combine all features
            features = pd.concat(all_features, ignore_index=True)

            # Run training with fixed hyperparameters
            logger.info(f"Training model with {feature_name} features")
            model_type = self.config['model_config']['model_type']
            training_pipeline = TrainingPipeline(
                model_type=model_type,
                feature_pipeline=feature_pipeline,
                config=training_config
            ).configure_data(data_provider)

            # Convert date strings to datetime objects
            from datetime import datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            training_result = training_pipeline.run_pipeline(
                start_date=start_dt,
                end_date=end_dt,
                symbols=symbols,
                model_name=f"feature_comparison_{feature_name}",
                hyperparameters=optimal_params
            )

            # Extract performance metrics from training results
            logger.info(f"Extracting performance metrics for {feature_name}")

            # For now, use validation metrics from training result
            # TODO: Implement proper backtesting with BacktestEngine
            if hasattr(training_result, 'validation_metrics') and training_result.validation_metrics:
                metrics = training_result.validation_metrics
            elif isinstance(training_result, dict) and 'validation_metrics' in training_result:
                metrics = training_result['validation_metrics']
            else:
                # Create basic metrics if none available
                metrics = {
                    'val_score': getattr(training_result, 'val_score', 0.5),
                    'mse': getattr(training_result, 'mse', 1.0),
                    'r2_score': getattr(training_result, 'r2_score', 0.0),
                }

            result = {
                'feature_name': feature_name,
                'feature_config': feature_config,
                'training_result': training_result,
                'performance_metrics': metrics,
                'success': True
            }

            logger.info(f"Experiment completed for {feature_name}")
            logger.info(f"  Validation Score: {metrics.get('val_score', 'N/A'):.3f}")
            logger.info(f"  MSE: {metrics.get('mse', 'N/A'):.3f}")
            logger.info(f"  R² Score: {metrics.get('r2_score', 'N/A'):.3f}")

            return result

        except Exception as e:
            logger.error(f"Experiment failed for {feature_name}: {e}")
            import traceback
            traceback.print_exc()

            return {
                'feature_name': feature_name,
                'feature_config': feature_config,
                'error': str(e),
                'success': False
            }

    def _create_data_provider(self):
        """Create data provider."""
        from trading_system.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider(
            max_retries=3,
            retry_delay=1.0,
            request_timeout=30,
            cache_enabled=True
        )

    def _create_model(self, hyperparameters: Dict[str, Any]):
        """Create model with hyperparameters."""
        model_config = self.config['model_config']
        model_type = model_config['model_type']

        factory = ModelFactory()
        model = factory.create_model(model_type, model_config.get('config', {}))

        # Set hyperparameters if model supports it
        if hasattr(model, 'set_hyperparameters'):
            model.set_hyperparameters(hyperparameters)

        return model

    def _create_training_config(self, feature_config: FeatureConfig) -> TrainingConfig:
        """Create training configuration."""
        config_dict = self.config['training_config'].copy()

        # Create TrainingConfig object with proper parameters
        return TrainingConfig(
            use_cross_validation=config_dict.get('use_cross_validation', True),
            cv_folds=config_dict.get('cv_folds', 5),
            purge_period=config_dict.get('purge_period', 21),
            embargo_period=config_dict.get('embargo_period', 5),
            validation_split=config_dict.get('validation_split', 0.2),
            early_stopping=config_dict.get('early_stopping', True),
            early_stopping_patience=config_dict.get('early_stopping_patience', 10),
            metrics_to_compute=config_dict.get('metrics_to_compute', ['r2', 'mse', 'mae']),
            log_experiment=config_dict.get('log_experiment', False),
            experiment_name=f"feature_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=config_dict.get('tags', {})
        )

    def _create_backtest_config(self) -> Dict[str, Any]:
        """Create backtest configuration."""
        return self.config['backtest_config'].copy()

    def _compare_results(self) -> Dict[str, Any]:
        """
        Compare results from all feature configurations.

        Returns:
            Dictionary with comparison results
        """
        successful_results = [r for r in self.results if r['success']]

        if not successful_results:
            logger.error("No successful experiments to compare")
            return {
                'comparison_table': pd.DataFrame(),
                'best_feature_name': None,
                'best_result': None,
                'primary_metric': 'val_score',
                'total_experiments': len(self.results),
                'successful_experiments': 0,
                'error': 'No successful experiments'
            }

        # Create comparison DataFrame
        comparison_data = []
        for result in successful_results:
            metrics = result['performance_metrics']
            comparison_data.append({
                'feature_name': result['feature_name'],
                'val_score': metrics.get('val_score', np.nan),
                'mse': metrics.get('mse', np.nan),
                'r2_score': metrics.get('r2_score', np.nan),
                'mae': metrics.get('mae', np.nan)
            })

        df = pd.DataFrame(comparison_data)

        # Sort by primary objective (validation score by default)
        primary_metric = self.config.get('comparison', {}).get('primary_metric', 'val_score')
        ascending = self.config.get('comparison', {}).get('ascending', False)

        if primary_metric not in df.columns:
            primary_metric = 'val_score'  # fallback to validation score

        df_sorted = df.sort_values(primary_metric, ascending=ascending)

        # Identify best result
        best_row = df_sorted.iloc[0]
        best_feature_name = best_row['feature_name']
        self.best_result = next(r for r in successful_results if r['feature_name'] == best_feature_name)

        comparison_results = {
            'comparison_table': df_sorted,
            'best_feature_name': best_feature_name,
            'best_result': self.best_result,
            'primary_metric': primary_metric,
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results)
        }

        logger.info(f"Comparison complete. Best feature set: {best_feature_name}")
        logger.info(f"Best {primary_metric}: {best_row[primary_metric]:.3f}")

        return comparison_results

    def _save_results(self, comparison_results: Dict[str, Any]) -> None:
        """Save results to files."""
        output_dir = Path(self.config.get('output', {}).get('directory', './feature_comparison_results'))
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save comparison table
        comparison_file = output_dir / f'feature_comparison_{timestamp}.csv'
        if not comparison_results['comparison_table'].empty:
            comparison_results['comparison_table'].to_csv(comparison_file, index=False)
            logger.info(f"Comparison table saved to {comparison_file}")
        else:
            # Create empty table with headers
            pd.DataFrame(columns=['feature_name', 'val_score', 'mse', 'r2_score', 'mae']).to_csv(comparison_file, index=False)
            logger.info(f"Empty comparison table saved to {comparison_file}")

        # Save detailed results
        detailed_results = []
        for result in self.results:
            if result['success']:
                detailed_results.append({
                    'feature_name': result['feature_name'],
                    'performance_metrics': result['performance_metrics'],
                    'success': True
                })
            else:
                detailed_results.append({
                    'feature_name': result['feature_name'],
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })

        results_file = output_dir / f'detailed_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {results_file}")

        # Save best feature configuration
        if self.best_result:
            best_config_file = output_dir / f'best_feature_config_{timestamp}.yaml'
            with open(best_config_file, 'w') as f:
                yaml.dump(self.best_result['feature_config'].__dict__, f, default_flow_style=False)
            logger.info(f"Best feature configuration saved to {best_config_file}")

        # Save summary report
        summary_file = output_dir / f'summary_report_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Feature Comparison Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")

            f.write(f"Configuration: {self.config_path}\n")
            f.write(f"Total Experiments: {comparison_results['total_experiments']}\n")
            f.write(f"Successful Experiments: {comparison_results['successful_experiments']}\n")
            f.write(f"Primary Metric: {comparison_results['primary_metric']}\n\n")

            if self.best_result:
                f.write(f"Best Feature Set: {comparison_results['best_feature_name']}\n")
                best_metrics = self.best_result['performance_metrics']
                f.write(f"Best Validation Score: {best_metrics.get('val_score', 'N/A'):.3f}\n")
                f.write(f"Best MSE: {best_metrics.get('mse', 'N/A'):.3f}\n")
                f.write(f"Best R² Score: {best_metrics.get('r2_score', 'N/A'):.3f}\n\n")
            else:
                f.write("No successful experiments - no best feature set identified\n\n")

            if not comparison_results['comparison_table'].empty:
                f.write("Full Results:\n")
                f.write(comparison_results['comparison_table'].to_string())
            else:
                f.write("No successful experiments to display\n")

            if 'error' in comparison_results:
                f.write(f"\nError: {comparison_results['error']}\n")

        logger.info(f"Summary report saved to {summary_file}")

    def run(self) -> Dict[str, Any]:
        """
        Run the complete feature comparison workflow.

        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting feature comparison workflow")

        # Create feature configurations
        feature_configs = self._create_feature_configs()
        if not feature_configs:
            raise ValueError("No valid feature configurations created")

        # Get optimal hyperparameters
        optimal_params = self._get_optimal_hyperparameters()
        logger.info(f"Using optimal hyperparameters: {optimal_params}")

        # Run experiments
        for feature_name, feature_config in feature_configs:
            result = self._run_single_experiment(feature_name, feature_config, optimal_params)
            self.results.append(result)

        # Compare results
        comparison_results = self._compare_results()

        # Save results
        self._save_results(comparison_results)

        logger.info("Feature comparison workflow completed successfully")
        return comparison_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Feature Comparison Workflow')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to feature comparison configuration file')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (shorter periods, fewer trials)')

    args = parser.parse_args()

    try:
        workflow = FeatureComparisonWorkflow(args.config, args.test_mode)
        results = workflow.run()

        print("\n" + "=" * 60)
        if results['successful_experiments'] > 0:
            print("🎉 Feature comparison completed successfully!")
            print(f"Best feature set: {results['best_feature_name']}")
            print(f"Best {results['primary_metric']}: {results['comparison_table'].iloc[0][results['primary_metric']]:.3f}")
        else:
            print("💥 Feature comparison completed with errors!")
            print("No successful experiments - check logs for details")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Feature comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())