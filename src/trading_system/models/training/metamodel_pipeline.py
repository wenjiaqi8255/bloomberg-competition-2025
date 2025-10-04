"""
MetaModel Training Pipeline

This module provides a specialized training pipeline for MetaModel that extends
the existing TrainingPipeline infrastructure while leveraging the StrategyDataCollector
for data preparation.

Key Features:
- Extends existing TrainingPipeline
- Integrates with StrategyDataCollector
- Supports MetaModel-specific configuration
- Maintains compatibility with existing infrastructure
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .training_pipeline import TrainingPipeline
from .metamodel_config import MetaModelTrainingConfig
from .trainer import TrainingResult
from ..model_persistence import ModelRegistry
from ...feature_engineering.pipeline import FeatureEngineeringPipeline
from ...data.strategy_data_collector import StrategyDataCollector
from ...orchestration.meta_model import MetaModel

logger = logging.getLogger(__name__)


class MetaModelTrainingPipeline(TrainingPipeline):
    """
    Specialized training pipeline for MetaModel that extends TrainingPipeline.

    This pipeline provides MetaModel-specific functionality while maintaining
    compatibility with the existing training infrastructure.
    """

    def __init__(self,
                 config: MetaModelTrainingConfig,
                 registry_path: Optional[str] = None):
        """
        Initialize the MetaModel training pipeline.

        Args:
            config: MetaModel-specific training configuration
            registry_path: Path for model registry
        """
        # Create a minimal feature pipeline (MetaModel doesn't need complex feature engineering)
        feature_pipeline = FeatureEngineeringPipeline.from_config(config.feature_config)

        # Initialize parent with converted training config
        super().__init__(
            model_type=f"metamodel_{config.method}",
            feature_pipeline=feature_pipeline,
            config=config.to_training_config(),
            registry_path=registry_path or "./models/"
        )

        # Store MetaModel-specific config
        self.metamodel_config = config

        # Initialize strategy data collector
        self.data_collector = StrategyDataCollector()

    def run_metamodel_pipeline(self,
                             model_name: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Run the complete MetaModel training pipeline.

        This method provides a MetaModel-specific workflow that:
        1. Collects strategy historical returns data
        2. Prepares training data using StrategyDataCollector
        3. Trains MetaModel using existing TrainingPipeline infrastructure
        4. Registers the trained model with appropriate artifacts

        Args:
            model_name: Optional name for model registration
            **kwargs: Additional pipeline parameters

        Returns:
            Pipeline execution results with MetaModel-specific information
        """
        logger.info(f"Starting MetaModel training pipeline for {self.metamodel_config.method}")

        # Generate model name if not provided
        if model_name is None:
            model_name = f"metamodel_{self.metamodel_config.method}"

        try:
            # Step 1: Collect strategy returns data
            logger.info("Step 1: Collecting strategy returns data...")
            strategy_returns, target_returns = self._collect_strategy_data(**kwargs)

            # Step 2: Prepare training data (convert to TrainingPipeline format)
            logger.info("Step 2: Preparing MetaModel training data...")
            X, y = self._prepare_metamodel_training_data(strategy_returns, target_returns)

            # Step 3: Create and train MetaModel using BaseModel interface
            logger.info("Step 3: Training MetaModel...")
            metamodel = self._create_metamodel()
            training_result = self._train_metamodel(metamodel, X, y)

            # Step 4: Register model with MetaModel-specific artifacts
            logger.info("Step 4: Registering MetaModel and artifacts...")
            model_id = self._register_metamodel(training_result, model_name)

            # Step 5: Generate MetaModel-specific report
            logger.info("Step 5: Generating MetaModel pipeline report...")
            pipeline_result = self._generate_metamodel_report(
                model_id, training_result, strategy_returns, X, y
            )

            logger.info(f"MetaModel training pipeline completed successfully. Model ID: {model_id}")
            return pipeline_result

        except Exception as e:
            logger.error(f"MetaModel training pipeline failed: {e}")
            raise RuntimeError(f"MetaModel pipeline failed: {e}")

    def _collect_strategy_data(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Collect strategy returns data using StrategyDataCollector.

        Args:
            **kwargs: Additional collection parameters

        Returns:
            Tuple of (strategy_returns_df, target_returns_series)
        """
        config = self.metamodel_config

        if config.data_source == 'backtest':
            logger.info(f"Collecting data from backtest results for strategies: {config.strategies}")

            strategy_returns, target_returns = self.data_collector.collect_from_backtest_results(
                strategy_names=config.strategies,
                start_date=config.start_date,
                end_date=config.end_date,
                target_benchmark=config.target_benchmark
            )

        elif config.data_source == 'synthetic':
            logger.info("Generating synthetic strategy data for testing")

            # Generate synthetic strategy configurations
            strategies_config = {}
            for strategy in config.strategies:
                strategies_config[strategy] = {
                    'annual_return': 0.08 + (hash(strategy) % 10) * 0.01,  # Varied returns
                    'annual_volatility': 0.12 + (hash(strategy) % 8) * 0.01,  # Varied volatilities
                    'correlation_factor': 0.2 + (hash(strategy) % 5) * 0.1  # Varied correlations
                }

            strategy_returns, target_returns = self.data_collector.create_synthetic_strategy_data(
                strategies_config=strategies_config,
                start_date=config.start_date,
                end_date=config.end_date
            )

        elif config.data_source == 'live':
            # For live data, would need to collect from live trading results
            raise NotImplementedError("Live data collection not yet implemented for MetaModel training")

        else:
            raise ValueError(f"Unsupported data source: {config.data_source}")

        logger.info(f"Collected strategy data: {strategy_returns.shape}, target data: {target_returns.shape}")
        return strategy_returns, target_returns

    def _prepare_metamodel_training_data(self,
                                       strategy_returns: pd.DataFrame,
                                       target_returns: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for MetaModel.

        For MetaModel, the features are just the strategy returns and the target
        is the portfolio returns. This method handles the data preparation.

        Args:
            strategy_returns: DataFrame of strategy returns (strategies as columns)
            target_returns: Series of target portfolio returns

        Returns:
            Tuple of (X, y) in sklearn format
        """
        # For MetaModel, we don't need complex feature engineering
        # The strategy returns are already in the correct format

        # Align data and remove NaN values
        aligned_data = pd.concat([strategy_returns, target_returns.rename('target')], axis=1, join='inner')
        aligned_data = aligned_data.dropna()

        if aligned_data.empty:
            raise ValueError("No overlapping data found after alignment and NaN removal")

        # Separate features and target
        X = aligned_data.drop(columns=['target'])
        y = aligned_data['target']

        logger.info(f"Prepared MetaModel training data: {len(X)} samples, {len(X.columns)} strategies")
        return X, y

    def _create_metamodel(self) -> MetaModel:
        """
        Create MetaModel instance based on configuration.

        Returns:
            Configured MetaModel instance
        """
        config = self.metamodel_config

        metamodel = MetaModel(
            method=config.method,
            alpha=config.alpha,
            positive_weights=config.positive_weights,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            weight_sum_constraint=config.weight_sum_constraint
        )

        # Set strategy weights if provided
        if config.strategy_weights:
            metamodel.strategy_weights = config.strategy_weights

        # Add metadata
        if config.tags:
            if isinstance(config.tags, list):
                # Convert list of tags to dict format
                for i, tag in enumerate(config.tags):
                    metamodel.metadata.tags[f'tag_{i+1}'] = tag
            elif isinstance(config.tags, dict):
                metamodel.metadata.tags.update(config.tags)
        metamodel.metadata.tags['pipeline_type'] = 'metamodel_training'
        metamodel.metadata.tags['data_source'] = config.data_source

        logger.info(f"Created MetaModel with method '{config.method}'")
        return metamodel

    def _train_metamodel(self, metamodel: MetaModel, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """
        Train MetaModel using existing trainer infrastructure.

        Args:
            metamodel: MetaModel instance to train
            X: Feature matrix (strategy returns)
            y: Target series (portfolio returns)

        Returns:
            Training result with MetaModel-specific information
        """
        logger.info("Training MetaModel using BaseModel interface...")

        # Use BaseModel's fit method
        trained_metamodel = metamodel.fit(X, y)

        # Calculate performance metrics using in-sample prediction
        # For MetaModel, we use the trained weights to compute portfolio returns
        predictions = self._compute_portfolio_returns(trained_metamodel, X)
        metrics = self._calculate_metamodel_metrics(y, predictions)

        # Create training result
        training_result = TrainingResult(
            model=trained_metamodel,
            training_time=0.0,  # Not tracked for simple MetaModel training
            validation_metrics=metrics,
            test_metrics=metrics  # For MetaModel, validation and test are the same
        )

        logger.info(f"MetaModel training completed. Strategy weights: {trained_metamodel.strategy_weights}")
        return training_result

    def _compute_portfolio_returns(self, metamodel: MetaModel, X: pd.DataFrame) -> np.ndarray:
        """
        Compute portfolio returns using trained MetaModel weights.

        Args:
            metamodel: Trained MetaModel
            X: Feature matrix (strategy returns)

        Returns:
            Portfolio returns as numpy array
        """
        if not metamodel.strategy_weights:
            raise ValueError("MetaModel has no strategy weights")

        # Compute weighted portfolio returns
        portfolio_returns = np.zeros(len(X))
        for strategy, weight in metamodel.strategy_weights.items():
            if strategy in X.columns:
                portfolio_returns += X[strategy].values * weight

        return portfolio_returns

    def _calculate_metamodel_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for MetaModel.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of performance metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

        # Add strategy-specific metrics if configured
        if self.metamodel_config.track_strategy_correlation:
            # Calculate correlation between predictions and individual strategies
            strategy_correlations = {}
            for strategy in self.metamodel_config.strategies:
                if hasattr(self, '_last_strategy_returns') and strategy in self._last_strategy_returns:
                    corr = pd.Series(y_pred).corr(self._last_strategy_returns[strategy])
                    strategy_correlations[f'corr_{strategy}'] = corr

            metrics.update(strategy_correlations)

        return metrics

    def _register_metamodel(self, training_result: TrainingResult, model_name: str) -> str:
        """
        Register trained MetaModel with appropriate artifacts.

        Args:
            training_result: Training result containing trained model
            model_name: Name for model registration

        Returns:
            Model ID from registry
        """
        config = self.metamodel_config

        # Prepare MetaModel-specific artifacts
        artifacts = {
            'metamodel_config': config.to_dict(),
            'strategies': config.strategies,
            'training_period': {
                'start': config.start_date.isoformat() if config.start_date else None,
                'end': config.end_date.isoformat() if config.end_date else None
            },
            'feature_pipeline': self.feature_pipeline  # Even though minimal, include for consistency
        }

        # Update model metadata with training metrics
        if training_result.validation_metrics:
            training_result.model.update_metadata(
                performance_metrics=training_result.validation_metrics
            )

        # Add strategy-specific metadata
        training_result.model.update_metadata(
            strategy_count=len(config.strategies),
            metamodel_method=config.method,
            data_source=config.data_source
        )

        # Register model
        model_id = self.registry.save_model_with_artifacts(
            model=training_result.model,
            model_name=model_name,
            artifacts=artifacts,
            tags={
                'model_type': 'metamodel',
                'method': config.method,
                'pipeline_type': 'metamodel_training',
                'training_date': datetime.now().isoformat()
            }
        )

        return model_id

    def _generate_metamodel_report(self,
                                 model_id: str,
                                 training_result: TrainingResult,
                                 strategy_returns: pd.DataFrame,
                                 X: pd.DataFrame,
                                 y: pd.Series) -> Dict[str, Any]:
        """
        Generate MetaModel-specific pipeline report.

        Args:
            model_id: ID of registered model
            training_result: Training results
            strategy_returns: Original strategy returns data
            X: Feature matrix used for training
            y: Target series used for training

        Returns:
            Comprehensive pipeline report
        """
        base_report = self._generate_pipeline_report(model_id, training_result, {}, X)

        # Add MetaModel-specific information
        metamodel_info = {
            'metamodel_config': self.metamodel_config.to_dict(),
            'strategy_info': {
                'strategies': list(strategy_returns.columns),
                'strategy_correlations': strategy_returns.corr().to_dict(),
                'strategy_returns_stats': strategy_returns.describe().to_dict()
            },
            'training_data_info': {
                'training_samples': len(X),
                'strategy_count': len(X.columns),
                'training_period': {
                    'start': X.index.min().isoformat(),
                    'end': X.index.max().isoformat()
                }
            },
            'model_weights': training_result.model.strategy_weights,
            'performance_analysis': {
                'weight_distribution': self._analyze_weight_distribution(training_result.model.strategy_weights),
                'strategy_contributions': self._analyze_strategy_contributions(training_result.model, X)
            }
        }

        # Merge with base report
        base_report.update(metamodel_info)

        # Log to experiment tracking if available
        try:
            import wandb
            if wandb.run:
                wandb.log(base_report)
                logger.info("MetaModel pipeline report logged to Weights & Biases")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log MetaModel pipeline report: {e}")

        return base_report

    def _analyze_weight_distribution(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the distribution of strategy weights.

        Args:
            weights: Strategy weights dictionary

        Returns:
            Weight distribution analysis
        """
        if not weights:
            return {}

        weight_values = list(weights.values())

        return {
            'mean': sum(weight_values) / len(weight_values),
            'min': min(weight_values),
            'max': max(weight_values),
            'std': (sum((w - sum(weight_values)/len(weight_values))**2 for w in weight_values) / len(weight_values))**0.5,
            'concentration': max(weight_values),  # Largest weight
            'effective_n': sum(w**2 for w in weight_values)**-1,  # Effective number of strategies
            'weights': weights
        }

    def _analyze_strategy_contributions(self, metamodel: MetaModel, X: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze individual strategy contributions to overall performance.

        Args:
            metamodel: Trained MetaModel
            X: Feature matrix (strategy returns)

        Returns:
            Strategy contribution analysis
        """
        if not metamodel.strategy_weights or X.empty:
            return {}

        contributions = {}
        for strategy, weight in metamodel.strategy_weights.items():
            if strategy in X.columns:
                # Weighted return contribution
                strategy_return = X[strategy].mean() * 252  # Annualized
                contributions[f'{strategy}_annualized_return'] = strategy_return
                contributions[f'{strategy}_contribution'] = strategy_return * weight

        return contributions