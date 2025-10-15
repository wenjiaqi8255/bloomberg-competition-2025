"""
MetaModel Trainer - Simplified Component
=======================================

This component trains a metamodel on base model predictions.
Following single responsibility principle - only handles metamodel training coordination.
"""

import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from src.trading_system.metamodel.meta_model import MetaModel
from src.trading_system.utils.performance import PerformanceMetrics
from src.trading_system.data.enhanced_strategy_data_collector import EnhancedStrategyDataCollector, DataCollectionError

logger = logging.getLogger(__name__)


class MetaModelTrainer:
    """
    Simple metamodel trainer with single responsibility: train metamodels using real strategy returns.

    INPUTS:
    - model_results: List[Dict] - Results from trained base models (model_id, performance_metrics, etc.)
    - data_config: Dict - Configuration with periods, universe, etc.
    - method: str - Metamodel method ('ridge', 'lasso', 'equal')
    - alpha: float - Regularization parameter

    OUTPUTS:
    - metamodel: MetaModel - Trained metamodel instance
    - weights: Dict[str, float] - Strategy weights from metamodel
    - performance_metrics: Dict - Sharpe ratio, returns, etc.
    - combined_returns: pd.Series - Metamodel predictions
    - target_returns: pd.Series - Benchmark for comparison

    RESPONSIBILITY: Only coordinate metamodel training using real strategy returns.
    """

    def __init__(self, model_results: List[Dict[str, Any]], data_config: Dict[str, Any]):
        """
        Initialize the metamodel trainer.

        Args:
            model_results: List of results from trained base models
            data_config: Data configuration containing periods, universe, etc.
        """
        self.model_results = model_results
        self.data_config = data_config
        self.data_collector = EnhancedStrategyDataCollector(data_dir="./results")
        logger.info(f"MetaModelTrainer initialized with {len(model_results)} base models")

    def train_simple_metamodel(self, method: str = 'ridge', alpha: float = 1.0) -> Dict[str, Any]:
        """
        Train a simple metamodel using real strategy returns.

        Args:
            method: Metamodel method ('ridge', 'lasso', 'equal')
            alpha: Regularization parameter for ridge/lasso

        Returns:
            Dictionary containing trained metamodel and performance metrics
        """
        logger.info(f"Training simple metamodel: method={method}, alpha={alpha}")

        # Step 1: Collect real strategy returns
        strategy_returns = self._collect_strategy_returns()
        target_returns = self._create_target_returns(strategy_returns)

        # Step 2: Train metamodel
        metamodel = MetaModel(method=method, alpha=alpha)
        metamodel.fit(strategy_returns, target_returns)

        # Step 3: Evaluate performance
        combined_returns = metamodel.predict(strategy_returns)
        metrics = PerformanceMetrics.calculate_all_metrics(combined_returns, target_returns)

        logger.info(f"Metamodel trained successfully")
        logger.info(f"Strategy weights: {metamodel.strategy_weights}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0.0):.4f}")

        return {
            'metamodel': metamodel,
            'weights': metamodel.strategy_weights,
            'performance_metrics': metrics,
            'combined_returns': combined_returns,
            'target_returns': target_returns,
            'base_models': [m['model_id'] for m in self.model_results]
        }

    def _collect_strategy_returns(self) -> pd.DataFrame:
        """
        Collect real strategy returns from base models.

        Returns:
            DataFrame of strategy returns

        Raises:
            ValueError: If data collection fails
        """
        strategy_names = [result['model_id'] for result in self.model_results]
        start_date = datetime.fromisoformat(self.data_config['periods']['test']['start'])
        end_date = datetime.fromisoformat(self.data_config['periods']['test']['end'])

        try:
            strategy_returns, benchmark_returns = self.data_collector.collect_from_backtest_results(
                strategy_names=strategy_names,
                start_date=start_date,
                end_date=end_date
            )
            self._benchmark_returns = benchmark_returns
            return strategy_returns
        except DataCollectionError as e:
            raise ValueError(f"Failed to collect strategy returns: {e}")

    def _create_target_returns(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Create target returns from benchmark or equal-weighted portfolio.

        Args:
            strategy_returns: Real strategy returns

        Returns:
            Target returns series
        """
        if hasattr(self, '_benchmark_returns') and self._benchmark_returns is not None:
            return self._benchmark_returns
        return strategy_returns.mean(axis=1)
