"""
Performance Evaluator for ML Models

Utility class for evaluating model performance with ML-specific metrics.
This works alongside the general performance utilities to provide
model-specific evaluation capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from scipy import stats
import warnings

from ..base.base_model import BaseModel
from ...utils.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive performance evaluator for ML models.

    Provides model-specific metrics including:
    - Regression metrics (R², MSE, MAE, etc.)
    - Classification metrics (accuracy, precision, etc.)
    - Financial metrics (IC, Rank IC, etc.)
    - Model comparison capabilities
    """

    @staticmethod
    def evaluate_model(model: BaseModel, X: pd.DataFrame, y: pd.Series,
                      task_type: str = 'auto', sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model to evaluate
            X: Feature DataFrame
            y: Target Series
            task_type: 'regression', 'classification', or 'auto'
            sample_weight: Optional sample weights

        Returns:
            Dictionary of performance metrics
        """
        if model.status != "trained":
            raise ValueError("Model must be trained before evaluation")

        try:
            # Make predictions
            predictions = model.predict(X)

            # Determine task type if auto
            if task_type == 'auto':
                task_type = PerformanceEvaluator._determine_task_type(y)

            # Calculate metrics based on task type
            if task_type == 'regression':
                metrics = PerformanceEvaluator._regression_metrics(y, predictions, sample_weight)
            elif task_type == 'classification':
                # For classification, we need class predictions
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if len(proba.shape) == 2 and proba.shape[1] == 2:
                        proba = proba[:, 1]  # Binary classification
                    class_pred = (proba > 0.5).astype(int)
                else:
                    class_pred = predictions

                metrics = PerformanceEvaluator._classification_metrics(y, class_pred, proba if 'proba' in locals() else None)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Add model-specific information
            metrics.update({
                'model_type': model.model_type,
                'training_samples': model.metadata.training_samples,
                'feature_count': len(model.metadata.features) if model.metadata.features else 0,
                'evaluation_samples': len(y)
            })

            # Add feature importance if available
            feature_importance = model.get_feature_importance()
            if feature_importance:
                metrics['top_feature_importance'] = max(feature_importance.values()) if feature_importance else 0.0

            logger.info(f"Evaluated {model.model_type}: {task_type} task, {metrics.get('r2', 'N/A')} R²")
            return metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return PerformanceEvaluator._empty_evaluation_metrics()

    @staticmethod
    def evaluate_financial_model(model: BaseModel, X: pd.DataFrame, y: pd.Series,
                                returns: Optional[pd.Series] = None,
                                benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate financial model with finance-specific metrics.

        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target returns
            returns: Actual returns (if different from y)
            benchmark_returns: Benchmark returns for relative metrics

        Returns:
            Dictionary of financial performance metrics
        """
        try:
            predictions = model.predict(X)

            # Use provided returns or target as returns
            actual_returns = returns if returns is not None else y

            # Basic ML metrics
            ml_metrics = PerformanceEvaluator._regression_metrics(actual_returns, predictions)

            # Financial-specific metrics
            financial_metrics = {}

            # Information Coefficient (IC)
            ic = np.corrcoef(actual_returns, predictions)[0, 1]
            financial_metrics['information_coefficient'] = ic if not np.isnan(ic) else 0.0

            # Rank IC (correlation of ranks)
            rank_ic = stats.spearmanr(actual_returns, predictions)[0]
            financial_metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0.0

            # Predictive accuracy (percentage of correct directional predictions)
            if len(actual_returns) > 1:
                direction_actual = (actual_returns > 0).astype(int)
                direction_pred = (predictions > 0).astype(int)
                directional_accuracy = (direction_actual == direction_pred).mean()
                financial_metrics['directional_accuracy'] = directional_accuracy
            else:
                financial_metrics['directional_accuracy'] = 0.0

            # Portfolio metrics (if we can construct portfolio from predictions)
            if len(predictions) > 10:  # Need sufficient observations
                # Create long-short portfolio based on predictions
                top_quintile = predictions >= np.percentile(predictions, 80)
                bottom_quintile = predictions <= np.percentile(predictions, 20)

                if top_quintile.sum() > 0 and bottom_quintile.sum() > 0:
                    portfolio_returns = actual_returns[top_quintile].mean() - actual_returns[bottom_quintile].mean()
                    financial_metrics['long_short_return'] = portfolio_returns

                    # Calculate Sharpe ratio for this portfolio
                    if len(actual_returns[top_quintile]) > 1:
                        long_short_series = pd.Series([
                            actual_returns[top_quintile].iloc[i] - actual_returns[bottom_quintile].iloc[i]
                            for i in range(min(len(actual_returns[top_quintile]), len(actual_returns[bottom_quintile])))
                        ])
                        financial_metrics['long_short_sharpe'] = PerformanceMetrics.sharpe_ratio(long_short_series)

            # Combine all metrics
            all_metrics = {**ml_metrics, **financial_metrics}

            # Add portfolio-level metrics if returns and benchmark are provided
            if benchmark_returns is not None and len(actual_returns) == len(benchmark_returns):
                portfolio_metrics = PerformanceMetrics.calculate_all_metrics(
                    actual_returns, benchmark_returns
                )
                # Add prefix to avoid conflicts
                for key, value in portfolio_metrics.items():
                    all_metrics[f'portfolio_{key}'] = value

            logger.info(f"Financial evaluation completed: IC={financial_metrics.get('information_coefficient', 0):.3f}")
            return all_metrics

        except Exception as e:
            logger.error(f"Financial model evaluation failed: {e}")
            return PerformanceEvaluator._empty_financial_metrics()

    @staticmethod
    def compare_models(models: List[BaseModel], X: pd.DataFrame, y: pd.Series,
                      task_type: str = 'auto') -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            models: List of trained models to compare
            X: Feature DataFrame
            y: Target Series
            task_type: Task type for evaluation

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model in models:
            try:
                metrics = PerformanceEvaluator.evaluate_model(model, X, y, task_type)
                metrics['model_name'] = model.__class__.__name__
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to evaluate model {model.__class__.__name__}: {e}")
                continue

        if not results:
            logger.warning("No models could be evaluated")
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)

        # Add rankings
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['model_name', 'training_samples', 'feature_count', 'evaluation_samples']:
                try:
                    ranking = comparison_df[col].rank(ascending=False, method='min')
                    comparison_df[f'{col}_rank'] = ranking.astype(int)
                except Exception:
                    continue

        return comparison_df

    @staticmethod
    def cross_validate_model(model_class, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5, model_config: Optional[Dict] = None,
                           task_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform cross-validation on a model class.

        Args:
            model_class: Model class to evaluate (not instance)
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of CV folds
            model_config: Configuration for model initialization
            task_type: Task type for evaluation

        Returns:
            Dictionary with CV results
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, KFold

            # Use TimeSeriesSplit for financial data
            if len(X) > 100:  # Use time series split for larger datasets
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            fold_results = []
            models = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
                try:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Initialize and train model
                    model = model_class(config=model_config or {})
                    model.fit(X_train, y_train)

                    # Evaluate on validation set
                    metrics = PerformanceEvaluator.evaluate_model(model, X_val, y_val, task_type)
                    metrics['fold'] = fold + 1
                    fold_results.append(metrics)
                    models.append(model)

                except Exception as e:
                    logger.warning(f"CV fold {fold + 1} failed: {e}")
                    continue

            if not fold_results:
                return {'error': 'All CV folds failed'}

            # Aggregate results
            results_df = pd.DataFrame(fold_results)
            numeric_columns = results_df.select_dtypes(include=[np.number]).columns

            summary = {
                'cv_folds': len(fold_results),
                'mean_metrics': results_df[numeric_columns].mean().to_dict(),
                'std_metrics': results_df[numeric_columns].std().to_dict(),
                'fold_results': fold_results,
                'models': models
            }

            logger.info(f"Cross-validation completed: {len(fold_results)} successful folds")
            return summary

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def _determine_task_type(y: pd.Series) -> str:
        """Determine if this is regression or classification task."""
        # Check if target is binary or multi-class
        unique_values = y.nunique()

        if unique_values <= 20 and y.dtype in ['int64', 'int32', 'bool']:
            # Likely classification
            return 'classification'
        else:
            # Likely regression
            return 'regression'

    @staticmethod
    def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray,
                          sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate regression metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics = {
                'r2': r2_score(y_true, y_pred, sample_weight=sample_weight),
                'mse': mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
                'mae': mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
                'mape': mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
            }

            # Add correlation-based metrics
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0

            # Add custom financial metrics
            if len(y_true) > 1:
                # Directional accuracy
                direction_correct = ((y_true > 0) == (pd.Series(y_pred, index=y_true.index) > 0)).mean()
                metrics['directional_accuracy'] = direction_correct

            return metrics

    @staticmethod
    def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

        # Add AUC if probabilities are available
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except Exception:
                metrics['auc_roc'] = 0.0

        return metrics

    @staticmethod
    def _empty_evaluation_metrics() -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'r2': 0.0, 'mse': float('inf'), 'rmse': float('inf'),
            'mae': float('inf'), 'mape': float('inf'),
            'correlation': 0.0, 'directional_accuracy': 0.0,
            'model_type': '', 'training_samples': 0,
            'feature_count': 0, 'evaluation_samples': 0
        }

    @staticmethod
    def _empty_financial_metrics() -> Dict[str, float]:
        """Return empty financial metrics dictionary."""
        base_metrics = PerformanceEvaluator._empty_evaluation_metrics()
        financial_metrics = {
            'information_coefficient': 0.0,
            'rank_ic': 0.0,
            'directional_accuracy': 0.0,
            'long_short_return': 0.0,
            'long_short_sharpe': 0.0
        }
        return {**base_metrics, **financial_metrics}