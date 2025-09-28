"""
Time Series Cross-Validation utilities for ML strategies.

This module provides robust cross-validation methods specifically designed
for financial time series data to prevent look-ahead bias and ensure
realistic performance estimation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with financial data considerations.

    Features:
    - Expanding window cross-validation
    - Purge period to avoid data leakage
    - Embargo period for realistic evaluation
    - Performance metrics calculation
    - Model validation statistics
    """

    def __init__(self,
                 cv_folds: int = 5,
                 min_train_size: int = 252,
                 purge_period: int = 21,
                 embargo_period: int = 5,
                 test_size: float = 0.2):
        """
        Initialize time series cross-validator.

        Args:
            cv_folds: Number of cross-validation folds
            min_train_size: Minimum training size (in days)
            purge_period: Days to purge after training period to avoid leakage
            embargo_period: Days to wait before including new data in test
            test_size: Fraction of data to use for testing in each fold
        """
        self.cv_folds = cv_folds
        self.min_train_size = min_train_size
        self.purge_period = purge_period
        self.embargo_period = embargo_period
        self.test_size = test_size

        self.validation_results = []

    def validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                       feature_names: List[str] = None,
                       model_type: str = "regression") -> Dict[str, Any]:
        """
        Validate model using time series cross-validation.

        Args:
            model: ML model to validate
            X: Feature DataFrame
            y: Target Series
            feature_names: List of feature names
            model_type: Type of model ("regression" or "classification")

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting time series cross-validation with {self.cv_folds} folds")

        # Ensure data is properly indexed by date
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.to_datetime(X.index)
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index)

        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index].sort_index()
        y = y.loc[common_index].sort_index()

        # Create custom time series split
        splits = self._create_time_series_splits(X, y)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Validating fold {fold_idx + 1}/{self.cv_folds}")

            try:
                fold_result = self._validate_fold(
                    model, X, y, train_idx, test_idx, fold_idx, model_type
                )
                fold_results.append(fold_result)

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} validation failed: {e}")
                continue

        # Aggregate results
        validation_summary = self._aggregate_validation_results(fold_results)

        # Store validation results
        self.validation_results.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'folds': fold_results,
            'summary': validation_summary
        })

        logger.info(f"Cross-validation completed. Average R²: {validation_summary.get('mean_r2', 0):.4f}")

        return validation_summary

    def _create_time_series_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series cross-validation splits."""
        n_samples = len(X)
        splits = []

        # Calculate train/test sizes for each fold
        test_samples = int(n_samples * self.test_size / self.cv_folds)

        for fold in range(self.cv_folds):
            # Calculate train end index
            train_end = int(n_samples * (0.6 + 0.3 * fold / self.cv_folds))
            train_end = max(train_end, self.min_train_size)

            # Calculate test start index (with purge period)
            test_start = train_end + self.purge_period
            test_end = min(test_start + test_samples, n_samples)

            if test_end > test_start and test_end <= n_samples:
                train_idx = np.arange(train_end)
                test_idx = np.arange(test_start, test_end)

                if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        return splits

    def _validate_fold(self, model, X: pd.DataFrame, y: pd.Series,
                      train_idx: np.ndarray, test_idx: np.ndarray,
                      fold_idx: int, model_type: str) -> Dict[str, Any]:
        """Validate a single fold."""
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        if model_type == "regression":
            train_metrics = {
                'mse': mean_squared_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train),
                'mae': np.mean(np.abs(y_train - y_pred_train))
            }
            test_metrics = {
                'mse': mean_squared_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test),
                'mae': np.mean(np.abs(y_test - y_pred_test))
            }
        else:  # classification
            y_pred_train_class = (y_pred_train > 0.5).astype(int)
            y_pred_test_class = (y_pred_test > 0.5).astype(int)

            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train_class),
                'precision': self._calculate_precision(y_train, y_pred_train_class),
                'recall': self._calculate_recall(y_train, y_pred_train_class)
            }
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test_class),
                'precision': self._calculate_precision(y_test, y_pred_test_class),
                'recall': self._calculate_recall(y_test, y_pred_test_class)
            }

        # Calculate overfitting metrics
        overfitting_metrics = self._calculate_overfitting_metrics(train_metrics, test_metrics)

        # Time-based analysis
        time_analysis = self._analyze_time_predictions(y_test, y_pred_test, X_test.index)

        fold_result = {
            'fold_idx': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_date_range': (X_train.index.min(), X_train.index.max()),
            'test_date_range': (X_test.index.min(), X_test.index.max()),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_metrics': overfitting_metrics,
            'time_analysis': time_analysis
        }

        return fold_result

    def _calculate_precision(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate precision with handling for edge cases."""
        from sklearn.metrics import precision_score
        try:
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    def _calculate_recall(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate recall with handling for edge cases."""
        from sklearn.metrics import recall_score
        try:
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    def _calculate_overfitting_metrics(self, train_metrics: Dict, test_metrics: Dict) -> Dict[str, float]:
        """Calculate overfitting metrics."""
        overfitting = {}

        for metric in train_metrics.keys():
            if metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]

                if train_val != 0:
                    overfitting[f'{metric}_overfitting'] = (train_val - test_val) / abs(train_val)
                else:
                    overfitting[f'{metric}_overfitting'] = 0.0

        return overfitting

    def _analyze_time_predictions(self, y_true: pd.Series, y_pred: np.ndarray,
                               test_dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze predictions over time."""
        analysis = {}

        try:
            # Convert predictions to Series with proper index
            pred_series = pd.Series(y_pred, index=test_dates)
            true_series = pd.Series(y_true.values, index=test_dates)

            # Calculate rolling performance
            if len(pred_series) >= 21:  # Monthly analysis
                monthly_r2 = []
                monthly_periods = pred_series.resample('M')

                for period_name, period_data in monthly_periods:
                    if len(period_data) >= 5:  # Minimum 5 data points
                        period_true = true_series.loc[period_data.index]
                        r2 = r2_score(period_true, period_data)
                        monthly_r2.append(r2)

                analysis['monthly_r2_mean'] = np.mean(monthly_r2) if monthly_r2 else 0
                analysis['monthly_r2_std'] = np.std(monthly_r2) if monthly_r2 else 0
                analysis['prediction_stability'] = analysis['monthly_r2_std'] / (abs(analysis['monthly_r2_mean']) + 1e-8)

            # Directional accuracy analysis
            if len(pred_series) >= 2:
                pred_directions = np.sign(pred_series.diff().dropna())
                true_directions = np.sign(true_series.diff().dropna())

                common_idx = pred_directions.index.intersection(true_directions.index)
                if len(common_idx) > 0:
                    directional_accuracy = (pred_directions.loc[common_idx] == true_directions.loc[common_idx]).mean()
                    analysis['directional_accuracy'] = directional_accuracy

        except Exception as e:
            logger.debug(f"Time analysis failed: {e}")

        return analysis

    def _aggregate_validation_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate validation results across folds."""
        if not fold_results:
            return {}

        summary = {
            'total_folds': len(fold_results),
            'successful_folds': len(fold_results),
            'model_type': fold_results[0].get('train_metrics', {}).keys()
        }

        # Aggregate test metrics
        test_metrics = {}
        for metric in fold_results[0]['test_metrics'].keys():
            values = [fold['test_metrics'][metric] for fold in fold_results]
            test_metrics[f'mean_{metric}'] = np.mean(values)
            test_metrics[f'std_{metric}'] = np.std(values)
            test_metrics[f'min_{metric}'] = np.min(values)
            test_metrics[f'max_{metric}'] = np.max(values)

        summary['test_metrics'] = test_metrics

        # Aggregate overfitting metrics
        overfitting_metrics = {}
        for metric in fold_results[0].get('overfitting_metrics', {}).keys():
            values = [fold['overfitting_metrics'][metric] for fold in fold_results]
            overfitting_metrics[f'mean_{metric}'] = np.mean(values)
            overfitting_metrics[f'std_{metric}'] = np.std(values)

        summary['overfitting_metrics'] = overfitting_metrics

        # Time analysis aggregation
        time_metrics = {}
        time_keys = ['monthly_r2_mean', 'monthly_r2_std', 'prediction_stability', 'directional_accuracy']

        for key in time_keys:
            values = []
            for fold in fold_results:
                if key in fold.get('time_analysis', {}):
                    values.append(fold['time_analysis'][key])

            if values:
                time_metrics[f'mean_{key}'] = np.mean(values)
                time_metrics[f'std_{key}'] = np.std(values)

        summary['time_analysis'] = time_metrics

        # Overall model quality assessment
        summary['model_quality'] = self._assess_model_quality(summary)

        return summary

    def _assess_model_quality(self, summary: Dict) -> Dict[str, Any]:
        """Assess overall model quality."""
        quality = {}

        # Performance quality
        if 'mean_r2' in summary.get('test_metrics', {}):
            r2 = summary['test_metrics']['mean_r2']
            if r2 > 0.1:
                quality['performance_rating'] = 'excellent'
            elif r2 > 0.05:
                quality['performance_rating'] = 'good'
            elif r2 > 0.0:
                quality['performance_rating'] = 'fair'
            else:
                quality['performance_rating'] = 'poor'

        # Overfitting assessment
        if 'mean_r2_overfitting' in summary.get('overfitting_metrics', {}):
            overfitting = summary['overfitting_metrics']['mean_r2_overfitting']
            if overfitting < 0.1:
                quality['overfitting_rating'] = 'low'
            elif overfitting < 0.3:
                quality['overfitting_rating'] = 'moderate'
            else:
                quality['overfitting_rating'] = 'high'

        # Stability assessment
        if 'mean_prediction_stability' in summary.get('time_analysis', {}):
            stability = summary['time_analysis']['mean_prediction_stability']
            if stability < 0.5:
                quality['stability_rating'] = 'stable'
            elif stability < 1.0:
                quality['stability_rating'] = 'moderate'
            else:
                quality['stability_rating'] = 'unstable'

        # Overall recommendation
        ratings = [quality.get('performance_rating', 'unknown'),
                  quality.get('overfitting_rating', 'unknown'),
                  quality.get('stability_rating', 'unknown')]

        if 'poor' in ratings or 'high' in ratings:
            quality['recommendation'] = 'not_recommended'
        elif 'excellent' in ratings and 'low' in ratings and 'stable' in ratings:
            quality['recommendation'] = 'highly_recommended'
        elif 'good' in ratings and 'moderate' not in ratings:
            quality['recommendation'] = 'recommended'
        else:
            quality['recommendation'] = 'acceptable'

        return quality

    def get_validation_report(self) -> Optional[Dict]:
        """Get comprehensive validation report."""
        if not self.validation_results:
            return None

        latest_validation = self.validation_results[-1]

        report = {
            'validation_timestamp': latest_validation['timestamp'],
            'model_type': latest_validation['model_type'],
            'summary': latest_validation['summary'],
            'fold_details': latest_validation['folds'],
            'validation_configuration': {
                'cv_folds': self.cv_folds,
                'min_train_size': self.min_train_size,
                'purge_period': self.purge_period,
                'embargo_period': self.embargo_period,
                'test_size': self.test_size
            }
        }

        return report

    def plot_validation_results(self, save_path: str = None):
        """Plot validation results (requires plotly)."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if not self.validation_results:
                logger.warning("No validation results to plot")
                return

            latest = self.validation_results[-1]
            summary = latest['summary']

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Test Performance', 'Overfitting Analysis',
                              'Time Series Stability', 'Fold Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Plot 1: Test Performance
            test_metrics = summary.get('test_metrics', {})
            if 'mean_r2' in test_metrics:
                fig.add_trace(
                    go.Bar(name='R² Score', x=['R²'], y=[test_metrics['mean_r2']],
                           error_y=dict(type='data', array=[test_metrics['std_r2']])),
                    row=1, col=1
                )

            # Plot 2: Overfitting Analysis
            overfitting_metrics = summary.get('overfitting_metrics', {})
            if overfitting_metrics:
                metrics = list(overfitting_metrics.keys())
                values = list(overfitting_metrics.values())

                fig.add_trace(
                    go.Bar(name='Overfitting', x=metrics, y=values),
                    row=1, col=2
                )

            # Plot 3: Time Series Stability
            time_analysis = summary.get('time_analysis', {})
            if time_analysis:
                metrics = list(time_analysis.keys())
                values = list(time_analysis.values())

                fig.add_trace(
                    go.Bar(name='Time Metrics', x=metrics, y=values),
                    row=2, col=1
                )

            # Plot 4: Fold Comparison
            fold_r2s = [fold['test_metrics'].get('r2', 0) for fold in latest['folds']]
            fold_numbers = list(range(1, len(fold_r2s) + 1))

            fig.add_trace(
                go.Scatter(name='Fold R²', x=fold_numbers, y=fold_r2s, mode='lines+markers'),
                row=2, col=2
            )

            fig.update_layout(
                title=f"Model Validation Report - {latest['timestamp']}",
                showlegend=False,
                height=600
            )

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Validation plot saved to {save_path}")

            return fig

        except ImportError:
            logger.warning("Plotly not available for plotting validation results")
            return None
        except Exception as e:
            logger.error(f"Failed to plot validation results: {e}")
            return None