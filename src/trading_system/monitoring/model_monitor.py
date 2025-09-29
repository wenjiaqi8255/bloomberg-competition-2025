"""
Model Monitoring and Validation System.

This module provides comprehensive model monitoring capabilities:
- Performance tracking and degradation detection
- Model drift detection
- Feature importance monitoring
- Prediction quality monitoring
- Automated alerts and reporting
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Comprehensive model monitoring system for trading strategies.

    This class provides:
    - Performance tracking over time
    - Model drift detection
    - Feature importance monitoring
    - Prediction quality analysis
    - Automated alerting
    """

    def __init__(self, monitor_path: str = "./monitoring/"):
        """
        Initialize model monitor.

        Args:
            monitor_path: Path to store monitoring data
        """
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(exist_ok=True, parents=True)

        # Performance history
        self.performance_history = {}
        self.model_metrics = {}
        self.feature_importance_history = {}
        self.prediction_stats = {}

        # Alert thresholds
        self.performance_threshold = 0.1  # 10% performance drop
        self.drift_threshold = 0.05     # 5% drift threshold
        self.prediction_threshold = 0.2 # 20% prediction change threshold

        # Load existing data
        self._load_monitoring_data()

    def _load_monitoring_data(self):
        """Load existing monitoring data."""
        try:
            # Load performance history
            perf_path = self.monitor_path / "performance_history.json"
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    self.performance_history = json.load(f)

            # Load model metrics
            metrics_path = self.monitor_path / "model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)

            # Load feature importance
            importance_path = self.monitor_path / "feature_importance.json"
            if importance_path.exists():
                with open(importance_path, 'r') as f:
                    self.feature_importance_history = json.load(f)

        except Exception as e:
            logger.warning(f"Error loading monitoring data: {e}")

    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        try:
            # Save performance history
            perf_path = self.monitor_path / "performance_history.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)

            # Save model metrics
            metrics_path = self.monitor_path / "model_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2, default=str)

            # Save feature importance
            importance_path = self.monitor_path / "feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance_history, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")

    def track_performance(self, model_name: str, timestamp: datetime,
                        predictions: np.ndarray, actuals: np.ndarray,
                        features: pd.DataFrame = None):
        """
        Track model performance over time.

        Args:
            model_name: Name of the model
            timestamp: Timestamp of prediction
            predictions: Model predictions
            actuals: Actual values
            features: Feature matrix (optional)
        """
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        if len(np.unique(predictions)) > 2:
            # Regression problem
            accuracy = None
        else:
            # Classification problem
            accuracy = accuracy_score(actuals, predictions > 0.5)

        # Create performance record
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'n_samples': len(predictions),
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'actual_mean': float(np.mean(actuals)),
            'actual_std': float(np.std(actuals))
        }

        # Store performance history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        self.performance_history[model_name].append(performance_record)

        # Check for performance degradation
        alerts = self._check_performance_degradation(model_name)

        # Track feature importance if provided
        if features is not None:
            self._track_feature_importance(model_name, timestamp, features)

        # Save monitoring data
        self._save_monitoring_data()

        return {
            'performance': performance_record,
            'alerts': alerts
        }

    def _check_performance_degradation(self, model_name: str) -> List[str]:
        """Check for performance degradation."""
        alerts = []

        if model_name not in self.performance_history:
            return alerts

        history = self.performance_history[model_name]
        if len(history) < 2:
            return alerts

        # Get recent performance
        recent_mse = history[-1]['mse']
        recent_r2 = history[-1]['r2']

        # Get baseline performance (average of last 10 records)
        baseline_records = history[-10:-1]
        if baseline_records:
            baseline_mse = np.mean([r['mse'] for r in baseline_records])
            baseline_r2 = np.mean([r['r2'] for r in baseline_records])

            # Check for significant degradation
            mse_change = (recent_mse - baseline_mse) / baseline_mse
            r2_change = (baseline_r2 - recent_r2) / abs(baseline_r2) if baseline_r2 != 0 else 0

            if mse_change > self.performance_threshold:
                alerts.append(f"MSE increased by {mse_change:.2%}")

            if r2_change > self.performance_threshold:
                alerts.append(f"R² decreased by {r2_change:.2%}")

        return alerts

    def _track_feature_importance(self, model_name: str, timestamp: datetime,
                                features: pd.DataFrame):
        """Track feature importance over time."""
        # Calculate simple feature importance using correlation with target
        if len(features.columns) == 0:
            return

        # Use variance as a simple importance measure
        importance = {}
        for col in features.columns:
            variance = features[col].var()
            if variance > 0:
                importance[col] = variance

        # Normalize importance
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        # Store feature importance
        if model_name not in self.feature_importance_history:
            self.feature_importance_history[model_name] = []

        importance_record = {
            'timestamp': timestamp.isoformat(),
            'importance': importance
        }

        self.feature_importance_history[model_name].append(importance_record)

        # Check for feature importance drift
        self._check_feature_drift(model_name)

    def _check_feature_drift(self, model_name: str) -> List[str]:
        """Check for feature importance drift."""
        alerts = []

        if model_name not in self.feature_importance_history:
            return alerts

        history = self.feature_importance_history[model_name]
        if len(history) < 2:
            return alerts

        # Get recent and baseline importance
        recent_importance = history[-1]['importance']
        baseline_importance = history[-2]['importance']

        # Check for significant changes
        for feature in recent_importance:
            if feature in baseline_importance:
                change = abs(recent_importance[feature] - baseline_importance[feature])
                if change > self.drift_threshold:
                    alerts.append(f"Feature importance drift detected for {feature}")

        return alerts

    def detect_model_drift(self, model_name: str, current_features: pd.DataFrame,
                          baseline_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Detect model drift using statistical methods.

        Args:
            model_name: Name of the model
            current_features: Current feature distribution
            baseline_features: Baseline feature distribution

        Returns:
            Dictionary with drift detection results
        """
        if baseline_features is None:
            # Use historical data as baseline
            if model_name not in self.feature_importance_history:
                return {'drift_detected': False, 'reason': 'No baseline data'}

            # For simplicity, use recent feature importance as baseline
            return {'drift_detected': False, 'reason': 'Baseline comparison not implemented'}

        # Calculate drift metrics
        drift_metrics = {}
        drift_detected = False

        # Compare feature distributions
        for feature in current_features.columns:
            if feature in baseline_features.columns:
                # Kolmogorov-Smirnov test for distribution difference
                try:
                    from scipy import stats
                    current_data = current_features[feature].dropna()
                    baseline_data = baseline_features[feature].dropna()

                    if len(current_data) > 10 and len(baseline_data) > 10:
                        ks_stat, p_value = stats.ks_2samp(current_data, baseline_data)
                        drift_metrics[feature] = {
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'drift_detected': p_value < 0.05
                        }

                        if p_value < 0.05:
                            drift_detected = True
                except ImportError:
                    logger.warning("SciPy not available for drift detection")
                    break
                except Exception as e:
                    logger.warning(f"Error calculating drift for {feature}: {e}")
                    continue

        return {
            'drift_detected': drift_detected,
            'drift_metrics': drift_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get performance summary for models."""
        if model_name:
            # Get summary for specific model
            if model_name in self.performance_history:
                history = self.performance_history[model_name]
                return {
                    'model_name': model_name,
                    'total_predictions': sum(r['n_samples'] for r in history),
                    'avg_mse': np.mean([r['mse'] for r in history]),
                    'avg_r2': np.mean([r['r2'] for r in history]),
                    'latest_performance': history[-1] if history else None,
                    'performance_trend': self._calculate_performance_trend(history)
                }
            else:
                return {'model_name': model_name, 'error': 'No performance data'}
        else:
            # Get summary for all models
            summaries = {}
            for model in self.performance_history.keys():
                summaries[model] = self.get_performance_summary(model)
            return summaries

    def _calculate_performance_trend(self, history: List[Dict]) -> str:
        """Calculate performance trend."""
        if len(history) < 5:
            return "insufficient_data"

        # Get recent MSE values
        recent_mse = [r['mse'] for r in history[-5:]]
        early_mse = [r['mse'] for r in history[-10:-5]] if len(history) > 5 else recent_mse

        recent_avg = np.mean(recent_mse)
        early_avg = np.mean(early_mse)

        if recent_avg < early_avg * 0.9:
            return "improving"
        elif recent_avg > early_avg * 1.1:
            return "degrading"
        else:
            return "stable"

    def generate_report(self, model_name: str = None) -> str:
        """Generate monitoring report."""
        summaries = self.get_performance_summary(model_name)

        report = f"Model Monitoring Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"

        if model_name:
            # Single model report
            summary = summaries.get(model_name, {})
            if 'error' in summary:
                report += f"Model {model_name}: {summary['error']}\n"
            else:
                report += f"Model: {model_name}\n"
                report += f"Total Predictions: {summary['total_predictions']:,}\n"
                report += f"Average MSE: {summary['avg_mse']:.4f}\n"
                report += f"Average R²: {summary['avg_r2']:.4f}\n"
                report += f"Performance Trend: {summary['performance_trend']}\n"

                if summary['latest_performance']:
                    latest = summary['latest_performance']
                    report += f"\nLatest Performance:\n"
                    report += f"  MSE: {latest['mse']:.4f}\n"
                    report += f"  R²: {latest['r2']:.4f}\n"
                    report += f"  Samples: {latest['n_samples']}\n"
        else:
            # All models report
            report += f"Total Models Monitored: {len(summaries)}\n\n"

            for model_name, summary in summaries.items():
                if 'error' not in summary:
                    report += f"{model_name}:\n"
                    report += f"  Predictions: {summary['total_predictions']:,}\n"
                    report += f"  MSE: {summary['avg_mse']:.4f}\n"
                    report += f"  R²: {summary['avg_r2']:.4f}\n"
                    report += f"  Trend: {summary['performance_trend']}\n\n"

        return report

    def get_alerts(self, model_name: str = None) -> List[Dict]:
        """Get recent alerts for models."""
        alerts = []

        if model_name:
            # Get alerts for specific model
            if model_name in self.performance_history:
                alerts.extend(self._check_performance_degradation(model_name))
        else:
            # Get alerts for all models
            for model in self.performance_history.keys():
                model_alerts = self._check_performance_degradation(model)
                for alert in model_alerts:
                    alerts.append({
                        'model': model,
                        'alert': alert,
                        'timestamp': datetime.now().isoformat()
                    })

        return alerts

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old monitoring data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for model_name in list(self.performance_history.keys()):
            # Clean performance history
            self.performance_history[model_name] = [
                r for r in self.performance_history[model_name]
                if datetime.fromisoformat(r['timestamp']) > cutoff_date
            ]

            # Clean feature importance history
            if model_name in self.feature_importance_history:
                self.feature_importance_history[model_name] = [
                    r for r in self.feature_importance_history[model_name]
                    if datetime.fromisoformat(r['timestamp']) > cutoff_date
                ]

        # Save cleaned data
        self._save_monitoring_data()

        logger.info(f"Cleaned up monitoring data older than {days_to_keep} days")