"""
Model Monitor for Serving and Degradation Detection

This module provides comprehensive model monitoring capabilities including:
- Performance degradation detection
- Data drift detection
- Prediction logging and analysis
- Automated alerting for model issues
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from scipy import stats
import warnings

from ..base.base_model import BaseModel
from ..utils.performance_evaluator import PerformanceEvaluator
from ...utils.experiment_tracking.interface import ExperimentTrackerInterface

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction record for monitoring."""
    timestamp: datetime
    model_id: str
    prediction_id: str
    features: Dict[str, float]
    prediction: float
    actual: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHealthStatus:
    """Model health status information."""
    model_id: str
    status: str  # 'healthy', 'warning', 'critical', 'degraded'
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ModelMonitor:
    """
    Comprehensive model monitoring system.

    Features:
    - Real-time performance tracking
    - Data drift detection
    - Prediction logging
    - Automated degradation alerts
    - Health reporting
    """

    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None,
                 tracker: Optional[ExperimentTrackerInterface] = None):
        """
        Initialize model monitor.

        Args:
            model_id: Unique identifier for the model
            config: Configuration dictionary with:
                - performance_window: Days for performance evaluation (default: 30)
                - degradation_threshold: Performance drop threshold (default: 0.2)
                - drift_threshold: Drift detection threshold (default: 0.1)
                - min_samples: Minimum samples for evaluation (default: 100)
                - alert_threshold: Number of issues before alert (default: 3)
                - log_path: Path for storing logs (default: "./logs/")
            tracker: Optional experiment tracker for logging monitoring events
        """
        self.model_id = model_id
        self.config = config or {}
        self.tracker = tracker

        # Configuration
        self.performance_window = self.config.get('performance_window', 30)
        self.degradation_threshold = self.config.get('degradation_threshold', 0.2)
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.min_samples = self.config.get('min_samples', 100)
        self.alert_threshold = self.config.get('alert_threshold', 3)
        self.log_path = Path(self.config.get('log_path', './logs/'))

        # Initialize storage
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.prediction_log: List[PredictionRecord] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.health_status = ModelHealthStatus(
            model_id=model_id,
            status='healthy',
            last_check=datetime.now()
        )

        # Baseline metrics (set during initialization or first evaluation)
        self.baseline_metrics: Dict[str, float] = {}

        # Initialize monitoring run if tracker provided
        self.monitor_run_id = None
        if self.tracker:
            try:
                from ...utils.experiment_tracking.config import ExperimentConfig
                monitor_config = ExperimentConfig(
                    project_name="model_monitoring",
                    experiment_name=f"monitor_{model_id}",
                    run_type="monitoring",
                    tags=["monitoring", model_id],
                    hyperparameters={},
                    metadata={
                        "model_id": model_id,
                        "monitoring_config": self.config,
                        "start_time": datetime.now().isoformat()
                    }
                )
                self.monitor_run_id = self.tracker.init_run(monitor_config)
                self.tracker.update_run_status("monitoring")
                logger.info(f"Initialized monitoring run {self.monitor_run_id} for model: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize monitoring run: {e}")

        logger.info(f"Initialized ModelMonitor for model: {model_id}")

    def log_prediction(self, features: Dict[str, float], prediction: float,
                      actual: Optional[float] = None, confidence: Optional[float] = None,
                      prediction_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a prediction for monitoring.

        Args:
            features: Feature values used for prediction
            prediction: Predicted value
            actual: Actual value (if available)
            confidence: Prediction confidence score
            prediction_id: Unique prediction identifier
            metadata: Additional metadata

        Returns:
            Prediction ID
        """
        if prediction_id is None:
            prediction_id = f"{self.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        record = PredictionRecord(
            timestamp=datetime.now(),
            model_id=self.model_id,
            prediction_id=prediction_id,
            features=features,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.prediction_log.append(record)

        # Keep only recent predictions (based on performance window)
        cutoff_date = datetime.now() - timedelta(days=self.performance_window * 2)
        self.prediction_log = [r for r in self.prediction_log if r.timestamp > cutoff_date]

        logger.debug(f"Logged prediction {prediction_id} for model {self.model_id}")
        return prediction_id

    def update_actual(self, prediction_id: str, actual: float) -> bool:
        """
        Update the actual value for a prediction.

        Args:
            prediction_id: Prediction identifier
            actual: Actual value

        Returns:
            True if updated successfully, False if not found
        """
        for record in self.prediction_log:
            if record.prediction_id == prediction_id:
                record.actual = actual
                logger.debug(f"Updated actual value for prediction {prediction_id}")
                return True

        logger.warning(f"Prediction {prediction_id} not found")
        return False

    def check_performance_degradation(self, model: BaseModel) -> Dict[str, Any]:
        """
        Check for model performance degradation.

        Args:
            model: The model being monitored

        Returns:
            Dictionary with degradation analysis
        """
        try:
            # Get recent predictions with actual values
            recent_predictions = [r for r in self.prediction_log
                                if r.actual is not None and
                                r.timestamp > datetime.now() - timedelta(days=self.performance_window)]

            if len(recent_predictions) < self.min_samples:
                return {
                    'status': 'insufficient_data',
                    'message': f'Need at least {self.min_samples} samples, got {len(recent_predictions)}',
                    'degradation_detected': False
                }

            # Extract data for evaluation
            X = pd.DataFrame([r.features for r in recent_predictions])
            y = pd.Series([r.actual for r in recent_predictions])

            # Current performance
            current_metrics = PerformanceEvaluator.evaluate_model(model, X, y)

            # Check against baseline if available
            if self.baseline_metrics:
                degradation_analysis = self._analyze_degradation(current_metrics, self.baseline_metrics)
            else:
                # Set current metrics as baseline
                self.baseline_metrics = current_metrics
                degradation_analysis = {
                    'status': 'baseline_set',
                    'message': 'Baseline metrics established',
                    'degradation_detected': False,
                    'baseline_metrics': current_metrics
                }

            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': current_metrics,
                'sample_count': len(recent_predictions)
            })

            # Keep only recent history
            self.performance_history = self.performance_history[-100:]  # Keep last 100 evaluations

            # Log to tracker if degradation detected
            if self.tracker and degradation_analysis.get('degradation_detected', False):
                try:
                    self.tracker.log_metrics({
                        'current_performance': current_metrics.get('r2', 0.0),
                        'baseline_performance': self.baseline_metrics.get('r2', 0.0),
                        'performance_drop': abs(current_metrics.get('r2', 0.0) - self.baseline_metrics.get('r2', 0.0)),
                        'degradation_detected': int(degradation_analysis.get('degradation_detected', False))
                    })

                    # Send alert
                    self.tracker.log_alert(
                        title="Model Performance Degradation",
                        text=f"Model {self.model_id} performance degraded. "
                             f"R² dropped from {self.baseline_metrics.get('r2', 0.0):.3f} to {current_metrics.get('r2', 0.0):.3f}",
                        level="warning"
                    )

                    logger.warning(f"Performance degradation detected for model {self.model_id}")
                except Exception as e:
                    logger.warning(f"Failed to log degradation to tracker: {e}")

            return degradation_analysis

        except Exception as e:
            logger.error(f"Performance degradation check failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'degradation_detected': False
            }

    def check_data_drift(self, reference_features: pd.DataFrame,
                         current_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Check for data drift in feature distributions.

        Args:
            reference_features: Reference feature distribution (training data)
            current_features: Current feature distribution (if None, uses recent predictions)

        Returns:
            Dictionary with drift analysis
        """
        try:
            # Use recent predictions if current_features not provided
            if current_features is None:
                recent_predictions = [r for r in self.prediction_log
                                   if r.timestamp > datetime.now() - timedelta(days=self.performance_window)]

                if len(recent_predictions) < self.min_samples:
                    return {
                        'status': 'insufficient_data',
                        'message': f'Need at least {self.min_samples} samples, got {len(recent_predictions)}',
                        'drift_detected': False
                    }

                current_features = pd.DataFrame([r.features for r in recent_predictions])

            # Check for common features
            common_features = set(reference_features.columns) & set(current_features.columns)
            if not common_features:
                return {
                    'status': 'error',
                    'message': 'No common features between reference and current data',
                    'drift_detected': False
                }

            # Calculate drift for each feature
            drift_results = {}
            for feature in common_features:
                ref_data = reference_features[feature].dropna()
                cur_data = current_features[feature].dropna()

                if len(ref_data) > 10 and len(cur_data) > 10:
                    # Kolmogorov-Smirnov test for distribution drift
                    ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, cur_data)

                    # Population Stability Index (PSI)
                    psi = self._calculate_psi(ref_data, cur_data)

                    drift_results[feature] = {
                        'ks_statistic': ks_statistic,
                        'ks_pvalue': ks_pvalue,
                        'psi': psi,
                        'drift_detected': ks_pvalue < 0.05 or psi > self.drift_threshold
                    }

            # Overall drift assessment
            drifted_features = [f for f, result in drift_results.items() if result['drift_detected']]
            drift_detected = len(drifted_features) > 0

            return {
                'status': 'completed',
                'drift_detected': drift_detected,
                'drifted_features': drifted_features,
                'drift_results': drift_results,
                'total_features_checked': len(common_features),
                'features_with_drift': len(drifted_features)
            }

        except Exception as e:
            logger.error(f"Data drift check failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'drift_detected': False
            }

    def get_health_status(self, model: BaseModel,
                         reference_features: Optional[pd.DataFrame] = None) -> ModelHealthStatus:
        """
        Get comprehensive health status for the model.

        Args:
            model: The model being monitored
            reference_features: Reference features for drift detection

        Returns:
            ModelHealthStatus object
        """
        try:
            issues = []
            recommendations = []
            metrics = {}

            # Check performance degradation
            degradation_result = self.check_performance_degradation(model)
            metrics['performance_status'] = degradation_result.get('status', 'unknown')
            metrics['degradation_detected'] = degradation_result.get('degradation_detected', False)

            if degradation_result.get('degradation_detected', False):
                issues.append("Performance degradation detected")
                recommendations.append("Consider retraining the model")

            # Check data drift if reference features provided
            if reference_features is not None:
                drift_result = self.check_data_drift(reference_features)
                metrics['drift_status'] = drift_result.get('status', 'unknown')
                metrics['drift_detected'] = drift_result.get('drift_detected', False)

                if drift_result.get('drift_detected', False):
                    issues.append(f"Data drift detected in {len(drift_result.get('drifted_features', []))} features")
                    recommendations.append("Update model with recent data patterns")

            # Check prediction volume
            recent_predictions = [r for r in self.prediction_log
                                if r.timestamp > datetime.now() - timedelta(days=1)]
            metrics['daily_predictions'] = len(recent_predictions)

            if len(recent_predictions) == 0:
                issues.append("No predictions in last 24 hours")
                recommendations.append("Check model serving pipeline")

            # Check actual value feedback rate
            predictions_with_actual = [r for r in recent_predictions if r.actual is not None]
            if len(recent_predictions) > 0:
                feedback_rate = len(predictions_with_actual) / len(recent_predictions)
                metrics['feedback_rate'] = feedback_rate

                if feedback_rate < 0.1:
                    issues.append("Low actual value feedback rate")
                    recommendations.append("Improve actual value collection process")

            # Determine overall status
            if len(issues) == 0:
                status = 'healthy'
            elif len(issues) <= self.alert_threshold:
                status = 'warning'
            else:
                status = 'critical'

            # Update health status
            old_status = self.health_status.status
            self.health_status = ModelHealthStatus(
                model_id=self.model_id,
                status=status,
                last_check=datetime.now(),
                issues=issues,
                metrics=metrics,
                recommendations=recommendations
            )

            # Log health status changes to tracker
            if self.tracker and old_status != status:
                try:
                    self.tracker.log_metrics({
                        'health_status_code': {'healthy': 0, 'warning': 1, 'critical': 2, 'error': 3, 'degraded': 4}.get(status, 0),
                        'num_issues': len(issues),
                        'daily_predictions': metrics.get('daily_predictions', 0),
                        'feedback_rate': metrics.get('feedback_rate', 0.0)
                    })

                    # Send alert for status changes
                    if status in ['critical', 'error']:
                        self.tracker.log_alert(
                            title=f"Model Health Status: {status.upper()}",
                            text=f"Model {self.model_id} health status changed to {status}. "
                                 f"Issues: {len(issues)}. Recommendations: {len(recommendations)}",
                            level="error"
                        )
                    elif status == 'warning' and old_status == 'healthy':
                        self.tracker.log_alert(
                            title="Model Health Status: Warning",
                            text=f"Model {self.model_id} health status changed to warning. "
                                 f"Issues detected: {issues}",
                            level="warning"
                        )

                    logger.info(f"Health status changed from {old_status} to {status} for model {self.model_id}")
                except Exception as e:
                    logger.warning(f"Failed to log health status to tracker: {e}")

            return self.health_status

        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return ModelHealthStatus(
                model_id=self.model_id,
                status='error',
                last_check=datetime.now(),
                issues=[f"Health check failed: {str(e)}"],
                metrics={},
                recommendations=["Check monitoring system"]
            )

    def generate_report(self, reference_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Args:
            reference_features: Reference features for drift analysis

        Returns:
            Dictionary with monitoring report
        """
        try:
            # Basic statistics
            total_predictions = len(self.prediction_log)
            recent_predictions = [r for r in self.prediction_log
                                if r.timestamp > datetime.now() - timedelta(days=self.performance_window)]
            predictions_with_actual = [r for r in recent_predictions if r.actual is not None]

            # Performance metrics (if we have actual values)
            performance_metrics = {}
            if predictions_with_actual:
                predictions = [r.prediction for r in predictions_with_actual]
                actuals = [r.actual for r in predictions_with_actual]

                performance_metrics = {
                    'mae': np.mean(np.abs(np.array(predictions) - np.array(actuals))),
                    'rmse': np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2)),
                    'correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0,
                    'directional_accuracy': np.mean(
                        (np.array(predictions) > 0) == (np.array(actuals) > 0)
                    )
                }

            # Prediction volume analysis
            daily_counts = {}
            for record in recent_predictions:
                date = record.timestamp.date()
                daily_counts[date] = daily_counts.get(date, 0) + 1

            # Feature statistics
            feature_stats = {}
            if recent_predictions:
                feature_df = pd.DataFrame([r.features for r in recent_predictions])
                for col in feature_df.columns:
                    feature_stats[col] = {
                        'mean': float(feature_df[col].mean()),
                        'std': float(feature_df[col].std()),
                        'min': float(feature_df[col].min()),
                        'max': float(feature_df[col].max()),
                        'missing_rate': float(feature_df[col].isna().mean())
                    }

            report = {
                'model_id': self.model_id,
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_period_days': self.performance_window,

                # Prediction statistics
                'total_predictions': total_predictions,
                'recent_predictions': len(recent_predictions),
                'predictions_with_actual': len(predictions_with_actual),
                'feedback_rate': len(predictions_with_actual) / len(recent_predictions) if recent_predictions else 0.0,

                # Performance metrics
                'performance_metrics': performance_metrics,
                'baseline_metrics': self.baseline_metrics,

                # Health status
                'health_status': {
                    'status': self.health_status.status,
                    'issues': self.health_status.issues,
                    'recommendations': self.health_status.recommendations,
                    'last_check': self.health_status.last_check.isoformat()
                },

                # Prediction volume
                'daily_prediction_counts': {str(k): v for k, v in daily_counts.items()},

                # Feature statistics
                'feature_statistics': feature_stats,

                # Configuration
                'monitoring_config': {
                    'performance_window': self.performance_window,
                    'degradation_threshold': self.degradation_threshold,
                    'min_samples': self.min_samples
                }
            }

            return report

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                'model_id': self.model_id,
                'error': str(e),
                'report_timestamp': datetime.now().isoformat()
            }

    def save_logs(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Save monitoring logs to file.

        Args:
            filepath: Path to save logs (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.log_path / f"{self.model_id}_monitoring_{timestamp}.json"

        filepath = Path(filepath)

        # Prepare data for serialization
        log_data = {
            'model_id': self.model_id,
            'prediction_log': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'prediction_id': r.prediction_id,
                    'features': r.features,
                    'prediction': r.prediction,
                    'actual': r.actual,
                    'confidence': r.confidence,
                    'metadata': r.metadata
                }
                for r in self.prediction_log[-1000:]  # Save last 1000 records
            ],
            'performance_history': self.performance_history,
            'health_status': {
                'status': self.health_status.status,
                'last_check': self.health_status.last_check.isoformat(),
                'issues': self.health_status.issues,
                'recommendations': self.health_status.recommendations
            },
            'baseline_metrics': self.baseline_metrics,
            'config': self.config
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved monitoring logs to {filepath}")
        return str(filepath)

    def _analyze_degradation(self, current_metrics: Dict[str, float],
                           baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance degradation against baseline."""
        degraded_metrics = {}
        degradation_detected = False

        # Key metrics to monitor
        key_metrics = ['r2', 'correlation', 'directional_accuracy', 'information_coefficient']

        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]

                # Calculate relative degradation
                if baseline_val != 0:
                    relative_change = abs(current_val - baseline_val) / abs(baseline_val)
                else:
                    relative_change = abs(current_val)

                degraded_metrics[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'relative_change': relative_change,
                    'degraded': relative_change > self.degradation_threshold
                }

                if degraded_metrics[metric]['degraded']:
                    degradation_detected = True

        # For metrics where lower is better (error metrics)
        error_metrics = ['mse', 'rmse', 'mae', 'mape']
        for metric in error_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]

                # For error metrics, degradation means increase in error
                if baseline_val != 0:
                    relative_change = (current_val - baseline_val) / baseline_val
                else:
                    relative_change = current_val

                degraded_metrics[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'relative_change': relative_change,
                    'degraded': relative_change > self.degradation_threshold
                }

                if degraded_metrics[metric]['degraded']:
                    degradation_detected = True

        return {
            'status': 'degradation_detected' if degradation_detected else 'healthy',
            'degradation_detected': degradation_detected,
            'degraded_metrics': degraded_metrics,
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics
        }

    def _calculate_psi(self, reference_data: pd.Series, current_data: pd.Series,
                      bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            _, bin_edges = pd.cut(reference_data, bins=bins, retbins=True, duplicates='drop')

            # Calculate frequencies
            ref_counts = pd.cut(reference_data, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            cur_counts = pd.cut(current_data, bins=bin_edges, include_lowest=True).value_counts(normalize=True)

            # Align bins
            all_bins = bin_edges[:-1]
            psi = 0.0

            for i in range(len(all_bins)):
                bin_label = f"({all_bins[i]:.3f}, {all_bins[i+1]:.3f}]"

                ref_pct = ref_counts.get(i+1, 0.0001)  # Avoid division by zero
                cur_pct = cur_counts.get(i+1, 0.0001)

                if ref_pct > 0 and cur_pct > 0:
                    psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

            return psi

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def stop_monitoring(self) -> None:
        """
        Stop monitoring and finish the tracking run.

        This method should be called when monitoring is complete to
        properly close the experiment tracking run.
        """
        if self.tracker and self.monitor_run_id:
            try:
                # Log final summary
                final_report = self.generate_report()
                self.tracker.log_artifact_from_dict(final_report, f"final_monitoring_report_{self.model_id}")

                # Finish the run
                self.tracker.finish_run(exit_code=0)
                logger.info(f"Finished monitoring run {self.monitor_run_id} for model {self.model_id}")
                self.monitor_run_id = None
            except Exception as e:
                logger.warning(f"Failed to finish monitoring run: {e}")

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get the historical metrics for dashboard visualization.

        Returns:
            List of historical metric dictionaries
        """
        return self.performance_history.copy()

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current monitoring metrics for dashboard.

        Returns:
            Dictionary of current metrics
        """
        recent_predictions = [r for r in self.prediction_log
                            if r.timestamp > datetime.now() - timedelta(days=1)]

        return {
            'model_id': self.model_id,
            'health_status': self.health_status.status,
            'last_check': self.health_status.last_check,
            'daily_predictions': len(recent_predictions),
            'total_predictions': len(self.prediction_log),
            'issues_count': len(self.health_status.issues),
            'recommendations_count': len(self.health_status.recommendations),
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.performance_history[-1]['metrics'] if self.performance_history else {}
        }