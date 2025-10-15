"""
Performance Monitor for Trading System

This module provides comprehensive performance monitoring and metrics collection
for the trading system, following SOLID principles and financial industry standards.

Key Features:
- Real-time performance metrics collection
- Memory usage monitoring
- Cache performance tracking
- Feature calculation timing
- Strategy execution monitoring
- Alert generation for performance issues

Design Principles:
- Single Responsibility: Only handles performance monitoring
- Open/Closed: Easy to extend with new metrics
- Dependency Inversion: Works with any monitoring backend
- DRY: Centralizes all performance logic
- KISS: Simple, focused implementation
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Centralized performance monitoring system for the trading application.

    This class provides real-time monitoring of key performance indicators,
    memory usage, cache performance, and system health metrics.
    """

    def __init__(self,
                 max_history_size: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the performance monitor.

        Args:
            max_history_size: Maximum number of data points to keep in memory
            alert_thresholds: Dictionary of performance thresholds for alerts
        """
        self.max_history_size = max_history_size
        self.alert_thresholds = alert_thresholds or {
            'memory_usage_percent': 80.0,
            'feature_calculation_time_sec': 5.0,
            'cache_hit_rate_min': 0.3,
            'error_rate_max': 0.05
        }

        # Performance data storage
        self._performance_data = defaultdict(lambda: deque(maxlen=max_history_size))
        self._alerts = deque(maxlen=100)
        self._lock = threading.RLock()

        # Monitoring state
        self._monitoring_enabled = True
        self._start_time = datetime.now()

        logger.info(f"PerformanceMonitor initialized with thresholds: {self.alert_thresholds}")

    def record_feature_calculation(self,
                                calculation_time: float,
                                symbols_count: int,
                                features_count: int,
                                cache_hit: bool = False):
        """
        Record feature calculation performance metrics.

        Args:
            calculation_time: Time taken for calculation in seconds
            symbols_count: Number of symbols processed
            features_count: Number of features calculated
            cache_hit: Whether result was retrieved from cache
        """
        if not self._monitoring_enabled:
            return

        timestamp = datetime.now()

        metrics = {
            'timestamp': timestamp,
            'calculation_time_sec': calculation_time,
            'symbols_per_second': symbols_count / calculation_time if calculation_time > 0 else 0,
            'features_per_second': features_count / calculation_time if calculation_time > 0 else 0,
            'symbols_count': symbols_count,
            'features_count': features_count,
            'cache_hit': cache_hit,
            'throughput_mb_per_sec': self._estimate_data_size(symbols_count, features_count) / calculation_time if calculation_time > 0 else 0
        }

        with self._lock:
            self._performance_data['feature_calculations'].append(metrics)

        # Check for performance alerts
        self._check_performance_alerts(metrics)

        logger.debug(f"Recorded feature calculation: {calculation_time:.4f}s, {symbols_count} symbols")

    def record_cache_performance(self,
                              cache_type: str,
                              hit: bool,
                              retrieval_time: float):
        """
        Record cache performance metrics.

        Args:
            cache_type: Type of cache (e.g., 'cross_sectional', 'technical')
            hit: Whether cache hit occurred
            retrieval_time: Time taken for cache operation
        """
        if not self._monitoring_enabled:
            return

        timestamp = datetime.now()

        metrics = {
            'timestamp': timestamp,
            'cache_type': cache_type,
            'hit': hit,
            'retrieval_time_sec': retrieval_time
        }

        with self._lock:
            self._performance_data['cache_performance'].append(metrics)

        logger.debug(f"Recorded cache performance: {cache_type}, hit={hit}, time={retrieval_time:.6f}s")

    def record_strategy_execution(self,
                                strategy_name: str,
                                execution_time: float,
                                signals_generated: int,
                                success: bool = True,
                                error_message: Optional[str] = None):
        """
        Record strategy execution performance metrics.

        Args:
            strategy_name: Name of the strategy
            execution_time: Time taken for strategy execution
            signals_generated: Number of trading signals generated
            success: Whether execution was successful
            error_message: Error message if execution failed
        """
        if not self._monitoring_enabled:
            return

        timestamp = datetime.now()

        metrics = {
            'timestamp': timestamp,
            'strategy_name': strategy_name,
            'execution_time_sec': execution_time,
            'signals_generated': signals_generated,
            'success': success,
            'error_message': error_message,
            'signals_per_second': signals_generated / execution_time if execution_time > 0 else 0
        }

        with self._lock:
            self._performance_data['strategy_executions'].append(metrics)

        # Check for performance alerts
        self._check_strategy_alerts(metrics)

        logger.debug(f"Recorded strategy execution: {strategy_name}, {execution_time:.4f}s, {signals_generated} signals")

    def record_system_metrics(self):
        """
        Record system-level performance metrics (memory, CPU, etc.).
        """
        if not self._monitoring_enabled:
            return

        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_used_gb = memory_info.used / (1024**3)
            memory_total_gb = memory_info.total / (1024**3)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Process-specific metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024**2)
            process_cpu_percent = process.cpu_percent()

            timestamp = datetime.now()

            metrics = {
                'timestamp': timestamp,
                'system_memory_percent': memory_percent,
                'system_memory_used_gb': memory_used_gb,
                'system_memory_total_gb': memory_total_gb,
                'system_cpu_percent': cpu_percent,
                'process_memory_mb': process_memory_mb,
                'process_cpu_percent': process_cpu_percent
            }

            with self._lock:
                self._performance_data['system_metrics'].append(metrics)

            # Check for system alerts
            self._check_system_alerts(metrics)

        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")

    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get a summary of performance metrics over the specified time window.

        Args:
            time_window_minutes: Time window in minutes for analysis

        Returns:
            Dictionary with performance summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        summary = {
            'time_window_minutes': time_window_minutes,
            'analysis_timestamp': datetime.now(),
            'uptime_hours': (datetime.now() - self._start_time).total_seconds() / 3600
        }

        with self._lock:
            # Feature calculations summary
            if 'feature_calculations' in self._performance_data:
                recent_calcs = [
                    m for m in self._performance_data['feature_calculations']
                    if m['timestamp'] >= cutoff_time
                ]

                if recent_calcs:
                    calc_times = [m['calculation_time_sec'] for m in recent_calcs]
                    symbols_counts = [m['symbols_count'] for m in recent_calcs]
                    cache_hits = sum(1 for m in recent_calcs if m['cache_hit'])

                    summary['feature_calculations'] = {
                        'total_calculations': len(recent_calcs),
                        'avg_time_sec': np.mean(calc_times),
                        'max_time_sec': np.max(calc_times),
                        'min_time_sec': np.min(calc_times),
                        'avg_symbols_per_calc': np.mean(symbols_counts),
                        'cache_hit_rate': cache_hits / len(recent_calcs),
                        'avg_throughput_symbols_per_sec': np.mean([m['symbols_per_second'] for m in recent_calcs])
                    }

            # Cache performance summary
            if 'cache_performance' in self._performance_data:
                recent_cache = [
                    m for m in self._performance_data['cache_performance']
                    if m['timestamp'] >= cutoff_time
                ]

                if recent_cache:
                    cache_hits = sum(1 for m in recent_cache if m['hit'])
                    retrieval_times = [m['retrieval_time_sec'] for m in recent_cache]

                    summary['cache_performance'] = {
                        'total_requests': len(recent_cache),
                        'cache_hit_rate': cache_hits / len(recent_cache),
                        'avg_retrieval_time_sec': np.mean(retrieval_times),
                        'cache_types': list(set(m['cache_type'] for m in recent_cache))
                    }

            # Strategy execution summary
            if 'strategy_executions' in self._performance_data:
                recent_strategies = [
                    m for m in self._performance_data['strategy_executions']
                    if m['timestamp'] >= cutoff_time
                ]

                if recent_strategies:
                    exec_times = [m['execution_time_sec'] for m in recent_strategies if m['success']]
                    success_rate = sum(1 for m in recent_strategies if m['success']) / len(recent_strategies)

                    summary['strategy_executions'] = {
                        'total_executions': len(recent_strategies),
                        'success_rate': success_rate,
                        'avg_execution_time_sec': np.mean(exec_times) if exec_times else 0,
                        'strategies_used': list(set(m['strategy_name'] for m in recent_strategies))
                    }

            # System metrics summary
            if 'system_metrics' in self._performance_data:
                recent_system = [
                    m for m in self._performance_data['system_metrics']
                    if m['timestamp'] >= cutoff_time
                ]

                if recent_system:
                    memory_usage = [m['system_memory_percent'] for m in recent_system]
                    cpu_usage = [m['system_cpu_percent'] for m in recent_system]

                    summary['system_metrics'] = {
                        'avg_memory_usage_percent': np.mean(memory_usage),
                        'max_memory_usage_percent': np.max(memory_usage),
                        'avg_cpu_usage_percent': np.mean(cpu_usage),
                        'max_cpu_usage_percent': np.max(cpu_usage)
                    }

            # Recent alerts
            recent_alerts = [
                alert for alert in self._alerts
                if alert['timestamp'] >= cutoff_time
            ]

            summary['alerts'] = {
                'total_alerts': len(recent_alerts),
                'alert_types': list(set(alert['type'] for alert in recent_alerts)),
                'recent_alerts': recent_alerts[-5:]  # Last 5 alerts
            }

        return summary

    def _estimate_data_size(self, symbols_count: int, features_count: int) -> float:
        """
        Estimate data size in megabytes.

        Args:
            symbols_count: Number of symbols
            features_count: Number of features per symbol

        Returns:
            Estimated data size in MB
        """
        # Rough estimate: each feature value ~8 bytes (float64)
        bytes_per_feature = 8
        total_bytes = symbols_count * features_count * bytes_per_feature

        # Add overhead for pandas DataFrame structure
        overhead_factor = 1.5
        total_bytes *= overhead_factor

        return total_bytes / (1024**2)  # Convert to MB

    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance-related alerts."""
        alerts = []

        # Feature calculation time alert
        if metrics['calculation_time_sec'] > self.alert_thresholds['feature_calculation_time_sec']:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'slow_feature_calculation',
                'message': f"Slow feature calculation: {metrics['calculation_time_sec']:.2f}s for {metrics['symbols_count']} symbols",
                'severity': 'warning'
            })

        # Add alerts to the alerts deque
        if alerts:
            with self._lock:
                self._alerts.extend(alerts)

    def _check_strategy_alerts(self, metrics: Dict[str, Any]):
        """Check for strategy execution alerts."""
        if not metrics['success']:
            alert = {
                'timestamp': datetime.now(),
                'type': 'strategy_execution_error',
                'message': f"Strategy {metrics['strategy_name']} failed: {metrics['error_message']}",
                'severity': 'error'
            }

            with self._lock:
                self._alerts.append(alert)

    def _check_system_alerts(self, metrics: Dict[str, Any]):
        """Check for system-level alerts."""
        alerts = []

        # Memory usage alert
        if metrics['system_memory_percent'] > self.alert_thresholds['memory_usage_percent']:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'high_memory_usage',
                'message': f"High memory usage: {metrics['system_memory_percent']:.1f}%",
                'severity': 'warning'
            })

        # Add alerts to the alerts deque
        if alerts:
            with self._lock:
                self._alerts.extend(alerts)

    def enable_monitoring(self):
        """Enable performance monitoring."""
        self._monitoring_enabled = True
        logger.info("Performance monitoring enabled")

    def disable_monitoring(self):
        """Disable performance monitoring."""
        self._monitoring_enabled = False
        logger.info("Performance monitoring disabled")

    def clear_history(self):
        """Clear all performance history."""
        with self._lock:
            for key in self._performance_data:
                self._performance_data[key].clear()
            self._alerts.clear()

        logger.info("Performance history cleared")

    def get_alerts(self, severity: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            severity: Filter by severity ('error', 'warning', 'info')
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        with self._lock:
            alerts = list(self._alerts)

        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]

        return alerts[-limit:] if limit else alerts

    def export_metrics(self, filepath: str, time_window_minutes: int = 60):
        """
        Export performance metrics to a CSV file.

        Args:
            filepath: Path to save the CSV file
            time_window_minutes: Time window for data to export
        """
        try:
            summary = self.get_performance_summary(time_window_minutes)

            # Convert summary to a flattend DataFrame for easy export
            flattened_data = []

            for category, data in summary.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        flattened_data.append({
                            'category': category,
                            'metric': key,
                            'value': value,
                            'timestamp': summary['analysis_timestamp']
                        })
                else:
                    flattened_data.append({
                        'category': 'summary',
                        'metric': category,
                        'value': data,
                        'timestamp': summary['analysis_timestamp']
                    })

            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False)

            logger.info(f"Performance metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get current real-time metrics.

        Returns:
            Dictionary with current system metrics
        """
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()

            return {
                'timestamp': datetime.now(),
                'system_memory_percent': memory_info.percent,
                'system_memory_used_gb': memory_info.used / (1024**3),
                'system_cpu_percent': psutil.cpu_percent(interval=None),
                'process_memory_mb': process.memory_info().rss / (1024**2),
                'process_cpu_percent': process.cpu_percent(),
                'uptime_hours': (datetime.now() - self._start_time).total_seconds() / 3600,
                'monitoring_enabled': self._monitoring_enabled
            }
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'monitoring_enabled': self._monitoring_enabled
            }

    def __str__(self):
        return f"PerformanceMonitor(enabled={self._monitoring_enabled}, alerts={len(self._alerts)})"

    def __repr__(self):
        return self.__str__()