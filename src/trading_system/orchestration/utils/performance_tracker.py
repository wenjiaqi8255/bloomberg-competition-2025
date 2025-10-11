"""
Performance Tracking Utilities
=============================

Unified performance tracking for orchestration components.
Replaces individual stats dictionaries with a consistent interface.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class OperationRecord:
    """Record of a single operation."""
    timestamp: datetime
    operation: str
    duration_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentPerformanceTracker:
    """
    Unified performance tracker for orchestration components.
    
    Replaces individual stats dictionaries across components with a consistent
    interface following the Single Responsibility Principle.
    """
    
    def __init__(self, component_name: str, max_history: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            component_name: Name of the component being tracked
            max_history: Maximum number of operation records to keep
        """
        self.component_name = component_name
        self.max_history = max_history
        
        # Operation tracking
        self.operation_history: List[OperationRecord] = []
        self.active_operations: Dict[str, datetime] = {}
        
        # Aggregated statistics
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_duration_ms': 0.0,
            'average_duration_ms': 0.0,
            'last_operation_time': None,
            'operations_per_hour': 0.0
        }
        
        # Component-specific counters
        self.counters: Dict[str, int] = {}
        
        logger.debug(f"Initialized performance tracker for {component_name}")

    def start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata for the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.active_operations[operation_id] = datetime.now()
        
        if metadata:
            # Store metadata for later use
            if not hasattr(self, '_operation_metadata'):
                self._operation_metadata = {}
            self._operation_metadata[operation_id] = metadata
        
        logger.debug(f"Started operation {operation} for {self.component_name}")
        return operation_id

    def end_operation(self, operation_id: str, success: bool = True, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End tracking an operation.
        
        Args:
            operation_id: Operation ID returned by start_operation
            success: Whether the operation was successful
            metadata: Additional metadata for the operation
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active operations")
            return
        
        start_time = self.active_operations.pop(operation_id)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get stored metadata
        stored_metadata = getattr(self, '_operation_metadata', {}).get(operation_id, {})
        if metadata:
            stored_metadata.update(metadata)
        
        # Create operation record
        record = OperationRecord(
            timestamp=start_time,
            operation=operation_id.split('_')[0],  # Extract operation name
            duration_ms=duration_ms,
            success=success,
            metadata=stored_metadata
        )
        
        # Add to history
        self.operation_history.append(record)
        
        # Update statistics
        self._update_stats(record)
        
        # Clean up metadata
        if hasattr(self, '_operation_metadata') and operation_id in self._operation_metadata:
            del self._operation_metadata[operation_id]
        
        # Trim history if needed
        if len(self.operation_history) > self.max_history:
            self.operation_history = self.operation_history[-self.max_history:]
        
        logger.debug(f"Ended operation {operation_id} for {self.component_name} ({duration_ms:.2f}ms)")

    def track_counter(self, counter_name: str, increment: int = 1) -> None:
        """
        Track a counter metric.
        
        Args:
            counter_name: Name of the counter
            increment: Amount to increment by
        """
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += increment

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        # Calculate operations per hour
        if self.stats['total_operations'] > 0 and self.stats['last_operation_time']:
            time_since_last = datetime.now() - self.stats['last_operation_time']
            hours_since_last = time_since_last.total_seconds() / 3600
            if hours_since_last > 0:
                self.stats['operations_per_hour'] = self.stats['total_operations'] / hours_since_last
        
        return {
            'component_name': self.component_name,
            'stats': self.stats.copy(),
            'counters': self.counters.copy(),
            'recent_operations': self._get_recent_operations(10),
            'success_rate': self._calculate_success_rate(),
            'performance_summary': self._generate_performance_summary()
        }

    def get_operation_history(self, hours: int = 24) -> List[OperationRecord]:
        """
        Get operation history for specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of operation records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [record for record in self.operation_history if record.timestamp >= cutoff_time]

    def reset_stats(self) -> None:
        """Reset all statistics and history."""
        self.operation_history.clear()
        self.active_operations.clear()
        self.counters.clear()
        
        # Reset stats
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_duration_ms': 0.0,
            'average_duration_ms': 0.0,
            'last_operation_time': None,
            'operations_per_hour': 0.0
        }
        
        # Clean up metadata
        if hasattr(self, '_operation_metadata'):
            self._operation_metadata.clear()
        
        logger.info(f"Reset performance statistics for {self.component_name}")

    def _update_stats(self, record: OperationRecord) -> None:
        """Update aggregated statistics with new operation record."""
        self.stats['total_operations'] += 1
        self.stats['total_duration_ms'] += record.duration_ms
        self.stats['last_operation_time'] = record.timestamp
        
        if record.success:
            self.stats['successful_operations'] += 1
        else:
            self.stats['failed_operations'] += 1
        
        # Update average duration
        if self.stats['total_operations'] > 0:
            self.stats['average_duration_ms'] = self.stats['total_duration_ms'] / self.stats['total_operations']

    def _get_recent_operations(self, count: int) -> List[Dict[str, Any]]:
        """Get recent operations as dictionaries."""
        recent = self.operation_history[-count:] if self.operation_history else []
        return [
            {
                'timestamp': record.timestamp.isoformat(),
                'operation': record.operation,
                'duration_ms': record.duration_ms,
                'success': record.success,
                'metadata': record.metadata
            }
            for record in recent
        ]

    def _calculate_success_rate(self) -> float:
        """Calculate success rate."""
        if self.stats['total_operations'] == 0:
            return 0.0
        return self.stats['successful_operations'] / self.stats['total_operations']

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.operation_history:
            return {'status': 'no_operations'}
        
        # Get recent operations (last hour)
        recent_operations = self.get_operation_history(hours=1)
        
        if not recent_operations:
            return {'status': 'no_recent_operations'}
        
        # Calculate performance metrics
        recent_durations = [op.duration_ms for op in recent_operations]
        recent_success_rate = sum(1 for op in recent_operations if op.success) / len(recent_operations)
        
        return {
            'status': 'active',
            'recent_operations_count': len(recent_operations),
            'recent_success_rate': recent_success_rate,
            'recent_avg_duration_ms': sum(recent_durations) / len(recent_durations),
            'recent_max_duration_ms': max(recent_durations),
            'recent_min_duration_ms': min(recent_durations)
        }


class ComponentPerformanceTrackerMixin:
    """
    Mixin class to add performance tracking to components.
    
    Components can inherit from this mixin to get automatic performance tracking
    without duplicating the tracking logic.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with performance tracking."""
        super().__init__(*args, **kwargs)
        component_name = self.__class__.__name__
        self.performance_tracker = ComponentPerformanceTracker(component_name)
    
    def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation."""
        return self.performance_tracker.start_operation(operation, metadata)
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """End tracking an operation."""
        self.performance_tracker.end_operation(operation_id, success, metadata)
    
    def track_counter(self, counter_name: str, increment: int = 1) -> None:
        """Track a counter metric."""
        self.performance_tracker.track_counter(counter_name, increment)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_tracker.get_stats()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_tracker.reset_stats()
