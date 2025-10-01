"""
Time Series Cross-Validation utilities for ML strategies.

This module provides robust cross-validation methods specifically designed
for financial time series data to prevent look-ahead bias and ensure
realistic performance estimation.

Unified validation system with:
- PurgedTimeSeriesSplit: Core method preventing look-ahead bias
- Expanding window: Special case of purged split
- Walk-forward: Factory method using purged split
- Comprehensive validation and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Generator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit

logger = logging.getLogger(__name__)


# Export main classes for convenience
__all__ = ['PurgedTimeSeriesSplit', 'TimeSeriesCV']


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Core time series cross-validator with purge and embargo periods.

    This validator prevents look-ahead bias by:
    1. Purge period: Gap between training and test sets
    2. Embargo period: Additional buffer after test set

    This is the foundational validation method - all other methods
    are implemented as special cases or factory methods of this core.

    Parameters:
    -----------
    n_splits : int
        Number of splits to generate
    purge_period : int
        Number of periods to purge between train and test
    embargo_period : int
        Number of periods to embargo after test set
    """

    def __init__(self, n_splits=5, purge_period=5, embargo_period=2):
        self.n_splits = n_splits
        self.purge_period = purge_period
        self.embargo_period = embargo_period

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters:
        -----------
        X : array-like
            Training data, where n_samples is the number of samples
        y : array-like
            Target variable for supervised learning
        groups : array-like
            Group labels for the samples

        Yields:
        -------
        train_idx : ndarray
            The training set indices for that split
        test_idx : ndarray
            The testing set indices for that split
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate test set size
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test set indices
            test_start = (i + 1) * test_size
            test_end = min((i + 2) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            # Training set indices with purge period
            train_end = test_start - self.purge_period - self.embargo_period
            train_indices = indices[:train_end]

            # Ensure we have enough training data
            if len(train_indices) < 10:  # Minimum training samples
                continue

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class TimeSeriesCV:
    """
    Unified Time Series Cross-Validation system.

    This class provides a single interface for all time series validation needs,
    built on the foundation of PurgedTimeSeriesSplit to prevent look-ahead bias.

    Key Methods:
    - purged_split(): Core method with customizable parameters
    - expanding_window(): Expanding window validation
    - walk_forward(): Walk-forward validation
    - validate_model(): Comprehensive model validation
    """

    def __init__(self, method='purged', n_splits=5, purge_period=5, embargo_period=2):
        """
        Initialize TimeSeriesCV with default parameters.

        Parameters:
        -----------
        method : str
            Validation method ('purged', 'expanding', 'walk_forward')
        n_splits : int
            Number of splits to generate
        purge_period : int
            Number of periods to purge between train and test
        embargo_period : int
            Number of periods to embargo after test set
        """
        self.method = method
        self.n_splits = n_splits
        self.purge_period = purge_period
        self.embargo_period = embargo_period

    def split(self, X, y=None, groups=None):
        """
        Generate cross-validation splits.

        This method provides a unified interface that delegates to the appropriate
        static method based on the configured validation method.

        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like, optional
            Target variable (not used)
        groups : array-like, optional
            Group labels (not used)

        Yields:
        -------
        train_idx : ndarray
            Training set indices
        test_idx : ndarray
            Test set indices
        """
        if self.method == 'purged':
            yield from self.purged_split(
                X,
                n_splits=self.n_splits,
                purge_period=self.purge_period,
                embargo_period=self.embargo_period
            )
        elif self.method == 'expanding':
            yield from self.expanding_window_split(
                X,
                n_splits=self.n_splits,
                purge_period=self.purge_period
            )
        elif self.method == 'walk_forward':
            # For walk-forward, we need datetime index
            if not hasattr(X, 'index'):
                raise ValueError("Walk-forward requires datetime index")
            for split_data in self.walk_forward_split(X, purge_period=self.purge_period):
                # walk_forward_split yields 6 items, we need the last 2
                _, _, _, _, train_idx, test_idx = split_data
                yield train_idx, test_idx
        else:
            raise ValueError(f"Unknown validation method: {self.method}")

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

    @staticmethod
    def purged_split(X, n_splits=5, purge_period=5, embargo_period=2) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Core purged time series split method.

        This is the foundational method that prevents look-ahead bias by adding
        purge and embargo periods between training and test sets.

        Parameters:
        -----------
        X : array-like
            Training data
        n_splits : int
            Number of splits to generate
        purge_period : int
            Number of periods to purge between train and test
        embargo_period : int
            Number of periods to embargo after test set

        Yields:
        -------
        train_idx : ndarray
            Training set indices
        test_idx : ndarray
            Test set indices
        """
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_period=purge_period, embargo_period=embargo_period)
        yield from cv.split(X)

    @staticmethod
    def expanding_window_split(X, n_splits=5, initial_train_size=0.5, test_size=0.2, purge_period=5) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Expanding window time series split.

        This validator uses an expanding training window implemented as a special
        case of purged split with dynamic train size.

        Parameters:
        -----------
        X : array-like
            Training data
        n_splits : int
            Number of splits to generate
        initial_train_size : float
            Initial training set size as fraction of total data
        test_size : float
            Test set size as fraction of total data
        purge_period : int
            Number of periods to purge between train and test

        Yields:
        -------
        train_idx : ndarray
            Training set indices (expanding)
        test_idx : ndarray
            Test set indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate window sizes
        initial_train_end = int(n_samples * initial_train_size)
        test_samples = int(n_samples * test_size)

        for i in range(n_splits):
            # Test set indices
            test_start = initial_train_end + i * test_samples + purge_period
            test_end = min(test_start + test_samples, n_samples)

            if test_end > n_samples:
                break

            test_indices = indices[test_start:test_end]

            # Training set indices (expanding)
            train_end = test_start - purge_period
            train_indices = indices[:train_end]

            if len(train_indices) >= 10 and len(test_indices) > 0:
                yield train_indices, test_indices

    @staticmethod
    def walk_forward_split(X, train_size=252, test_size=21, step_size=21, purge_period=5) -> Generator[Tuple[datetime, datetime, datetime, datetime, np.ndarray, np.ndarray], None, None]:
        """
        Walk-forward validation for practical backtesting.

        This validator simulates real trading conditions with fixed training
        window and rolling test window.

        Parameters:
        -----------
        X : pd.DataFrame
            Training data with datetime index
        train_size : int
            Number of samples in training window
        test_size : int
            Number of samples in test window
        step_size : int
            Number of samples to move forward each iteration
        purge_period : int
            Number of periods to purge between train and test

        Yields:
        -------
        train_start : datetime
            Training start date
        train_end : datetime
            Training end date
        test_start : datetime
            Test start date
        test_end : datetime
            Test end date
        train_idx : ndarray
            Training indices
        test_idx : ndarray
            Test indices
        """
        if not hasattr(X, 'index'):
            raise ValueError("X must have a datetime index")

        dates = X.index
        n_samples = len(dates)

        # Calculate number of splits
        n_splits = (n_samples - train_size - purge_period) // step_size

        for i in range(n_splits):
            # Calculate indices
            train_start_idx = i * step_size
            train_end_idx = train_start_idx + train_size
            test_start_idx = train_end_idx + purge_period
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n_samples:
                break

            # Get dates
            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx - 1]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx - 1]

            # Get indices
            train_idx = np.arange(train_start_idx, train_end_idx)
            test_idx = np.arange(test_start_idx, test_end_idx)

            yield train_start, train_end, test_start, test_end, train_idx, test_idx

    @staticmethod
    def validate_model(model, X: pd.DataFrame, y: pd.Series,
                      method='purged', n_splits=5, purge_period=5,
                      embargo_period=2, scoring='neg_mean_squared_error') -> Dict[str, Any]:
        """
        Validate model using time series cross-validation.

        Parameters:
        -----------
        model : object
            Scikit-learn compatible model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        method : str
            Validation method ('purged', 'expanding', 'walk_forward')
        n_splits : int
            Number of CV splits
        purge_period : int
            Number of periods to purge between train and test
        embargo_period : int
            Number of periods to embargo after test set
        scoring : str
            Scoring method

        Returns:
        --------
        dict
            Validation results with scores, predictions, and metrics
        """
        scores = []
        predictions = []
        actuals = []

        # Get cross-validation splits
        if method == 'purged':
            splits = TimeSeriesCV.purged_split(X, n_splits=n_splits, purge_period=purge_period, embargo_period=embargo_period)
        elif method == 'expanding':
            splits = TimeSeriesCV.expanding_window_split(X, n_splits=n_splits, purge_period=purge_period)
        elif method == 'walk_forward':
            # For walk-forward, we need datetime index
            if not hasattr(X, 'index'):
                raise ValueError("Walk-forward requires datetime index")
            splits = TimeSeriesCV.walk_forward_split(X, train_size=252, test_size=21, purge_period=purge_period)
        else:
            raise ValueError(f"Unknown validation method: {method}")

        for split_data in splits:
            try:
                if method == 'walk_forward':
                    # Walk-forward yields 6 items
                    train_start, train_end, test_start, test_end, train_idx, test_idx = split_data
                else:
                    # Other methods yield 2 items
                    train_idx, test_idx = split_data

                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate score
                if scoring == 'neg_mean_squared_error':
                    score = -mean_squared_error(y_test, y_pred)
                elif scoring == 'r2':
                    score = r2_score(y_test, y_pred)
                elif scoring == 'accuracy':
                    score = accuracy_score(y_test, (y_pred > 0).astype(int))
                else:
                    score = 0

                scores.append(score)
                predictions.extend(y_pred)
                actuals.extend(y_test.values)

            except Exception as e:
                logger.warning(f"Validation fold failed: {e}")
                continue

        return {
            'method': method,
            'scores': scores,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'predictions': predictions,
            'actuals': actuals,
            'n_folds': len(scores)
        }

    # Backward compatibility aliases
    @staticmethod
    def purged_time_series_split(X, n_splits=5, purge_period=5, embargo_period=2):
        """Backward compatibility alias for purged_split"""
        return TimeSeriesCV.purged_split(X, n_splits, purge_period, embargo_period)

    @staticmethod
    def expanding_window_splitter(X, n_splits=5, initial_train_size=0.5, test_size=0.2, purge_period=5):
        """Backward compatibility alias for expanding_window_split"""
        return TimeSeriesCV.expanding_window_split(X, n_splits, initial_train_size, test_size, purge_period)