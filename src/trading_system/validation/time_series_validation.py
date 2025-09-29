"""
Time Series Validation for ML Trading Strategies.

This module provides robust time series cross-validation methods:
- PurgedTimeSeriesSplit: Avoids look-ahead bias with purge and embargo periods
- ExpandingWindowSplitter: Expanding window validation for time series
- CombinatorialPurgedCV: Advanced validation method from financial ML
- WalkForwardValidator: Practical walk-forward validation implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validator with purge and embargo periods.

    This validator prevents look-ahead bias by:
    1. Purge period: Gap between training and test sets
    2. Embargo period: Additional buffer after test set

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


class ExpandingWindowSplitter(BaseCrossValidator):
    """
    Expanding window time series cross-validator.

    This validator uses an expanding training window:
    - Training set grows with each split
    - Test set size remains constant
    - No overlap between test sets

    Parameters:
    -----------
    n_splits : int
        Number of splits to generate
    initial_train_size : float
        Initial training set size as fraction of total data
    test_size : float
        Test set size as fraction of total data
    purge_period : int
        Number of periods to purge between train and test
    """

    def __init__(self, n_splits=5, initial_train_size=0.5, test_size=0.2, purge_period=5):
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.purge_period = purge_period

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate window sizes
        initial_train_end = int(n_samples * self.initial_train_size)
        test_size = int(n_samples * self.test_size)

        for i in range(self.n_splits):
            # Test set indices
            test_start = initial_train_end + i * test_size + self.purge_period
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples:
                break

            test_indices = indices[test_start:test_end]

            # Training set indices (expanding)
            train_end = test_start - self.purge_period
            train_indices = indices[:train_end]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation for financial time series.

    This method combines multiple train/test splits to reduce variance
    in validation estimates, while preventing look-ahead bias.

    Parameters:
    -----------
    n_splits : int
        Number of splits to generate
    n_test_groups : int
        Number of test groups to use
    purge_period : int
        Number of periods to purge between train and test
    embargo_period : int
        Number of periods to embargo after test set
    """

    def __init__(self, n_splits=10, n_test_groups=3, purge_period=5, embargo_period=2):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_period = purge_period
        self.embargo_period = embargo_period

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Create test groups
        test_group_size = n_samples // self.n_splits
        test_groups = []

        for i in range(self.n_splits):
            start = i * test_group_size
            end = min((i + 1) * test_group_size, n_samples)
            test_groups.append(indices[start:end])

        # Generate combinatorial splits
        for i in range(self.n_splits):
            # Select test groups (non-overlapping)
            test_indices = []
            for j in range(self.n_test_groups):
                group_idx = (i + j) % self.n_splits
                test_indices.extend(test_groups[group_idx])

            test_indices = np.array(test_indices)

            # Training indices (all other groups with purge)
            train_indices = []
            for j, group in enumerate(test_groups):
                if j not in [(i + k) % self.n_splits for k in range(self.n_test_groups)]:
                    # Check purge period
                    group_min = group.min()
                    group_max = group.max()

                    # Purge if too close to test set
                    if (group_max + self.purge_period + self.embargo_period < test_indices.min() or
                        group_min > test_indices.max() + self.embargo_period):
                        train_indices.extend(group)

            train_indices = np.array(train_indices)

            if len(train_indices) > 10 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation for practical backtesting.

    This validator simulates real trading conditions:
    - Fixed training window
    - Rolling test window
    - Realistic retraining frequency

    Parameters:
    -----------
    train_size : int
        Number of samples in training window
    test_size : int
        Number of samples in test window
    step_size : int
        Number of samples to move forward each iteration
    purge_period : int
        Number of periods to purge between train and test
    """

    def __init__(self, train_size=252, test_size=21, step_size=21, purge_period=5):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_period = purge_period

    def split(self, X, y=None):
        """
        Generate walk-forward splits.

        Parameters:
        -----------
        X : pd.DataFrame or pd.Series
            Training data with datetime index
        y : pd.Series
            Target variable

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
        n_splits = (n_samples - self.train_size - self.purge_period) // self.step_size

        for i in range(n_splits):
            # Calculate indices
            train_start_idx = i * self.step_size
            train_end_idx = train_start_idx + self.train_size
            test_start_idx = train_end_idx + self.purge_period
            test_end_idx = test_start_idx + self.test_size

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


class ModelValidator:
    """
    Comprehensive model validation for trading strategies.

    This class provides methods to validate ML models with proper
    time series techniques and performance metrics.
    """

    def __init__(self, cv_method='purged', purge_period=5, embargo_period=2):
        """
        Initialize model validator.

        Parameters:
        -----------
        cv_method : str
            Cross-validation method ('purged', 'expanding', 'combinatorial')
        purge_period : int
            Number of periods to purge between train and test
        embargo_period : int
            Number of periods to embargo after test set
        """
        self.cv_method = cv_method
        self.purge_period = purge_period
        self.embargo_period = embargo_period

        # Initialize cross-validator
        self.cv = self._get_cv()

    def _get_cv(self):
        """Get cross-validator based on method."""
        if self.cv_method == 'purged':
            return PurgedTimeSeriesSplit(
                n_splits=5,
                purge_period=self.purge_period,
                embargo_period=self.embargo_period
            )
        elif self.cv_method == 'expanding':
            return ExpandingWindowSplitter(
                n_splits=5,
                purge_period=self.purge_period
            )
        elif self.cv_method == 'combinatorial':
            return CombinatorialPurgedCV(
                n_splits=10,
                purge_period=self.purge_period,
                embargo_period=self.embargo_period
            )
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")

    def validate_model(self, model, X, y, scoring='neg_mean_squared_error'):
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
        scoring : str
            Scoring method

        Returns:
        --------
        dict
            Validation results
        """
        scores = []
        predictions = []
        actuals = []

        for train_idx, test_idx in self.cv.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            try:
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
            'scores': scores,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'predictions': predictions,
            'actuals': actuals,
            'n_folds': len(scores)
        }

    def walk_forward_validation(self, model, X, y, train_size=252, test_size=21):
        """
        Perform walk-forward validation.

        Parameters:
        -----------
        model : object
            Scikit-learn compatible model
        X : pd.DataFrame
            Feature matrix with datetime index
        y : pd.Series
            Target variable
        train_size : int
            Training window size
        test_size : int
            Test window size

        Returns:
        --------
        dict
            Walk-forward validation results
        """
        validator = WalkForwardValidator(
            train_size=train_size,
            test_size=test_size,
            purge_period=self.purge_period
        )

        results = []
        predictions = []
        actuals = []

        for (train_start, train_end, test_start, test_end,
             train_idx, test_idx) in validator.split(X, y):

            try:
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'mse': mse,
                    'r2': r2,
                    'n_samples': len(y_test)
                })

                predictions.extend(y_pred)
                actuals.extend(y_test.values)

            except Exception as e:
                logger.warning(f"Walk-forward fold failed: {e}")
                continue

        return {
            'results': results,
            'predictions': predictions,
            'actuals': actuals,
            'mean_mse': np.mean([r['mse'] for r in results]) if results else 0,
            'mean_r2': np.mean([r['r2'] for r in results]) if results else 0,
            'n_folds': len(results)
        }