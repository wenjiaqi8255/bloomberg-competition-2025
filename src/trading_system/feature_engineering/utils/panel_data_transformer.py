"""
Panel Data Transformer

This module provides utilities for transforming data between different formats
commonly used in financial time series and cross-sectional analysis.

Key Transformations:
- Time series format → Panel format (date, symbol)
- Panel format → Cross-sectional slices
- Data validation and completeness checks

Design Principles:
- Stateless transformations (pure functions)
- Defensive data validation
- Clear error messages
"""

import logging
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PanelDataTransformer:
    """
    Utility class for transforming financial data between formats.
    
    Supported Formats:
    1. Time Series: MultiIndex(symbol, date) - Traditional format
    2. Panel: MultiIndex(date, symbol) - For cross-sectional analysis
    3. Wide: date as index, symbols as columns - For visualization
    
    Example:
        # Transform to panel format for Fama-MacBeth
        features_panel, target_panel = PanelDataTransformer.to_panel_format(
            features, target
        )
        
        # Validate panel data quality
        is_valid = PanelDataTransformer.validate_panel_data(features_panel)
        
        # Extract cross-section for specific date
        cross_section = PanelDataTransformer.get_cross_section(
            features_panel, datetime(2024, 1, 15)
        )
    """
    
    @staticmethod
    def to_panel_format(
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Convert time series format to panel format.
        
        Transform from MultiIndex(symbol, date) to MultiIndex(date, symbol).
        This makes it easy to slice by date for cross-sectional analysis.
        
        Args:
            features: DataFrame with MultiIndex(symbol, date)
            target: Optional Series with MultiIndex(symbol, date)
        
        Returns:
            Tuple of (features_panel, target_panel) with MultiIndex(date, symbol)
        
        Raises:
            ValueError: If input data doesn't have the expected format
        """
        # Validate input
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Features must have MultiIndex")
        
        if len(features.index.names) != 2:
            raise ValueError(f"Expected 2-level MultiIndex, got {len(features.index.names)} levels")
        
        # Determine current index order
        index_names = features.index.names
        
        # Check if already in panel format (date, symbol)
        if index_names[0] == 'date' and index_names[1] == 'symbol':
            logger.debug("Data already in panel format (date, symbol)")
            features_panel = features.sort_index()
            target_panel = target.sort_index() if target is not None else None
            return features_panel, target_panel
        
        # Check if in time series format (symbol, date)
        elif index_names[0] == 'symbol' and index_names[1] == 'date':
            logger.debug("Converting from time series (symbol, date) to panel (date, symbol) format")
            features_panel = features.swaplevel(0, 1).sort_index()
            target_panel = target.swaplevel(0, 1).sort_index() if target is not None else None
            return features_panel, target_panel
        
        else:
            raise ValueError(f"Unexpected MultiIndex names: {index_names}. "
                           f"Expected ('symbol', 'date') or ('date', 'symbol')")
    
    @staticmethod
    def to_time_series_format(
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Convert panel format to time series format.
        
        Transform from MultiIndex(date, symbol) to MultiIndex(symbol, date).
        This is useful for symbol-by-symbol analysis.
        
        Args:
            features: DataFrame with MultiIndex(date, symbol)
            target: Optional Series with MultiIndex(date, symbol)
        
        Returns:
            Tuple of (features_ts, target_ts) with MultiIndex(symbol, date)
        """
        # Validate input
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Features must have MultiIndex")
        
        index_names = features.index.names
        
        # Check if already in time series format
        if index_names[0] == 'symbol' and index_names[1] == 'date':
            logger.debug("Data already in time series format (symbol, date)")
            features_ts = features.sort_index()
            target_ts = target.sort_index() if target is not None else None
            return features_ts, target_ts
        
        # Convert from panel to time series
        elif index_names[0] == 'date' and index_names[1] == 'symbol':
            logger.debug("Converting from panel (date, symbol) to time series (symbol, date) format")
            features_ts = features.swaplevel(0, 1).sort_index()
            target_ts = target.swaplevel(0, 1).sort_index() if target is not None else None
            return features_ts, target_ts
        
        else:
            raise ValueError(f"Unexpected MultiIndex names: {index_names}")
    
    @staticmethod
    def validate_panel_data(
        data: pd.DataFrame,
        min_symbols: int = 3,
        min_observations_per_date: int = 3,
        max_missing_pct: float = 0.5
    ) -> bool:
        """
        Validate panel data quality and completeness.
        
        Checks:
        1. Sufficient number of symbols
        2. Sufficient cross-sectional observations at each date
        3. Not too much missing data
        4. No dates with all NaN values
        
        Args:
            data: DataFrame with MultiIndex(date, symbol)
            min_symbols: Minimum number of unique symbols required
            min_observations_per_date: Minimum observations required per date
            max_missing_pct: Maximum allowable percentage of missing data
        
        Returns:
            True if validation passes, False otherwise
        """
        if data.empty:
            logger.error("Panel data validation failed: DataFrame is empty")
            return False
        
        if not isinstance(data.index, pd.MultiIndex):
            logger.error("Panel data validation failed: Not a MultiIndex")
            return False
        
        # Get index levels
        try:
            dates = data.index.get_level_values('date').unique()
            symbols = data.index.get_level_values('symbol').unique()
        except KeyError as e:
            logger.error(f"Panel data validation failed: Missing expected index level {e}")
            return False
        
        # Check 1: Sufficient symbols
        if len(symbols) < min_symbols:
            logger.error(f"Panel data validation failed: Only {len(symbols)} symbols, "
                        f"need at least {min_symbols}")
            return False
        
        # Check 2: Sufficient observations per date
        insufficient_dates = []
        for date in dates:
            try:
                cross_section = data.xs(date, level='date')
                if len(cross_section) < min_observations_per_date:
                    insufficient_dates.append((date, len(cross_section)))
            except Exception as e:
                logger.warning(f"Error checking date {date}: {e}")
                insufficient_dates.append((date, 0))
        
        if insufficient_dates:
            logger.warning(f"Panel data has {len(insufficient_dates)} dates with <{min_observations_per_date} observations")
            if len(insufficient_dates) > len(dates) * 0.5:  # More than 50% of dates
                logger.error("Panel data validation failed: Too many dates with insufficient observations")
                return False
        
        # Check 3: Missing data percentage
        total_values = data.shape[0] * data.shape[1]
        missing_values = data.isnull().sum().sum()
        missing_pct = missing_values / total_values
        
        if missing_pct > max_missing_pct:
            logger.error(f"Panel data validation failed: {missing_pct:.1%} missing data, "
                        f"threshold is {max_missing_pct:.1%}")
            return False
        
        # Check 4: No all-NaN rows
        all_nan_rows = data.isnull().all(axis=1).sum()
        if all_nan_rows > 0:
            logger.warning(f"Panel data has {all_nan_rows} rows with all NaN values")
            if all_nan_rows > len(data) * 0.1:  # More than 10%
                logger.error("Panel data validation failed: Too many all-NaN rows")
                return False
        
        logger.info(f"Panel data validation passed: {len(dates)} dates, {len(symbols)} symbols, "
                   f"{missing_pct:.1%} missing")
        return True
    
    @staticmethod
    def get_cross_section(
        data: pd.DataFrame,
        date: datetime
    ) -> pd.DataFrame:
        """
        Extract cross-sectional slice for a specific date.
        
        Args:
            data: Panel DataFrame with MultiIndex(date, symbol)
            date: Date to extract
        
        Returns:
            DataFrame with cross-sectional data (symbol as index)
        
        Raises:
            ValueError: If date not found
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex")
        
        try:
            cross_section = data.xs(date, level='date')
            logger.debug(f"Extracted cross-section for {date}: {len(cross_section)} observations")
            return cross_section
        except KeyError:
            # Try to find nearest date
            available_dates = data.index.get_level_values('date').unique()
            nearest_date = min(available_dates, key=lambda d: abs((d - date).total_seconds()))
            logger.warning(f"Date {date} not found, using nearest date {nearest_date}")
            return data.xs(nearest_date, level='date')
    
    @staticmethod
    def get_time_series(
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Extract time series for a specific symbol.
        
        Args:
            data: Panel DataFrame with MultiIndex(date, symbol)
            symbol: Symbol to extract
        
        Returns:
            DataFrame with time series data (date as index)
        
        Raises:
            ValueError: If symbol not found
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex")
        
        try:
            time_series = data.xs(symbol, level='symbol')
            logger.debug(f"Extracted time series for {symbol}: {len(time_series)} observations")
            return time_series
        except KeyError:
            raise ValueError(f"Symbol {symbol} not found in data")
    
    @staticmethod
    def balance_panel(
        data: pd.DataFrame,
        method: str = 'intersection'
    ) -> pd.DataFrame:
        """
        Balance panel data by ensuring consistent date-symbol coverage.
        
        Args:
            data: Unbalanced panel data
            method: Balancing method
                - 'intersection': Keep only date-symbol pairs present in all features
                - 'forward_fill': Forward fill missing values
                - 'drop': Drop any rows with missing values
        
        Returns:
            Balanced panel DataFrame
        """
        if method == 'intersection':
            # Keep only complete cases
            balanced = data.dropna()
            logger.info(f"Balanced panel using intersection: {len(balanced)} / {len(data)} rows retained")
            return balanced
        
        elif method == 'forward_fill':
            # Forward fill within each symbol
            balanced = data.copy()
            
            # Group by symbol and forward fill
            symbols = balanced.index.get_level_values('symbol').unique()
            filled_dfs = []
            
            for symbol in symbols:
                symbol_data = balanced.xs(symbol, level='symbol')
                symbol_data_filled = symbol_data.ffill()
                
                # Reconstruct MultiIndex
                symbol_data_filled.index = pd.MultiIndex.from_product(
                    [[symbol_data_filled.index.name], symbol_data_filled.index],
                    names=['symbol', 'date']
                )
                filled_dfs.append(symbol_data_filled)
            
            balanced = pd.concat(filled_dfs)
            logger.info(f"Balanced panel using forward fill")
            return balanced
        
        elif method == 'drop':
            balanced = data.dropna()
            logger.info(f"Balanced panel by dropping missing: {len(balanced)} / {len(data)} rows retained")
            return balanced
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
    
    @staticmethod
    def get_panel_summary(data: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics for panel data.
        
        Args:
            data: Panel DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        if not isinstance(data.index, pd.MultiIndex):
            return {'error': 'Not a MultiIndex DataFrame'}
        
        try:
            dates = data.index.get_level_values('date').unique()
            symbols = data.index.get_level_values('symbol').unique()
            
            # Calculate completeness
            expected_obs = len(dates) * len(symbols)
            actual_obs = len(data)
            completeness = actual_obs / expected_obs
            
            # Missing data
            total_values = data.shape[0] * data.shape[1]
            missing_values = data.isnull().sum().sum()
            missing_pct = missing_values / total_values
            
            # Observations per date
            obs_per_date = [len(data.xs(date, level='date')) for date in dates[:10]]  # Sample first 10
            avg_obs_per_date = np.mean([len(data.xs(date, level='date')) for date in dates])
            
            summary = {
                'n_dates': len(dates),
                'n_symbols': len(symbols),
                'n_observations': actual_obs,
                'expected_observations': expected_obs,
                'completeness': completeness,
                'missing_pct': missing_pct,
                'avg_obs_per_date': avg_obs_per_date,
                'date_range': (dates.min(), dates.max()),
                'sample_obs_per_date': obs_per_date
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}


