"""
Data Validation Utilities

Centralized validation logic following DRY principles.
Moved from types/data_types.py to follow SOLID design.
"""

import pandas as pd
from typing import List, Dict


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """
    Data validation utilities.

    Centralized validation for price data, factor data, signals, and portfolios.
    Moved from types/data_types.py to follow SOLID single responsibility principle.
    """

    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate price data DataFrame.

        Args:
            df: DataFrame with price data
            symbol: Asset symbol for validation

        Returns:
            True if valid, raises DataValidationError otherwise

        Raises:
            DataValidationError: If data is invalid
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            raise DataValidationError(f"Empty DataFrame for symbol {symbol}")

        # Check for missing values
        if df[required_columns].isnull().all().any():
            raise DataValidationError(f"All values missing in required columns for {symbol}")

        # Validate price ranges
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] <= 0).any():
                raise DataValidationError(f"Non-positive prices found in {col} for {symbol}")

        # Validate High-Low relationships
        invalid_hl = (df['High'] < df['Low']).any()
        if invalid_hl:
            raise DataValidationError(f"High < Low found for {symbol}")

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).any()

        if invalid_ohlc:
            raise DataValidationError(f"Invalid OHLC relationships found for {symbol}")

        # Validate volume
        if (df['Volume'] < 0).any():
            raise DataValidationError(f"Negative volume found for {symbol}")

        return True

    @staticmethod
    def validate_factor_data(df: pd.DataFrame, data_source: str) -> bool:
        """
        Validate factor data DataFrame.

        Args:
            df: DataFrame with factor data
            data_source: Name of the data source for validation

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check required FF5 columns
        required_columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required factor columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            raise DataValidationError(f"Empty factor DataFrame for source {data_source}")

        # Check for missing values
        if df[required_columns].isnull().all().any():
            raise DataValidationError(f"All values missing in required factor columns for {data_source}")

        # Validate factor return ranges (should be reasonable)
        for col in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
            if ((df[col] < -1.0) | (df[col] > 1.0)).any():
                raise DataValidationError(f"Factor returns out of range for {col} in {data_source}")

        # Validate risk-free rate range
        if ((df['RF'] < -0.1) | (df['RF'] > 0.1)).any():
            raise DataValidationError(f"Risk-free rate out of range for {data_source}")

        return True

    @staticmethod
    def validate_signals(signals: pd.DataFrame, symbols: List[str]) -> bool:
        """
        Validate trading signals DataFrame.

        Args:
            signals: DataFrame with trading signals
            symbols: Expected symbols in the signals

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check that all expected symbols are present
        missing_symbols = [sym for sym in symbols if sym not in signals.columns]
        if missing_symbols:
            raise DataValidationError(f"Missing signals for symbols: {missing_symbols}")

        # Check signal values are within valid range
        for symbol in symbols:
            if symbol in signals.columns:
                signal_values = signals[symbol].dropna()
                if ((signal_values < 0) | (signal_values > 1)).any():
                    raise DataValidationError(f"Invalid signal values for {symbol}")

        return True

    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
        """
        Validate portfolio weights.

        Args:
            weights: Dictionary of symbol -> weight mappings

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            raise DataValidationError(f"Portfolio weights sum to {total_weight}, expected 1.0")

        # Check individual weights are valid
        for symbol, weight in weights.items():
            if weight < 0 or weight > 1:
                raise DataValidationError(f"Invalid weight {weight} for {symbol}")

        return True