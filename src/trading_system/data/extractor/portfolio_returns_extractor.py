"""
Portfolio Returns Extractor - Pure Functions for Data Processing
================================================================

This module provides pure functions for extracting returns from portfolio files.
Following functional programming principles - no side effects, no state.

Design Principles:
- Pure functions only (no state, no side effects)
- Single responsibility for each function
- Type hints for all functions
- Comprehensive error handling
- Financial industry standards compliance
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PortfolioReturnsExtractor:
    """
    Pure functions collection for extracting returns from portfolio data files.

    This class contains only static methods - no state, no side effects.
    Each function is independently testable and follows functional programming principles.
    """

    @staticmethod
    def extract_returns_from_portfolio(file_path: Union[str, Path]) -> pd.Series:
        """
        Extract daily returns from a portfolio CSV file.

        Handles different portfolio file formats found in the results directory.
        Follows financial industry standards for returns calculation.

        Args:
            file_path: Path to portfolio CSV file

        Returns:
            pd.Series: Daily returns indexed by date

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or no data found
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {file_path}")

        try:
            # Load portfolio data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # Handle different portfolio file formats
            returns_series = PortfolioReturnsExtractor._extract_returns_from_dataframe(df)

            if returns_series.empty:
                raise ValueError(f"No valid returns data found in {file_path}")

            logger.info(f"Extracted {len(returns_series)} returns from {file_path.name}")
            return returns_series

        except Exception as e:
            logger.error(f"Failed to extract returns from {file_path}: {e}")
            raise ValueError(f"Error processing portfolio file {file_path}: {e}")

    @staticmethod
    def _extract_returns_from_dataframe(df: pd.DataFrame) -> pd.Series:
        """
        Extract returns series from a portfolio dataframe.

        Handles different column names and data formats.

        Args:
            df: Portfolio dataframe

        Returns:
            pd.Series: Daily returns
        """
        if df.empty:
            return pd.Series(dtype=float)

        # Method 1: Direct returns column
        if 'returns' in df.columns:
            return df['returns'].dropna()

        # Method 2: Calculate from portfolio_value column
        if 'portfolio_value' in df.columns:
            portfolio_values = df['portfolio_value'].dropna()
            if len(portfolio_values) < 2:
                return pd.Series(dtype=float)

            # Calculate simple returns (financial industry standard)
            returns = portfolio_values.pct_change().dropna()
            return returns

        # Method 3: Handle multi-column format (ml_strategy format)
        # Find the first numeric column that looks like portfolio values
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # Skip columns that are clearly not portfolio values
            if col.lower() in ['cash', 'positions', 'trades']:
                continue

            # Check if this column looks like portfolio values
            values = df[col].dropna()
            if len(values) > 10 and values.min() > 0:  # Portfolio values should be positive
                # Test if this could be portfolio values by checking for reasonable variance
                if (values.max() / values.min()) < 10:  # Not too much variance for portfolio values
                    returns = values.pct_change().dropna()
                    if len(returns) > 10:
                        logger.info(f"Using column '{col}' for portfolio values")
                        return returns

        # Method 4: Last resort - use first numeric column
        if len(numeric_columns) > 0:
            logger.warning(f"Using first numeric column '{numeric_columns[0]}' as portfolio values")
            values = df[numeric_columns[0]].dropna()
            if len(values) > 1:
                return values.pct_change().dropna()

        logger.error("No suitable portfolio value column found")
        return pd.Series(dtype=float)

    @staticmethod
    def align_returns_series(returns_list: list, method: str = 'inner') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align multiple returns series to common dates.

        Args:
            returns_list: List of (name, pd.Series) tuples
            method: Alignment method ('inner', 'outer', 'left', 'right')

        Returns:
            Tuple of (aligned_returns_df, common_dates_mask)
        """
        if not returns_list:
            return pd.DataFrame(), pd.Series(dtype=bool)

        # Create dataframe from all series
        returns_dict = {name: series for name, series in returns_list}
        aligned_df = pd.DataFrame(returns_dict)

        # Apply alignment method
        if method == 'inner':
            aligned_df = aligned_df.dropna()
        elif method == 'outer':
            aligned_df = aligned_df.fillna(0.0)  # Fill missing returns with 0
        elif method in ['left', 'right']:
            # Keep the index of the first series
            first_series = returns_list[0][1]
            aligned_df = aligned_df.reindex(first_series.index, method='ffill').fillna(0.0)

        logger.info(f"Aligned {len(returns_list)} returns series to {len(aligned_df)} common dates")
        return aligned_df

    @staticmethod
    def create_equal_weighted_target(returns_df: pd.DataFrame) -> pd.Series:
        """
        Create equal-weighted target returns from strategy returns.

        This is the standard academic benchmark for strategy combination.

        Args:
            returns_df: DataFrame of strategy returns

        Returns:
            pd.Series: Equal-weighted target returns
        """
        if returns_df.empty:
            return pd.Series(dtype=float)

        # Equal weighted portfolio (financial industry standard)
        target_returns = returns_df.mean(axis=1)

        logger.info(f"Created equal-weighted target from {len(returns_df.columns)} strategies")
        return target_returns

    @staticmethod
    def calculate_strategy_statistics(returns_df: pd.DataFrame) -> dict:
        """
        Calculate basic statistics for each strategy.

        Args:
            returns_df: DataFrame of strategy returns

        Returns:
            dict: Statistics for each strategy
        """
        if returns_df.empty:
            return {}

        stats = {}

        for strategy in returns_df.columns:
            series = returns_df[strategy].dropna()

            if len(series) == 0:
                continue

            # Annualized return (252 trading days per year - industry standard)
            annual_return = series.mean() * 252

            # Annualized volatility
            annual_volatility = series.std() * np.sqrt(252)

            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

            # Maximum drawdown
            cumulative_returns = (1 + series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            stats[strategy] = {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_observations': len(series),
                'start_date': series.index.min(),
                'end_date': series.index.max()
            }

        return stats

    @staticmethod
    def validate_returns_data(returns_df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate returns data for financial reasonableness.

        Args:
            returns_df: DataFrame of strategy returns

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if returns_df.empty:
            issues.append("Returns dataframe is empty")
            return False, issues

        # Check for extreme values (outside reasonable daily return range)
        extreme_threshold = 0.5  # 50% daily return is extreme
        extreme_values = (returns_df.abs() > extreme_threshold).any()
        if extreme_values.any():
            extreme_cols = returns_df.columns[extreme_values].tolist()
            issues.append(f"Extreme daily returns found in columns: {extreme_cols}")

        # Check for missing data
        missing_data = returns_df.isnull().sum()
        if missing_data.any():
            missing_cols = missing_data[missing_data > 0].to_dict()
            issues.append(f"Missing data found: {missing_cols}")

        # Check for constant returns (suspicious)
        for col in returns_df.columns:
            if returns_df[col].nunique() <= 2:
                issues.append(f"Column '{col}' has constant or near-constant values")

        # Check for reasonable volatility range
        for col in returns_df.columns:
            vol = returns_df[col].std()
            if vol < 1e-6:  # Essentially zero volatility
                issues.append(f"Column '{col}' has near-zero volatility")
            elif vol > 0.1:  # Very high daily volatility (>10%)
                issues.append(f"Column '{col}' has very high volatility: {vol:.2%}")

        return len(issues) == 0, issues