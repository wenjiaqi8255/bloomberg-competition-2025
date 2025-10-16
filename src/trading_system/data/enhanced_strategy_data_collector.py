"""
Enhanced Strategy Data Collector
================================

Enhanced version of StrategyDataCollector with improved error handling,
data validation, and support for the new standardized returns format.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataCollectionError(Exception):
    """Exception raised when data collection fails."""
    pass


class EnhancedStrategyDataCollector:
    """
    Enhanced strategy data collector with single responsibility: collect real strategy returns.

    INPUTS:
    - strategy_names: List[str] - Names of strategies to collect
    - start_date: datetime - Start date for collection
    - end_date: datetime - End date for collection
    - data_dir: str - Directory containing strategy results

    OUTPUTS:
    - strategy_returns: pd.DataFrame - Columns=strategies, Index=dates, Values=returns
    - target_returns: pd.Series - Index=dates, Values=benchmark returns

    RESPONSIBILITY: Only collect and validate real strategy returns data.
    """
    def __init__(self, data_dir: str = "./results"):
        """
        Initialize the enhanced data collector.

        Args:
            data_dir: Directory containing strategy results files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_backtest_results(self,
                                     strategy_names: List[str],
                                     start_date: datetime,
                                     end_date: datetime,
                                     target_benchmark: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Collect strategy returns from backtest result files with enhanced validation.
        
        Args:
            strategy_names: List of strategy names to collect data for
            start_date: Start date for data collection
            end_date: End date for data collection
            target_benchmark: Optional benchmark symbol for target returns

        Returns:
            Tuple of (strategy_returns_df, target_returns_series)
            
        Raises:
            DataCollectionError: If data collection fails
        """
        logger.info(f"Collecting returns for {len(strategy_names)} strategies")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # 1. Collect all strategy returns
        all_returns = {}
        missing_strategies = []
        
        for strategy_name in strategy_names:
            returns_file = self._get_returns_file_path(strategy_name)
            
            if not returns_file.exists():
                logger.error(f"Returns file not found: {returns_file}")
                missing_strategies.append(strategy_name)
                continue
            
            try:
                # Load and validate returns data
                returns = self._load_and_validate_returns_file(returns_file, start_date, end_date)
                all_returns[strategy_name] = returns
                logger.info(f"✓ Loaded {len(returns)} observations for {strategy_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {strategy_name}: {e}")
                missing_strategies.append(strategy_name)
        
        # 2. Check for missing strategies
        if missing_strategies:
            raise DataCollectionError(
                f"Failed to collect returns for {len(missing_strategies)} strategies:\n"
                f"{missing_strategies}\n"
                f"Expected files:\n" + 
                "\n".join([str(self._get_returns_file_path(s)) for s in missing_strategies])
            )
        
        # 3. Align time series
        returns_df = pd.DataFrame(all_returns)
        
        # 4. Handle missing data
        returns_df = self._handle_missing_data(returns_df)
        
        # 5. Validate final data quality
        self._validate_returns_data(returns_df)
        
        # 6. Generate target returns
        if target_benchmark:
            target_returns = self._get_benchmark_returns(target_benchmark, returns_df.index)
        else:
            # Use equal-weighted portfolio as benchmark
            target_returns = returns_df.mean(axis=1)
            logger.info("Using equal-weighted strategy portfolio as benchmark")
        
        logger.info(f"Successfully collected {len(returns_df)} observations "
                   f"for {len(returns_df.columns)} strategies")
        
        return returns_df, target_returns

    def _get_returns_file_path(self, strategy_name: str) -> Path:
        """
        Get the standardized path for strategy returns file.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Path to the returns file
        """
        return self.data_dir / strategy_name / "strategy_returns.csv"

    def _load_and_validate_returns_file(self, 
                                      file_path: Path, 
                                      start_date: datetime, 
                                      end_date: datetime) -> pd.Series:
        """
        Load and validate a single returns file.
        
        Args:
            file_path: Path to the returns file
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered returns series
            
        Raises:
            ValueError: If file format is invalid
        """
        try:
            # Load the file
            returns_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Validate format
            if 'daily_return' not in returns_df.columns:
                raise ValueError(f"Missing 'daily_return' column in {file_path}")
            
            # Extract returns series
            returns = returns_df['daily_return']
            
            # Filter by date range
            mask = (returns.index >= start_date) & (returns.index <= end_date)
            filtered_returns = returns.loc[mask]
            
            if len(filtered_returns) == 0:
                raise ValueError(f"No data in date range for {file_path}")
            
            return filtered_returns
            
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")

    def _handle_missing_data(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the returns matrix.
        
        Args:
            returns_df: Returns DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        missing_pct = returns_df.isnull().sum() / len(returns_df)
        
        if missing_pct.max() > 0.1:  # More than 10% missing
            problematic = missing_pct[missing_pct > 0.1]
            logger.warning(
                f"High missing data rate detected:\n{problematic}"
            )
        
        # Forward fill first, then fill remaining with 0 (no trading day)
        returns_df = returns_df.fillna(method='ffill')
        returns_df = returns_df.fillna(0)
        
        return returns_df

    def _validate_returns_data(self, returns_df: pd.DataFrame):
        """
        Validate the quality of returns data.
        
        Args:
            returns_df: Returns DataFrame to validate
            
        Raises:
            ValueError: If data quality issues are found
        """
        # 1. Check for empty DataFrame
        if returns_df.empty:
            raise ValueError("Returns matrix is empty")
        
        # 2. Check minimum number of strategies
        if len(returns_df.columns) < 2:
            raise ValueError(
                f"Need at least 2 strategies, got {len(returns_df.columns)}"
            )
        
        # 3. Check minimum number of observations
        min_observations = 20  # At least 20 trading days
        if len(returns_df) < min_observations:
            raise ValueError(
                f"Insufficient data: {len(returns_df)} observations, "
                f"need at least {min_observations}"
            )
        
        # 4. Check for extreme returns
        extreme_mask = returns_df.abs() > 0.5  # Daily returns > 50%
        if extreme_mask.any().any():
            extreme_counts = extreme_mask.sum()
            logger.warning(
                f"Extreme returns (>50%) detected:\n"
                f"{extreme_counts[extreme_counts > 0]}"
            )
        
        # 5. Check for zero variance strategies
        zero_variance = returns_df.std() == 0
        if zero_variance.any():
            logger.warning(
                f"Strategies with zero variance:\n"
                f"{returns_df.columns[zero_variance].tolist()}"
            )
        
        # 6. Check time series continuity
        date_diff = returns_df.index.to_series().diff()
        max_gap = date_diff.max()
        if max_gap > pd.Timedelta(days=5):
            logger.warning(f"Time series has gaps up to {max_gap}")
        
        logger.info("✓ Returns data validation passed")

    def _get_benchmark_returns(self, benchmark: str, date_index: pd.DatetimeIndex) -> pd.Series:
        """
        Get benchmark returns for the specified date range.
        
        Args:
            benchmark: Benchmark symbol (e.g., 'SPY')
            date_index: Date index to align with
            
        Returns:
            Benchmark returns series
        """
        # This is a simplified implementation
        # In a real system, you would fetch benchmark data from a data provider
        logger.warning(f"Benchmark returns for {benchmark} not implemented, using zero returns")
        return pd.Series(0.0, index=date_index, name=benchmark)

    def validate_returns_file(self, file_path: str) -> bool:
        """
        Validate a single returns file format.
        
        Args:
            file_path: Path to the returns file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            returns_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check required columns
            if 'daily_return' not in returns_df.columns:
                logger.error(f"Missing 'daily_return' column in {file_path}")
                return False
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(returns_df['daily_return']):
                logger.error(f"'daily_return' column is not numeric in {file_path}")
                return False
            
            # Check for reasonable values
            returns = returns_df['daily_return']
            if (returns.abs() > 1.0).any():  # Returns > 100%
                logger.warning(f"Extreme returns detected in {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate {file_path}: {e}")
            return False

    def get_collection_summary(self, strategy_names: List[str]) -> Dict[str, Any]:
        """
        Get a summary of data collection status for strategies.
        
        Args:
            strategy_names: List of strategy names to check
            
        Returns:
            Summary dictionary with collection status
        """
        summary = {
            'total_strategies': len(strategy_names),
            'available_strategies': [],
            'missing_strategies': [],
            'invalid_strategies': [],
            'file_paths': {}
        }
        
        for strategy_name in strategy_names:
            file_path = self._get_returns_file_path(strategy_name)
            summary['file_paths'][strategy_name] = str(file_path)
            
            if not file_path.exists():
                summary['missing_strategies'].append(strategy_name)
            elif not self.validate_returns_file(str(file_path)):
                summary['invalid_strategies'].append(strategy_name)
            else:
                summary['available_strategies'].append(strategy_name)
        
        return summary




