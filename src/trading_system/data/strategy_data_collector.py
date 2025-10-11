"""
Strategy Data Collector

This module provides functionality to collect historical strategy returns
for MetaModel training. It extracts strategy performance data from various
sources and formats it for training pipeline integration.

Key Features:
- Collect strategy returns from backtest results
- Align time series data across multiple strategies
- Format data for TrainingPipeline compatibility
- Support multiple data sources (backtest results, live trading, etc.)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StrategyDataCollector:
    """
    Collects and processes historical strategy returns data for MetaModel training.

    This class bridges the gap between strategy execution results and MetaModel
    training data preparation.
    """

    def __init__(self, data_dir: str = "./results"):
        """
        Initialize the data collector.

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
        Collect strategy returns from backtest result files.

        Args:
            strategy_names: List of strategy names to collect data for
            start_date: Start date for data collection
            end_date: End date for data collection
            target_benchmark: Optional benchmark symbol for target returns

        Returns:
            Tuple of (strategy_returns_df, target_returns_series)
        """
        logger.info(f"Collecting strategy returns for {len(strategy_names)} strategies "
                   f"from {start_date.date()} to {end_date.date()}")

        strategy_returns = {}

        for strategy_name in strategy_names:
            try:
                returns_file = self.data_dir / f"{strategy_name}_returns.csv"

                if returns_file.exists():
                    # Load returns from file
                    strategy_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)

                    # Filter by date range
                    mask = (strategy_df.index >= start_date) & (strategy_df.index <= end_date)
                    filtered_returns = strategy_df.loc[mask]

                    if not filtered_returns.empty:
                        # Use the 'returns' column or first numeric column
                        returns_col = 'returns' if 'returns' in filtered_returns.columns else filtered_returns.select_dtypes(include=[np.number]).columns[0]
                        strategy_returns[strategy_name] = filtered_returns[returns_col]
                        logger.info(f"Loaded {len(filtered_returns)} return observations for {strategy_name}")
                    else:
                        logger.warning(f"No data found for {strategy_name} in specified date range")
                else:
                    logger.warning(f"Returns file not found for strategy: {strategy_name}")

            except Exception as e:
                logger.error(f"Failed to load returns for {strategy_name}: {e}")
                continue

        if not strategy_returns:
            raise ValueError("No strategy returns data could be collected")

        # Create aligned DataFrame
        strategy_returns_df = pd.DataFrame(strategy_returns)

        # Remove rows with any NaN values (ensure all strategies have data for each date)
        strategy_returns_df = strategy_returns_df.dropna()

        logger.info(f"Aligned strategy returns DataFrame shape: {strategy_returns_df.shape}")

        # Generate target returns
        if target_benchmark:
            target_returns = self._get_benchmark_returns(target_benchmark, strategy_returns_df.index)
        else:
            # Use equal-weighted portfolio of strategies as target
            target_returns = strategy_returns_df.mean(axis=1)
            logger.info("Using equal-weighted strategy portfolio as target returns")

        logger.info(f"Target returns series shape: {target_returns.shape}")

        return strategy_returns_df, target_returns

    def collect_from_portfolio_files(self,
                                   strategy_patterns: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   target_benchmark: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Collect strategy returns from portfolio files using pattern matching.

        This method extracts returns from portfolio CSV files using file patterns.
        It uses the PortfolioReturnsExtractor utility for pure data processing.

        Args:
            strategy_patterns: List of file patterns (e.g., ['ml_strategy_*', 'ff5_*'])
            start_date: Start date for data collection
            end_date: End date for data collection
            target_benchmark: Optional benchmark symbol for target returns

        Returns:
            Tuple of (strategy_returns_df, target_returns_series)
        """
        from src.trading_system.data.extractor.portfolio_returns_extractor import PortfolioReturnsExtractor

        logger.info(f"Collecting strategy returns from portfolio files using {len(strategy_patterns)} patterns "
                   f"from {start_date.date()} to {end_date.date()}")

        strategy_returns = {}

        # Find all matching portfolio files
        portfolio_files = []
        for pattern in strategy_patterns:
            matching_files = list(self.data_dir.glob(f"{pattern}_portfolio.csv"))
            portfolio_files.extend(matching_files)

        logger.info(f"Found {len(portfolio_files)} matching portfolio files")

        # Extract returns from each portfolio file
        for portfolio_file in portfolio_files:
            try:
                # Extract strategy name from filename
                strategy_name = portfolio_file.stem.replace('_portfolio', '')

                # Extract returns using the pure function utility
                returns_series = PortfolioReturnsExtractor.extract_returns_from_portfolio(portfolio_file)

                if not returns_series.empty:
                    # Filter by date range
                    mask = (returns_series.index >= start_date) & (returns_series.index <= end_date)
                    filtered_returns = returns_series.loc[mask]

                    if not filtered_returns.empty:
                        strategy_returns[strategy_name] = filtered_returns
                        logger.info(f"Extracted {len(filtered_returns)} returns for {strategy_name}")
                    else:
                        logger.warning(f"No data found for {strategy_name} in specified date range")
                else:
                    logger.warning(f"No returns extracted from {portfolio_file}")

            except Exception as e:
                logger.error(f"Failed to extract returns from {portfolio_file}: {e}")
                continue

        if not strategy_returns:
            raise ValueError("No strategy returns data could be collected from portfolio files")

        # Align returns series using the pure function utility
        returns_list = [(name, series) for name, series in strategy_returns.items()]
        aligned_returns_df = PortfolioReturnsExtractor.align_returns_series(returns_list, method='inner')

        logger.info(f"Aligned strategy returns DataFrame shape: {aligned_returns_df.shape}")

        # Validate data quality
        is_valid, issues = PortfolioReturnsExtractor.validate_returns_data(aligned_returns_df)
        if not is_valid:
            logger.warning(f"Data quality issues found: {issues}")

        # Generate target returns
        if target_benchmark and target_benchmark.lower() != 'equal_weighted':
            target_returns = self._get_benchmark_returns(target_benchmark, aligned_returns_df.index)
        else:
            # Use equal-weighted portfolio (financial industry standard)
            target_returns = PortfolioReturnsExtractor.create_equal_weighted_target(aligned_returns_df)
            logger.info("Using equal-weighted strategy portfolio as target returns")

        # Calculate and log strategy statistics
        stats = PortfolioReturnsExtractor.calculate_strategy_statistics(aligned_returns_df)
        logger.info("Strategy statistics:")
        for strategy, stat in stats.items():
            logger.info(f"  {strategy}: Return={stat['annual_return']:.2%}, "
                       f"Vol={stat['annual_volatility']:.2%}, "
                       f"Sharpe={stat['sharpe_ratio']:.2f}")

        logger.info(f"Target returns series shape: {target_returns.shape}")

        return aligned_returns_df, target_returns

    def collect_from_live_results(self,
                                 strategy_results: Dict[str, pd.DataFrame],
                                 target_benchmark: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Collect strategy returns from live trading results.

        Args:
            strategy_results: Dictionary of strategy name to returns DataFrame
            target_benchmark: Optional benchmark symbol for target returns

        Returns:
            Tuple of (strategy_returns_df, target_returns_series)
        """
        logger.info(f"Processing live results for {len(strategy_results)} strategies")

        # Align all strategy returns
        strategy_returns_df = pd.DataFrame(strategy_results)
        strategy_returns_df = strategy_returns_df.dropna()

        # Generate target returns
        if target_benchmark:
            target_returns = self._get_benchmark_returns(target_benchmark, strategy_returns_df.index)
        else:
            # Use equal-weighted portfolio as target
            target_returns = strategy_returns_df.mean(axis=1)

        logger.info(f"Live data processed - Shape: {strategy_returns_df.shape}")

        return strategy_returns_df, target_returns

    def _get_benchmark_returns(self, benchmark_symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Get benchmark returns for the specified dates.

        Args:
            benchmark_symbol: Symbol like 'SPY' for S&P 500
            dates: Date index for returns

        Returns:
            Series of benchmark returns
        """
        try:
            from ..data.yfinance_provider import YFinanceProvider

            provider = YFinanceProvider()

            # Download benchmark data
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = (dates.max() + timedelta(days=1)).strftime('%Y-%m-%d')

            benchmark_data = provider.fetch_data([benchmark_symbol], start_date, end_date)

            if benchmark_symbol in benchmark_data:
                price_data = benchmark_data[benchmark_symbol]['Close']
                returns = price_data.pct_change().dropna()

                # Align with requested dates
                aligned_returns = returns.reindex(dates, method='ffill').dropna()

                logger.info(f"Loaded benchmark returns for {benchmark_symbol}: {len(aligned_returns)} observations")
                return aligned_returns
            else:
                logger.warning(f"Could not fetch data for benchmark {benchmark_symbol}")
                return pd.Series(0.0, index=dates)  # Return zero returns as fallback

        except Exception as e:
            logger.error(f"Failed to fetch benchmark returns: {e}")
            return pd.Series(0.0, index=dates)  # Return zero returns as fallback

    def create_synthetic_strategy_data(self,
                                      strategies_config: Dict[str, Dict[str, Any]],
                                      start_date: datetime,
                                      end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create synthetic strategy returns for testing/demo purposes.

        Returns data in MultiIndex format to match existing TrainingPipeline expectations:
        - Features: MultiIndex (symbol, date) with strategy returns as features
        - Target: MultiIndex (symbol, date) with target returns

        Args:
            strategies_config: Dictionary of strategy configurations
            start_date: Start date for synthetic data
            end_date: End date for synthetic data

        Returns:
            Tuple of (strategy_features_df, target_returns_series)
        """
        logger.info("Generating synthetic strategy returns data in MultiIndex format")

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)

        np.random.seed(42)  # For reproducible results

        strategy_returns = {}

        for strategy_name, config in strategies_config.items():
            # Generate returns based on strategy configuration
            annual_return = config.get('annual_return', 0.08)
            annual_volatility = config.get('annual_volatility', 0.15)
            correlation_factor = config.get('correlation_factor', 0.3)

            daily_return = annual_return / 252
            daily_volatility = annual_volatility / np.sqrt(252)

            # Generate correlated returns
            base_returns = np.random.normal(daily_return, daily_volatility, n_days)

            # Add some correlation with other strategies
            correlated_noise = correlation_factor * np.random.normal(0, daily_volatility, n_days)
            strategy_return_series = base_returns + correlated_noise

            strategy_returns[strategy_name] = pd.Series(strategy_return_series, index=date_range)

        # Create DataFrame (single index format)
        strategy_returns_df = pd.DataFrame(strategy_returns)

        # Create target returns (weighted combination + some noise)
        target_weights = [0.4, 0.3, 0.3]  # Example weights
        if len(strategy_returns_df.columns) >= len(target_weights):
            target_returns = (
                strategy_returns_df.iloc[:, :len(target_weights)].dot(target_weights) +
                np.random.normal(0, 0.001, n_days)  # Add some noise
            )
        else:
            target_returns = strategy_returns_df.mean(axis=1)

        target_returns = pd.Series(target_returns, index=date_range)

        # Convert to MultiIndex format like TrainingPipeline expects
        # For MetaModel, we treat each strategy as a "symbol" with its returns as features
        feature_data = []
        target_data = []

        for strategy in strategy_returns_df.columns:
            # Create features for each strategy (using its own returns)
            strategy_data = pd.DataFrame({
                'strategy_return': strategy_returns_df[strategy],
                'volatility_5d': strategy_returns_df[strategy].rolling(5).std().fillna(0),
                'momentum_10d': strategy_returns_df[strategy].rolling(10).mean().fillna(0),
                'ma_5d': strategy_returns_df[strategy].rolling(5).mean().fillna(0)
            })

            # Add symbol column for MultiIndex
            strategy_data['symbol'] = strategy
            strategy_data.index.name = 'date'
            strategy_data = strategy_data.set_index('symbol', append=True).reorder_levels(['symbol', 'date'])

            feature_data.append(strategy_data)

            # Create target data for this strategy (using overall target returns)
            target_series = target_returns.copy()
            target_series.name = 'target'
            target_df = target_series.to_frame()
            target_df['symbol'] = strategy
            target_df.index.name = 'date'
            target_df = target_df.set_index('symbol', append=True).reorder_levels(['symbol', 'date'])

            target_data.append(target_df['target'])

        # Combine all strategies
        if feature_data:
            strategy_features = pd.concat(feature_data)
            target_returns_multi = pd.concat(target_data)
        else:
            raise ValueError("No feature data generated")

        logger.info(f"Generated synthetic MultiIndex data - Features shape: {strategy_features.shape}, Target shape: {target_returns_multi.shape}")

        return strategy_features, target_returns_multi

    def save_collected_data(self,
                           strategy_returns: pd.DataFrame,
                           target_returns: pd.Series,
                           output_dir: str,
                           experiment_name: str) -> str:
        """
        Save collected data for later use.

        Args:
            strategy_returns: DataFrame of strategy returns
            target_returns: Series of target returns
            output_dir: Directory to save data
            experiment_name: Name for this data collection

        Returns:
            Path to the saved data directory
        """
        output_path = Path(output_dir) / f"metamodel_data_{experiment_name}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save strategy returns
        strategy_returns.to_csv(output_path / "strategy_returns.csv")

        # Save target returns
        target_returns.to_csv(output_path / "target_returns.csv")

        # Save metadata
        metadata = {
            'experiment_name': experiment_name,
            'collection_date': datetime.now().isoformat(),
            'strategies': list(strategy_returns.columns),
            'date_range': {
                'start': strategy_returns.index.min().isoformat(),
                'end': strategy_returns.index.max().isoformat()
            },
            'observations': len(strategy_returns),
            'summary_stats': {
                'strategy_returns': strategy_returns.describe().to_dict(),
                'target_returns': target_returns.describe().to_dict()
            }
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Data saved to: {output_path}")
        return str(output_path)

    def load_collected_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load previously collected data.

        Args:
            data_path: Path to the collected data directory

        Returns:
            Tuple of (strategy_returns_df, target_returns_series, metadata)
        """
        data_path = Path(data_path)

        # Load data files
        strategy_returns = pd.read_csv(data_path / "strategy_returns.csv", index_col=0, parse_dates=True)
        target_returns = pd.read_csv(data_path / "target_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]

        # Load metadata
        with open(data_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded collected data from: {data_path}")
        return strategy_returns, target_returns, metadata