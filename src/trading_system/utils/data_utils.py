"""
Data Utilities Module

Common data processing utilities extracted from BaseStrategy to eliminate duplication
and provide consistent data handling across all strategies.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import warnings


class DataProcessor:
    """
    Utility class for common data processing operations.
    Extracted from BaseStrategy to follow DRY principle.
    """

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data

        Args:
            prices: DataFrame with price data
            method: 'simple' for simple returns, 'log' for log returns

        Returns:
            DataFrame with returns
        """
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")

    @staticmethod
    def calculate_volatility(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling volatility

        Args:
            returns: DataFrame with returns data
            window: Rolling window for volatility calculation

        Returns:
            DataFrame with volatility values
        """
        return returns.rolling(window=window).std() * np.sqrt(window)

    @staticmethod
    def prepare_data(prices: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare price data for strategy processing

        Args:
            prices: Raw price data
            config: Configuration dictionary with processing parameters

        Returns:
            Processed data DataFrame
        """
        # Fill missing values
        if config.get('fill_method', 'ffill') == 'ffill':
            prices = prices.fillna(method='ffill')
        else:
            prices = prices.fillna(method='bfill')

        # Remove outliers if specified
        if config.get('remove_outliers', False):
            threshold = config.get('outlier_threshold', 3.0)
            for column in prices.columns:
                mean = prices[column].mean()
                std = prices[column].std()
                prices[column] = prices[column].clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std
                )

        return prices

    @staticmethod
    def validate_price_data(prices: pd.DataFrame, min_periods: int = 252) -> bool:
        """
        Validate price data for processing

        Args:
            prices: DataFrame with price data
            min_periods: Minimum number of observations required

        Returns:
            True if data is valid, False otherwise
        """
        if len(prices) < min_periods:
            warnings.warn(f"Insufficient data: {len(prices)} < {min_periods}")
            return False

        if prices.isnull().all().any():
            warnings.warn("Columns with all missing values found")
            return False

        if (prices <= 0).any().any():
            warnings.warn("Non-positive prices found")
            return False

        return True

    @staticmethod
    def normalize_data(data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize data using specified method

        Args:
            data: DataFrame to normalize
            method: 'zscore', 'minmax', or 'robust'

        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")

    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in data

        Args:
            data: DataFrame with potential missing values
            method: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'

        Returns:
            DataFrame with missing values handled
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate()
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError("Method must be 'forward_fill', 'backward_fill', 'interpolate', or 'drop'")

    @staticmethod
    def resample_data(data: pd.DataFrame, frequency: str, method: str = 'last') -> pd.DataFrame:
        """
        Resample data to different frequency

        Args:
            data: DataFrame to resample
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            method: Aggregation method ('last', 'first', 'mean', 'sum')

        Returns:
            Resampled DataFrame
        """
        if method == 'last':
            return data.resample(frequency).last()
        elif method == 'first':
            return data.resample(frequency).first()
        elif method == 'mean':
            return data.resample(frequency).mean()
        elif method == 'sum':
            return data.resample(frequency).sum()
        else:
            raise ValueError("Method must be 'last', 'first', 'mean', or 'sum'")

    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for returns

        Args:
            returns: DataFrame with returns data
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix
        """
        return returns.corr(method=method)

    @staticmethod
    def calculate_drawdown(returns: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdown metrics

        Args:
            returns: Series of returns

        Returns:
            DataFrame with drawdown information
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        return pd.DataFrame({
            'cumulative_return': cumulative,
            'peak': peak,
            'drawdown': drawdown
        })

    @staticmethod
    def clean_data(data: pd.DataFrame,
                   remove_duplicates: bool = True,
                   remove_outliers: bool = False,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Clean data by removing duplicates and outliers

        Args:
            data: DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')

        Returns:
            Cleaned DataFrame
        """
        result = data.copy()

        if remove_duplicates:
            result = result.drop_duplicates()

        if remove_outliers:
            for column in result.select_dtypes(include=[np.number]).columns:
                if outlier_method == 'iqr':
                    Q1 = result[column].quantile(0.25)
                    Q3 = result[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result = result[(result[column] >= lower_bound) & (result[column] <= upper_bound)]
                elif outlier_method == 'zscore':
                    z_scores = np.abs((result[column] - result[column].mean()) / result[column].std())
                    result = result[z_scores < 3]

        return result

    @staticmethod
    def create_lagged_features(data: pd.DataFrame, lags: list) -> pd.DataFrame:
        """
        Create lagged features for time series analysis

        Args:
            data: DataFrame with original data
            lags: List of lag periods to create

        Returns:
            DataFrame with lagged features
        """
        lagged_data = {}

        for column in data.columns:
            lagged_data[column] = data[column]
            for lag in lags:
                lagged_data[f'{column}_lag_{lag}'] = data[column].shift(lag)

        return pd.DataFrame(lagged_data)

    @staticmethod
    def calculate_momentum(prices: pd.DataFrame, periods: list = [21, 63, 252]) -> pd.DataFrame:
        """
        Calculate momentum indicators for different time periods.

        Args:
            prices: DataFrame with price data
            periods: List of lookback periods in days

        Returns:
            DataFrame with momentum indicators
        """
        momentum_data = {}

        for period in periods:
            momentum = prices.pct_change(period)
            if isinstance(momentum, pd.DataFrame):
                # Multi-column case
                for col in momentum.columns:
                    momentum_data[f'{col}_momentum_{period}d'] = momentum[col]
            else:
                # Single column case
                momentum_data[f'momentum_{period}d'] = momentum

        return pd.DataFrame(momentum_data)

    @staticmethod
    def calculate_rolling_statistics(data: pd.DataFrame, window: int,
                                   stats: list = ['mean', 'std']) -> pd.DataFrame:
        """
        Calculate rolling statistics for data.

        Args:
            data: DataFrame with data
            window: Rolling window size
            stats: List of statistics to calculate ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with rolling statistics
        """
        result_data = {}

        for column in data.columns:
            for stat in stats:
                if stat == 'mean':
                    result_data[f'{column}_rolling_mean_{window}'] = data[column].rolling(window).mean()
                elif stat == 'std':
                    result_data[f'{column}_rolling_std_{window}'] = data[column].rolling(window).std()
                elif stat == 'min':
                    result_data[f'{column}_rolling_min_{window}'] = data[column].rolling(window).min()
                elif stat == 'max':
                    result_data[f'{column}_rolling_max_{window}'] = data[column].rolling(window).max()

        return pd.DataFrame(result_data)

    @staticmethod
    def calculate_rolling_returns(prices: pd.DataFrame, windows: list = [21, 63, 252]) -> pd.DataFrame:
        """
        Calculate rolling returns for different windows.

        Args:
            prices: DataFrame with price data
            windows: List of rolling windows in days

        Returns:
            DataFrame with rolling returns
        """
        returns_data = {}

        for window in windows:
            rolling_returns = prices.pct_change(window)
            returns_data[f'return_{window}d'] = rolling_returns

        return pd.DataFrame(returns_data)

    @staticmethod
    def calculate_rank_features(data: pd.DataFrame, windows: list = [20, 60, 252]) -> pd.DataFrame:
        """
        Calculate ranking features for data.

        Args:
            data: DataFrame with data
            windows: List of rolling windows for ranking

        Returns:
            DataFrame with rank features
        """
        rank_data = {}

        for window in windows:
            for column in data.columns:
                rank_data[f'{column}_rank_{window}d'] = data[column].rolling(window).rank(pct=True)

        return pd.DataFrame(rank_data)

    @staticmethod
    def identify_outliers(data: pd.DataFrame, method: str = 'iqr',
                         threshold: float = 1.5) -> pd.DataFrame:
        """
        Identify outliers in data.

        Args:
            data: DataFrame with data
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with boolean indicators for outliers
        """
        outliers = pd.DataFrame(index=data.index, columns=data.columns, dtype=bool)

        for column in data.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[column] = (data[column] < lower_bound) | (data[column] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers[column] = z_scores > threshold

            elif method == 'modified_zscore':
                median = data[column].median()
                mad = np.median(np.abs(data[column] - median))
                modified_z_scores = 0.6745 * (data[column] - median) / mad
                outliers[column] = np.abs(modified_z_scores) > threshold

        return outliers

    @staticmethod
    def calculate_technical_indicators_basic(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with basic technical indicators
        """
        indicators = pd.DataFrame(index=data.index)

        if 'Close' in data.columns:
            # Simple Moving Averages
            indicators['sma_20'] = data['Close'].rolling(20).mean()
            indicators['sma_50'] = data['Close'].rolling(50).mean()
            indicators['sma_200'] = data['Close'].rolling(200).mean()

            # Exponential Moving Averages
            indicators['ema_12'] = data['Close'].ewm(span=12).mean()
            indicators['ema_26'] = data['Close'].ewm(span=26).mean()

            # Price relative to moving averages
            indicators['price_to_sma20'] = data['Close'] / indicators['sma_20']
            indicators['price_to_sma200'] = data['Close'] / indicators['sma_200']

        if 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
            # True Range
            tr1 = data['High'] - data['Low']
            tr2 = abs(data['High'] - data['Close'].shift(1))
            tr3 = abs(data['Low'] - data['Close'].shift(1))
            indicators['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Average True Range
            indicators['atr_14'] = indicators['true_range'].rolling(14).mean()

        if 'Volume' in data.columns and 'Close' in data.columns:
            # Volume moving average
            indicators['volume_sma_20'] = data['Volume'].rolling(20).mean()

            # Volume price trend
            price_change = data['Close'].pct_change()
            volume_change = data['Volume'].pct_change()
            indicators['vpt'] = (price_change * data['Volume']).cumsum()

            # On-balance volume
            obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                          np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
            indicators['obv'] = pd.Series(obv, index=data.index).cumsum()

        return indicators

    @staticmethod
    def calculate_price_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based momentum features.

        Args:
            prices: DataFrame with price data

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=prices.index)

        # Returns
        returns = prices.pct_change()
        features['returns_1d'] = returns
        features['returns_5d'] = returns.rolling(5).mean()
        features['returns_21d'] = returns.rolling(21).mean()
        features['returns_63d'] = returns.rolling(63).mean()
        features['returns_252d'] = returns.rolling(252).mean()

        # Momentum strength
        features['momentum_strength_21d'] = abs(returns.rolling(21).mean())
        features['momentum_strength_63d'] = abs(returns.rolling(63).mean())

        # Momentum acceleration
        features['momentum_acceleration'] = features['returns_21d'] - features['returns_63d']

        # Momentum relative to different periods
        for period in [21, 63, 252]:
            momentum = prices / prices.shift(period) - 1
            features[f'momentum_{period}d'] = momentum

        return features

    @staticmethod
    def calculate_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional features across multiple assets.

        Args:
            data: DataFrame with data for multiple assets (columns = assets)

        Returns:
            DataFrame with cross-sectional features
        """
        features = pd.DataFrame(index=data.index)

        # Cross-sectional ranks
        features['cs_rank'] = data.rank(axis=1, pct=True).mean(axis=1)

        # Cross-sectional z-scores
        data_mean = data.mean(axis=1)
        data_std = data.std(axis=1)
        z_scores = data.subtract(data_mean, axis=0).div(data_std, axis=0)
        features['cs_zscore'] = z_scores.mean(axis=1)

        # Cross-sectional momentum
        returns = data.pct_change()
        features['cs_momentum'] = returns.rank(axis=1, pct=True).mean(axis=1)

        # Market beta (simplified - correlation with equal-weighted market)
        market_returns = returns.mean(axis=1)
        for asset in data.columns:
            if len(returns) > 30:  # Need sufficient data
                rolling_cov = returns[asset].rolling(60).cov(market_returns)
                rolling_var = market_returns.rolling(60).var()
                beta = rolling_cov / rolling_var
                features[f'beta_{asset}'] = beta

        return features

    @staticmethod
    def calculate_sentiment_features(volume_data: pd.DataFrame,
                                   price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market sentiment features from volume and price data.

        Args:
            volume_data: DataFrame with volume data
            price_data: DataFrame with price data

        Returns:
            DataFrame with sentiment features
        """
        features = pd.DataFrame(index=price_data.index)

        # Handle single-column case (most common)
        if isinstance(price_data, pd.Series) or price_data.shape[1] == 1:
            price_series = price_data if isinstance(price_data, pd.Series) else price_data.iloc[:, 0]
            volume_series = volume_data if isinstance(volume_data, pd.Series) else volume_data.iloc[:, 0]

            # Volume price trend
            vpt = (price_series.pct_change() * volume_series).cumsum()
            features['vpt_momentum'] = vpt.pct_change(21)

            # On-balance volume momentum
            obv = np.where(price_series > price_series.shift(1), volume_series,
                          np.where(price_series < price_series.shift(1), -volume_series, 0))
            obv_series = pd.Series(obv, index=price_data.index).cumsum()
            features['obv_momentum'] = obv_series.pct_change(21)

            # Volume surge detection
            volume_ma = volume_series.rolling(20).mean()
            volume_surge = volume_series / volume_ma
            features['volume_surge'] = (volume_surge > 2.0).astype(int)

            # Price-volume divergence
            price_trend = price_series.rolling(20).mean().pct_change(5)
            volume_trend = volume_series.rolling(20).mean().pct_change(5)
            features['price_volume_divergence'] = price_trend * volume_trend
        else:
            # Multi-column case - calculate for each column separately
            for col in price_data.columns:
                if col in volume_data.columns:
                    price_series = price_data[col]
                    volume_series = volume_data[col]

                    # Volume price trend
                    vpt = (price_series.pct_change() * volume_series).cumsum()
                    features[f'{col}_vpt_momentum'] = vpt.pct_change(21)

                    # On-balance volume momentum
                    obv = np.where(price_series > price_series.shift(1), volume_series,
                                  np.where(price_series < price_series.shift(1), -volume_series, 0))
                    obv_series = pd.Series(obv, index=price_data.index).cumsum()
                    features[f'{col}_obv_momentum'] = obv_series.pct_change(21)

                    # Volume surge detection
                    volume_ma = volume_series.rolling(20).mean()
                    volume_surge = volume_series / volume_ma
                    features[f'{col}_volume_surge'] = (volume_surge > 2.0).astype(int)

                    # Price-volume divergence
                    price_trend = price_series.rolling(20).mean().pct_change(5)
                    volume_trend = volume_series.rolling(20).mean().pct_change(5)
                    features[f'{col}_price_volume_divergence'] = price_trend * volume_trend

        return features