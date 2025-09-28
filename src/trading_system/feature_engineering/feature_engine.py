"""
Feature Engine for ML-based trading strategies.

This module provides comprehensive feature engineering capabilities:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Momentum features at multiple time horizons
- Volatility measures and risk metrics
- Theoretical features (Hurst exponent)
- Feature normalization and selection
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Feature engineering engine for ML trading strategies.

    Features computed:
    1. Momentum Features: Returns at multiple horizons (1, 3, 6, 12 months)
    2. Technical Indicators: RSI, MACD, Bollinger Bands, ADX, etc.
    3. Volatility Features: Historical volatility, ATR, GARCH-like measures
    4. Theoretical Features: Hurst exponent for trend persistence
    5. Relative Strength: Performance relative to benchmark
    6. Volume Features: Volume trends and anomalies

    All features are designed to avoid look-ahead bias.
    """

    def __init__(self,
                 lookback_periods: List[int] = None,
                 momentum_periods: List[int] = None,
                 volatility_windows: List[int] = None,
                 include_technical: bool = True,
                 include_theoretical: bool = True,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize feature engine.

        Args:
            lookback_periods: List of lookback periods for features (default: [20, 50, 200])
            momentum_periods: List of periods for momentum calculations (default: [21, 63, 126, 252])
            volatility_windows: List of windows for volatility calculations (default: [20, 60])
            include_technical: Whether to include technical indicators
            include_theoretical: Whether to include theoretical features
            benchmark_symbol: Symbol for relative strength calculations
        """
        self.lookback_periods = lookback_periods or [20, 50, 200]
        self.momentum_periods = momentum_periods or [21, 63, 126, 252]  # 1, 3, 6, 12 months
        self.volatility_windows = volatility_windows or [20, 60]
        self.include_technical = include_technical
        self.include_theoretical = include_theoretical
        self.benchmark_symbol = benchmark_symbol

        # Feature cache for performance
        self.feature_cache = {}

    def compute_features(self, price_data: Dict[str, pd.DataFrame],
                        start_date: datetime = None,
                        end_date: datetime = None,
                        benchmark_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all symbols in the universe.

        Args:
            price_data: Dictionary of price DataFrames for each symbol
            start_date: Start date for feature computation
            end_date: End date for feature computation
            benchmark_data: Benchmark price data for relative calculations

        Returns:
            Dictionary of feature DataFrames for each symbol
        """
        logger.info(f"Computing features for {len(price_data)} symbols")

        feature_dfs = {}

        for symbol, data in price_data.items():
            try:
                features = self._compute_symbol_features(
                    data, symbol, start_date, end_date, benchmark_data
                )
                feature_dfs[symbol] = features
                logger.debug(f"Computed {len(features.columns)} features for {symbol}")

            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")
                continue

        logger.info(f"Successfully computed features for {len(feature_dfs)} symbols")
        return feature_dfs

    def _compute_symbol_features(self, data: pd.DataFrame, symbol: str,
                                start_date: datetime = None,
                                end_date: datetime = None,
                                benchmark_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute features for a single symbol.

        Args:
            data: Price DataFrame for the symbol
            symbol: Symbol name
            start_date: Start date for computation
            end_date: End date for computation
            benchmark_data: Benchmark price data

        Returns:
            DataFrame with all features for the symbol
        """
        # Filter data by date range
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask = mask & (data.index >= start_date)
            if end_date:
                mask = mask & (data.index <= end_date)
            data = data[mask].copy()

        # Initialize features DataFrame
        features = pd.DataFrame(index=data.index)

        # 1. Basic Price Features
        features = pd.concat([features, self._compute_price_features(data)], axis=1)

        # 2. Momentum Features
        features = pd.concat([features, self._compute_momentum_features(data)], axis=1)

        # 3. Volatility Features
        features = pd.concat([features, self._compute_volatility_features(data)], axis=1)

        # 4. Technical Indicators
        if self.include_technical:
            features = pd.concat([features, self._compute_technical_indicators(data)], axis=1)

        # 5. Theoretical Features
        if self.include_theoretical:
            features = pd.concat([features, self._compute_theoretical_features(data)], axis=1)

        # 6. Relative Strength Features (if benchmark data available)
        if benchmark_data is not None and symbol != self.benchmark_symbol:
            features = pd.concat([features, self._compute_relative_features(data, benchmark_data, symbol)], axis=1)

        # Clean up infinite values and NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

        return features

    def _compute_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute basic price-based features."""
        features = pd.DataFrame(index=data.index)

        # Price levels
        features['price'] = data['Close']
        features['price_sma_20'] = data['Close'].rolling(20).mean()
        features['price_sma_50'] = data['Close'].rolling(50).mean()
        features['price_sma_200'] = data['Close'].rolling(200).mean()

        # Price position relative to moving averages
        features['price_above_sma20'] = (data['Close'] > features['price_sma_20']).astype(int)
        features['price_above_sma50'] = (data['Close'] > features['price_sma_50']).astype(int)
        features['price_above_sma200'] = (data['Close'] > features['price_sma_200']).astype(int)

        # Price acceleration (second derivative)
        returns = data['Close'].pct_change()
        features['price_acceleration'] = returns.diff()

        return features

    def _compute_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum features at multiple time horizons."""
        features = pd.DataFrame(index=data.index)

        for period in self.momentum_periods:
            # Raw returns
            features[f'return_{period}d'] = data['Close'].pct_change(period)

            # Log returns
            features[f'log_return_{period}d'] = np.log(data['Close'] / data['Close'].shift(period))

            # Risk-adjusted momentum (returns divided by volatility)
            volatility = data['Close'].pct_change().rolling(period).std()
            features[f'risk_adj_momentum_{period}d'] = features[f'return_{period}d'] / (volatility + 1e-8)

            # Momentum rank (percentile of current momentum vs history)
            momentum_window = min(period * 2, 252)
            features[f'momentum_rank_{period}d'] = features[f'return_{period}d'].rolling(momentum_window).rank(pct=True)

        # Momentum divergence (short-term vs long-term)
        if len(self.momentum_periods) >= 2:
            short_period = min(self.momentum_periods)
            long_period = max(self.momentum_periods)
            features['momentum_divergence'] = features[f'return_{short_period}d'] - features[f'return_{long_period}d']

        return features

    def _compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility and risk features."""
        features = pd.DataFrame(index=data.index)

        returns = data['Close'].pct_change()

        for window in self.volatility_windows:
            # Historical volatility
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

            # Volatility of volatility (vol clustering)
            features[f'vol_of_vol_{window}d'] = features[f'volatility_{window}d'].rolling(window).std()

            # Volatility rank (relative to recent history)
            vol_window = min(window * 3, 252)
            features[f'volatility_rank_{window}d'] = features[f'volatility_{window}d'].rolling(vol_window).rank(pct=True)

        # High-Low volatility
        features['hl_volatility_20d'] = ((data['High'] - data['Low']) / data['Close']).rolling(20).mean() * np.sqrt(252)

        # Parkinson's volatility estimator
        features['parkinson_vol_20d'] = np.sqrt((1 / (4 * np.log(2))) *
                                              (np.log(data['High'] / data['Low'])**2).rolling(20).mean()) * np.sqrt(252)

        # GARCH-like volatility (simplified)
        features['garch_like_vol'] = self._compute_garch_volatility(returns)

        return features

    def _compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators using ta library."""
        try:
            # Use ta library to add all technical indicators
            # Handle both lowercase and uppercase column names
            open_col = 'Open' if 'Open' in data.columns else 'open'
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'

            ta_data = add_all_ta_features(
                df=data,
                open=open_col,
                high=high_col,
                low=low_col,
                close=close_col,
                volume=volume_col,
                fillna=False
            )

            # Select relevant indicators to avoid dimensionality explosion
            relevant_indicators = [
                # Momentum indicators
                'momentum_rsi', 'momentum_stoch', 'momentum_stoch_signal',
                'momentum_williams_r', 'momentum_uo', 'momentum_mfi',

                # Trend indicators
                'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
                'trend_cci', 'trend_dpo', 'trend_ichimoku_conv',

                # Volatility indicators
                'volatility_bbh', 'volatility_bbl', 'volatility_bbm',
                'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
                'volatility_kch', 'volatility_kcl', 'volatility_kcm',
                'volatility_dch', 'volatility_dcl', 'volatility_dcm',
                'volatility_atr',

                # Volume indicators
                'volume_adi', 'volume_vpt', 'volume_fi',
                'volume_em', 'volume_sma_em', 'volume_vwap',

                # Other
                'others_dr', 'others_cr'
            ]

            # Filter columns that exist in the data
            available_indicators = [col for col in relevant_indicators if col in ta_data.columns]

            return ta_data[available_indicators].copy()

        except Exception as e:
            logger.warning(f"Failed to compute technical indicators: {e}")
            return pd.DataFrame(index=data.index)

    def _compute_theoretical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute theoretical features like Hurst exponent."""
        features = pd.DataFrame(index=data.index)

        # Hurst exponent for trend persistence
        features['hurst_126d'] = self._compute_rolling_hurst(data['Close'], window=126)
        features['hurst_63d'] = self._compute_rolling_hurst(data['Close'], window=63)

        # Autocorrelation features
        returns = data['Close'].pct_change().dropna()
        for lag in [1, 5, 10]:
            features[f'autocorr_lag_{lag}'] = returns.rolling(60).apply(lambda x: x.autocorr(lag=lag))

        # Skewness and kurtosis of returns
        features['returns_skew_20d'] = returns.rolling(20).skew()
        features['returns_kurt_20d'] = returns.rolling(20).kurt()

        return features

    def _compute_relative_features(self, data: pd.DataFrame, benchmark_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute relative strength features against benchmark."""
        features = pd.DataFrame(index=data.index)

        # Align data with benchmark
        aligned_data, aligned_benchmark = data.align(benchmark_data, join='inner', axis=0)

        if len(aligned_data) == 0:
            logger.warning(f"No overlapping data between {symbol} and benchmark")
            return features

        # Relative returns
        symbol_returns = aligned_data['Close'].pct_change()
        benchmark_returns = aligned_benchmark['Close'].pct_change()

        features['relative_return_20d'] = (1 + symbol_returns.rolling(20).mean()) / (1 + benchmark_returns.rolling(20).mean()) - 1
        features['relative_return_60d'] = (1 + symbol_returns.rolling(60).mean()) / (1 + benchmark_returns.rolling(60).mean()) - 1

        # Beta calculation
        features['beta_60d'] = self._compute_rolling_beta(symbol_returns, benchmark_returns, window=60)

        # Alpha calculation (CAPM alpha)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        features['alpha_60d'] = features['relative_return_60d'] - risk_free_rate * 60

        # Tracking error
        features['tracking_error_60d'] = (symbol_returns - benchmark_returns).rolling(60).std() * np.sqrt(252)

        return features

    def _compute_garch_volatility(self, returns: pd.Series, omega: float = 0.1,
                                 alpha: float = 0.1, beta: float = 0.85) -> pd.Series:
        """Compute simplified GARCH volatility."""
        volatility = pd.Series(index=returns.index, dtype=float)

        # Initialize with historical volatility
        volatility.iloc[20] = returns.iloc[:20].std()

        for i in range(21, len(returns)):
            if pd.isna(returns.iloc[i-1]) or pd.isna(volatility.iloc[i-1]):
                volatility.iloc[i] = volatility.iloc[i-1] if i > 0 else 0.02
            else:
                # GARCH(1,1) formula
                volatility.iloc[i] = np.sqrt(
                    omega + alpha * returns.iloc[i-1]**2 + beta * volatility.iloc[i-1]**2
                )

        return volatility * np.sqrt(252)  # Annualize

    def _compute_rolling_hurst(self, price_series: pd.Series, window: int = 126) -> pd.Series:
        """Compute rolling Hurst exponent."""
        hurst_values = pd.Series(index=price_series.index, dtype=float)

        for i in range(window, len(price_series)):
            subset = price_series.iloc[i-window:i]
            hurst_values.iloc[i] = self._calculate_hurst_exponent(subset.dropna())

        return hurst_values

    def _calculate_hurst_exponent(self, price_series: pd.Series) -> float:
        """Calculate Hurst exponent for a price series."""
        try:
            # Calculate price differences
            lags = range(2, min(20, len(price_series) // 4))

            tau = []
            for lag in lags:
                # Calculate range and standard deviation
                diff = price_series.diff(lag).dropna()
                if len(diff) == 0:
                    continue

                range_val = diff.max() - diff.min()
                std_val = diff.std()

                if std_val > 0:
                    tau.append(np.log(range_val / std_val))

            if len(tau) < 3:
                return 0.5  # Random walk

            # Linear regression to find Hurst exponent
            x = np.log(lags[:len(tau)])
            y = np.array(tau)

            slope, _, _, _, _ = stats.linregress(x, y)
            return slope

        except Exception as e:
            logger.debug(f"Failed to calculate Hurst exponent: {e}")
            return 0.5  # Default to random walk

    def _compute_rolling_beta(self, asset_returns: pd.Series,
                            benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
        """Compute rolling beta against benchmark."""
        covariance = asset_returns.rolling(window).cov(benchmark_returns)
        benchmark_variance = benchmark_returns.rolling(window).var()

        beta = covariance / (benchmark_variance + 1e-8)
        return beta.fillna(1.0)  # Default to beta = 1

    def normalize_features(self, features: Dict[str, pd.DataFrame],
                          method: str = 'robust') -> Dict[str, pd.DataFrame]:
        """
        Normalize features to prevent scale dominance.

        Args:
            features: Dictionary of feature DataFrames
            method: Normalization method ('robust', 'standard', 'minmax')

        Returns:
            Dictionary of normalized feature DataFrames
        """
        from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_features = {}

        for symbol, feature_df in features.items():
            try:
                # Fit scaler on non-NaN data
                valid_data = feature_df.dropna()
                if len(valid_data) > 10:  # Need minimum samples
                    scaler.fit(valid_data)
                    normalized = scaler.transform(feature_df.fillna(0))
                    normalized_features[symbol] = pd.DataFrame(
                        normalized, index=feature_df.index, columns=feature_df.columns
                    )
                else:
                    normalized_features[symbol] = feature_df.copy()

            except Exception as e:
                logger.warning(f"Failed to normalize features for {symbol}: {e}")
                normalized_features[symbol] = feature_df.copy()

        return normalized_features

    def select_features(self, features: Dict[str, pd.DataFrame],
                       target: pd.Series, method: str = 'univariate',
                       max_features: int = 50) -> List[str]:
        """
        Select most predictive features.

        Args:
            features: Dictionary of feature DataFrames
            target: Target variable for feature selection
            method: Feature selection method ('univariate', 'importance')
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        try:
            # Combine all features for selection
            all_features = pd.concat(features.values(), axis=0)
            all_features = all_features.dropna()

            if len(all_features) == 0 or len(target) == 0:
                return list(all_features.columns)[:max_features]

            # Align features with target
            aligned_features, aligned_target = all_features.align(target, join='inner', axis=0)

            if method == 'univariate':
                from sklearn.feature_selection import SelectKBest, f_regression
                selector = SelectKBest(score_func=f_regression, k=max_features)
                selector.fit(aligned_features, aligned_target)
                selected = aligned_features.columns[selector.get_support()].tolist()

            elif method == 'importance':
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(aligned_features, aligned_target)
                importance_df = pd.DataFrame({
                    'feature': aligned_features.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                selected = importance_df.head(max_features)['feature'].tolist()

            else:
                raise ValueError(f"Unknown feature selection method: {method}")

            logger.info(f"Selected {len(selected)} features using {method} method")
            return selected

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            # Fallback to all features
            if features:
                return list(list(features.values())[0].columns)[:max_features]
            return []

    def get_feature_info(self) -> Dict:
        """Get information about computed features."""
        return {
            'lookback_periods': self.lookback_periods,
            'momentum_periods': self.momentum_periods,
            'volatility_windows': self.volatility_windows,
            'include_technical': self.include_technical,
            'include_theoretical': self.include_theoretical,
            'benchmark_symbol': self.benchmark_symbol,
            'feature_categories': [
                'price_features',
                'momentum_features',
                'volatility_features',
                'technical_indicators',
                'theoretical_features',
                'relative_features'
            ]
        }