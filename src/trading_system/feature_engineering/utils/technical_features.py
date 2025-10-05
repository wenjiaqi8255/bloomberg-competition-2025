"""
Technical Features Calculator - Optimized Implementation.

This module provides efficient technical indicator calculations
without the overhead of the previous complex architecture.
"""

import logging
import numpy as np
import pandas as pd

# Disable debug logging to avoid format string issues
logging.getLogger(__name__).setLevel(logging.INFO)
from typing import Dict, List, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """
    Efficient technical indicator calculator.

    This class provides optimized implementations of common technical
    indicators used in quantitative trading strategies.
    """

    def __init__(self):
        """Initialize calculator."""
        logger.debug("Initialized TechnicalIndicatorCalculator")

    # ========== Group-Aware Functions for Multi-Stock Processing ==========

    def _calculate_momentum_for_group(self, group_df: pd.DataFrame, periods: List[int],
                                     return_methods: List[str] = None) -> pd.DataFrame:
        """
        Calculate momentum features for a single stock group.

        This function operates on a single stock's data and returns momentum features.
        It's designed to be used with pandas groupby operations.
        """
        return_methods = return_methods or ["simple"]
        features = pd.DataFrame(index=group_df.index)
        prices = group_df['Close']

        for period in periods:
            # Price momentum (simple returns)
            if "simple" in return_methods:
                momentum = prices / prices.shift(period) - 1
                features[f'momentum_{period}d'] = momentum
                logger.debug(f"DEBUG: momentum_{period}d has {momentum.isnull().sum()} NaN values")

            # Log returns
            if "log" in return_methods:
                log_returns = np.log(prices / prices.shift(period))
                features[f'log_return_{period}d'] = log_returns
                logger.debug(f"DEBUG: log_return_{period}d has {log_returns.isnull().sum()} NaN values")

            # Exponential returns (weighted momentum)
            if "exponential" in return_methods:
                exp_weights = np.exp(np.linspace(-1, 0, period))
                exp_weights = exp_weights / exp_weights.sum()  # Normalize weights
                exp_momentum = prices.rolling(period).apply(
                    lambda x: np.sum(exp_weights * (x / x.iloc[0] - 1)) if len(x) == period else np.nan
                )
                features[f'exp_momentum_{period}d'] = exp_momentum
                logger.debug(f"DEBUG: exp_momentum_{period}d has {exp_momentum.isnull().sum()} NaN values")

            # Risk-adjusted momentum (only if simple momentum was calculated)
            if "simple" in return_methods:
                volatility = prices.pct_change().rolling(period).std()
                risk_adj_momentum = (prices / prices.shift(period) - 1) / volatility
                features[f'risk_adj_momentum_{period}d'] = risk_adj_momentum
                logger.debug(f"DEBUG: risk_adj_momentum_{period}d has {risk_adj_momentum.isnull().sum()} NaN values")

                # Momentum rank (cross-sectional ranking will be handled at higher level)
                momentum_rank = momentum.rolling(window=252, min_periods=126).rank(pct=True)
                features[f'momentum_rank_{period}d'] = momentum_rank
                logger.debug(f"DEBUG: momentum_rank_{period}d has {momentum_rank.isnull().sum()} NaN values")

        # Technical indicators within momentum function
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi_14'] = rsi
        logger.debug(f"DEBUG: rsi_14 has {rsi.isnull().sum()} NaN values")

        # Stochastic Oscillator
        k_period = 14
        d_period = 3
        low_min = group_df['Low'].rolling(window=k_period).min()
        high_max = group_df['High'].rolling(window=k_period).max()
        k_percent = 100 * ((group_df['Close'] - low_min) / (high_max - low_min + 1e-8))
        d_percent = k_percent.rolling(window=d_period).mean()
        features['stochastic_k'] = k_percent
        features['stochastic_d'] = d_percent
        logger.debug(f"DEBUG: stochastic_k has {k_percent.isnull().sum()} NaN values")
        logger.debug(f"DEBUG: stochastic_d has {d_percent.isnull().sum()} NaN values")

        # Williams %R
        williams_r = -100 * ((high_max - group_df['Close']) / (high_max - low_min + 1e-8))
        features['williams_r'] = williams_r
        logger.debug(f"DEBUG: williams_r has {williams_r.isnull().sum()} NaN values")

        # Money Flow Index (MFI)
        if 'Volume' in group_df.columns:
            typical_price = (group_df['High'] + group_df['Low'] + group_df['Close']) / 3
            money_flow = typical_price * group_df['Volume']

            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()

            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            features['mfi'] = mfi
            logger.debug(f"DEBUG: mfi has {mfi.isnull().sum()} NaN values")

        return features

    def _calculate_volatility_for_group(self, group_df: pd.DataFrame, windows: List[int],
                                       volatility_methods: List[str] = None) -> pd.DataFrame:
        """
        Calculate volatility features for a single stock group.

        This function operates on a single stock's data and returns volatility features.
        It's designed to be used with pandas groupby operations.
        """
        volatility_methods = volatility_methods or ["std"]
        features = pd.DataFrame(index=group_df.index)
        returns = group_df['Close'].pct_change()

        for window in windows:
            # Standard deviation volatility
            if "std" in volatility_methods:
                vol = returns.rolling(window).std()
                features[f'volatility_{window}d'] = vol
                logger.debug(f"DEBUG: volatility_{window}d has {vol.isnull().sum()} NaN values")

                # Volatility of volatility
                vol_of_vol = vol.rolling(window).std()
                features[f'vol_of_vol_{window}d'] = vol_of_vol
                logger.debug(f"DEBUG: vol_of_vol_{window}d has {vol_of_vol.isnull().sum()} NaN values")

            # Parkinson volatility (using high-low range)
            if "parkinson" in volatility_methods:
                hl_range = np.log(group_df['High'] / group_df['Low'])
                parkinson_vol = np.sqrt((hl_range ** 2).rolling(window).sum() / (4 * np.log(2)))
                features[f'parkinson_vol_{window}d'] = parkinson_vol
                logger.debug(f"DEBUG: parkinson_vol_{window}d has {parkinson_vol.isnull().sum()} NaN values")

            # Garman-Klass volatility
            if "garman_klass" in volatility_methods:
                log_hl = np.log(group_df['High'] / group_df['Low'])
                log_co = np.log(group_df['Close'] / group_df['Open'])

                gk_vol = np.sqrt(
                    (0.5 * log_hl ** 2) -
                    (2 * np.log(2) - 1) * log_co ** 2
                ).rolling(window).mean()

                features[f'gk_vol_{window}d'] = gk_vol
                logger.debug(f"DEBUG: gk_vol_{window}d has {gk_vol.isnull().sum()} NaN values")

            # Volatility ranking (cross-sectional ranking will be handled at higher level)
            if "std" in volatility_methods:
                vol_rank = vol.rolling(window=252, min_periods=126).rank(pct=True)
                features[f'volatility_rank_{window}d'] = vol_rank
                logger.debug(f"DEBUG: volatility_rank_{window}d has {vol_rank.isnull().sum()} NaN values")

        # Additional volatility measures
        # Parkinson volatility (longer-term)
        parkinson_vol = np.sqrt(
            ((np.log(group_df['High'] / group_df['Low']) ** 2).rolling(window=20).sum()) /
            (4 * len(group_df) * np.log(2))
        )
        features['parkinson_volatility'] = parkinson_vol
        logger.debug(f"DEBUG: parkinson_volatility has {parkinson_vol.isnull().sum()} NaN values")

        # Garman-Klass volatility (longer-term)
        log_hl = np.log(group_df['High'] / group_df['Low'])
        log_co = np.log(group_df['Close'] / group_df['Open'])

        garman_klass_vol = np.sqrt(
            ((0.5 * log_hl ** 2) - (2 * np.log(2) - 1) * log_co ** 2).rolling(window=20).sum()
        )
        features['garman_klass_volatility'] = garman_klass_vol
        logger.debug(f"DEBUG: garman_klass_volatility has {garman_klass_vol.isnull().sum()} NaN values")

        # Range volatility
        range_vol = (group_df['High'] - group_df['Low']) / group_df['Close']
        features['range_volatility'] = range_vol
        logger.debug(f"DEBUG: range_volatility has {range_vol.isnull().sum()} NaN values")

        return features

    def _calculate_technical_for_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a single stock group.

        This function operates on a single stock's data and returns technical indicator features.
        It's designed to be used with pandas groupby operations.
        """
        features = pd.DataFrame(index=group_df.index)
        prices = group_df['Close']
        lows = group_df['Low']
        highs = group_df['High']
        opens = group_df['Open']
        volumes = group_df['Volume'] if 'Volume' in group_df.columns else None

        # Simple Moving Averages
        for period in [10, 20, 50, 200]:
            sma = prices.rolling(window=period).mean()
            features[f'sma_{period}'] = sma
            logger.debug(f"DEBUG: sma_{period} has {sma.isnull().sum()} NaN values")

            # Price above SMA
            features[f'price_above_sma_{period}'] = (prices > sma).astype(int)
            logger.debug(f"DEBUG: price_above_sma_{period} has {((prices > sma).astype(int)).isnull().sum()} NaN values")

        # Exponential Moving Averages
        for period in [12, 26]:
            ema = prices.ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            logger.debug(f"DEBUG: ema_{period} has {ema.isnull().sum()} NaN values")

        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal

        features['macd_line'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram
        logger.debug(f"DEBUG: macd_line has {macd_line.isnull().sum()} NaN values")
        logger.debug(f"DEBUG: macd_signal has {macd_signal.isnull().sum()} NaN values")
        logger.debug(f"DEBUG: macd_histogram has {macd_histogram.isnull().sum()} NaN values")

        # Bollinger Bands
        sma_20 = prices.rolling(window=20).mean()
        std_20 = prices.rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)

        features['sma_20'] = sma_20  # Already added above, but keeping for consistency
        features['bb_upper'] = bb_upper
        features['bb_middle'] = sma_20
        features['bb_lower'] = bb_lower

        # Bollinger Band position and width
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        bb_width = (bb_upper - bb_lower) / sma_20

        features['bb_position'] = bb_position
        features['bb_width'] = bb_width
        logger.debug(f"DEBUG: bb_position has {bb_position.isnull().sum()} NaN values")
        logger.debug(f"DEBUG: bb_width has {bb_width.isnull().sum()} NaN values")

        # ADX (Average Directional Index)
        # Calculate directional movement
        high_diff = highs.diff()
        low_diff = -lows.diff()

        dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # True Range
        tr1 = highs - lows
        tr2 = abs(highs - prices.shift(1))
        tr3 = abs(lows - prices.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # Convert to pandas Series for rolling operations
        # Flatten arrays if needed
        dm_plus_flat = dm_plus.flatten() if dm_plus.ndim > 1 else dm_plus
        dm_minus_flat = dm_minus.flatten() if dm_minus.ndim > 1 else dm_minus

        # true_range is already a pandas Series from np.maximum, just ensure it's Series
        if hasattr(true_range, 'values'):
            tr_flat = true_range.values.flatten() if true_range.values.ndim > 1 else true_range.values
        else:
            tr_flat = true_range.flatten() if hasattr(true_range, 'flatten') else true_range

        dm_plus_series = pd.Series(dm_plus_flat, index=group_df.index)
        dm_minus_series = pd.Series(dm_minus_flat, index=group_df.index)
        tr_series = pd.Series(tr_flat, index=group_df.index)

        # Smoothed values
        atr = tr_series.rolling(window=14).mean()
        di_plus = 100 * (dm_plus_series.rolling(window=14).mean() / atr)
        di_minus = 100 * (dm_minus_series.rolling(window=14).mean() / atr)

        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=14).mean()

        features['adx'] = adx
        logger.debug(f"DEBUG: adx has {adx.isnull().sum()} NaN values")

        # Commodity Channel Index (CCI)
        typical_price = (highs + lows + prices) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        features['cci'] = cci
        logger.debug(f"DEBUG: cci has {cci.isnull().sum()} NaN values")

        # Volume-based indicators (if volume data available)
        if volumes is not None:
            # Volume Price Trend
            vpt = volumes * prices.pct_change()
            vpt_cumsum = vpt.cumsum()
            features['volume_price_trend'] = vpt_cumsum
            logger.debug(f"DEBUG: volume_price_trend has {vpt_cumsum.isnull().sum()} NaN values")

        return features
