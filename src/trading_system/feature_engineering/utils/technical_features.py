"""
Technical Features Calculator - Optimized Implementation.

This module provides efficient technical indicator calculations
without the overhead of the previous complex architecture.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
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

    def compute_momentum_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate momentum features for given periods."""
        features = pd.DataFrame(index=data.index)
        prices = data['Close']

        for period in periods:
            # Price momentum
            momentum = prices / prices.shift(period) - 1
            features[f'momentum_{period}d'] = momentum

            # Log returns
            features[f'log_return_{period}d'] = np.log(prices / prices.shift(period))

            # Risk-adjusted momentum
            volatility = prices.pct_change().rolling(period).std()
            features[f'risk_adj_momentum_{period}d'] = momentum / (volatility + 1e-8)

            # Momentum rank
            lookback = min(period * 2, 252)
            features[f'momentum_rank_{period}d'] = momentum.rolling(lookback).rank(pct=True)

        # Momentum divergence (short-term vs long-term)
        if len(periods) >= 2:
            short_period, long_period = min(periods), max(periods)
            features['momentum_divergence'] = (
                features[f'momentum_{short_period}d'] - features[f'momentum_{long_period}d']
            )

        # RSI
        features['rsi_14'] = self._calculate_rsi(prices, 14)

        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(data)
        features['stochastic_k'] = stoch_k
        features['stochastic_d'] = stoch_d

        # Williams %R
        features['williams_r'] = self._calculate_williams_r(data)

        # Money Flow Index
        features['mfi'] = self._calculate_mfi(data)

        return features

    def compute_volatility_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate volatility features for given windows."""
        features = pd.DataFrame(index=data.index)
        returns = data['Close'].pct_change()

        for window in windows:
            # Historical volatility
            vol = returns.rolling(window).std() * np.sqrt(252)
            features[f'volatility_{window}d'] = vol

            # Volatility of volatility
            features[f'vol_of_vol_{window}d'] = vol.rolling(window).std()

            # Volatility rank
            rank_window = min(window * 3, 252)
            features[f'volatility_rank_{window}d'] = vol.rolling(rank_window).rank(pct=True)

        # Parkinson volatility
        features['parkinson_volatility'] = self._calculate_parkinson_volatility(data)

        # Garman-Klass volatility
        features['garman_klass_volatility'] = self._calculate_garman_klass_volatility(data)

        # Realized range volatility
        features['range_volatility'] = self._calculate_range_volatility(data)

        return features

    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        features = pd.DataFrame(index=data.index)

        # Moving averages
        for period in [10, 20, 50, 200]:
            sma = data['Close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_above_sma_{period}'] = (data['Close'] > sma).astype(int)

        # Exponential moving averages
        for period in [12, 26]:
            ema = data['Close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema

        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal

        features['macd_line'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram

        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # ADX
        features['adx'] = self._calculate_adx(data)

        # Commodity Channel Index
        features['cci'] = self._calculate_cci(data)

        return features

    def compute_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=data.index)

        if 'Volume' not in data.columns:
            logger.warning("Volume data not available")
            return features

        volume = data['Volume']
        prices = data['Close']

        # Volume moving averages
        for period in [20, 60]:
            vol_sma = volume.rolling(period).mean()
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio_{period}'] = volume / vol_sma

        # On-Balance Volume
        features['obv'] = self._calculate_obv(data)

        # Volume Price Trend
        features['vpt'] = self._calculate_vpt(data)

        # VWAP deviation
        vwap = self._calculate_vwap(data)
        features['vwap'] = vwap
        features['vwap_deviation'] = (prices - vwap) / vwap

        # Accumulation/Distribution Line
        features['accumulation_distribution'] = self._calculate_accumulation_distribution(data)

        # Chaikin Money Flow
        features['chaikin_money_flow'] = self._calculate_chaikin_money_flow(data)

        return features

    def compute_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity features."""
        features = pd.DataFrame(index=data.index)

        if 'Volume' not in data.columns:
            logger.warning("Volume data not available for liquidity features")
            return features

        prices = data['Close']
        volume = data['Volume']

        # Amihud illiquidity ratio
        daily_returns = prices.pct_change()
        dollar_volume = prices * volume
        features['amihud_illiquidity'] = abs(daily_returns) / (dollar_volume + 1e-8)

        # Zero return days (liquidity proxy)
        features['zero_return_days_20d'] = (daily_returns == 0).rolling(20).sum() / 20

        # Price impact
        features['price_impact'] = abs(daily_returns) / (volume + 1e-8)

        # Turnover ratio (approximate)
        market_cap_proxy = prices * volume * 252
        features['turnover_ratio'] = volume / (market_cap_proxy + 1e-8)

        return features

    def compute_mean_reversion_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate mean reversion features."""
        features = pd.DataFrame(index=data.index)
        prices = data['Close']

        for period in periods:
            # Distance from moving average
            sma = prices.rolling(period).mean()
            features[f'price_vs_sma_{period}'] = (prices - sma) / sma

        # Bollinger Band mean reversion
        bb_upper, bb_lower, _ = self._calculate_bollinger_bands(prices)
        features['bb_mean_reversion'] = (bb_lower + bb_upper) / 2 - prices

        # Rate of change
        for period in [5, 10]:
            features[f'roc_{period}d'] = prices.pct_change(period)

        return features

    def compute_trend_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate trend-following features."""
        features = pd.DataFrame(index=data.index)
        prices = data['Close']

        # Moving average crossovers
        if len(periods) >= 2:
            short_period, long_period = periods[0], periods[1]
            sma_short = prices.rolling(short_period).mean()
            sma_long = prices.rolling(long_period).mean()
            features[f'sma_crossover_{short_period}_{long_period}'] = sma_short - sma_long

        # EMA crossovers
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        features['ema_crossover_12_26'] = ema_12 - ema_26

        # Trend strength
        for period in [20, 60]:
            returns = prices.pct_change()
            features[f'trend_strength_{period}d'] = abs(returns.rolling(period).mean())

        # Directional movement
        dm_plus, dm_minus = self._calculate_directional_movement(data)
        features['dm_plus'] = dm_plus
        features['dm_minus'] = dm_minus
        features['dm_net'] = dm_plus - dm_minus

        return features

    # ========================================================================
    # Technical Indicator Helper Methods
    # ========================================================================

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min + 1e-8))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = data['High'].rolling(window=period).max()
        low_min = data['Low'].rolling(window=period).min()
        return -100 * (high_max - data['Close']) / (high_max - low_min + 1e-8)

    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        money_ratio = positive_mf / (negative_mf + 1e-8)
        return 100 - (100 / (1 + money_ratio))

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified version)."""
        dm_plus, dm_minus = self._calculate_directional_movement(data)
        tr = self._calculate_true_range(data)

        # Smoothed values
        smooth_dm_plus = dm_plus.ewm(span=period).mean()
        smooth_dm_minus = dm_minus.ewm(span=period).mean()
        smooth_tr = tr.ewm(span=period).mean()

        # Directional indicators
        di_plus = 100 * smooth_dm_plus / (smooth_tr + 1e-8)
        di_minus = 100 * smooth_dm_minus / (smooth_tr + 1e-8)

        # Directional movement index
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = dx.ewm(span=period).mean()

        return adx

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad + 1e-8)

    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility."""
        hl_ratio = np.log(data['High'] / data['Low'])
        return np.sqrt((hl_ratio**2).rolling(period).mean() / (4 * np.log(2))) * np.sqrt(252)

    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility."""
        log_hl = np.log(data['High'] / data['Low'])
        log_co = np.log(data['Close'] / data['Open'])

        term1 = 0.5 * log_hl**2
        term2 = (2 * np.log(2) - 1) * log_co**2

        return np.sqrt((term1 - term2).rolling(period).mean()) * np.sqrt(252)

    def _calculate_range_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate range-based volatility."""
        high_low_range = data['High'] - data['Low']
        close_open_range = abs(data['Close'] - data['Open'])
        range_measure = (high_low_range + close_open_range) / 2
        return range_measure.rolling(period).mean() * np.sqrt(252)

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                      np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))
        return pd.Series(obv, index=data.index).cumsum()

    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = data['Close'].pct_change()
        vpt = (price_change * data['Volume']).cumsum()
        return vpt

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / (data['Volume'].cumsum() + 1e-8)
        return vwap

    def _calculate_accumulation_distribution(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        high_low = data['High'] - data['Low']
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (high_low + 1e-8)
        ad_line = (clv * data['Volume']).cumsum()
        return ad_line

    def _calculate_chaikin_money_flow(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        high_low = data['High'] - data['Low']
        mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (high_low + 1e-8) * data['Volume']
        return mfv.rolling(period).sum() / (data['Volume'].rolling(period).sum() + 1e-8)

    def _calculate_directional_movement(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Plus and Minus Directional Movement."""
        up_move = data['High'].diff()
        down_move = -data['Low'].diff()

        dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        return pd.Series(dm_plus, index=data.index), pd.Series(dm_minus, index=data.index)

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr