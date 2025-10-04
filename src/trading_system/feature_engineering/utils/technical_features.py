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
        # logger.info(f"DEBUG: Computing momentum features for periods {periods} on {len(data)} rows")
        features = pd.DataFrame(index=data.index)
        prices = data['Close']

        initial_nan_count = 0
        nan_report = {}

        for period in periods:
            # logger.debug(f"DEBUG: Computing {period}-day momentum features...")
            period_nan_count = 0

            # Price momentum
            momentum = prices / prices.shift(period) - 1
            momentum_nan = momentum.isnull().sum()
            period_nan_count += momentum_nan
            features[f'momentum_{period}d'] = momentum
            nan_report[f'momentum_{period}d'] = momentum_nan
            logger.debug(f"DEBUG: momentum_{period}d has {momentum_nan} NaN values ({momentum_nan/len(momentum)*100:.1f}%)")

            # Log returns
            log_returns = np.log(prices / prices.shift(period))
            log_returns_nan = log_returns.isnull().sum()
            period_nan_count += log_returns_nan
            features[f'log_return_{period}d'] = log_returns
            nan_report[f'log_return_{period}d'] = log_returns_nan
            logger.debug(f"DEBUG: log_return_{period}d has {log_returns_nan} NaN values ({log_returns_nan/len(log_returns)*100:.1f}%)")

            # Risk-adjusted momentum
            volatility = prices.pct_change().rolling(period).std()
            risk_adj_momentum = momentum / (volatility + 1e-8)
            risk_adj_nan = risk_adj_momentum.isnull().sum()
            period_nan_count += risk_adj_nan
            features[f'risk_adj_momentum_{period}d'] = risk_adj_momentum
            nan_report[f'risk_adj_momentum_{period}d'] = risk_adj_nan
            logger.debug(f"DEBUG: risk_adj_momentum_{period}d has {risk_adj_nan} NaN values ({risk_adj_nan/len(risk_adj_momentum)*100:.1f}%)")

            # Momentum rank
            lookback = min(period * 2, 252)
            momentum_rank = momentum.rolling(lookback).rank(pct=True)
            rank_nan = momentum_rank.isnull().sum()
            period_nan_count += rank_nan
            features[f'momentum_rank_{period}d'] = momentum_rank
            nan_report[f'momentum_rank_{period}d'] = rank_nan
            logger.debug(f"DEBUG: momentum_rank_{period}d has {rank_nan} NaN values ({rank_nan/len(momentum_rank)*100:.1f}%)")

            logger.info(f"DEBUG: Period {period} total NaN values: {period_nan_count}")
            initial_nan_count += period_nan_count

        # Momentum divergence (short-term vs long-term)
        if len(periods) >= 2:
            short_period, long_period = min(periods), max(periods)
            momentum_divergence = (
                features[f'momentum_{short_period}d'] - features[f'momentum_{long_period}d']
            )
            div_nan = momentum_divergence.isnull().sum()
            initial_nan_count += div_nan
            features['momentum_divergence'] = momentum_divergence
            nan_report['momentum_divergence'] = div_nan
            logger.debug(f"DEBUG: momentum_divergence has {div_nan} NaN values ({div_nan/len(momentum_divergence)*100:.1f}%)")

        # RSI
        # logger.debug("DEBUG: Computing RSI...")
        rsi = self._calculate_rsi(prices, 14)
        rsi_nan = rsi.isnull().sum()
        initial_nan_count += rsi_nan
        features['rsi_14'] = rsi
        nan_report['rsi_14'] = rsi_nan
        logger.debug(f"DEBUG: rsi_14 has {rsi_nan} NaN values ({rsi_nan/len(rsi)*100:.1f}%)")

        # Stochastic Oscillator
        # logger.debug("DEBUG: Computing Stochastic Oscillator...")
        stoch_k, stoch_d = self._calculate_stochastic(data)
        stoch_k_nan = stoch_k.isnull().sum()
        stoch_d_nan = stoch_d.isnull().sum()
        initial_nan_count += stoch_k_nan + stoch_d_nan
        features['stochastic_k'] = stoch_k
        features['stochastic_d'] = stoch_d
        nan_report['stochastic_k'] = stoch_k_nan
        nan_report['stochastic_d'] = stoch_d_nan
        logger.debug(f"DEBUG: stochastic_k has {stoch_k_nan} NaN values ({stoch_k_nan/len(stoch_k)*100:.1f}%)")
        logger.debug(f"DEBUG: stochastic_d has {stoch_d_nan} NaN values ({stoch_d_nan/len(stoch_d)*100:.1f}%)")

        # Williams %R
        # logger.debug("DEBUG: Computing Williams %R...")
        williams_r = self._calculate_williams_r(data)
        wr_nan = williams_r.isnull().sum()
        initial_nan_count += wr_nan
        features['williams_r'] = williams_r
        nan_report['williams_r'] = wr_nan
        logger.debug(f"DEBUG: williams_r has {wr_nan} NaN values ({wr_nan/len(williams_r)*100:.1f}%)")

        # Money Flow Index
        # logger.debug("DEBUG: Computing Money Flow Index...")
        mfi = self._calculate_mfi(data)
        mfi_nan = mfi.isnull().sum()
        initial_nan_count += mfi_nan
        features['mfi'] = mfi
        nan_report['mfi'] = mfi_nan
        logger.debug(f"DEBUG: mfi has {mfi_nan} NaN values ({mfi_nan/len(mfi)*100:.1f}%)")

        # Final momentum features summary
        total_momentum_features = len(features.columns)
        final_total_nan = features.isnull().sum().sum()
        # logger.info(f"DEBUG: Momentum features computation complete:")
        # logger.info(f"DEBUG: - Total features created: {total_momentum_features}")
        logger.info(f"DEBUG: - Total NaN values: {final_total_nan} out of {len(features) * total_momentum_features} possible values")
        logger.info(f"DEBUG: - NaN percentage: {final_total_nan/(len(features) * total_momentum_features)*100:.2f}%")
        logger.info(f"DEBUG: - NaN breakdown: {nan_report}")

        return features

    def compute_volatility_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate volatility features for given windows."""
        # logger.info(f"DEBUG: Computing volatility features for windows {windows} on {len(data)} rows")
        features = pd.DataFrame(index=data.index)
        returns = data['Close'].pct_change()

        nan_report = {}

        for window in windows:
            # logger.debug(f"DEBUG: Computing {window}-day volatility features...")
            period_nan_count = 0

            # Historical volatility
            vol = returns.rolling(window).std() * np.sqrt(252)
            vol_nan = vol.isnull().sum()
            period_nan_count += vol_nan
            features[f'volatility_{window}d'] = vol
            nan_report[f'volatility_{window}d'] = vol_nan
            logger.debug(f"DEBUG: volatility_{window}d has {vol_nan} NaN values ({vol_nan/len(vol)*100:.1f}%)")

            # Volatility of volatility
            vol_of_vol = vol.rolling(window).std()
            vov_nan = vol_of_vol.isnull().sum()
            period_nan_count += vov_nan
            features[f'vol_of_vol_{window}d'] = vol_of_vol
            nan_report[f'vol_of_vol_{window}d'] = vov_nan
            logger.debug(f"DEBUG: vol_of_vol_{window}d has {vov_nan} NaN values ({vov_nan/len(vol_of_vol)*100:.1f}%)")

            # Volatility rank
            rank_window = min(window * 3, 252)
            vol_rank = vol.rolling(rank_window).rank(pct=True)
            rank_nan = vol_rank.isnull().sum()
            period_nan_count += rank_nan
            features[f'volatility_rank_{window}d'] = vol_rank
            nan_report[f'volatility_rank_{window}d'] = rank_nan
            logger.debug(f"DEBUG: volatility_rank_{window}d has {rank_nan} NaN values ({rank_nan/len(vol_rank)*100:.1f}%)")

            logger.info(f"DEBUG: Window {window} total NaN values: {period_nan_count}")

        # Additional volatility measures
        # logger.debug("DEBUG: Computing Parkinson volatility...")
        parkinson_vol = self._calculate_parkinson_volatility(data)
        parkinson_nan = parkinson_vol.isnull().sum()
        nan_report['parkinson_volatility'] = parkinson_nan
        features['parkinson_volatility'] = parkinson_vol
        logger.debug(f"DEBUG: parkinson_volatility has {parkinson_nan} NaN values ({parkinson_nan/len(parkinson_vol)*100:.1f}%)")

        # logger.debug("DEBUG: Computing Garman-Klass volatility...")
        garman_klass_vol = self._calculate_garman_klass_volatility(data)
        gk_nan = garman_klass_vol.isnull().sum()
        nan_report['garman_klass_volatility'] = gk_nan
        features['garman_klass_volatility'] = garman_klass_vol
        logger.debug(f"DEBUG: garman_klass_volatility has {gk_nan} NaN values ({gk_nan/len(garman_klass_vol)*100:.1f}%)")

        # logger.debug("DEBUG: Computing range volatility...")
        range_vol = self._calculate_range_volatility(data)
        range_nan = range_vol.isnull().sum()
        nan_report['range_volatility'] = range_nan
        features['range_volatility'] = range_vol
        logger.debug(f"DEBUG: range_volatility has {range_nan} NaN values ({range_nan/len(range_vol)*100:.1f}%)")

        # Final volatility features summary
        total_volatility_features = len(features.columns)
        final_total_nan = features.isnull().sum().sum()
        # logger.info(f"DEBUG: Volatility features computation complete:")
        # logger.info(f"DEBUG: - Total features created: {total_volatility_features}")
        logger.info(f"DEBUG: - Total NaN values: {final_total_nan} out of {len(features) * total_volatility_features} possible values")
        logger.info(f"DEBUG: - NaN percentage: {final_total_nan/(len(features) * total_volatility_features)*100:.2f}%")
        logger.info(f"DEBUG: - NaN breakdown: {nan_report}")

        return features

    def compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        # logger.info(f"DEBUG: Computing technical indicators on {len(data)} rows")
        features = pd.DataFrame(index=data.index)

        initial_nan_count = data['Close'].isnull().sum()
        logger.info(f"DEBUG: Input prices have {initial_nan_count} NaN values ({initial_nan_count/len(data)*100:.1f}%)")
        nan_report = {}

        # Moving averages
        # logger.debug("DEBUG: Computing simple moving averages...")
        for period in [10, 20, 50, 200]:
            sma = data['Close'].rolling(period).mean()
            sma_nan = sma.isnull().sum()
            features[f'sma_{period}'] = sma
            features[f'price_above_sma_{period}'] = (data['Close'] > sma).astype(int)
            nan_report[f'sma_{period}'] = sma_nan
            logger.debug(f"DEBUG: sma_{period} has {sma_nan} NaN values ({sma_nan/len(sma)*100:.1f}%)")

        # Exponential moving averages
        # logger.debug("DEBUG: Computing exponential moving averages...")
        for period in [12, 26]:
            ema = data['Close'].ewm(span=period).mean()
            ema_nan = ema.isnull().sum()
            features[f'ema_{period}'] = ema
            nan_report[f'ema_{period}'] = ema_nan
            logger.debug(f"DEBUG: ema_{period} has {ema_nan} NaN values ({ema_nan/len(ema)*100:.1f}%)")

        # MACD
        # logger.debug("DEBUG: Computing MACD...")
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal

        macd_line_nan = macd_line.isnull().sum()
        macd_signal_nan = macd_signal.isnull().sum()
        macd_histogram_nan = macd_histogram.isnull().sum()

        features['macd_line'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_histogram

        nan_report['macd_line'] = macd_line_nan
        nan_report['macd_signal'] = macd_signal_nan
        nan_report['macd_histogram'] = macd_histogram_nan

        logger.debug(f"DEBUG: macd_line has {macd_line_nan} NaN values ({macd_line_nan/len(macd_line)*100:.1f}%)")
        logger.debug(f"DEBUG: macd_signal has {macd_signal_nan} NaN values ({macd_signal_nan/len(macd_signal)*100:.1f}%)")
        logger.debug(f"DEBUG: macd_histogram has {macd_histogram_nan} NaN values ({macd_histogram_nan/len(macd_histogram)*100:.1f}%)")

        # Bollinger Bands
        logger.debug("DEBUG: Computing Bollinger Bands...")
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['Close'])
        bb_upper_nan = bb_upper.isnull().sum()
        bb_lower_nan = bb_lower.isnull().sum()
        bb_middle_nan = bb_middle.isnull().sum()

        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle

        bb_position_nan = features['bb_position'].isnull().sum()
        bb_width_nan = features['bb_width'].isnull().sum()

        nan_report['bb_upper'] = bb_upper_nan
        nan_report['bb_lower'] = bb_lower_nan
        nan_report['bb_middle'] = bb_middle_nan
        nan_report['bb_position'] = bb_position_nan
        nan_report['bb_width'] = bb_width_nan

        logger.debug(f"DEBUG: bb_upper has {bb_upper_nan} NaN values ({bb_upper_nan/len(bb_upper)*100:.1f}%)")
        logger.debug(f"DEBUG: bb_middle has {bb_middle_nan} NaN values ({bb_middle_nan/len(bb_middle)*100:.1f}%)")
        logger.debug(f"DEBUG: bb_lower has {bb_lower_nan} NaN values ({bb_lower_nan/len(bb_lower)*100:.1f}%)")
        logger.debug(f"DEBUG: bb_position has {bb_position_nan} NaN values ({bb_position_nan/len(features['bb_position'])*100:.1f}%)")
        logger.debug(f"DEBUG: bb_width has {bb_width_nan} NaN values ({bb_width_nan/len(features['bb_width'])*100:.1f}%)")

        # ADX
        logger.debug("DEBUG: Computing ADX...")
        adx = self._calculate_adx(data)
        adx_nan = adx.isnull().sum()
        features['adx'] = adx
        nan_report['adx'] = adx_nan
        logger.debug(f"DEBUG: adx has {adx_nan} NaN values ({adx_nan/len(adx)*100:.1f}%)")

        # Commodity Channel Index
        logger.debug("DEBUG: Computing CCI...")
        cci = self._calculate_cci(data)
        cci_nan = cci.isnull().sum()
        features['cci'] = cci
        nan_report['cci'] = cci_nan
        # logger.debug(f"DEBUG: cci has {cci_nan} NaN values ({cci_nan/len(cci)*100:.1f}%)")

        # Summary for technical indicators
        total_technical_nan = sum(nan_report.values())
        # logger.info(f"DEBUG: Technical indicators summary - Total NaN values: {total_technical_nan}")
        # logger.debug(f"DEBUG: Technical indicators NaN breakdown: {nan_report}")

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
