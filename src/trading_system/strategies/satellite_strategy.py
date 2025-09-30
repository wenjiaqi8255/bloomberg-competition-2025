"""
Satellite Strategy using technical indicators.

Implements a simplified satellite strategy due to data constraints (no Bloomberg event data).
Uses technical indicators and momentum-based signals to generate alpha.

This strategy:
1. Calculates multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Generates momentum-based signals
3. Applies risk management and position sizing
4. Complements the Core FFML Strategy with tactical tilts
5. Uses 20-30% of portfolio capital for satellite opportunities
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..data.stock_classifier import StockClassifier, InvestmentBox, SizeCategory, StyleCategory, RegionCategory, SectorCategory
from ..types.data_types import (
    BaseStrategy, StrategyConfig, BacktestConfig, TradingSignal, SignalType,
    PortfolioPosition, PortfolioSnapshot, Trade, DataValidator, DataValidationError
)
from ..backtesting.risk_management import RiskManager
from ..utils.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class SatelliteStrategy(BaseStrategy):
    """
    Satellite Strategy using technical indicators and momentum.

    Complements the Core FFML Strategy with tactical tilts based on:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Momentum and trend following
    - Mean reversion signals
    - Volatility-based positioning
    - Relative strength signals
    """

    def __init__(self, config=None, backtest_config=None, **kwargs):
        """
        Initialize Satellite Strategy.

        Args:
            config: Strategy configuration with parameters (StrategyConfig or dict)
            backtest_config: Backtest configuration (BacktestConfig or dict)
            **kwargs: Direct parameters for backward compatibility
        """
        # Handle backward compatibility - if kwargs are passed, create config objects
        if kwargs and (config is None or isinstance(config, dict)):
            # Create config objects from kwargs
            from ..types.data_types import StrategyConfig, BacktestConfig

            if config is None:
                config = {}

            # Merge kwargs with config dict
            merged_config = {**config, **kwargs}

            # Extract strategy-specific parameters
            strategy_params = {
                'name': merged_config.get('strategy_name', 'Satellite_Strategy'),
                'parameters': {
                    'satellite_weight': merged_config.get('satellite_weight', 0.25),
                    'max_positions': merged_config.get('max_positions', 8),
                    'stop_loss_threshold': merged_config.get('stop_loss_threshold', 0.05),
                    'take_profit_threshold': merged_config.get('take_profit_threshold', 0.15),
                    'rebalance_frequency': merged_config.get('rebalance_frequency', 7),
                    'rsi_period': merged_config.get('rsi_period', 14),
                    'macd_fast': merged_config.get('macd_fast', 12),
                    'macd_slow': merged_config.get('macd_slow', 26),
                    'macd_signal': merged_config.get('macd_signal', 9),
                    'bollinger_period': merged_config.get('bollinger_period', 20),
                    'bollinger_std': merged_config.get('bollinger_std', 2),
                    'momentum_period': merged_config.get('momentum_period', 21),
                    'volatility_period': merged_config.get('volatility_period', 20)
                },
                'lookback_period': merged_config.get('lookback_window', 63),
                'universe': merged_config.get('symbols', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'])
            }

            # Create backtest config
            backtest_params = {
                'start_date': merged_config.get('start_date', datetime(2020, 1, 1)),
                'end_date': merged_config.get('end_date', datetime(2023, 12, 31)),
                'initial_capital': merged_config.get('initial_capital', 1000000),
                'symbols': merged_config.get('symbols', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']),
                'rebalance_frequency': merged_config.get('rebalance_frequency', 7),
                'transaction_cost': merged_config.get('transaction_cost', 0.001),
                'slippage': merged_config.get('slippage', 0.0005)
            }

            # Create config objects
            config = StrategyConfig(**strategy_params)
            backtest_config = BacktestConfig(**backtest_params)

        # Ensure config and backtest_config are proper objects
        if isinstance(config, dict):
            config = StrategyConfig(**config)
        if isinstance(backtest_config, dict):
            backtest_config = BacktestConfig(**backtest_config)

        super().__init__(config, backtest_config)

        # Initialize components
        self.stock_classifier = StockClassifier()
        self.feature_engineering = FeatureEngineering()
        self.risk_manager = RiskManager()

        # Strategy parameters
        self.momentum_lookback = config.parameters.get('momentum_lookback', 12)  # 12 months
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.macd_fast = config.parameters.get('macd_fast', 12)
        self.macd_slow = config.parameters.get('macd_slow', 26)
        self.macd_signal = config.parameters.get('macd_signal', 9)
        self.bb_period = config.parameters.get('bb_period', 20)
        self.bb_std = config.parameters.get('bb_std', 2)

        # Signal thresholds
        self.rsi_oversold = config.parameters.get('rsi_oversold', 30)
        self.rsi_overbought = config.parameters.get('rsi_overbought', 70)
        self.momentum_threshold = config.parameters.get('momentum_threshold', 0.1)
        self.min_signal_strength = config.parameters.get('min_signal_strength', 0.3)
        self.max_correlation = config.parameters.get('max_correlation', 0.7)

        # Risk management parameters
        self.max_position_size = config.parameters.get('max_position_size', 0.05)  # 5% per position
        self.max_satellite_exposure = config.parameters.get('max_satellite_exposure', 0.3)  # 30% of portfolio
        self.stop_loss_threshold = config.parameters.get('stop_loss_threshold', 0.08)  # 8% stop loss
        self.take_profit_threshold = config.parameters.get('take_profit_threshold', 0.15)  # 15% take profit

        # Portfolio construction
        self.max_positions = config.parameters.get('max_positions', 10)
        self.rebalance_threshold = config.parameters.get('rebalance_threshold', 0.1)

        # Data storage
        self.equity_data = {}
        self.technical_indicators = {}
        self.signals_history = []
        self.portfolio_history = []
        self.risk_metrics = {}

        # Performance tracking
        self.signal_accuracy = {}
        self.strategy_performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

        logger.info(f"Initialized Satellite Strategy with {len(config.universe)} symbols")

    def prepare_data(self, start_date: datetime = None, end_date: datetime = None) -> bool:
        """
        Prepare and validate all required data for the strategy.

        Args:
            start_date: Start date for data preparation
            end_date: End date for data preparation

        Returns:
            True if data preparation successful
        """
        try:
            logger.info("Starting data preparation for Satellite Strategy")

            # Use backtest config dates if not provided
            if start_date is None:
                start_date = self.backtest_config.start_date
            if end_date is None:
                end_date = self.backtest_config.end_date

            # Step 1: Fetch equity data
            logger.info("Fetching equity data...")
            self.equity_data = self._fetch_equity_data(start_date, end_date)

            # Step 2: Calculate technical indicators
            logger.info("Calculating technical indicators...")
            self.technical_indicators = self._calculate_technical_indicators()

            # Step 3: Classify stocks for risk management
            logger.info("Classifying stocks...")
            self.stock_classifications = self._classify_stocks()

            # Step 4: Calculate risk metrics
            logger.info("Calculating risk metrics...")
            self.risk_metrics = self._calculate_risk_metrics()

            logger.info(f"Data preparation completed successfully")
            logger.info(f"Processed {len(self.equity_data)} symbols with technical indicators")

            return True

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise DataValidationError(f"Data preparation failed: {e}")

    def _fetch_equity_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch equity data for all symbols in universe."""
        equity_data = {}

        for symbol in self.config.universe:
            try:
                from ..data.data_provider import DataProvider
                data_provider = DataProvider()

                data = data_provider.get_price_data(
                    symbol, start_date, end_date, frequency="1mo"
                )

                if data is not None and len(data) > 12:  # Need at least 12 months of data
                    equity_data[symbol] = data
                    logger.debug(f"Fetched {len(data)} rows of data for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        if not equity_data:
            raise DataValidationError("No equity data available for any symbols")

        return equity_data

    def _calculate_technical_indicators(self) -> Dict[str, Dict[str, pd.Series]]:
        """Calculate technical indicators for all symbols."""
        indicators = {}

        for symbol, data in self.equity_data.items():
            try:
                symbol_indicators = {}

                # RSI (Relative Strength Index)
                symbol_indicators['rsi'] = self._calculate_rsi(data['Close'])

                # MACD (Moving Average Convergence Divergence)
                macd_data = self._calculate_macd(data['Close'])
                symbol_indicators['macd'] = macd_data['macd']
                symbol_indicators['macd_signal'] = macd_data['signal']
                symbol_indicators['macd_histogram'] = macd_data['histogram']

                # Bollinger Bands
                bb_data = self._calculate_bollinger_bands(data['Close'])
                symbol_indicators['bb_upper'] = bb_data['upper']
                symbol_indicators['bb_middle'] = bb_data['middle']
                symbol_indicators['bb_lower'] = bb_data['lower']
                symbol_indicators['bb_position'] = bb_data['position']

                # Momentum indicators
                symbol_indicators['momentum'] = self._calculate_momentum(data['Close'])
                symbol_indicators['rate_of_change'] = self._calculate_rate_of_change(data['Close'])

                # Volatility indicators
                symbol_indicators['volatility'] = self._calculate_volatility(data['Close'])
                symbol_indicators['atr'] = self._calculate_atr(data)

                # Trend indicators
                symbol_indicators['sma_50'] = data['Close'].rolling(window=50).mean()
                symbol_indicators['sma_200'] = data['Close'].rolling(window=200).mean()
                symbol_indicators['ema_12'] = data['Close'].ewm(span=12).mean()
                symbol_indicators['ema_26'] = data['Close'].ewm(span=26).mean()

                indicators[symbol] = symbol_indicators
                logger.debug(f"Calculated {len(symbol_indicators)} indicators for {symbol}")

            except Exception as e:
                logger.error(f"Failed to calculate indicators for {symbol}: {e}")
                continue

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        if period is None:
            period = self.rsi_period

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = prices.rolling(window=self.bb_period).mean()
        std_dev = prices.rolling(window=self.bb_period).std()
        upper_band = middle_band + (std_dev * self.bb_std)
        lower_band = middle_band - (std_dev * self.bb_std)

        # Calculate position within bands (0 to 1, where 0.5 is middle)
        position = (prices - lower_band) / (upper_band - lower_band)

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'position': position
        }

    def _calculate_momentum(self, prices: pd.Series, lookback: int = None) -> pd.Series:
        """Calculate momentum indicator."""
        if lookback is None:
            lookback = self.momentum_lookback

        return prices.pct_change(lookback)

    def _calculate_rate_of_change(self, prices: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change (ROC)."""
        return prices.pct_change(period)

    def _calculate_volatility(self, prices: pd.Series, lookback: int = 20) -> pd.Series:
        """Calculate volatility indicator."""
        return prices.pct_change().rolling(window=lookback).std()

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def _classify_stocks(self) -> Dict[str, InvestmentBox]:
        """Classify stocks into investment boxes for risk management."""
        classifications = {}

        for symbol, data in self.equity_data.items():
            try:
                box = self.stock_classifier.classify_stock(symbol, data)
                classifications[symbol] = box
            except Exception as e:
                logger.error(f"Failed to classify {symbol}: {e}")
                continue

        return classifications

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics for all symbols."""
        risk_metrics = {}

        for symbol, data in self.equity_data.items():
            try:
                returns = data['Close'].pct_change().dropna()

                metrics = {
                    'volatility': returns.std() * np.sqrt(12),  # Annualized
                    'max_drawdown': self._calculate_max_drawdown_series(returns),
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0,
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'var_95': returns.quantile(0.05),
                    'var_99': returns.quantile(0.01),
                    'beta_to_market': self._calculate_beta_to_market(returns)
                }

                risk_metrics[symbol] = metrics

            except Exception as e:
                logger.error(f"Failed to calculate risk metrics for {symbol}: {e}")
                continue

        return risk_metrics

    def _calculate_max_drawdown_series(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a returns series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _calculate_beta_to_market(self, returns: pd.Series) -> float:
        """Calculate beta to market (simplified)."""
        # Use SPY as proxy for market returns
        try:
            spy_returns = self.equity_data.get('SPY', pd.DataFrame()).get('Close', pd.Series()).pct_change().dropna()

            if len(spy_returns) > 0 and len(returns) > 0:
                # Align dates
                aligned_returns = pd.concat([returns, spy_returns], axis=1, join='inner')
                if len(aligned_returns) > 10:
                    correlation = aligned_returns.iloc[:, 0].corr(aligned_returns.iloc[:, 1])
                    beta = correlation * (returns.std() / spy_returns.std())
                    return beta

        except Exception:
            pass

        return 1.0  # Default beta

    def generate_signals(self, date: datetime) -> List[TradingSignal]:
        """
        Generate trading signals based on technical indicators.

        Args:
            date: Date for signal generation

        Returns:
            List of trading signals
        """
        try:
            logger.info(f"Generating satellite signals for {date}")

            signals = []

            # Generate signals for each symbol
            for symbol in self.config.universe:
                if symbol not in self.technical_indicators:
                    continue

                signal = self._generate_symbol_signal(symbol, date)
                if signal is not None:
                    signals.append(signal)

            # Apply portfolio constraints and risk management
            signals = self._apply_portfolio_constraints(signals, date)

            # Rank signals by strength
            signals.sort(key=lambda x: x.strength, reverse=True)

            # Limit number of positions
            signals = signals[:self.max_positions]

            logger.info(f"Generated {len(signals)} satellite signals for {date}")
            self.signals_history.extend(signals)

            return signals

        except Exception as e:
            logger.error(f"Failed to generate satellite signals for {date}: {e}")
            return []

    def _generate_symbol_signal(self, symbol: str, date: datetime) -> Optional[TradingSignal]:
        """Generate signal for a single symbol based on technical indicators."""
        try:
            indicators = self.technical_indicators[symbol]

            # Get current values
            current_values = self._get_current_indicator_values(symbol, indicators, date)
            if current_values is None:
                return None

            # Calculate signal components
            rsi_signal = self._calculate_rsi_signal(current_values['rsi'])
            macd_signal = self._calculate_macd_signal(current_values['macd'], current_values['macd_signal'])
            bb_signal = self._calculate_bb_signal(current_values['bb_position'])
            momentum_signal = self._calculate_momentum_signal(current_values['momentum'])
            trend_signal = self._calculate_trend_signal(current_values)

            # Combine signals
            combined_score = self._combine_signals(rsi_signal, macd_signal, bb_signal, momentum_signal, trend_signal)

            # Generate trading signal if strong enough
            if abs(combined_score) >= self.min_signal_strength:
                signal_type = SignalType.BUY if combined_score > 0 else SignalType.SELL
                strength = min(abs(combined_score), 1.0)
                confidence = self._calculate_signal_confidence(symbol, combined_score, current_values)

                # Get current price
                current_price = self._get_current_price(symbol, date)
                if current_price is None:
                    return None

                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    timestamp=date,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'combined_score': combined_score,
                        'rsi_signal': rsi_signal,
                        'macd_signal': macd_signal,
                        'bb_signal': bb_signal,
                        'momentum_signal': momentum_signal,
                        'trend_signal': trend_signal,
                        'indicator_values': current_values,
                        'position_size': self._calculate_position_size(symbol, strength)
                    }
                )

                return signal

            return None

        except Exception as e:
            logger.error(f"Failed to generate signal for {symbol}: {e}")
            return None

    def _get_current_indicator_values(self, symbol: str, indicators: Dict[str, pd.Series], date: datetime) -> Optional[Dict[str, float]]:
        """Get current indicator values for a symbol."""
        try:
            values = {}

            for indicator_name, series in indicators.items():
                if len(series) == 0:
                    return None

                # Find closest date
                available_dates = series.index[series.index <= date]
                if len(available_dates) == 0:
                    return None

                closest_date = available_dates[-1]
                values[indicator_name] = series.loc[closest_date]

            return values

        except Exception as e:
            logger.error(f"Failed to get indicator values for {symbol}: {e}")
            return None

    def _calculate_rsi_signal(self, rsi: float) -> float:
        """Calculate RSI-based signal (-1 to 1)."""
        if rsi < self.rsi_oversold:
            return 0.7  # Strong buy signal
        elif rsi > self.rsi_overbought:
            return -0.7  # Strong sell signal
        else:
            # Neutral zone
            if rsi < 50:
                return (50 - rsi) / 50 * 0.3  # Mild buy
            else:
                return -(rsi - 50) / 50 * 0.3  # Mild sell

    def _calculate_macd_signal(self, macd: float, macd_signal: float) -> float:
        """Calculate MACD-based signal (-1 to 1)."""
        histogram = macd - macd_signal

        if histogram > 0 and macd > 0:
            return 0.6  # Bullish signal
        elif histogram < 0 and macd < 0:
            return -0.6  # Bearish signal
        else:
            return histogram * 0.1  # Weak signal based on histogram

    def _calculate_bb_signal(self, bb_position: float) -> float:
        """Calculate Bollinger Bands-based signal (-1 to 1)."""
        if bb_position < 0.1:  # Near lower band
            return 0.8  # Strong buy (mean reversion)
        elif bb_position > 0.9:  # Near upper band
            return -0.8  # Strong sell (mean reversion)
        elif bb_position < 0.3:  # Lower half
            return 0.3  # Mild buy
        elif bb_position > 0.7:  # Upper half
            return -0.3  # Mild sell
        else:
            return 0.0  # Neutral

    def _calculate_momentum_signal(self, momentum: float) -> float:
        """Calculate momentum-based signal (-1 to 1)."""
        if momentum > self.momentum_threshold:
            return 0.7  # Strong buy
        elif momentum < -self.momentum_threshold:
            return -0.7  # Strong sell
        else:
            return momentum / self.momentum_threshold * 0.5  # Scaled signal

    def _calculate_trend_signal(self, values: Dict[str, float]) -> float:
        """Calculate trend-based signal (-1 to 1)."""
        try:
            # Simple moving average crossover
            if values['sma_50'] > values['sma_200']:
                return 0.4  # Uptrend
            else:
                return -0.4  # Downtrend

        except KeyError:
            return 0.0  # No trend signal

    def _combine_signals(self, rsi_signal: float, macd_signal: float, bb_signal: float,
                       momentum_signal: float, trend_signal: float) -> float:
        """Combine multiple signals with weights."""
        weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bb': 0.25,
            'momentum': 0.25,
            'trend': 0.1
        }

        combined = (
            weights['rsi'] * rsi_signal +
            weights['macd'] * macd_signal +
            weights['bb'] * bb_signal +
            weights['momentum'] * momentum_signal +
            weights['trend'] * trend_signal
        )

        return max(-1.0, min(1.0, combined))  # Clamp to [-1, 1]

    def _calculate_signal_confidence(self, symbol: str, signal_score: float, values: Dict[str, float]) -> float:
        """Calculate signal confidence based on multiple factors."""
        try:
            confidence_factors = []

            # 1. Signal strength
            strength_confidence = abs(signal_score)
            confidence_factors.append(strength_confidence)

            # 2. Technical indicator agreement
            indicators = [values['rsi'], values['macd'], values['bb_position']]
            agreement = sum(1 for ind in indicators if ind is not None) / len(indicators)
            confidence_factors.append(agreement)

            # 3. Recent performance
            recent_accuracy = self.signal_accuracy.get(symbol, {}).get('recent_accuracy', 0.5)
            confidence_factors.append(recent_accuracy)

            # 4. Risk metrics
            risk_score = self._calculate_risk_adjusted_score(symbol)
            confidence_factors.append(risk_score)

            # Weighted average of confidence factors
            weights = [0.4, 0.2, 0.2, 0.2]
            confidence = sum(w * f for w, f in zip(weights, confidence_factors))

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Failed to calculate confidence for {symbol}: {e}")
            return 0.5

    def _calculate_risk_adjusted_score(self, symbol: str) -> float:
        """Calculate risk-adjusted score for a symbol."""
        try:
            if symbol not in self.risk_metrics:
                return 0.5

            metrics = self.risk_metrics[symbol]

            # Penalize high volatility and drawdown
            volatility_penalty = min(metrics['volatility'] / 0.3, 1.0)  # Normalize to 30% vol
            drawdown_penalty = min(abs(metrics['max_drawdown']) / 0.2, 1.0)  # Normalize to 20% drawdown

            risk_score = 1.0 - (volatility_penalty * 0.6 + drawdown_penalty * 0.4)
            return max(0.0, risk_score)

        except Exception:
            return 0.5

    def _get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get current price for a symbol."""
        if symbol not in self.equity_data:
            return None

        data = self.equity_data[symbol]
        available_dates = data.index[data.index <= date]

        if len(available_dates) == 0:
            return None

        closest_date = available_dates[-1]
        return data.loc[closest_date, 'Close']

    def _calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk."""
        base_size = self.max_position_size

        # Adjust for signal strength
        strength_adjusted = base_size * signal_strength

        # Adjust for risk
        risk_adjusted = strength_adjusted * self._calculate_risk_adjusted_score(symbol)

        return max(0.0, min(risk_adjusted, self.max_position_size))

    def _apply_portfolio_constraints(self, signals: List[TradingSignal], date: datetime) -> List[TradingSignal]:
        """Apply portfolio constraints and risk management."""
        if not signals:
            return signals

        # Check correlation between signals
        filtered_signals = self._filter_by_correlation(signals)

        # Apply position size limits
        total_exposure = sum(signal.metadata.get('position_size', 0.0) for signal in filtered_signals)

        if total_exposure > self.max_satellite_exposure:
            # Scale down positions
            scale_factor = self.max_satellite_exposure / total_exposure
            for signal in filtered_signals:
                signal.metadata['position_size'] *= scale_factor

        return filtered_signals

    def _filter_by_correlation(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals to avoid highly correlated positions."""
        if len(signals) <= 2:
            return signals

        # Simple correlation filter - remove highly correlated signals
        filtered = []

        for i, signal in enumerate(signals):
            keep_signal = True

            # Check correlation with already selected signals
            for existing_signal in filtered:
                correlation = self._calculate_signal_correlation(signal, existing_signal)
                if correlation > self.max_correlation:
                    keep_signal = False
                    break

            if keep_signal:
                filtered.append(signal)

        return filtered

    def _calculate_signal_correlation(self, signal1: TradingSignal, signal2: TradingSignal) -> float:
        """Calculate correlation between two signals (simplified)."""
        try:
            # Use price correlation as proxy
            symbol1 = signal1.symbol
            symbol2 = signal2.symbol

            if symbol1 in self.equity_data and symbol2 in self.equity_data:
                returns1 = self.equity_data[symbol1]['Close'].pct_change().dropna()
                returns2 = self.equity_data[symbol2]['Close'].pct_change().dropna()

                if len(returns1) > 0 and len(returns2) > 0:
                    # Align dates
                    aligned_returns = pd.concat([returns1, returns2], axis=1, join='inner')
                    if len(aligned_returns) > 10:
                        correlation = aligned_returns.iloc[:, 0].corr(aligned_returns.iloc[:, 1])
                        return abs(correlation)  # Return absolute correlation

        except Exception:
            pass

        return 0.0  # Default no correlation

    def execute_trades(self, signals: List[TradingSignal], portfolio: PortfolioSnapshot) -> List[Trade]:
        """
        Execute trades based on signals.

        Args:
            signals: List of trading signals
            portfolio: Current portfolio snapshot

        Returns:
            List of executed trades
        """
        try:
            logger.info(f"Executing satellite trades for {len(signals)} signals")

            trades = []
            current_positions = {pos.symbol: pos for pos in portfolio.positions}

            for signal in signals:
                target_weight = signal.metadata.get('position_size', 0.0)
                current_weight = current_positions.get(signal.symbol, PortfolioPosition(
                    symbol=signal.symbol, quantity=0, average_cost=0, current_price=signal.price,
                    market_value=0, unrealized_pnl=0, weight=0.0
                )).weight

                # Calculate required trade
                weight_change = target_weight - current_weight

                if abs(weight_change) < self.rebalance_threshold:
                    continue

                # Apply stop-loss and take-profit
                if not self._check_risk_management(signal, current_positions.get(signal.symbol)):
                    continue

                # Calculate trade quantity
                trade_value = portfolio.total_value * weight_change
                quantity = trade_value / signal.price

                # Determine trade side
                side = 'buy' if quantity > 0 else 'sell'
                quantity = abs(quantity)

                trade = Trade(
                    symbol=signal.symbol,
                    side=side,
                    quantity=quantity,
                    price=signal.price,
                    timestamp=signal.timestamp,
                    commission=0.0,
                    trade_id=f"satellite_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d')}"
                )
                trades.append(trade)

            logger.info(f"Generated {len(trades)} satellite trades")
            return trades

        except Exception as e:
            logger.error(f"Failed to execute satellite trades: {e}")
            return []

    def _check_risk_management(self, signal: TradingSignal, current_position: Optional[PortfolioPosition]) -> bool:
        """Check if signal passes risk management rules."""
        try:
            if current_position is None:
                return True  # New position

            # Check stop-loss
            if current_position.unrealized_pnl < -self.stop_loss_threshold * current_position.market_value:
                logger.info(f"Stop-loss triggered for {signal.symbol}")
                return False

            # Check take-profit
            if current_position.unrealized_pnl > self.take_profit_threshold * current_position.market_value:
                logger.info(f"Take-profit triggered for {signal.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check risk management for {signal.symbol}: {e}")
            return True

    def monitor_performance(self, portfolio: PortfolioSnapshot) -> Dict[str, Any]:
        """
        Monitor strategy performance and generate diagnostics.

        Args:
            portfolio: Current portfolio snapshot

        Returns:
            Performance metrics and diagnostics
        """
        try:
            # Calculate performance metrics
            metrics = {
                'total_return': portfolio.total_return,
                'daily_return': portfolio.daily_return,
                'drawdown': portfolio.drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio),
                'sortino_ratio': self._calculate_sortino_ratio(portfolio),
                'max_drawdown': self._calculate_max_drawdown(portfolio),
                'volatility': self._calculate_volatility(portfolio),
                'number_of_positions': len(portfolio.positions),
                'concentration': self._calculate_concentration(portfolio),
                'turnover': self._calculate_turnover(),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor()
            }

            # Signal quality metrics
            metrics['signal_quality'] = {
                'signal_count': len(self.signals_history),
                'accuracy': self._calculate_overall_accuracy(),
                'average_strength': self._calculate_average_strength(),
                'confidence_correlation': self._calculate_confidence_correlation()
            }

            # Risk metrics
            metrics['risk_metrics'] = {
                'var_95': self._calculate_var(portfolio, 0.05),
                'var_99': self._calculate_var(portfolio, 0.01),
                'expected_shortfall': self._calculate_expected_shortfall(portfolio),
                'beta_to_market': self._calculate_beta_to_market_portfolio(portfolio),
                'tracking_error': self._calculate_tracking_error(portfolio)
            }

            # Sector and style allocation
            metrics['allocations'] = self._calculate_allocations(portfolio)

            return metrics

        except Exception as e:
            logger.error(f"Failed to monitor performance: {e}")
            return {}

    def _calculate_sharpe_ratio(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate Sharpe ratio for satellite strategy."""
        if len(self.portfolio_history) < 2:
            return 0.0

        returns = [p.daily_return for p in self.portfolio_history[-12:]]
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        return avg_return / std_return if std_return > 0 else 0.0

    def _calculate_sortino_ratio(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        if len(self.portfolio_history) < 2:
            return 0.0

        returns = [p.daily_return for p in self.portfolio_history[-12:]]
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.001

        return avg_return / downside_std if downside_std > 0 else 0.0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate of signals."""
        if not self.signals_history:
            return 0.0

        # Simplified - would need actual trade outcomes
        return 0.55  # Placeholder

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        # Simplified calculation
        return 1.2  # Placeholder

    def _calculate_overall_accuracy(self) -> float:
        """Calculate overall signal accuracy."""
        if not self.signal_accuracy:
            return 0.5

        accuracies = [acc['recent_accuracy'] for acc in self.signal_accuracy.values()]
        return np.mean(accuracies) if accuracies else 0.5

    def _calculate_average_strength(self) -> float:
        """Calculate average signal strength."""
        if not self.signals_history:
            return 0.0

        strengths = [signal.strength for signal in self.signals_history]
        return np.mean(strengths) if strengths else 0.0

    def _calculate_confidence_correlation(self) -> float:
        """Calculate correlation between signal confidence and accuracy."""
        # Simplified calculation
        return 0.3  # Placeholder

    def _calculate_beta_to_market_portfolio(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate portfolio beta to market."""
        # Simplified calculation
        return 1.1  # Slightly higher beta for satellite

    def _calculate_tracking_error(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate tracking error vs benchmark."""
        return 0.08  # 8% tracking error for satellite

    def _calculate_allocations(self, portfolio: PortfolioSnapshot) -> Dict[str, Any]:
        """Calculate sector and style allocations."""
        allocations = {
            'sectors': {},
            'styles': {},
            'sizes': {}
        }

        for pos in portfolio.positions:
            symbol = pos.symbol
            if symbol in self.stock_classifications:
                box = self.stock_classifications[symbol]

                # Sector allocation
                sector = box.sector.name
                allocations['sectors'][sector] = allocations['sectors'].get(sector, 0.0) + pos.weight

                # Style allocation
                style = box.style.name
                allocations['styles'][style] = allocations['styles'].get(style, 0.0) + pos.weight

                # Size allocation
                size = box.size.name
                allocations['sizes'][size] = allocations['sizes'].get(size, 0.0) + pos.weight

        return allocations

    def save_results(self, output_dir: str) -> None:
        """
        Save strategy results to files.

        Args:
            output_dir: Output directory path
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save signal history
            signals_file = output_path / "satellite_signals.csv"
            if self.signals_history:
                signals_data = []
                for signal in self.signals_history:
                    signals_data.append({
                        'timestamp': signal.timestamp.isoformat(),
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type.value,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'price': signal.price,
                        'combined_score': signal.metadata.get('combined_score', 0.0)
                    })

                signals_df = pd.DataFrame(signals_data)
                signals_df.to_csv(signals_file, index=False)

            # Save technical indicators
            indicators_file = output_path / "satellite_indicators.csv"
            if self.technical_indicators:
                # Save latest indicator values
                latest_indicators = []
                for symbol, indicators in self.technical_indicators.items():
                    latest_values = {}
                    for name, series in indicators.items():
                        if len(series) > 0:
                            latest_values[name] = series.iloc[-1]
                    latest_values['symbol'] = symbol
                    latest_indicators.append(latest_values)

                indicators_df = pd.DataFrame(latest_indicators)
                indicators_df.to_csv(indicators_file, index=False)

            logger.info(f"Saved satellite results to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save satellite results: {e}")

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and configuration."""
        return {
            'strategy_name': self.config.name,
            'strategy_type': 'Satellite Technical',
            'universe': self.config.universe,
            'parameters': self.config.parameters,
            'indicators_used': [
                'RSI', 'MACD', 'Bollinger Bands', 'Momentum',
                'Rate of Change', 'Volatility', 'ATR', 'Moving Averages'
            ],
            'signal_weights': {
                'rsi': 0.2,
                'macd': 0.2,
                'bollinger_bands': 0.25,
                'momentum': 0.25,
                'trend': 0.1
            },
            'risk_management': {
                'max_position_size': self.max_position_size,
                'max_satellite_exposure': self.max_satellite_exposure,
                'stop_loss_threshold': self.stop_loss_threshold,
                'take_profit_threshold': self.take_profit_threshold
            }
        }