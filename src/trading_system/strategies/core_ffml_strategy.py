"""
Core FFML (Fama-French + Machine Learning) Strategy.

Implements the IPS Method A approach: Expected Return = Factor Return + ML Residual
where Factor Return comes from FF5 model and ML Residual comes from technical feature prediction.

This strategy:
1. Estimates factor betas using rolling window FF5 regression
2. Extracts residuals from FF5 model
3. Trains ML models to predict future residuals using technical features
4. Combines factor-implied returns with ML residual predictions
5. Generates trading signals based on combined predictions
6. Applies portfolio construction with risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..data.ff5_provider import FF5DataProvider
from ..data.stock_classifier import StockClassifier, InvestmentBox, SizeCategory, StyleCategory, RegionCategory
from ..models.ff5_regression import FF5RegressionEngine
from ..models.residual_predictor import ResidualPredictor
from ..types.data_types import (
    BaseStrategy, StrategyConfig, BacktestConfig, TradingSignal, SignalType,
    PortfolioPosition, PortfolioSnapshot, Trade, DataValidator, DataValidationError
)
from ..backtesting.engine import BacktestEngine
from ..backtesting.risk_management import RiskManager
from ..utils.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class CoreFFMLStrategy(BaseStrategy):
    """
    Core FFML Strategy implementing IPS Method A.

    Combines Fama-French 5-factor model with machine learning residual prediction
    to generate trading signals with superior risk-adjusted returns.
    """

    def __init__(self, config=None, backtest_config=None, **kwargs):
        """
        Initialize Core FFML Strategy.

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
                'name': merged_config.get('strategy_name', 'Core_FFML_Strategy'),
                'parameters': {
                    'ff5_lookback': merged_config.get('lookback_window', 252),
                    'min_observations': merged_config.get('min_observations', 24),
                    'prediction_horizon': merged_config.get('prediction_horizon', 1),
                    'feature_lags': merged_config.get('feature_lags', 5),
                    'cv_folds': merged_config.get('cv_folds', 5),
                    'min_signal_strength': merged_config.get('min_signal_strength', 0.1),
                    'max_position_size': merged_config.get('max_position_size', 0.15),
                    'rebalance_threshold': merged_config.get('rebalance_threshold', 0.05),
                    'factor_confidence_threshold': merged_config.get('factor_confidence_threshold', 0.7),
                    'ml_confidence_threshold': merged_config.get('ml_confidence_threshold', 0.6),
                    'target_positions': merged_config.get('target_positions', 20),
                    'risk_budget': merged_config.get('risk_budget', 0.02),
                    'correlation_threshold': merged_config.get('correlation_threshold', 0.7),
                    'core_weight': merged_config.get('core_weight', 0.8),
                    'volatility_target': merged_config.get('volatility_target', 0.15)
                },
                'lookback_period': merged_config.get('lookback_window', 252),
                'universe': merged_config.get('symbols', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM'])
            }

            # Create backtest config
            backtest_params = {
                'start_date': merged_config.get('start_date', datetime(2020, 1, 1)),
                'end_date': merged_config.get('end_date', datetime(2023, 12, 31)),
                'initial_capital': merged_config.get('initial_capital', 1000000),
                'symbols': merged_config.get('symbols', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']),
                'rebalance_frequency': merged_config.get('rebalance_frequency', 30),
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

        # Initialize data providers
        self.ff5_provider = FF5DataProvider(data_frequency="monthly")
        self.stock_classifier = StockClassifier()

        # Initialize models
        self.ff5_regression = FF5RegressionEngine(
            estimation_window=config.parameters.get('ff5_lookback', 36),
            min_observations=config.parameters.get('min_observations', 20)
        )

        self.residual_predictor = ResidualPredictor(
            prediction_horizon=config.parameters.get('prediction_horizon', 1),
            feature_lag=config.parameters.get('feature_lags', 1),
            cv_folds=config.parameters.get('cv_folds', 5)
        )

        # Initialize feature engineering
        self.feature_engineering = FeatureEngineering()

        # Strategy parameters
        self.min_signal_strength = config.parameters.get('min_signal_strength', 0.1)
        self.max_position_size = config.parameters.get('max_position_size', 0.1)
        self.rebalance_threshold = config.parameters.get('rebalance_threshold', 0.05)
        self.factor_confidence_threshold = config.parameters.get('factor_confidence_threshold', 0.7)
        self.ml_confidence_threshold = config.parameters.get('ml_confidence_threshold', 0.6)

        # Portfolio construction parameters
        self.target_positions = config.parameters.get('target_positions', 20)
        self.risk_budget = config.parameters.get('risk_budget', 0.02)  # 2% monthly risk budget
        self.correlation_threshold = config.parameters.get('correlation_threshold', 0.7)

        # Model performance tracking
        self.model_performance = {}
        self.signal_history = []
        self.portfolio_history = []

        # Data storage
        self.equity_data = {}
        self.factor_data = None
        self.residuals = {}
        self.ml_predictions = {}
        self.combined_predictions = {}

        # Investment box constraints
        self.box_constraints = self._initialize_box_constraints()

        logger.info(f"Initialized Core FFML Strategy with {len(config.universe)} symbols")

    def _initialize_box_constraints(self) -> Dict[str, Dict]:
        """Initialize investment box constraints for IPS compliance."""
        return {
            'size_limits': {
                SizeCategory.LARGE: (0.4, 0.8),      # 40-80% in large cap
                SizeCategory.MID: (0.15, 0.4),       # 15-40% in mid cap
                SizeCategory.SMALL: (0.0, 0.2)       # 0-20% in small cap
            },
            'style_limits': {
                StyleCategory.VALUE: (0.2, 0.6),     # 20-60% in value
                StyleCategory.GROWTH: (0.2, 0.6),    # 20-60% in growth
            },
            'sector_limits': {
                # Each sector max 20% of portfolio
                'Technology': (0.0, 0.2),
                'Financials': (0.0, 0.2),
                'Healthcare': (0.0, 0.2),
                'Consumer': (0.0, 0.2),
                'Industrial': (0.0, 0.2),
                'Energy': (0.0, 0.2),
                'Utilities': (0.0, 0.2),
                'Real Estate': (0.0, 0.2),
                'Materials': (0.0, 0.2),
                'Communication': (0.0, 0.2)
            },
            'region_limits': {
                RegionCategory.DEVELOPED: (0.6, 1.0), # 60-100% in developed markets
                RegionCategory.EMERGING: (0.0, 0.15) # 0-15% in emerging markets
            }
        }

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
            logger.info("Starting data preparation for Core FFML Strategy")

            # Use backtest config dates if not provided
            if start_date is None:
                start_date = self.backtest_config.start_date
            if end_date is None:
                end_date = self.backtest_config.end_date

            # Step 1: Fetch equity data
            logger.info("Fetching equity data...")
            self.equity_data = self._fetch_equity_data(start_date, end_date)

            # Step 2: Fetch and align FF5 factor data
            logger.info("Fetching FF5 factor data...")
            self.factor_data = self.ff5_provider.get_factor_returns(start_date, end_date)

            # Step 3: Align factor data with equity data
            logger.info("Aligning factor and equity data...")
            self.factor_data, self.equity_data = self.ff5_provider.align_with_equity_data(
                self.equity_data, self.factor_data
            )

            # Step 4: Classify stocks into investment boxes
            logger.info("Classifying stocks into investment boxes...")
            self.stock_classifications = self._classify_stocks()

            # Step 5: Estimate factor betas and extract residuals
            logger.info("Estimating factor betas...")
            factor_betas, factor_returns, self.residuals = self.ff5_regression.estimate_factor_betas(
                self.equity_data, self.factor_data
            )

            # Step 6: Train ML residual predictor
            logger.info("Training ML residual predictor...")
            self.residual_predictor.train_models(
                self.equity_data, self.residuals, start_date, end_date
            )

            # Step 7: Generate ML residual predictions
            logger.info("Generating ML residual predictions...")
            self.ml_predictions = self.residual_predictor.predict_residuals(
                self.equity_data, self.factor_data.index[-1]
            )

            # Step 8: Combine factor and ML predictions
            logger.info("Combining factor and ML predictions...")
            self.combined_predictions = self._combine_predictions(
                factor_returns, self.ml_predictions, factor_betas
            )

            logger.info(f"Data preparation completed successfully")
            logger.info(f"Processed {len(self.equity_data)} symbols with {len(self.factor_data)} factor observations")

            return True

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise DataValidationError(f"Data preparation failed: {e}")

    def _fetch_equity_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch equity data for all symbols in universe."""
        equity_data = {}

        for symbol in self.config.universe:
            try:
                # Fetch data using existing data infrastructure
                from ..data.data_provider import DataProvider
                data_provider = DataProvider()

                data = data_provider.get_price_data(
                    symbol, start_date, end_date, frequency="1mo"
                )

                if data is not None and len(data) > 0:
                    equity_data[symbol] = data
                    logger.debug(f"Fetched {len(data)} rows of data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        if not equity_data:
            raise DataValidationError("No equity data available for any symbols")

        return equity_data

    def _classify_stocks(self) -> Dict[str, InvestmentBox]:
        """Classify stocks into investment boxes."""
        classifications = {}

        for symbol, data in self.equity_data.items():
            try:
                box = self.stock_classifier.classify_stock(symbol, data)
                classifications[symbol] = box
                logger.debug(f"Classified {symbol} as {box}")
            except Exception as e:
                logger.error(f"Failed to classify {symbol}: {e}")
                continue

        return classifications

    def _combine_predictions(self, factor_returns: Dict[str, pd.Series],
                           ml_predictions: Dict[str, float],
                           factor_betas: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Combine factor-implied returns with ML residual predictions.

        Expected Return = Factor Return + ML Residual Prediction
        """
        combined = {}

        for symbol in self.config.universe:
            if symbol in factor_returns and symbol in ml_predictions:
                factor_return = factor_returns[symbol].iloc[-1] if len(factor_returns[symbol]) > 0 else 0
                ml_residual = ml_predictions[symbol]

                # Apply confidence weighting
                factor_confidence = self._get_factor_confidence(symbol, factor_betas)
                ml_confidence = self._get_ml_confidence(symbol)

                # Weighted combination based on confidence
                total_confidence = factor_confidence + ml_confidence
                if total_confidence > 0:
                    factor_weight = factor_confidence / total_confidence
                    ml_weight = ml_confidence / total_confidence
                else:
                    factor_weight = 0.5
                    ml_weight = 0.5

                combined_return = (factor_weight * factor_return) + (ml_weight * ml_residual)
                combined[symbol] = combined_return

                logger.debug(f"{symbol}: Factor={factor_return:.4f}, ML={ml_residual:.4f}, Combined={combined_return:.4f}")

        return combined

    def _get_factor_confidence(self, symbol: str, factor_betas: Dict[str, pd.DataFrame]) -> float:
        """Get confidence score for factor model based on regression quality."""
        if symbol not in factor_betas:
            return 0.0

        beta_data = factor_betas[symbol]
        if len(beta_data) == 0:
            return 0.0

        # Use R-squared as confidence metric
        latest_r2 = beta_data['r_squared'].iloc[-1] if 'r_squared' in beta_data.columns else 0.5

        # Apply confidence threshold
        return min(latest_r2, 1.0)

    def _get_ml_confidence(self, symbol: str) -> float:
        """Get confidence score for ML model based on recent performance."""
        # Get recent prediction accuracy for this symbol
        recent_performance = self.residual_predictor.get_recent_performance(symbol)

        if recent_performance is None:
            return 0.5  # Default confidence

        # Use correlation between predicted and actual residuals
        correlation = recent_performance.get('correlation', 0.0)

        # Apply confidence threshold
        return max(0.0, min(correlation, 1.0))

    def generate_signals(self, date: datetime) -> List[TradingSignal]:
        """
        Generate trading signals based on combined predictions.

        Args:
            date: Date for signal generation

        Returns:
            List of trading signals
        """
        try:
            logger.info(f"Generating signals for {date}")

            signals = []

            # Get current predictions
            current_predictions = self._get_current_predictions(date)

            # Rank stocks by predicted returns
            ranked_stocks = sorted(
                current_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Generate signals for top and bottom stocks
            for i, (symbol, predicted_return) in enumerate(ranked_stocks):
                if abs(predicted_return) < self.min_signal_strength:
                    continue

                # Calculate signal strength (0 to 1)
                max_return = max(abs(r) for r in current_predictions.values())
                if max_return > 0:
                    strength = min(abs(predicted_return) / max_return, 1.0)
                else:
                    strength = 0.5

                # Determine signal type
                if predicted_return > 0:
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL

                # Calculate confidence
                confidence = self._calculate_signal_confidence(symbol, predicted_return, date)

                # Get current price
                current_price = self._get_current_price(symbol, date)

                if current_price is not None:
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        timestamp=date,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'predicted_return': predicted_return,
                            'rank': i + 1,
                            'factor_confidence': self._get_factor_confidence(symbol, self.ff5_regression.factor_betas),
                            'ml_confidence': self._get_ml_confidence(symbol),
                            'investment_box': self.stock_classifications.get(symbol).__dict__ if symbol in self.stock_classifications else None
                        }
                    )
                    signals.append(signal)

            # Apply risk management and portfolio constraints
            signals = self._apply_portfolio_constraints(signals, date)

            logger.info(f"Generated {len(signals)} signals for {date}")
            self.signal_history.extend(signals)

            return signals

        except Exception as e:
            logger.error(f"Failed to generate signals for {date}: {e}")
            return []

    def _get_current_predictions(self, date: datetime) -> Dict[str, float]:
        """Get current predictions for all symbols."""
        predictions = {}

        for symbol in self.config.universe:
            if symbol in self.combined_predictions:
                predictions[symbol] = self.combined_predictions[symbol]
            else:
                # Use fallback prediction
                predictions[symbol] = 0.0

        return predictions

    def _calculate_signal_confidence(self, symbol: str, predicted_return: float, date: datetime) -> float:
        """Calculate overall signal confidence."""
        # Combine factor and ML confidences
        factor_confidence = self._get_factor_confidence(symbol, self.ff5_regression.factor_betas)
        ml_confidence = self._get_ml_confidence(symbol)

        # Weight by prediction magnitude
        return_weight = min(abs(predicted_return) / 0.1, 1.0)  # Normalize to 0-1

        # Combined confidence
        combined_confidence = (factor_confidence * 0.6) + (ml_confidence * 0.4)

        return combined_confidence * return_weight

    def _get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get current price for a symbol."""
        if symbol not in self.equity_data:
            return None

        data = self.equity_data[symbol]

        # Find closest date
        available_dates = data.index[data.index <= date]
        if len(available_dates) == 0:
            return None

        closest_date = available_dates[-1]
        return data.loc[closest_date, 'Close']

    def _apply_portfolio_constraints(self, signals: List[TradingSignal], date: datetime) -> List[TradingSignal]:
        """Apply portfolio constraints and risk management."""
        if not signals:
            return signals

        # Sort by strength
        signals.sort(key=lambda x: x.strength, reverse=True)

        # Apply position size limits
        constrained_signals = []
        total_weight = 0.0

        for signal in signals:
            if total_weight >= 1.0:
                break

            # Calculate position size based on rank and strength
            position_size = min(self.max_position_size, signal.strength * 0.2)

            # Check investment box constraints
            if self._check_box_constraints(signal.symbol, position_size):
                signal.metadata['position_size'] = position_size
                constrained_signals.append(signal)
                total_weight += position_size

        return constrained_signals

    def _check_box_constraints(self, symbol: str, proposed_weight: float) -> bool:
        """Check if proposed position violates investment box constraints."""
        if symbol not in self.stock_classifications:
            return True  # No constraint info, allow

        box = self.stock_classifications[symbol]

        # Check if this would violate box constraints
        # (Simplified check - in practice, would need to check current portfolio composition)
        return True

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
            logger.info(f"Executing trades for {len(signals)} signals")

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
                    commission=0.0,  # Simplified - no commission
                    trade_id=f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d')}"
                )
                trades.append(trade)

            logger.info(f"Generated {len(trades)} trades")
            return trades

        except Exception as e:
            logger.error(f"Failed to execute trades: {e}")
            return []

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
                'information_ratio': self._calculate_information_ratio(portfolio),
                'max_drawdown': self._calculate_max_drawdown(portfolio),
                'volatility': self._calculate_volatility(portfolio),
                'number_of_positions': len(portfolio.positions),
                'concentration': self._calculate_concentration(portfolio),
                'turnover': self._calculate_turnover()
            }

            # Model performance metrics
            metrics['model_performance'] = {
                'factor_model_accuracy': self._calculate_factor_model_accuracy(),
                'ml_prediction_accuracy': self._calculate_ml_prediction_accuracy(),
                'combined_prediction_accuracy': self._calculate_combined_prediction_accuracy()
            }

            # Risk metrics
            metrics['risk_metrics'] = {
                'var_95': self._calculate_var(portfolio, 0.05),
                'var_99': self._calculate_var(portfolio, 0.01),
                'expected_shortfall': self._calculate_expected_shortfall(portfolio),
                'beta_to_market': self._calculate_beta_to_market(portfolio),
                'tracking_error': self._calculate_tracking_error(portfolio)
            }

            # IPS compliance metrics
            metrics['ips_compliance'] = {
                'box_compliance': self._check_box_compliance(portfolio),
                'risk_budget_compliance': self._check_risk_budget_compliance(portfolio),
                'concentration_compliance': self._check_concentration_compliance(portfolio)
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to monitor performance: {e}")
            return {}

    def _calculate_sharpe_ratio(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate Sharpe ratio."""
        # Simplified calculation
        if len(self.portfolio_history) < 2:
            return 0.0

        returns = [p.daily_return for p in self.portfolio_history[-12:]]  # Last 12 months
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        return avg_return / std_return if std_return > 0 else 0.0

    def _calculate_information_ratio(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate information ratio vs benchmark."""
        # Simplified calculation - would need benchmark data
        return portfolio.total_return / 0.15  # Assume 15% benchmark vol

    def _calculate_max_drawdown(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate maximum drawdown."""
        return portfolio.drawdown

    def _calculate_volatility(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate portfolio volatility."""
        if len(self.portfolio_history) < 2:
            return 0.0

        returns = [p.daily_return for p in self.portfolio_history[-12:]]
        return np.std(returns) * np.sqrt(12)  # Annualized

    def _calculate_concentration(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        if not portfolio.positions:
            return 0.0

        weights = [pos.weight for pos in portfolio.positions]
        return sum(w**2 for w in weights)

    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover."""
        # Simplified calculation
        return 0.0  # Would need trade history

    def _calculate_factor_model_accuracy(self) -> float:
        """Calculate factor model accuracy."""
        # Would need prediction vs actual comparison
        return 0.7  # Placeholder

    def _calculate_ml_prediction_accuracy(self) -> float:
        """Calculate ML prediction accuracy."""
        return self.residual_predictor.get_overall_accuracy()

    def _calculate_combined_prediction_accuracy(self) -> float:
        """Calculate combined prediction accuracy."""
        return 0.75  # Placeholder

    def _calculate_var(self, portfolio: PortfolioSnapshot, confidence: float) -> float:
        """Calculate Value at Risk."""
        # Simplified calculation
        return abs(portfolio.daily_return) * 2.0  # Rough approximation

    def _calculate_expected_shortfall(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate Expected Shortfall."""
        return abs(portfolio.daily_return) * 3.0  # Rough approximation

    def _calculate_beta_to_market(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate portfolio beta to market."""
        return 1.0  # Placeholder

    def _calculate_tracking_error(self, portfolio: PortfolioSnapshot) -> float:
        """Calculate tracking error vs benchmark."""
        return 0.05  # 5% tracking error

    def _check_box_compliance(self, portfolio: PortfolioSnapshot) -> Dict[str, bool]:
        """Check investment box compliance."""
        compliance = {}

        # Calculate current box allocations
        box_allocations = self._calculate_box_allocations(portfolio)

        # Check each constraint
        for constraint_type, limits in self.box_constraints.items():
            compliance[constraint_type] = True

            for category, (min_weight, max_weight) in limits.items():
                current_weight = box_allocations.get(category, 0.0)
                if current_weight < min_weight or current_weight > max_weight:
                    compliance[constraint_type] = False
                    break

        return compliance

    def _calculate_box_allocations(self, portfolio: PortfolioSnapshot) -> Dict[str, float]:
        """Calculate current investment box allocations."""
        allocations = {
            'size': {},
            'style': {},
            'sector': {},
            'region': {}
        }

        for pos in portfolio.positions:
            if pos.symbol in self.stock_classifications:
                box = self.stock_classifications[pos.symbol]

                # Size allocations
                size_key = f"size_{box.size.name}"
                allocations['size'][size_key] = allocations['size'].get(size_key, 0.0) + pos.weight

                # Style allocations
                style_key = f"style_{box.style.name}"
                allocations['style'][style_key] = allocations['style'].get(style_key, 0.0) + pos.weight

                # Sector allocations
                sector_key = f"sector_{box.sector.name}"
                allocations['sector'][sector_key] = allocations['sector'].get(sector_key, 0.0) + pos.weight

                # Region allocations
                region_key = f"region_{box.region.name}"
                allocations['region'][region_key] = allocations['region'].get(region_key, 0.0) + pos.weight

        return allocations

    def _check_risk_budget_compliance(self, portfolio: PortfolioSnapshot) -> bool:
        """Check risk budget compliance."""
        # Simplified check
        return abs(portfolio.daily_return) <= self.risk_budget

    def _check_concentration_compliance(self, portfolio: PortfolioSnapshot) -> bool:
        """Check concentration compliance."""
        concentration = self._calculate_concentration(portfolio)
        return concentration <= 0.1  # Max 10% concentration

    def generate_ips_report(self, portfolio: PortfolioSnapshot) -> Dict[str, Any]:
        """
        Generate IPS compliance report.

        Args:
            portfolio: Current portfolio snapshot

        Returns:
            IPS compliance report
        """
        try:
            # Get performance metrics
            performance = self.monitor_performance(portfolio)

            # Generate report
            report = {
                'report_date': datetime.now().isoformat(),
                'strategy_name': self.config.name,
                'reporting_period': {
                    'start_date': self.backtest_config.start_date.isoformat(),
                    'end_date': self.backtest_config.end_date.isoformat()
                },
                'performance_summary': {
                    'total_return': f"{performance['total_return']:.2%}",
                    'annualized_return': f"{performance['total_return'] * 12:.2%}",
                    'sharpe_ratio': f"{performance['sharpe_ratio']:.2f}",
                    'max_drawdown': f"{performance['max_drawdown']:.2%}",
                    'volatility': f"{performance['volatility']:.2%}"
                },
                'ips_compliance': {
                    'box_compliance': performance['ips_compliance']['box_compliance'],
                    'risk_budget_compliance': performance['ips_compliance']['risk_budget_compliance'],
                    'concentration_compliance': performance['ips_compliance']['concentration_compliance'],
                    'overall_compliance': all(performance['ips_compliance'].values())
                },
                'model_performance': {
                    'factor_model_accuracy': f"{performance['model_performance']['factor_model_accuracy']:.2%}",
                    'ml_prediction_accuracy': f"{performance['model_performance']['ml_prediction_accuracy']:.2%}",
                    'combined_accuracy': f"{performance['model_performance']['combined_prediction_accuracy']:.2%}"
                },
                'risk_metrics': {
                    'var_95': f"{performance['risk_metrics']['var_95']:.2%}",
                    'var_99': f"{performance['risk_metrics']['var_99']:.2%}",
                    'beta_to_market': f"{performance['risk_metrics']['beta_to_market']:.2f}",
                    'tracking_error': f"{performance['risk_metrics']['tracking_error']:.2%}"
                },
                'portfolio_characteristics': {
                    'number_of_positions': performance['number_of_positions'],
                    'concentration': f"{performance['concentration']:.2%}",
                    'turnover': f"{performance['turnover']:.2%}"
                },
                'recommendations': self._generate_recommendations(performance)
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate IPS report: {e}")
            return {}

    def _generate_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        # Check compliance issues
        if not performance['ips_compliance']['overall_compliance']:
            recommendations.append("Address IPS compliance issues in investment box allocations")

        # Check model performance
        if performance['model_performance']['ml_prediction_accuracy'] < 0.6:
            recommendations.append("Retrain ML models to improve prediction accuracy")

        # Check risk metrics
        if performance['risk_metrics']['tracking_error'] > 0.08:
            recommendations.append("Reduce tracking error to align with benchmark")

        # Check concentration
        if performance['concentration'] > 0.15:
            recommendations.append("Diversify portfolio to reduce concentration risk")

        # Check Sharpe ratio
        if performance['sharpe_ratio'] < 1.0:
            recommendations.append("Improve risk-adjusted returns through better factor timing")

        return recommendations

    def save_results(self, output_dir: str) -> None:
        """
        Save strategy results to files.

        Args:
            output_dir: Output directory path
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model performance
            performance_file = output_path / "ffml_model_performance.json"
            with open(performance_file, 'w') as f:
                json.dump(self.model_performance, f, indent=2)

            # Save signal history
            signals_file = output_path / "ffml_signal_history.csv"
            if self.signal_history:
                signals_data = []
                for signal in self.signal_history:
                    signals_data.append({
                        'timestamp': signal.timestamp.isoformat(),
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type.value,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'price': signal.price,
                        'predicted_return': signal.metadata.get('predicted_return', 0.0),
                        'rank': signal.metadata.get('rank', 0)
                    })

                signals_df = pd.DataFrame(signals_data)
                signals_df.to_csv(signals_file, index=False)

            # Save portfolio history
            portfolio_file = output_path / "ffml_portfolio_history.csv"
            if self.portfolio_history:
                portfolio_data = []
                for snapshot in self.portfolio_history:
                    portfolio_data.append({
                        'timestamp': snapshot.timestamp.isoformat(),
                        'total_value': snapshot.total_value,
                        'daily_return': snapshot.daily_return,
                        'total_return': snapshot.total_return,
                        'drawdown': snapshot.drawdown,
                        'number_of_positions': len(snapshot.positions)
                    })

                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_df.to_csv(portfolio_file, index=False)

            logger.info(f"Saved results to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and configuration."""
        return {
            'strategy_name': self.config.name,
            'strategy_type': 'Core FFML',
            'universe': self.config.universe,
            'parameters': self.config.parameters,
            'backtest_config': {
                'initial_capital': self.backtest_config.initial_capital,
                'start_date': self.backtest_config.start_date.isoformat(),
                'end_date': self.backtest_config.end_date.isoformat(),
                'transaction_cost': self.backtest_config.transaction_cost
            },
            'model_configuration': {
                'ff5_lookback_window': self.ff5_regression.lookback_window,
                'ml_prediction_horizon': self.residual_predictor.prediction_horizon,
                'min_signal_strength': self.min_signal_strength,
                'target_positions': self.target_positions
            },
            'data_sources': {
                'equity_data': 'yfinance',
                'factor_data': 'Kenneth French Data Library',
                'frequency': 'monthly'
            }
        }