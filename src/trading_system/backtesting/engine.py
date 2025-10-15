"""
Unified Backtest Engine - Performance-Optimized Delegated Implementation

A balanced backtesting engine that delegates responsibilities to specialized components
while maintaining high performance through optimized batch processing and caching.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..config.backtest import BacktestConfig
from .types.results import BacktestResults
from ..utils.performance import PerformanceMetrics
from ..utils.risk import RiskCalculator
from .costs.transaction_costs import TransactionCostModel, TradeDirection
from .utils.validators import validate_inputs, align_data_periods, clean_price_data, filter_strategy_signals
from ..types import Position, Trade

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Unified backtest engine with academic-grade capabilities and performance optimization.

    Features:
    - Delegated architecture following SOLID principles
    - Performance-optimized batch processing and caching
    - Uses existing specialized components (DRY principle)
    - Clean, maintainable codebase balanced with performance needs
    - Backward compatibility with existing interfaces

    Architecture Balance:
    - Delegates complex calculations to specialized components (SOLID)
    - Maintains performance-critical optimizations (caching, batching)
    - Avoids code duplication through component reuse (DRY)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config

        # Initialize core components - DELEGATED to specialized classes
        self.performance_calculator = PerformanceMetrics()
        self.transaction_cost_model = TransactionCostModel(
            commission_rate=config.commission_rate,
            spread_rate=config.spread_rate,
            slippage_rate=config.slippage_rate,
            short_borrow_rate=config.short_borrow_rate
        )
        self.risk_calculator = RiskCalculator()

        # Portfolio state - keep minimal state
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.cash_balance = config.initial_capital
        self.positions: Dict[str, Position] = {}

        # Trading records - keep minimal records
        self.trades: List[Trade] = []
        self.portfolio_values: pd.Series = pd.Series(dtype=float)
        self.daily_returns: pd.Series = pd.Series(dtype=float)

        # Market data references
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None

        # Performance tracking
        self.peak_value: float = config.initial_capital
        self.total_costs: float = 0.0

        logger.info(f"Initialized BacktestEngine with ${config.initial_capital:,.0f} capital")

    def run_backtest(self,
                    strategy_signals: Dict[datetime, List[Any]],
                    price_data: Dict[str, pd.DataFrame],
                    benchmark_data: Optional[pd.DataFrame] = None,
                    **kwargs) -> BacktestResults:
        """
        Run a complete backtest.

        Args:
            strategy_signals: Dictionary mapping dates to trading signals
            price_data: Dictionary mapping symbols to price DataFrames
            benchmark_data: Optional benchmark price data
            **kwargs: Additional parameters (ignored for simplicity)

        Returns:
            Complete backtest results
        """
        try:
            logger.info("Starting backtest execution")

            # DELEGATE to existing validators - use established validation logic
            validate_inputs(strategy_signals, price_data, benchmark_data)

            # Filter strategy signals to only include symbols with available price data
            available_symbols = set(price_data.keys())
            filtered_signals = filter_strategy_signals(strategy_signals, available_symbols)

            self.price_data = clean_price_data(price_data)
            self.benchmark_data = benchmark_data
            self.strategy_signals = filtered_signals

            # Preload price data for performance optimization
            self._price_cache, self._sorted_dates_cache = self._preload_price_data()

            # DELEGATE to existing align_data_periods function
            aligned_signals = align_data_periods(strategy_signals, self.price_data)

            # Get trading calendar
            trading_days = self._get_trading_calendar(aligned_signals)

            # Initialize time series
            self._initialize_time_series(trading_days)

            # Execute backtest
            logger.info(f"Executing backtest over {len(trading_days)} trading days")

            for date in trading_days:
                # Execute trades for this date
                daily_signals = aligned_signals.get(date, [])
                self._execute_daily_trades(date, daily_signals)

                # Update portfolio value
                self._update_portfolio_value(date)

            # Calculate final results
            results = self._calculate_results()

            logger.info(f"Backtest completed: ${results.final_value:,.0f} "
                       f"({results.total_return:.2%} return, "
                       f"Sharpe: {results.sharpe_ratio:.2f})")

            return results

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise

    def _get_trading_calendar(self, strategy_signals: Dict[datetime, List[Any]]) -> List[datetime]:
        """Get sorted list of trading days."""
        signal_dates = set(strategy_signals.keys())

        # Add price dates to ensure full coverage
        if self.price_data:
            first_symbol_data = next(iter(self.price_data.values()))
            price_dates = set(first_symbol_data.index.to_pydatetime())
            trading_days = sorted((signal_dates | price_dates))
        else:
            trading_days = sorted(signal_dates)

        if not trading_days:
            raise ValueError("No trading days found")

        logger.info(f"Trading calendar: {len(trading_days)} days from "
                   f"{trading_days[0].date()} to {trading_days[-1].date()}")

        return trading_days

    def _initialize_time_series(self, trading_days: List[datetime]) -> None:
        """Initialize time series for portfolio tracking."""
        self.portfolio_values = pd.Series(
            index=trading_days,
            data=self.initial_capital,
            dtype=float
        )

        self.daily_returns = pd.Series(
            index=trading_days,
            data=0.0,
            dtype=float
        )

    def _execute_daily_trades(self, date: datetime, signals: List[Any]) -> None:
        """Execute all trades for a given date using batch processing for performance."""
        if not signals:
            return

        # Batch process signals for improved performance while maintaining clean architecture
        try:
            # Extract all signal data at once
            signal_data = self._extract_signal_data_batch(signals)

            # Batch price lookups using cached data
            symbols = list(signal_data.keys())
            price_data = self._get_batch_prices(symbols, date)

            # Filter signals with valid prices
            valid_signals = {
                symbol: data for symbol, data in signal_data.items()
                if symbol in price_data and price_data[symbol] is not None
            }

            if not valid_signals:
                logger.debug(f"No valid price data for signals on {date}")
                return

            # Batch calculate target positions and risk constraints
            trade_candidates = self._calculate_trade_candidates_batch(valid_signals, price_data, date)

            # Batch transaction cost calculation - DELEGATED to TransactionCostModel
            trades_with_costs = self._calculate_transaction_costs_batch(trade_candidates)

            # Execute valid trades
            executed_trades = self._execute_trades_batch(trades_with_costs, date)

            # Update positions and cash in batch
            self._update_positions_batch(executed_trades)

            # Record trades
            self.trades.extend(executed_trades)

            logger.debug(f"Executed {len(executed_trades)} trades on {date}")

        except Exception as e:
            logger.warning(f"Batch trade execution failed on {date}: {e}")

    def _execute_trade(self, date: datetime, signal: Any) -> Optional[Trade]:
        """Execute a single trade."""
        # Extract signal information
        symbol = self._extract_signal_symbol(signal)
        signal_strength = self._extract_signal_strength(signal)
        signal_type = self._extract_signal_type(signal)

        if not symbol or signal_strength == 0:
            return None

        # Get current price
        current_price = self._get_current_price(symbol, date)
        if current_price is None:
            logger.warning(f"No price data for {symbol} on {date}")
            return None

        # Calculate target position
        target_weight = signal_strength * self.config.position_limit
        target_value = self.current_capital * target_weight

        # Get current position
        current_position = self.positions.get(symbol, self._create_empty_position(symbol))
        current_value = current_position.market_value
        value_change = target_value - current_value

        # Check rebalance threshold
        if abs(value_change) < self.current_capital * self.config.rebalance_threshold:
            return None

        # Calculate trade quantity
        trade_quantity = value_change / current_price
        trade_value = abs(trade_quantity * current_price)

        # Apply risk checks
        if not self._check_risk_constraints(symbol, target_weight, current_price):
            return None

        # Calculate transaction costs
        direction = self._determine_trade_direction(trade_quantity)
        cost_breakdown = self.transaction_cost_model.calculate_cost_breakdown(
            trade_value=trade_value,
            direction=direction
        )

        trade_cost = cost_breakdown['total']

        # Check cash availability for buys
        if trade_quantity > 0 and value_change + trade_cost > self.cash_balance:
            logger.warning(f"Insufficient cash for {symbol} trade")
            return None

        # Execute trade
        trade = self._create_trade(
            symbol=symbol,
            side=direction.value,
            quantity=abs(trade_quantity),
            price=current_price,
            timestamp=date,
            commission=cost_breakdown['commission'],
            trade_id=f"{symbol}_{date.strftime('%Y%m%d_%H%M%S')}"
        )

        # Update position
        self._update_position(trade)

        # Update cash
        if trade_quantity > 0:  # Buy
            self.cash_balance -= value_change + trade_cost
        else:  # Sell
            self.cash_balance += value_change - trade_cost

        self.total_costs += trade_cost

        logger.debug(f"Executed {direction.value} {abs(trade_quantity):.1f} {symbol} "
                    f"@ ${current_price:.2f}, cost: ${trade_cost:.2f}")

        return trade

    def _extract_signal_symbol(self, signal: Any) -> str:
        """Extract symbol from signal."""
        if hasattr(signal, 'symbol'):
            return signal.symbol
        elif isinstance(signal, dict) and 'symbol' in signal:
            return signal['symbol']
        return ""

    def _extract_signal_strength(self, signal: Any) -> float:
        """Extract signal strength (-1 to 1)."""
        if hasattr(signal, 'strength'):
            return signal.strength
        elif isinstance(signal, dict):
            return signal.get('strength', 0)
        return 0

    def _extract_signal_type(self, signal: Any) -> str:
        """Extract signal type."""
        if hasattr(signal, 'signal_type'):
            return signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
        elif isinstance(signal, dict):
            return signal.get('signal_type', 'unknown')
        return 'unknown'

    # RESTORED: Performance-optimized batch processing methods
# Maintaining clean architecture while ensuring good performance

    def _preload_price_data(self) -> Tuple[Dict[str, Dict[datetime, float]], Dict[str, List[datetime]]]:
        """Preload price data into optimized cache structure for fast lookups.

        Returns:
            Tuple of (price_cache, sorted_dates_cache):
            - price_cache: {symbol: {date: price}}
            - sorted_dates_cache: {symbol: [sorted_dates_desc]}
        """
        price_cache = {}
        sorted_dates_cache = {}

        if not self.price_data:
            return price_cache, sorted_dates_cache

        logger.info("Preloading price data for performance optimization")

        for symbol, price_df in self.price_data.items():
            # Convert DataFrame to dict for O(1) lookups
            price_cache[symbol] = {}

            # Get and sort dates once during preloading
            sorted_dates = sorted(price_df.index.tolist())

            for date in sorted_dates:
                # Convert pandas Timestamp to datetime for consistency
                date_key = date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date
                price_cache[symbol][date_key] = float(price_df.loc[date, 'Close'])

            # Store pre-sorted dates (descending for efficient search)
            sorted_dates_cache[symbol] = [
                date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date
                for date in reversed(sorted_dates)
            ]

        total_prices = sum(len(symbol_cache) for symbol_cache in price_cache.values())
        logger.info(f"Preloaded {total_prices} price points for {len(price_cache)} symbols")

        return price_cache, sorted_dates_cache

    def _get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get current price for a symbol using cached data for O(1) lookup."""
        if symbol not in self._price_cache:
            return None

        symbol_cache = self._price_cache[symbol]
        sorted_dates = self._sorted_dates_cache.get(symbol, [])

        # For small datasets, direct lookup is faster than binary search
        # Try exact match first
        if date in symbol_cache:
            return symbol_cache[date]

        # Find the latest price on or before the requested date
        # Use pre-sorted dates for efficient search
        for d in sorted_dates:
            if d <= date:
                return symbol_cache[d]

        return None

    # Batch processing methods for performance optimization
    def _extract_signal_data_batch(self, signals: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Extract signal data for all signals in batch."""
        signal_data = {}

        for signal in signals:
            symbol = self._extract_signal_symbol(signal)
            strength = self._extract_signal_strength(signal)
            signal_type = self._extract_signal_type(signal)

            if symbol and strength != 0:
                signal_data[symbol] = {
                    'strength': strength,
                    'signal_type': signal_type,
                    'original_signal': signal
                }

        return signal_data

    def _get_batch_prices(self, symbols: List[str], date: datetime) -> Dict[str, Optional[float]]:
        """Get current prices for multiple symbols in batch."""
        price_data = {}

        for symbol in symbols:
            price_data[symbol] = self._get_current_price(symbol, date)

        return price_data

    def _calculate_trade_candidates_batch(self, valid_signals: Dict[str, Dict[str, Any]],
                                        price_data: Dict[str, float], date: datetime) -> List[Dict[str, Any]]:
        """Calculate trade candidates for all valid signals in batch."""
        trade_candidates = []

        for symbol, signal_info in valid_signals.items():
            current_price = price_data[symbol]
            signal_strength = signal_info['strength']

            # Calculate target position
            target_weight = signal_strength * self.config.position_limit
            target_value = self.current_capital * target_weight

            # Get current position
            current_position = self.positions.get(symbol, self._create_empty_position(symbol))
            current_value = current_position.market_value
            value_change = target_value - current_value

            # Check rebalance threshold
            if abs(value_change) < self.current_capital * self.config.rebalance_threshold:
                continue

            # Calculate trade quantity
            trade_quantity = value_change / current_price
            trade_value = abs(trade_quantity * current_price)

            # Apply risk checks
            if not self._check_risk_constraints(symbol, target_weight, current_price):
                continue

            # Check cash availability for buys
            direction = self._determine_trade_direction(trade_quantity)
            if trade_quantity > 0 and value_change > self.cash_balance:
                logger.debug(f"Insufficient cash for {symbol} trade")
                continue

            trade_candidates.append({
                'symbol': symbol,
                'side': direction.value,
                'quantity': abs(trade_quantity),
                'price': current_price,
                'value_change': value_change,
                'trade_value': trade_value,
                'direction': direction,
                'timestamp': date,
                'signal_info': signal_info
            })

        return trade_candidates

    def _calculate_transaction_costs_batch(self, trade_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate transaction costs for all trade candidates in batch - DELEGATED to TransactionCostModel."""
        trades_with_costs = []

        for candidate in trade_candidates:
            trade_value = candidate['trade_value']
            direction = candidate['direction']

            # DELEGATE to TransactionCostModel - use existing comprehensive calculation
            cost_breakdown = self.transaction_cost_model.calculate_cost_breakdown(
                trade_value=trade_value,
                direction=direction
            )

            trade_cost = cost_breakdown['total']

            # Update candidate with cost information
            candidate_with_cost = candidate.copy()
            candidate_with_cost.update({
                'cost_breakdown': cost_breakdown,
                'trade_cost': trade_cost,
                'commission': cost_breakdown['commission']
            })

            # Final cash availability check
            value_change = candidate['value_change']
            if direction == TradeDirection.BUY and value_change + trade_cost > self.cash_balance:
                logger.debug(f"Insufficient cash for {candidate['symbol']} trade after costs")
                continue

            trades_with_costs.append(candidate_with_cost)

        return trades_with_costs

    def _execute_trades_batch(self, trades_with_costs: List[Dict[str, Any]], date: datetime) -> List[Trade]:
        """Execute all valid trades in batch."""
        executed_trades = []

        for trade_data in trades_with_costs:
            # Create trade object
            trade = self._create_trade(
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                timestamp=trade_data['timestamp'],
                commission=trade_data['commission'],
                trade_id=f"{trade_data['symbol']}_{date.strftime('%Y%m%d_%H%M%S')}"
            )

            executed_trades.append(trade)

        return executed_trades

    def _update_positions_batch(self, executed_trades: List[Trade]) -> None:
        """Update positions and cash for all executed trades in batch."""
        total_cash_change = 0
        total_costs = 0

        for trade in executed_trades:
            # Calculate sign based on trade side
            is_buy = trade.side.upper() == 'BUY'
            quantity = trade.quantity if is_buy else -trade.quantity
            value_change = quantity * trade.price
            trade_cost = trade.commission

            # Update position
            self._update_position(trade)

            # Track cash changes
            if is_buy:
                total_cash_change -= (value_change + trade_cost)
            else:
                total_cash_change += (value_change - trade_cost)

            total_costs += trade_cost

        # Apply cash changes once
        self.cash_balance += total_cash_change
        self.total_costs += total_costs

        logger.debug(f"Batch update: {len(executed_trades)} trades, "
                    f"cash change: ${total_cash_change:.2f}, costs: ${total_costs:.2f}")

    def _determine_trade_direction(self, quantity: float) -> TradeDirection:
        """Determine trade direction from quantity."""
        if quantity > 0:
            return TradeDirection.BUY
        elif quantity < 0:
            return TradeDirection.SELL
        else:
            return TradeDirection.BUY  # Default

    def _check_risk_constraints(self, symbol: str, target_weight: float, price: float) -> bool:
        """Check if trade passes risk constraints."""
        # Position size limit
        if abs(target_weight) > self.config.position_limit:
            logger.debug(f"Position size limit exceeded for {symbol}: {target_weight:.2%}")
            return False

        # Drawdown limit (simplified check)
        if self.current_capital < self.initial_capital * (1 - self.config.max_drawdown_limit):
            logger.debug(f"Max drawdown limit reached: {(self.current_capital/self.initial_capital - 1):.2%}")
            return False

        return True

    def _create_empty_position(self, symbol: str) -> Position:
        """Create an empty position with default values."""
        return Position(
            symbol=symbol,
            quantity=0.0,
            average_cost=0.0,
            current_price=0.0,
            market_value=0.0,
            unrealized_pnl=0.0,
            weight=0.0
        )

    def _create_trade(self, symbol: str, side: str, quantity: float,
                     price: float, timestamp: datetime, commission: float = 0.0,
                     trade_id: Optional[str] = None) -> Trade:
        """Create a trade record."""
        return Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            trade_id=trade_id
        )

    def _update_position(self, trade: Trade) -> None:
        """Update position after trade execution."""
        symbol = trade.symbol

        if symbol not in self.positions:
            self.positions[symbol] = self._create_empty_position(symbol)

        position = self.positions[symbol]
        trade_value = trade.quantity * trade.price

        if trade.side == 'buy':
            # Update average cost
            total_cost = position.quantity * position.average_cost + trade_value
            total_quantity = position.quantity + trade.quantity
            position.average_cost = total_cost / total_quantity if total_quantity > 0 else trade.price
            position.quantity += trade.quantity
        else:
            # Reduce position
            position.quantity -= trade.quantity
            if position.quantity < 0:
                position.quantity = 0
                position.average_cost = 0

        position.current_price = trade.price
        position.market_value = position.quantity * position.current_price
        position.unrealized_pnl = position.market_value - (position.quantity * position.average_cost)

    def _update_portfolio_value(self, date: datetime) -> None:
        """Update portfolio value with current market prices."""
        # Calculate total position value
        total_position_value = 0

        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = self._get_current_price(symbol, date)
                if current_price:
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - (position.quantity * position.average_cost)

                total_position_value += position.market_value

        # Update portfolio value
        previous_capital = self.current_capital
        self.current_capital = self.cash_balance + total_position_value

        # Update time series
        if date in self.portfolio_values.index:
            self.portfolio_values.loc[date] = self.current_capital

            # Calculate daily return
            if previous_capital > 0:
                daily_return = (self.current_capital - previous_capital) / previous_capital
                self.daily_returns.loc[date] = daily_return

        # Update peak value
        if self.current_capital > self.peak_value:
            self.peak_value = self.current_capital

    def _calculate_results(self) -> BacktestResults:
        """Calculate final backtest results using delegated PerformanceMetrics."""
        try:
            # Calculate performance metrics using existing PerformanceMetrics
            returns_clean = self.daily_returns.replace(0, np.nan).dropna()

            if len(returns_clean) < 2:
                logger.warning("Insufficient data for performance calculation")
                return self._empty_results()

            # Calculate benchmark returns if available
            benchmark_returns = None
            if self.benchmark_data is not None:
                benchmark_prices = self.benchmark_data['Close']
                benchmark_returns = benchmark_prices.pct_change().dropna()
                # Align with portfolio dates
                common_dates = self.daily_returns.index.intersection(benchmark_returns.index)
                benchmark_returns = benchmark_returns.loc[common_dates]

            # DELEGATE to PerformanceMetrics - use existing comprehensive calculation
            performance_metrics = self.performance_calculator.calculate_all_metrics(
                returns=returns_clean,
                benchmark_returns=benchmark_returns,
                risk_free_rate=self.config.risk_free_rate,
                periods_per_year=252
            )

            # DELEGATE risk calculations to RiskCalculator
            risk_metrics = self._calculate_risk_metrics(returns_clean, benchmark_returns)

            # Calculate turnover rate
            turnover_rate = self._calculate_turnover_rate()

            # Create results using existing BacktestResults class
            results = BacktestResults(
                portfolio_values=self.portfolio_values,
                daily_returns=self.daily_returns,
                benchmark_returns=benchmark_returns,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                trades=self.trades,
                transaction_costs=self.total_costs,
                start_date=self.portfolio_values.index[0] if len(self.portfolio_values) > 0 else None,
                end_date=self.portfolio_values.index[-1] if len(self.portfolio_values) > 0 else None,
                initial_capital=self.initial_capital,
                final_value=self.current_capital,
                turnover_rate=turnover_rate
            )

            return results

        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return self._empty_results()

    def _calculate_risk_metrics(self, portfolio_returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all risk metrics using the unified RiskCalculator.

        Args:
            portfolio_returns: Series of portfolio daily returns.
            benchmark_returns: Series of benchmark daily returns.

        Returns:
            Dictionary of comprehensive risk metrics.
        """
        # Note: This is a simplified risk calculation.
        # A more advanced version would use the full returns matrix of all assets.
        # For now, we calculate portfolio-level risk based on the final return series.

        risk_metrics = {
            'value_at_risk_95': self.risk_calculator.value_at_risk(portfolio_returns, 0.95),
            'expected_shortfall_95': self.risk_calculator.expected_shortfall(portfolio_returns, 0.95),
            'downside_deviation': self.risk_calculator.downside_deviation(portfolio_returns)
        }

        if benchmark_returns is not None:
            risk_metrics['beta_to_market'] = self.risk_calculator.portfolio_beta(
                portfolio_returns, benchmark_returns
            )
            risk_metrics['tracking_error'] = self.risk_calculator.tracking_error(
                portfolio_returns, benchmark_returns
            )

        # Drawdown risk is part of performance metrics but we can add it here if needed
        drawdown_info = self.risk_calculator.drawdown_risk(portfolio_returns)
        risk_metrics.update(drawdown_info)

        return risk_metrics

    def _calculate_turnover_rate(self) -> float:
        """Calculate annual turnover rate."""
        if len(self.trades) == 0:
            return 0.0

        total_trade_value = sum(trade.quantity * trade.price for trade in self.trades)
        avg_portfolio_value = self.portfolio_values.mean()
        if avg_portfolio_value == 0:
            return 0.0

        days_elapsed = len(self.daily_returns)
        if days_elapsed == 0:
            return 0.0

        annualized_turnover = (total_trade_value / avg_portfolio_value) * (252 / days_elapsed)
        return annualized_turnover

    def _empty_results(self) -> BacktestResults:
        """Return empty results for failed calculations - create minimal valid results."""
        # Create a minimal valid portfolio series with initial capital
        # This ensures BacktestResults validation passes
        today = pd.Timestamp.now().normalize()
        portfolio_values = pd.Series([self.initial_capital], index=[today])
        daily_returns = pd.Series([0.0], index=[today])

        # Create basic performance metrics for no trades scenario
        performance_metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0
        }

        return BacktestResults(
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            performance_metrics=performance_metrics,
            trades=[],
            transaction_costs=0.0,
            initial_capital=self.initial_capital,
            final_value=self.current_capital
        )

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        active_positions = {k: v for k, v in self.positions.items() if v.quantity > 0}

        return {
            'total_value': self.current_capital,
            'cash_balance': self.cash_balance,
            'positions_value': self.current_capital - self.cash_balance,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'active_positions': len(active_positions),
            'total_trades': len(self.trades),
            'total_costs': self.total_costs,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'weight': pos.market_value / self.current_capital if self.current_capital > 0 else 0
                }
                for symbol, pos in active_positions.items()
            }
        }