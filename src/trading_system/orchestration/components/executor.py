"""
Trade Executor - Signal to Trade Conversion and Execution

Converts trading signals into executable trades, applies execution logic,
manages trade timing, and tracks execution performance.

Extracted from SystemOrchestrator to follow Single Responsibility Principle.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ...types.signals import TradingSignal
from ...types.enums import SignalType
from ...types.portfolio import Portfolio, Position, Trade
from ...types.enums import AssetClass
from ..utils.performance_tracker import ComponentPerformanceTrackerMixin
from ..utils.config_validator import ComponentConfigValidator

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_EXECUTED = "partially_executed"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExecutionConfig: #TODO: read from config
    """Configuration for trade execution."""
    # Order preferences
    default_order_type: OrderType = OrderType.MARKET
    max_order_size_percent: float = 1.0  # Max 100% of portfolio per order (for portfolio optimization)
    min_order_size_usd: float = 1000  # Minimum order size $1,000

    # Execution timing
    execution_delay_seconds: int = 30  # Delay before execution
    max_execution_attempts: int = 3
    execution_timeout_seconds: int = 300  # 5 minutes timeout

    # Market impact protection
    max_daily_volume_participation: float = 0.20  # Max 20% of daily volume
    volatility_adjustment: bool = True
    spread_adjustment: bool = True

    # Risk controls
    max_positions_per_day: int = 10
    max_trades_per_symbol_per_day: int = 3
    cooling_period_hours: int = 1  # Hours between trades in same symbol

    # Slippage and cost estimates
    expected_slippage_bps: int = 5  # 5 basis points
    expected_spread_bps: int = 2   # 2 basis points
    commission_rate: float = 0.001  # 0.1%

    # Execution venues (simplified)
    primary_venue: str = "primary_market"
    backup_venues: List[str] = None

    def __post_init__(self):
        if self.backup_venues is None:
            self.backup_venues = ["dark_pool", "ecn"]


@dataclass
class Order:
    """Trade order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK

    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_time: datetime = None
    submitted_time: Optional[datetime] = None
    executed_time: Optional[datetime] = None
    executed_quantity: int = 0
    executed_price: Optional[float] = None

    # Metadata
    source_signal: Optional[TradingSignal] = None
    strategy_name: Optional[str] = None
    venue: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


class TradeExecutor(ComponentPerformanceTrackerMixin):
    """
    Executes trades based on trading signals.

    Responsibilities:
    - Convert signals to executable orders
    - Apply execution logic and risk controls
    - Manage order routing and execution
    - Track execution performance
    - Handle execution failures and retries
    """

    def __init__(self, config: ExecutionConfig):
        """
        Initialize trade executor.

        Args:
            config: Execution configuration
        """
        super().__init__()
        self.config = config

        # Execution state
        self.pending_orders: List[Order] = []
        self.completed_trades: List[Trade] = []
        self.execution_history: List[Dict[str, Any]] = []

        # Daily tracking
        self.daily_trades: Dict[str, List[datetime]] = {}
        self.symbol_last_trade: Dict[str, datetime] = {}

        # Performance tracking is now handled by ComponentPerformanceTrackerMixin

        # Risk tracking
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()

        logger.info("Initialized TradeExecutor")
        logger.info(f"Default order type: {config.default_order_type.value}")
        logger.info(f"Max order size: {config.max_order_size_percent:.1%}")

    def execute(self, signals: List[TradingSignal], portfolio: Portfolio,
                market_data: Optional[Dict[str, Any]] = None) -> List[Trade]:
        """
        Execute trades based on signals.

        Args:
            signals: List of trading signals
            portfolio: Current portfolio state
            market_data: Optional market data for execution

        Returns:
            List of executed trades
        """
        operation_id = self.track_operation("execute_trades", {"signal_count": len(signals)})
        
        try:
            logger.info(f"Executing {len(signals)} trading signals")

            # Reset daily counters if needed
            self._reset_daily_counters()

            # Step 1: Convert signals to orders
            orders = self._signals_to_orders(signals, portfolio)
            self.track_counter("orders_created", len(orders))

            # Step 2: Apply execution constraints
            filtered_orders = self._apply_execution_constraints(orders, portfolio)
            self.track_counter("orders_filtered", len(orders) - len(filtered_orders))

            # Step 3: Execute orders
            executed_trades = self._execute_orders(filtered_orders, market_data)
            self.track_counter("trades_executed", len(executed_trades))

            # Step 4: Update tracking
            self._update_execution_tracking(executed_trades)

            logger.info(f"Trade execution completed: {len(executed_trades)} trades executed")
            self.end_operation(operation_id, success=True, result={"trades_executed": len(executed_trades)})
            return executed_trades

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self.end_operation(operation_id, success=False, result={"error": str(e)})
            return []

    def _reset_daily_counters(self) -> None:
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_order_count = 0
            self.daily_trades.clear()
            self.last_reset_date = today
            logger.debug("Reset daily execution counters")

    def _signals_to_orders(self, signals: List[TradingSignal], portfolio: Portfolio) -> List[Order]:
        """Convert trading signals to executable orders."""
        orders = []

        for signal in signals:
            try:
                # Create order from signal
                order = self._create_order_from_signal(signal, portfolio)
                if order:
                    orders.append(order)
                    logger.debug(f"Created order {order.id} for {signal.symbol}")

            except Exception as e:
                logger.error(f"Failed to create order from signal {signal.symbol}: {e}")
                continue

        logger.info(f"Converted {len(signals)} signals to {len(orders)} orders")
        return orders

    def _create_order_from_signal(self, signal: TradingSignal, portfolio: Portfolio) -> Optional[Order]:
        """Create order from a single trading signal."""
        if signal.strength <= 0:
            return None

        # Calculate order quantity
        current_position = portfolio.positions.get(signal.symbol)
        current_quantity = current_position.quantity if current_position else 0

        # Calculate target position based on signal strength and portfolio value
        portfolio_value = portfolio.total_value
        target_value_usd = portfolio_value * signal.strength

        # Get price (simplified - would use market data)
        if current_position and current_position.current_price > 0:
            price = current_position.current_price
        else:
            # Would get from market data provider
            price = 100.0  # Simplified

        target_quantity = int(target_value_usd / price)
        order_quantity = target_quantity - current_quantity

        # Apply minimum order size
        order_value_usd = abs(order_quantity * price)
        if order_value_usd < self.config.min_order_size_usd:
            logger.debug(f"Order {signal.symbol} below minimum size (${order_value_usd:.2f})")
            return None

        # Apply maximum order size constraint
        max_order_value = portfolio_value * self.config.max_order_size_percent
        if order_value_usd > max_order_value:
            order_quantity = int(max_order_value / price) * (1 if order_quantity > 0 else -1)

        # Apply cooling period constraint
        if not self._check_cooling_period(signal.symbol):
            logger.debug(f"Order {signal.symbol} blocked by cooling period")
            return None

        # Determine order side based on final order_quantity (not signal type)
        # This ensures consistency between quantity and side after all constraints are applied
        side = OrderSide.BUY if order_quantity > 0 else OrderSide.SELL

        # Create order
        order = Order(
            id=f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=signal.symbol,
            side=side,
            order_type=self.config.default_order_type,
            quantity=abs(order_quantity),
            source_signal=signal,
            strategy_name=getattr(signal, 'strategy_name', 'unknown')
        )

        # Set limit price for non-market orders
        if order.order_type == OrderType.LIMIT:
            if side == OrderSide.BUY:
                order.price = price * 1.001  # 0.1% above market
            else:
                order.price = price * 0.999  # 0.1% below market

        return order

    def _check_cooling_period(self, symbol: str) -> bool:
        """Check if symbol is in cooling period."""
        if symbol not in self.symbol_last_trade:
            return True

        time_since_last_trade = datetime.now() - self.symbol_last_trade[symbol]
        cooling_period_hours = timedelta(hours=self.config.cooling_period_hours)

        return time_since_last_trade >= cooling_period_hours

    def _apply_execution_constraints(self, orders: List[Order], portfolio: Portfolio) -> List[Order]:
        """Apply execution constraints to filter orders."""
        filtered_orders = []

        # Check daily order limit
        if self.daily_order_count >= self.config.max_positions_per_day:
            logger.warning(f"Daily order limit ({self.config.max_positions_per_day}) reached")
            return []

        for order in orders:
            # Check trades per symbol per day limit
            symbol_trades_today = len(self.daily_trades.get(order.symbol, []))
            if symbol_trades_today >= self.config.max_trades_per_symbol_per_day:
                logger.debug(f"Symbol {order.symbol} daily trade limit reached")
                continue

            # Check portfolio has sufficient cash for buy orders
            if order.side == OrderSide.BUY:
                required_cash = order.quantity * order.price if order.price else order.quantity * 100  # Simplified
                available_cash = portfolio.cash_balance
                if required_cash > available_cash * 1.001:  # Allow 0.1% overdraft for commission costs
                    logger.debug(f"Insufficient cash for {order.symbol}: need ${required_cash:.2f}, have ${available_cash:.2f}")
                    continue

            # Check position exists for sell orders
            if order.side == OrderSide.SELL:
                current_position = portfolio.positions.get(order.symbol)
                if not current_position or current_position.quantity < order.quantity:
                    logger.debug(f"Insufficient position for {order.symbol}: want {order.quantity}, have {current_position.quantity if current_position else 0}")
                    continue

            filtered_orders.append(order)

        logger.info(f"Execution constraints: {len(orders)} -> {len(filtered_orders)} orders")
        return filtered_orders

    def _execute_orders(self, orders: List[Order], market_data: Optional[Dict[str, Any]] = None) -> List[Trade]:
        """Execute orders and return trades."""
        executed_trades = []

        for order in orders:
            try:
                trade = self._execute_single_order(order, market_data)
                if trade:
                    executed_trades.append(trade)
                    logger.debug(f"Executed trade: {trade.symbol} {trade.quantity} @ ${trade.price:.2f}")

            except Exception as e:
                logger.error(f"Failed to execute order {order.id}: {e}")
                order.status = ExecutionStatus.FAILED
                self.track_counter("failed_executions", 1)
                continue

        return executed_trades

    def _execute_single_order(self, order: Order, market_data: Optional[Dict[str, Any]] = None) -> Optional[Trade]:
        """Execute a single order."""
        # Update order status
        order.status = ExecutionStatus.SUBMITTED
        order.submitted_time = datetime.now()
        self.track_counter("orders_submitted", 1)
        self.daily_order_count += 1

        # Simulate execution delay
        import time
        time.sleep(0.1)  # Simulate processing time

        # Get execution price (simplified - would use market data)
        if market_data and order.symbol in market_data:
            execution_price = market_data[order.symbol].get('price', 100.0)
        else:
            execution_price = 100.0  # Simplified

        # Apply slippage
        slippage_bps = self.config.expected_slippage_bps
        if order.side == OrderSide.BUY:
            execution_price *= (1 + slippage_bps / 10000)
        else:
            execution_price *= (1 - slippage_bps / 10000)

        # Calculate commission
        trade_value = order.quantity * execution_price
        commission = trade_value * self.config.commission_rate

        # Create trade record
        trade = Trade(
            symbol=order.symbol,
            side=order.side.value.lower(),  # Convert OrderSide.BUY -> 'buy'
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.now(),
            commission=commission,
            trade_id=order.id
        )

        # Update order status
        order.status = ExecutionStatus.EXECUTED
        order.executed_time = datetime.now()
        order.executed_quantity = order.quantity
        order.executed_price = execution_price

        # Update tracking
        self._update_symbol_tracking(order.symbol)

        # Update statistics
        self.track_counter("successful_executions", 1)
        self.track_counter("total_commission_usd", int(commission * 100))  # Convert to cents for integer tracking
        self.track_counter("total_slippage_bps", slippage_bps)

        return trade

    def _update_symbol_tracking(self, symbol: str) -> None:
        """Update symbol tracking for cooling periods."""
        now = datetime.now()
        self.symbol_last_trade[symbol] = now

        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []
        self.daily_trades[symbol].append(now)

    def _update_execution_tracking(self, trades: List[Trade]) -> None:
        """Update execution tracking and history."""
        # Store completed trades
        self.completed_trades.extend(trades)

        # Update average execution time
        if self.execution_stats['total_orders'] > 0:
            total_time = sum([
                (trade.timestamp - trade.timestamp).total_seconds() * 1000
                for trade in trades
            ])
            self.execution_stats['average_execution_time_ms'] = total_time / len(trades)

        # Create execution record
        execution_record = {
            'timestamp': datetime.now(),
            'trades_executed': len(trades),
            'total_value': sum(trade.quantity * trade.price for trade in trades),
            'total_commission': sum(trade.commission for trade in trades),
            'symbols_traded': list(set(trade.symbol for trade in trades))
        }

        self.execution_history.append(execution_record)

        # Keep history manageable
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get execution performance metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.completed_trades if t.timestamp >= cutoff_date]

        if not recent_trades:
            return {
                'period_days': days,
                'total_trades': 0,
                'success_rate': 0,
                'average_slippage_bps': 0,
                'total_commission_usd': 0
            }

        total_trades = len(recent_trades)
        total_commission = sum(trade.commission for trade in recent_trades)
        total_value = sum(trade.quantity * trade.price for trade in recent_trades)

        # Calculate average slippage (simplified)
        avg_slippage_bps = self.config.expected_slippage_bps  # Would calculate from actual data

        # Get performance stats from the unified tracker
        performance_stats = self.get_performance_stats()
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'success_rate': performance_stats['stats']['successful_operations'] / max(1, performance_stats['stats']['total_operations']),
            'average_slippage_bps': avg_slippage_bps,
            'total_commission_usd': total_commission,
            'total_value_usd': total_value,
            'commission_rate_pct': (total_commission / total_value * 100) if total_value > 0 else 0,
            'stats': performance_stats,
            'recent_activity': self.execution_history[-10:]  # Last 10 execution records
        }

    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders."""
        return [order for order in self.pending_orders if order.status == ExecutionStatus.PENDING]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self.pending_orders:
            if order.id == order_id and order.status == ExecutionStatus.PENDING:
                order.status = ExecutionStatus.CANCELLED
                logger.info(f"Cancelled order {order_id}")
                return True
        return False

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate executor configuration."""
        config_dict = {
            'max_order_size_percent': self.config.max_order_size_percent,
            'min_order_size_usd': self.config.min_order_size_usd,
            'max_positions_per_day': self.config.max_positions_per_day,
            'commission_rate': self.config.commission_rate,
            'cooling_period_hours': self.config.cooling_period_hours
        }
        
        return ComponentConfigValidator.validate_executor_config(config_dict)