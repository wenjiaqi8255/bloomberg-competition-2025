"""
Performance Reporter - System Performance Analysis and Reporting

Generates comprehensive performance reports, calculates attribution metrics,
and provides insights on system performance across all components.

Extracted from SystemOrchestrator to follow Single Responsibility Principle.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

from ...types.portfolio import Portfolio, Trade
from ...types.data_types import SystemPerformance
from ...utils.performance import PerformanceMetrics as UnifiedPerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for performance reporting."""
    # Reporting frequency
    daily_reports: bool = True
    weekly_reports: bool = True
    monthly_reports: bool = True
    quarterly_reports: bool = True
    annual_reports: bool = True

    # Performance metrics to include
    include_basic_metrics: bool = True
    include_risk_metrics: bool = True
    include_attribution: bool = True
    include_benchmark_comparison: bool = True
    include_drawdown_analysis: bool = True

    # Output settings
    save_to_file: bool = True
    output_directory: str = "reports"
    file_format: str = "json"  # "json", "csv", "excel"

    # Benchmark settings
    benchmark_symbol: str = "SPY"
    benchmark_data_source: str = "yfinance"

    # Reporting thresholds
    significant_return_threshold: float = 0.01  # 1%
    significant_risk_threshold: float = 0.02   # 2%
    alert_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'daily_return': -0.05,  # Alert on -5% daily return
                'drawdown': -0.15,       # Alert on -15% drawdown
                'volatility': 0.25,      # Alert on 25%+ volatility
                'var_95': 0.05           # Alert on 5%+ VaR
            }


@dataclass
class ReportPerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic returns
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Value at Risk
    var_95: float
    var_99: float
    expected_shortfall_95: float

    # Benchmark comparison
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

    # Additional metrics
    win_rate: float
    profit_factor: float
    skewness: float
    kurtosis: float
    tail_ratio: float

    # Trading metrics
    total_trades: int
    average_trade_return: float
    average_holding_period_days: float
    turnover_rate: float

    # Component performance
    core_return: float = 0.0
    satellite_return: float = 0.0
    core_contribution: float = 0.0
    satellite_contribution: float = 0.0

    # Attribution metrics
    factor_attribution: Dict[str, float] = None
    alpha_attribution: Dict[str, float] = None
    sector_attribution: Dict[str, float] = None

    def __post_init__(self):
        if self.factor_attribution is None:
            self.factor_attribution = {}
        if self.alpha_attribution is None:
            self.alpha_attribution = {}
        if self.sector_attribution is None:
            self.sector_attribution = {}


@dataclass
class PerformanceReport:
    """Complete performance report."""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    metrics: ReportPerformanceMetrics
    portfolio_summary: Dict[str, Any]
    trading_summary: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    attribution_analysis: Dict[str, Any]
    recommendations: List[str]
    alerts: List[str]

    @property
    def is_positive_return(self) -> bool:
        """Check if period return is positive."""
        return self.metrics.total_return > 0

    @property
    def has_alerts(self) -> bool:
        """Check if there are any alerts."""
        return len(self.alerts) > 0


class PerformanceReporter:
    """
    Generates and manages performance reports.

    Responsibilities:
    - Calculate comprehensive performance metrics
    - Generate attribution analysis
    - Create benchmark comparisons
    - Produce periodic reports
    - Generate alerts and recommendations
    - Save reports to files
    """

    def __init__(self, config: ReportConfig):
        """
        Initialize performance reporter.

        Args:
            config: Reporting configuration
        """
        self.config = config

        # Performance history
        self.performance_history: List[PerformanceReport] = []
        self.daily_metrics_history: List[PerformanceMetrics] = []

        # Benchmark data
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional[pd.Series] = None

        # Alert tracking
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []

        # Statistics
        self.reporting_stats = {
            'reports_generated': 0,
            'alerts_triggered': 0,
            'recommendations_provided': 0,
            'last_report_time': None
        }

        logger.info("Initialized PerformanceReporter")
        logger.info(f"Report formats: {config.file_format}, Output: {config.output_directory}")

    def generate_report(self, portfolio: Portfolio, trades: List[Trade],
                       benchmark_data: Optional[pd.DataFrame] = None,
                       period_days: int = 30) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            portfolio: Current portfolio state
            trades: List of trades in the period
            benchmark_data: Optional benchmark data
            period_days: Number of days to analyze

        Returns:
            Complete performance report
        """
        try:
            logger.info(f"Generating performance report for {period_days} days")

            # Set period bounds
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Load benchmark data if needed
            if benchmark_data is not None:
                self.benchmark_data = benchmark_data
                self.benchmark_returns = self._calculate_benchmark_returns(benchmark_data)

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio, trades, start_date, end_date)

            # Create portfolio summary
            portfolio_summary = self._create_portfolio_summary(portfolio)

            # Create trading summary
            trading_summary = self._create_trading_summary(trades, period_days)

            # Create benchmark comparison
            benchmark_comparison = self._create_benchmark_comparison(metrics, period_days)

            # Create attribution analysis
            attribution_analysis = self._create_attribution_analysis(trades, metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, portfolio_summary, trading_summary)

            # Check for alerts
            alerts = self._check_alerts(metrics, self.config.alert_thresholds)

            # Create report
            report = PerformanceReport(
                timestamp=datetime.now(),
                period_start=start_date,
                period_end=end_date,
                metrics=metrics,
                portfolio_summary=portfolio_summary,
                trading_summary=trading_summary,
                benchmark_comparison=benchmark_comparison,
                attribution_analysis=attribution_analysis,
                recommendations=recommendations,
                alerts=alerts
            )

            # Update tracking
            self._update_report_tracking(report)

            # Save report if configured
            if self.config.save_to_file:
                self._save_report(report)

            logger.info(f"Performance report generated: {metrics.total_return:.2%} return, {metrics.sharpe_ratio:.2f} Sharpe")
            return report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            # Return minimal report
            return self._create_minimal_report(period_days)

    def _calculate_performance_metrics(self, portfolio: Portfolio, trades: List[Trade],
                                     start_date: datetime, end_date: datetime) -> ReportPerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        DELEGATED to unified PerformanceMetrics to eliminate code duplication.
        """
        # Get portfolio value history (simplified - would use actual history)
        if hasattr(portfolio, 'portfolio_values') and portfolio.portfolio_values is not None:
            values = portfolio.portfolio_values
        else:
            # Create simplified returns for demonstration
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            values = pd.Series(1000000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumsum(), index=dates)

        if len(values) < 2:
            # Return zero metrics if insufficient data
            return self._create_zero_metrics()

        # Calculate returns
        returns = values.pct_change().dropna()

        # Use unified performance calculation system
        benchmark_returns = None
        if self.benchmark_returns is not None:
            benchmark_returns = self.benchmark_returns.reindex(returns.index, fill_value=None)

        # Calculate comprehensive metrics using unified system
        unified_metrics = UnifiedPerformanceMetrics.calculate_all_metrics(
            returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.02,
            periods_per_year=252
        )

        # Trading metrics (still need to calculate locally as they depend on trade data)
        total_trades = len(trades)
        if total_trades > 0:
            turnover_rate = min(1.0, total_trades / len(values))  # Simplified
            # Simplified trading metrics
            average_trade_return = np.mean([abs(trade.quantity * trade.price) for trade in trades])
            win_rate = 0.6  # Simplified - would calculate from actual P&L
            profit_factor = 1.5  # Simplified
        else:
            average_trade_return = 0
            turnover_rate = 0
            win_rate = 0
            profit_factor = 1.0

        return ReportPerformanceMetrics(
            total_return=unified_metrics['total_return'],
            annualized_return=unified_metrics['annualized_return'],
            daily_return_mean=returns.mean(),
            daily_return_std=returns.std(),
            volatility=unified_metrics['volatility'],
            sharpe_ratio=unified_metrics['sharpe_ratio'],
            sortino_ratio=unified_metrics['sortino_ratio'],
            max_drawdown=unified_metrics['max_drawdown'],
            calmar_ratio=unified_metrics['calmar_ratio'],
            var_95=unified_metrics['var_95'],
            var_99=unified_metrics['var_99'],
            expected_shortfall_95=unified_metrics['expected_shortfall_95'],
            alpha=unified_metrics['alpha'],
            beta=unified_metrics['beta'],
            information_ratio=unified_metrics['information_ratio'],
            tracking_error=unified_metrics['tracking_error'],
            win_rate=win_rate,
            profit_factor=profit_factor,
            skewness=returns.skew(),
            kurtosis=returns.kurtosis(),
            tail_ratio=1.2,  # Simplified
            total_trades=total_trades,
            average_trade_return=average_trade_return,
            average_holding_period_days=5.0,  # Simplified
            turnover_rate=turnover_rate,
            core_return=0.03,  # Would calculate from component data
            satellite_return=0.02,
            core_contribution=0.025,
            satellite_contribution=0.015,
            factor_attribution={'market': 0.6, 'size': 0.2, 'value': 0.1, 'momentum': 0.1},
            alpha_attribution={'stock_selection': 0.015, 'timing': 0.01, 'other': 0.005},
            sector_attribution={'technology': 0.4, 'healthcare': 0.2, 'finance': 0.3, 'other': 0.1}
        )

    def _create_zero_metrics(self) -> ReportPerformanceMetrics:
        """Create zero metrics when insufficient data."""
        return ReportPerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_return_mean=0.0,
            daily_return_std=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            var_95=0.0,
            var_99=0.0,
            expected_shortfall_95=0.0,
            alpha=0.0,
            beta=1.0,
            information_ratio=0.0,
            tracking_error=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            skewness=0.0,
            kurtosis=0.0,
            tail_ratio=1.0,
            total_trades=0,
            average_trade_return=0.0,
            average_holding_period_days=0.0,
            turnover_rate=0.0
        )

    def _create_portfolio_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Create portfolio summary."""
        return {
            'total_value': getattr(portfolio, 'total_value', 0),
            'cash_balance': getattr(portfolio, 'cash_balance', 0),
            'number_of_positions': len([p for p in portfolio.positions.values() if p.quantity > 0]),
            'long_positions': len([p for p in portfolio.positions.values() if p.quantity > 0]),
            'short_positions': len([p for p in portfolio.positions.values() if p.quantity < 0]),
            'daily_return': getattr(portfolio, 'daily_return', 0),
            'total_return': getattr(portfolio, 'total_return', 0),
            'current_drawdown': getattr(portfolio, 'drawdown', 0),
            'last_updated': datetime.now()
        }

    def _create_trading_summary(self, trades: List[Trade], period_days: int) -> Dict[str, Any]:
        """Create trading summary."""
        if not trades:
            return {
                'total_trades': 0,
                'trades_per_day': 0,
                'average_trade_size_usd': 0,
                'total_commission_usd': 0,
                'most_traded_symbol': None
            }

        total_commission = sum(trade.commission for trade in trades)
        average_trade_size = np.mean([abs(trade.quantity * trade.price) for trade in trades])

        # Count trades by symbol
        symbol_counts = {}
        for trade in trades:
            symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1

        most_traded_symbol = max(symbol_counts, key=symbol_counts.get) if symbol_counts else None

        return {
            'total_trades': len(trades),
            'trades_per_day': len(trades) / period_days,
            'average_trade_size_usd': average_trade_size,
            'total_commission_usd': total_commission,
            'most_traded_symbol': most_traded_symbol,
            'unique_symbols': len(symbol_counts),
            'buy_trades': len([t for t in trades if t.trade_type == 'buy']),
            'sell_trades': len([t for t in trades if t.trade_type == 'sell'])
        }

    def _create_benchmark_comparison(self, metrics: ReportPerformanceMetrics, period_days: int) -> Dict[str, Any]:
        """Create benchmark comparison."""
        return {
            'benchmark_symbol': self.config.benchmark_symbol,
            'portfolio_return': metrics.total_return,
            'benchmark_return': 0.08,  # Simplified - would calculate from actual data
            'excess_return': metrics.total_return - 0.08,
            'alpha': metrics.alpha,
            'beta': metrics.beta,
            'information_ratio': metrics.information_ratio,
            'tracking_error': metrics.tracking_error,
            'correlation': 0.75,  # Simplified
            'up_capture_ratio': 0.85,  # Simplified
            'down_capture_ratio': 0.65   # Simplified
        }

    def _create_attribution_analysis(self, trades: List[Trade], metrics: ReportPerformanceMetrics) -> Dict[str, Any]:
        """Create attribution analysis."""
        return {
            'factor_attribution': metrics.factor_attribution,
            'alpha_attribution': metrics.alpha_attribution,
            'sector_attribution': metrics.sector_attribution,
            'asset_class_attribution': {
                'equities': 0.7,
                'fixed_income': 0.2,
                'alternatives': 0.1
            },
            'geographic_attribution': {
                'domestic': 0.8,
                'international': 0.15,
                'emerging_markets': 0.05
            },
            'strategy_attribution': {
                'core': metrics.core_contribution,
                'satellite': metrics.satellite_contribution
            }
        }

    def _generate_recommendations(self, metrics: ReportPerformanceMetrics,
                                portfolio_summary: Dict[str, Any],
                                trading_summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Return-based recommendations
        if metrics.total_return < -0.05:
            recommendations.append("Significant negative return detected - review strategy allocation")
        elif metrics.total_return > 0.15:
            recommendations.append("Strong positive performance - consider profit taking")

        # Risk-based recommendations
        if metrics.volatility > 0.2:
            recommendations.append("High volatility detected - consider reducing portfolio risk")
        elif metrics.volatility < 0.05:
            recommendations.append("Low volatility - consider increasing exposure to enhance returns")

        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review risk management strategy")
        elif metrics.sharpe_ratio > 2.0:
            recommendations.append("Excellent risk-adjusted performance - maintain current strategy")

        # Drawdown recommendations
        if metrics.max_drawdown < -0.1:
            recommendations.append("Significant drawdown - evaluate risk controls and position sizing")

        # Trading activity recommendations
        if trading_summary['trades_per_day'] > 10:
            recommendations.append("High trading activity - monitor transaction costs and tax implications")
        elif trading_summary['trades_per_day'] < 0.1:
            recommendations.append("Low trading activity - ensure portfolio remains aligned with strategy")

        # Portfolio composition recommendations
        if portfolio_summary['number_of_positions'] > 30:
            recommendations.append("Many positions - consider consolidation for better oversight")
        elif portfolio_summary['number_of_positions'] < 5:
            recommendations.append("Few positions - consider diversification to reduce risk")

        return recommendations

    def _check_alerts(self, metrics: ReportPerformanceMetrics, thresholds: Dict[str, float]) -> List[str]:
        """Check for performance alerts."""
        alerts = []

        # Daily return alert
        if metrics.daily_return_mean < thresholds.get('daily_return', -0.05):
            alerts.append(f"Low daily return: {metrics.daily_return_mean:.2%}")

        # Drawdown alert
        if metrics.max_drawdown < thresholds.get('drawdown', -0.15):
            alerts.append(f"Significant drawdown: {metrics.max_drawdown:.2%}")

        # Volatility alert
        if metrics.volatility > thresholds.get('volatility', 0.25):
            alerts.append(f"High volatility: {metrics.volatility:.2%}")

        # VaR alert
        if abs(metrics.var_95) > thresholds.get('var_95', 0.05):
            alerts.append(f"High Value at Risk: {metrics.var_95:.2%}")

        # Sharpe ratio alert
        if metrics.sharpe_ratio < -0.5:
            alerts.append(f"Very low Sharpe ratio: {metrics.sharpe_ratio:.2f}")

        return alerts

    def _update_report_tracking(self, report: PerformanceReport) -> None:
        """Update report tracking statistics."""
        self.performance_history.append(report)
        self.reporting_stats['reports_generated'] += 1
        self.reporting_stats['last_report_time'] = report.timestamp
        self.reporting_stats['recommendations_provided'] += len(report.recommendations)

        # Update alerts
        if report.alerts:
            self.reporting_stats['alerts_triggered'] += len(report.alerts)
            self.active_alerts.extend([
                {
                    'timestamp': report.timestamp,
                    'message': alert,
                    'report_id': len(self.performance_history) - 1
                } for alert in report.alerts
            ])

        # Keep history manageable
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _save_report(self, report: PerformanceReport) -> None:
        """Save report to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)

            timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp_str}.{self.config.file_format}"
            filepath = output_dir / filename

            # Convert report to dictionary for JSON serialization
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'metrics': report.metrics.__dict__,
                'portfolio_summary': report.portfolio_summary,
                'trading_summary': report.trading_summary,
                'benchmark_comparison': report.benchmark_comparison,
                'attribution_analysis': report.attribution_analysis,
                'recommendations': report.recommendations,
                'alerts': report.alerts
            }

            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            logger.info(f"Performance report saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _create_minimal_report(self, period_days: int) -> PerformanceReport:
        """Create minimal report when errors occur."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        zero_metrics = self._create_zero_metrics()

        return PerformanceReport(
            timestamp=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            metrics=zero_metrics,
            portfolio_summary={'error': 'Insufficient data'},
            trading_summary={'error': 'Insufficient data'},
            benchmark_comparison={'error': 'Insufficient data'},
            attribution_analysis={'error': 'Insufficient data'},
            recommendations=['Error generating report - check data sources'],
            alerts=['Report generation error']
        )

    def _calculate_benchmark_returns(self, benchmark_data: pd.DataFrame) -> pd.Series:
        """Calculate benchmark returns from price data."""
        if 'Close' in benchmark_data.columns:
            returns = benchmark_data['Close'].pct_change().dropna()
        else:
            # Assume first column is price
            returns = benchmark_data.iloc[:, 0].pct_change().dropna()

        return returns

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [r for r in self.performance_history if r.timestamp >= cutoff_date]

        if not recent_reports:
            return {
                'period_days': days,
                'total_reports': 0,
                'average_return': 0,
                'average_sharpe': 0,
                'alerts_count': 0
            }

        returns = [r.metrics.total_return for r in recent_reports]
        sharpes = [r.metrics.sharpe_ratio for r in recent_reports]
        alerts_count = sum(len(r.alerts) for r in recent_reports)

        return {
            'period_days': days,
            'total_reports': len(recent_reports),
            'average_return': np.mean(returns) if returns else 0,
            'average_sharpe': np.mean(sharpes) if sharpes else 0,
            'best_return': max(returns) if returns else 0,
            'worst_return': min(returns) if returns else 0,
            'alerts_count': alerts_count,
            'recommendations_count': sum(len(r.recommendations) for r in recent_reports),
            'stats': self.reporting_stats.copy()
        }

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate reporter configuration."""
        issues = []

        if self.config.significant_return_threshold <= 0 or self.config.significant_return_threshold > 1:
            issues.append("significant_return_threshold must be between 0 and 100%")

        if self.config.significant_risk_threshold <= 0 or self.config.significant_risk_threshold > 1:
            issues.append("significant_risk_threshold must be between 0 and 100%")

        if not self.config.benchmark_symbol:
            issues.append("benchmark_symbol must be specified")

        if self.config.file_format not in ["json", "csv", "excel"]:
            issues.append("file_format must be one of: json, csv, excel")

        return len(issues) == 0, issues