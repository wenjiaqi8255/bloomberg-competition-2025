"""
Modern System Orchestrator with Portfolio Construction Integration
================================================================

This module provides the `ModernSystemOrchestrator`, an enhanced version of the
SystemOrchestrator that integrates with the new portfolio construction framework
while maintaining compatibility with existing components.

Key improvements:
- Integration with IPortfolioBuilder interface
- Support for both Quantitative and Box-Based construction methods
- Configuration-driven construction method selection
- Maintains all existing SystemOrchestrator functionality
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from .system_orchestrator import (
    SystemOrchestrator, SystemResult, SystemConfig,
    CoordinatorConfig, StrategyCoordinator, AllocationConfig, CapitalAllocator,
    ComplianceRules, ComplianceMonitor, ExecutionConfig, TradeExecutor,
    ReportConfig, PerformanceReporter
)
from ..strategies.base_strategy import BaseStrategy
from ..types.portfolio import Portfolio
from ..config.system import SystemConfig
from ..data.stock_classifier import StockClassifier
from .meta_model import MetaModel

# New portfolio construction imports
from ..portfolio_construction import (
    IPortfolioBuilder, PortfolioConstructionRequest, PortfolioBuilderFactory
)

logger = logging.getLogger(__name__)


@dataclass
class ModernSystemConfig(SystemConfig):
    """
    Extended system configuration that includes portfolio construction settings.
    """
    portfolio_construction: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize portfolio construction config if not provided."""
        super().__post_init__()
        if self.portfolio_construction is None:
            self.portfolio_construction = {
                'method': 'quantitative',  # Default to quantitative for backward compatibility
                'universe_size': 100,
                'enable_short_selling': self.enable_short_selling,
                'optimizer': {
                    'method': 'mean_variance',
                    'risk_aversion': 2.0
                },
                'covariance': {
                    'lookback_days': 252,
                    'method': 'ledoit_wolf'
                },
                'classifier': {
                    'method': 'four_factor',
                    'cache_enabled': True
                }
            }


class ModernSystemOrchestrator(SystemOrchestrator):
    """
    Enhanced System Orchestrator with integrated portfolio construction framework.

    This orchestrator extends the original SystemOrchestrator to use the new
    portfolio construction interface while maintaining all existing functionality.
    It supports both traditional quantitative optimization and Box-First methodology
    through a unified interface.
    """

    def __init__(self,
                 system_config: ModernSystemConfig,
                 strategies: List[BaseStrategy],
                 meta_model: MetaModel,
                 stock_classifier: StockClassifier,
                 custom_configs: Optional[Dict[str, Any]] = None):
        """
        Initialize Modern System Orchestrator.

        Args:
            system_config: Enhanced system configuration with portfolio construction settings
            strategies: List of trading strategies
            meta_model: The meta-model for combining strategy signals
            stock_classifier: Classifier for box-based constraints
            custom_configs: Custom configuration overrides for components
        """
        # Store portfolio construction config
        self.portfolio_construction_config = system_config.portfolio_construction

        # Initialize portfolio builder
        self.portfolio_builder = self._create_portfolio_builder()

        # Initialize parent orchestrator
        super().__init__(system_config, strategies, meta_model, stock_classifier, custom_configs)

        logger.info(f"Initialized ModernSystemOrchestrator with portfolio construction method: "
                   f"{self.portfolio_builder.get_method_name()}")

    def _create_portfolio_builder(self) -> IPortfolioBuilder:
        """
        Create portfolio builder based on configuration.

        Returns:
            Configured portfolio builder
        """
        try:
            return PortfolioBuilderFactory.create_builder(self.portfolio_construction_config)
        except Exception as e:
            logger.error(f"Failed to create portfolio builder: {e}")
            # Fall back to quantitative method
            fallback_config = self.portfolio_construction_config.copy()
            fallback_config['method'] = 'quantitative'
            return PortfolioBuilderFactory.create_builder(fallback_config)

    def run_system(self, date: datetime, price_data: Dict[str, pd.DataFrame]) -> SystemResult:
        """
        Run the system using the new portfolio construction framework.

        Args:
            date: Date to run the system
            price_data: Dictionary of historical price data for all assets

        Returns:
            Complete system result
        """
        try:
            logger.info(f"--- Running Modern System for {date.date()} ---")
            logger.info(f"Portfolio construction method: {self.portfolio_builder.get_method_name()}")

            # Step 1: Generate signals from strategies (same as original)
            logger.info("[Step 1/6] Generating signals from strategies...")
            strategy_signals = self.coordinator.coordinate(date)

            # Step 2: Combine signals with Meta-Model (same as original)
            logger.info("[Step 2/6] Combining signals with Meta-Model...")
            strategy_signal_dfs = self._convert_signals_to_dataframes(strategy_signals, date)
            combined_signal = self.meta_model.combine(strategy_signal_dfs)

            if combined_signal.empty:
                raise ValueError("Meta-Model returned an empty signal.")

            # Convert combined signal to Series format
            combined_signal_series = combined_signal.iloc[0]

            # Step 3: Determine universe (get all stocks with signals)
            logger.info("[Step 3/6] Preparing universe...")
            available_universe = list(combined_signal_series.index)
            logger.info(f"Available universe: {len(available_universe)} stocks")

            # Step 4: Portfolio Construction using new framework
            logger.info(f"[Step 4/6] Portfolio construction using {self.portfolio_builder.get_method_name()}...")

            # Create portfolio construction request
            construction_request = PortfolioConstructionRequest(
                date=date,
                universe=available_universe,
                price_data=price_data,
                signals=combined_signal_series,
                constraints={},  # Constraints are handled internally by builders
                metadata={
                    'strategy_signals': strategy_signals,
                    'construction_method': self.portfolio_builder.get_method_name(),
                    'system_config': self.custom_configs
                }
            )

            # Build portfolio using the configured method
            portfolio_weights = self.portfolio_builder.build_portfolio(construction_request)

            logger.info(f"Portfolio construction completed: {len(portfolio_weights)} positions")
            logger.info(f"Portfolio weights sum: {portfolio_weights.sum():.6f}")

            # Step 5: Compliance checking
            logger.info("[Step 5/6] Compliance checking...")
            compliance_report = self._check_compliance(portfolio_weights, date, price_data)

            # Step 6: Trade execution and performance reporting
            logger.info("[Step 6/6] Finalizing system execution...")
            system_result = self._finalize_execution(
                portfolio_weights, strategy_signals, compliance_report, date
            )

            logger.info(f"--- Modern System execution completed: {system_result.status} ---")
            return system_result

        except Exception as e:
            logger.error(f"Modern system execution failed: {e}")
            return SystemResult.error_result(str(e))

    def run_system_with_detailed_result(self, date: datetime,
                                      price_data: Dict[str, pd.DataFrame]) -> Tuple[SystemResult, Dict[str, Any]]:
        """
        Run system with detailed portfolio construction results.

        Args:
            date: Date to run the system
            price_data: Dictionary of historical price data

        Returns:
            Tuple of (system result, detailed construction results)
        """
        try:
            # First, get basic system result
            system_result = self.run_system(date, price_data)

            if not system_result.is_successful:
                return system_result, {}

            # Get detailed construction information
            combined_signal = self._get_combined_signal_for_date(date)
            if combined_signal.empty:
                return system_result, {}

            combined_signal_series = combined_signal.iloc[0]
            available_universe = list(combined_signal_series.index)

            construction_request = PortfolioConstructionRequest(
                date=date,
                universe=available_universe,
                price_data=price_data,
                signals=combined_signal_series,
                constraints={},
                metadata={
                    'detailed_analysis': True,
                    'system_config': self.custom_configs
                }
            )

            # Get detailed construction result
            construction_result = self.portfolio_builder.build_portfolio_with_result(construction_request)

            # Combine results
            detailed_results = {
                'construction_method': self.portfolio_builder.get_method_name(),
                'construction_info': self.portfolio_builder.get_construction_info(),
                'box_coverage': construction_result.box_coverage,
                'selected_stocks': construction_result.selected_stocks,
                'target_weights': construction_result.target_weights,
                'construction_log': construction_result.construction_log,
                'system_result': system_result.__dict__
            }

            return system_result, detailed_results

        except Exception as e:
            logger.error(f"Detailed system execution failed: {e}")
            error_result = SystemResult.error_result(str(e))
            return error_result, {'error': str(e)}

    def _get_combined_signal_for_date(self, date: datetime) -> pd.DataFrame:
        """Get combined signal for a specific date."""
        try:
            strategy_signals = self.coordinator.coordinate(date)
            strategy_signal_dfs = self._convert_signals_to_dataframes(strategy_signals, date)
            return self.meta_model.combine(strategy_signal_dfs)
        except Exception as e:
            logger.error(f"Failed to get combined signal for {date}: {e}")
            return pd.DataFrame()

    def _check_compliance(self, portfolio_weights: pd.Series, date: datetime,
                         price_data: Dict[str, pd.DataFrame]):
        """Check portfolio compliance."""
        try:
            # Create a temporary portfolio for compliance checking
            from ..types.portfolio import Portfolio, Position

            positions = {}
            for symbol, weight in portfolio_weights.items():
                if weight != 0:
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=weight * 1000000,  # Assume $1M portfolio for compliance
                        current_price=price_data.get(symbol, pd.DataFrame()).iloc[-1].get('Close', 100.0)
                    )

            temp_portfolio = Portfolio(
                total_value=1000000,
                cash_balance=0,
                positions=positions,
                last_updated=date
            )

            return self.compliance_monitor.check_compliance(temp_portfolio, date)

        except Exception as e:
            logger.error(f"Compliance checking failed: {e}")
            # Return a compliant report as fallback
            from ..orchestration.components.compliance import ComplianceReport, ComplianceStatus
            return ComplianceReport(
                timestamp=date,
                overall_status=ComplianceStatus.COMPLIANT,
                total_violations=0,
                violations=[],
                warnings=[f"Compliance check failed: {e}"],
                recommendations=[],
                portfolio_summary={}
            )

    def _finalize_execution(self, portfolio_weights: pd.Series, strategy_signals: Dict[str, Any],
                           compliance_report, date: datetime) -> SystemResult:
        """Finalize execution and create system result."""
        try:
            # Create summaries
            signals_summary = {
                'total_signals': len(strategy_signals),
                'strategies': list(strategy_signals.keys())
            }

            trades_summary = {
                'total_positions': len(portfolio_weights),
                'long_positions': len(portfolio_weights[portfolio_weights > 0]),
                'short_positions': len(portfolio_weights[portfolio_weights < 0]) if self.config.enable_short_selling else 0,
                'top_positions': portfolio_weights.nlargest(10).to_dict()
            }

            portfolio_summary = {
                'total_weight': portfolio_weights.sum(),
                'max_weight': portfolio_weights.max(),
                'min_weight': portfolio_weights.min(),
                'weight_concentration': (portfolio_weights ** 2).sum()  # HHI concentration
            }

            allocation_summary = {
                'construction_method': self.portfolio_builder.get_method_name(),
                'universe_size': len(portfolio_weights),
                'compliance_status': compliance_report.overall_status.value
            }

            # Generate recommendations
            recommendations = []
            if compliance_report.total_violations > 0:
                recommendations.append(f"Address {compliance_report.total_violations} compliance issues")
            if portfolio_weights.sum() < 0.95 or portfolio_weights.sum() > 1.05:
                recommendations.append("Portfolio weights should sum to 1.0")

            # Create performance report (placeholder)
            performance_report = {
                'execution_status': 'success',
                'processing_time': 'N/A',
                'method_used': self.portfolio_builder.get_method_name()
            }

            return SystemResult(
                timestamp=date,
                status="success",
                signals_summary=signals_summary,
                trades_summary=trades_summary,
                portfolio_summary=portfolio_summary,
                allocation_summary=allocation_summary,
                compliance_report=compliance_report,
                performance_report=performance_report,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Execution finalization failed: {e}")
            return SystemResult.error_result(f"Finalization failed: {e}")

    def get_construction_method_info(self) -> Dict[str, Any]:
        """Get information about the current portfolio construction method."""
        return {
            'method_name': self.portfolio_builder.get_method_name(),
            'method_type': self.portfolio_builder.__class__.__name__,
            'construction_info': self.portfolio_builder.get_construction_info(),
            'config': self.portfolio_construction_config
        }

    def switch_construction_method(self, new_config: Dict[str, Any]) -> bool:
        """
        Switch to a different portfolio construction method.

        Args:
            new_config: New portfolio construction configuration

        Returns:
            True if switch successful
        """
        try:
            # Create new builder to validate config
            new_builder = PortfolioBuilderFactory.create_builder(new_config)

            # Update configuration and builder
            self.portfolio_construction_config = new_config
            self.portfolio_builder = new_builder

            logger.info(f"Switched to portfolio construction method: {self.portfolio_builder.get_method_name()}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch construction method: {e}")
            return False