"""
Complete System Integration Script.

This script demonstrates the full integration of the Week 3 IPS Core system:
1. Core FFML Strategy (70-80% capital)
2. Satellite Strategy (20-30% capital)
3. System Orchestrator for integration and IPS compliance
4. Comprehensive reporting and monitoring

Usage:
    python run_complete_system.py

The script will:
- Initialize all system components
- Run the complete trading system
- Generate IPS compliance reports
- Save all results to output directory
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_system.orchestrator.system_orchestrator import SystemOrchestrator, SystemConfig
from trading_system.strategies.core_ffml_strategy import CoreFFMLStrategy
from trading_system.strategies.satellite_strategy import SatelliteStrategy
from trading_system.types.data_types import StrategyConfig, BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def create_backtest_config() -> BacktestConfig:
    """Create backtest configuration."""
    return BacktestConfig(
        initial_capital=1_000_000,  # $1M initial capital
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        symbols=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'LQD', 'GLD', 'XLK', 'XLF'],
        transaction_cost=0.001,
        benchmark_symbol='SPY',
        rebalance_frequency='monthly'
    )


def create_core_strategy_config() -> StrategyConfig:
    """Create Core FFML strategy configuration."""
    return StrategyConfig(
        name="Core_FFML_Strategy",
        parameters={
            'ff5_lookback': 60,
            'min_observations': 24,
            'prediction_horizon': 1,
            'feature_lags': 5,
            'cv_folds': 5,
            'min_signal_strength': 0.1,
            'max_position_size': 0.1,
            'target_positions': 20,
            'risk_budget': 0.02,
            'correlation_threshold': 0.7,
            'factor_confidence_threshold': 0.7,
            'ml_confidence_threshold': 0.6,
            'rebalance_threshold': 0.05
        },
        lookback_period=60,
        universe=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'LQD', 'GLD'],
        allocation_method='equal_weight'
    )


def create_satellite_strategy_config() -> StrategyConfig:
    """Create Satellite strategy configuration."""
    return StrategyConfig(
        name="Satellite_Technical_Strategy",
        parameters={
            'momentum_lookback': 12,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'momentum_threshold': 0.1,
            'min_signal_strength': 0.3,
            'max_correlation': 0.7,
            'max_position_size': 0.05,
            'max_satellite_exposure': 0.3,
            'stop_loss_threshold': 0.08,
            'take_profit_threshold': 0.15,
            'max_positions': 10,
            'rebalance_threshold': 0.1
        },
        lookback_period=24,
        universe=['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLU', 'XLP', 'XLY'],
        allocation_method='equal_weight'
    )


def create_system_config() -> SystemConfig:
    """Create system orchestrator configuration."""
    return SystemConfig(
        # Capital allocation
        core_allocation_min=0.70,
        core_allocation_max=0.80,
        satellite_allocation_min=0.20,
        satellite_allocation_max=0.30,

        # Rebalancing parameters
        rebalance_threshold=0.05,
        rebalance_frequency='monthly',

        # Risk management
        max_drawdown_threshold=0.15,
        var_threshold=0.10,
        tracking_error_threshold=0.05,

        # IPS compliance
        strict_box_compliance=True,
        compliance_report_frequency='monthly',

        # Model governance
        model_retraining_threshold=0.10,
        model_monitoring_frequency='weekly',

        # Reporting
        reporting_frequency='monthly',
        performance_benchmark='SPY'
    )


def initialize_strategies(backtest_config: BacktestConfig) -> tuple:
    """Initialize Core and Satellite strategies."""
    logger.info("Initializing strategies...")

    # Create strategy configurations
    core_config = create_core_strategy_config()
    satellite_config = create_satellite_strategy_config()

    # Initialize strategies
    core_strategy = CoreFFMLStrategy(core_config, backtest_config)
    satellite_strategy = SatelliteStrategy(satellite_config, backtest_config)

    logger.info("Strategies initialized successfully")
    return core_strategy, satellite_strategy


def run_complete_system():
    """Run the complete trading system."""
    try:
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE WEEK 3 IPS CORE SYSTEM")
        logger.info("=" * 60)

        # Create configurations
        logger.info("Creating system configurations...")
        backtest_config = create_backtest_config()
        system_config = create_system_config()

        # Initialize strategies
        core_strategy, satellite_strategy = initialize_strategies(backtest_config)

        # Initialize system orchestrator
        logger.info("Initializing system orchestrator...")
        orchestrator = SystemOrchestrator(
            system_config=system_config,
            core_strategy=core_strategy,
            satellite_strategy=satellite_strategy,
            backtest_config=backtest_config
        )

        # Initialize system
        logger.info("Initializing complete system...")
        if not orchestrator.initialize_system():
            logger.error("System initialization failed")
            return False

        logger.info("System initialization completed successfully")

        # Run system for multiple dates
        logger.info("Running system simulation...")
        simulation_dates = [
            datetime(2023, 1, 31),
            datetime(2023, 2, 28),
            datetime(2023, 3, 31),
            datetime(2023, 4, 30),
            datetime(2023, 5, 31),
            datetime(2023, 6, 30),
            datetime(2023, 7, 31),
            datetime(2023, 8, 31),
            datetime(2023, 9, 30),
            datetime(2023, 10, 31),
            datetime(2023, 11, 30),
            datetime(2023, 12, 31)
        ]

        all_results = []
        for date in simulation_dates:
            logger.info(f"Processing date: {date.strftime('%Y-%m-%d')}")
            result = orchestrator.run_system(date)
            all_results.append(result)

            if result['status'] == 'error':
                logger.error(f"System execution failed for {date}: {result.get('message', 'Unknown error')}")
            else:
                logger.info(f"Date processed successfully - Portfolio value: ${result['portfolio']['total_value']:,.0f}")

        # Generate final IPS compliance report
        logger.info("Generating IPS compliance report...")
        ips_report = orchestrator.generate_ips_compliance_report()

        # Save all results
        output_dir = project_root / "results" / f"complete_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Saving results to {output_dir}...")

        orchestrator.save_system_results(str(output_dir))

        # Save simulation results
        simulation_file = output_dir / "simulation_results.json"
        with open(simulation_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save IPS report
        ips_file = output_dir / "ips_compliance_report.json"
        with open(ips_file, 'w') as f:
            json.dump(ips_report, f, indent=2, default=str)

        # Print summary
        logger.info("=" * 60)
        logger.info("SYSTEM EXECUTION SUMMARY")
        logger.info("=" * 60)

        if all_results:
            final_result = all_results[-1]
            logger.info(f"Final Portfolio Value: ${final_result['portfolio']['total_value']:,.0f}")
            logger.info(f"Total Return: {final_result['portfolio']['total_return']:.2%}")
            logger.info(f"Max Drawdown: {final_result['portfolio']['drawdown']:.2%}")
            logger.info(f"Number of Positions: {final_result['portfolio']['number_of_positions']}")

        logger.info(f"Core Allocation: {final_result['allocation']['core_weight']:.1%}")
        logger.info(f"Satellite Allocation: {final_result['allocation']['satellite_weight']:.1%}")
        logger.info(f"Cash Allocation: {final_result['allocation']['cash_weight']:.1%}")

        logger.info(f"IPS Compliance: {'✓' if final_result['compliance']['overall_compliance'] else '✗'}")
        logger.info(f"Rebalance Needed: {'✓' if final_result['rebalance_needed'] else '✗'}")

        # Print system information
        system_info = orchestrator.get_system_info()
        logger.info("\nSystem Information:")
        logger.info(f"- Strategies initialized: {len(system_info['strategies'])}")
        logger.info(f"- Performance history: {len(system_info['performance_history_length'])} records")
        logger.info(f"- Trade history: {len(system_info['trade_history_length'])} trades")
        logger.info(f"- System initialized: {system_info['system_initialized']}")

        logger.info("\n" + "=" * 60)
        logger.info("WEEK 3 IPS CORE SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Check the following files:")
        logger.info(f"- {output_dir}/ips_compliance_report.json")
        logger.info(f"- {output_dir}/simulation_results.json")
        logger.info(f"- {output_dir}/system_portfolio_history.csv")
        logger.info(f"- {output_dir}/system_allocation_history.csv")
        logger.info(f"- {output_dir}/system_compliance_history.csv")
        logger.info(f"- {output_dir}/system_performance_history.csv")
        logger.info(f"- {output_dir}/system_trade_history.csv")

        return True

    except Exception as e:
        logger.error(f"System execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    """Main execution function."""
    success = run_complete_system()

    if success:
        logger.info("Week 3 IPS Core System - EXECUTION SUCCESSFUL")
        sys.exit(0)
    else:
        logger.error("Week 3 IPS Core System - EXECUTION FAILED")
        sys.exit(1)