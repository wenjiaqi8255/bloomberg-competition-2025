"""
Strategy Evaluation Demo
========================

Demonstrates the new strategy evaluation capabilities:
1. Automatic signal quality evaluation in BaseStrategy
2. Position metrics and concentration risk analysis
3. Strategy health checks and diagnostic reports
4. Comprehensive metrics aggregation in StrategyRunner

This shows how the PortfolioCalculator utilities are now
fully integrated into the strategy evaluation workflow.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading_system.strategy_backtest.strategy_runner import create_strategy_runner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_strategy_evaluation():
    """
    Demonstrate strategy evaluation features.
    
    This demo runs a backtest and shows:
    - Real-time signal quality snapshots during signal generation
    - Comprehensive metrics aggregation in the runner
    - Health checks and diagnostic reports
    """
    
    logger.info("=" * 70)
    logger.info("STRATEGY EVALUATION DEMO")
    logger.info("=" * 70)
    
    # Create strategy runner with a configuration
    # You can use any strategy config (ML, FF5, Dual Momentum, etc.)
    runner = create_strategy_runner(
        config_path="configs/dual_momentum_config.yaml",
        use_wandb=False  # Disable wandb for demo
    )
    
    # Initialize components
    runner.initialize()
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Check Strategy Initial State")
    logger.info("=" * 70)
    
    # Get initial diagnostic report (before signals generated)
    initial_report = runner.strategy.get_diagnostic_report()
    logger.info(f"Initial diagnostic report: {initial_report}")
    
    # Get initial health check
    initial_health = runner.strategy.get_health_check()
    logger.info(f"Initial health check: {initial_health}")
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Run Backtest (Signal Generation)")
    logger.info("=" * 70)
    logger.info("Watch for automatic signal quality snapshots during signal generation...")
    
    # Run the backtest
    # This will trigger:
    # - generate_signals() in the strategy
    # - _evaluate_and_cache_signals() automatically after each signal generation
    # - Logs showing signal quality, position metrics, concentration risk
    results = runner.run_strategy(experiment_name="strategy_evaluation_demo")
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Review Post-Backtest Metrics")
    logger.info("=" * 70)
    
    # Get comprehensive diagnostic report after backtest
    final_report = runner.strategy.get_diagnostic_report()
    logger.info("\nFinal Diagnostic Report:")
    logger.info(f"  Status: {final_report.get('status')}")
    logger.info(f"  Signal Generation Count: {final_report.get('signal_generation_count')}")
    logger.info(f"  Number of Assets: {final_report.get('number_of_assets')}")
    logger.info(f"  Concentration Risk: {final_report.get('concentration_risk', 0):.3f}")
    
    if final_report.get('signal_quality'):
        logger.info(f"\n  Signal Quality Metrics:")
        for key, value in final_report['signal_quality'].items():
            logger.info(f"    - {key}: {value}")
    
    if final_report.get('position_metrics'):
        logger.info(f"\n  Position Metrics:")
        for key, value in final_report['position_metrics'].items():
            logger.info(f"    - {key}: {value}")
    
    # Get final health check
    final_health = runner.strategy.get_health_check()
    logger.info("\nFinal Health Check:")
    logger.info(f"  Is Healthy: {final_health.get('is_healthy')}")
    logger.info(f"  Warnings: {final_health.get('warnings', [])}")
    logger.info(f"  Checks Performed: {final_health.get('checks_performed', [])}")
    
    # Get current snapshot
    snapshot = runner.strategy.get_current_snapshot()
    logger.info(f"\nCurrent Snapshot:")
    for key, value in snapshot.items():
        logger.info(f"  - {key}: {value}")
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Review Aggregated Metrics from StrategyRunner")
    logger.info("=" * 70)
    
    # The strategy_metrics from the runner now include comprehensive evaluation
    strategy_metrics = results.get('strategy_metrics', {})
    
    logger.info("\nStrategy-Specific Metrics (from StrategyRunner):")
    
    # Signal quality metrics
    if 'signal_quality' in strategy_metrics:
        logger.info("\n  Signal Quality (aggregated):")
        for key, value in strategy_metrics['signal_quality'].items():
            logger.info(f"    - {key}: {value}")
    
    # Position metrics
    if 'position_metrics' in strategy_metrics:
        logger.info("\n  Position Metrics (aggregated):")
        for key, value in strategy_metrics['position_metrics'].items():
            logger.info(f"    - {key}: {value}")
    
    # Concentration and turnover
    logger.info(f"\n  Concentration Risk (HHI): {strategy_metrics.get('concentration_risk_hhi', 0):.3f}")
    logger.info(f"  Portfolio Turnover: {strategy_metrics.get('portfolio_turnover', 0):.3f}")
    
    # Top/worst contributors
    if 'top_contributors' in strategy_metrics:
        logger.info("\n  Top Contributors:")
        for symbol, contribution in strategy_metrics['top_contributors'][:3]:
            logger.info(f"    - {symbol}: {contribution:.4f}")
    
    if 'worst_contributors' in strategy_metrics:
        logger.info("\n  Worst Contributors:")
        for symbol, contribution in strategy_metrics['worst_contributors'][:3]:
            logger.info(f"    - {symbol}: {contribution:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Performance Metrics")
    logger.info("=" * 70)
    
    # Show backtest performance for context
    performance_metrics = results.get('performance_metrics', {})
    logger.info(f"\nBacktest Performance:")
    logger.info(f"  Total Return: {performance_metrics.get('total_return', 0):.2%}")
    logger.info(f"  Annualized Return: {performance_metrics.get('annualized_return', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
    
    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("1. ✓ Strategy automatically evaluates signals during generation")
    logger.info("2. ✓ Real-time snapshots provide current state visibility")
    logger.info("3. ✓ Health checks detect potential issues")
    logger.info("4. ✓ StrategyRunner aggregates comprehensive metrics")
    logger.info("5. ✓ PortfolioCalculator utilities fully integrated")
    
    # Cleanup
    runner.cleanup()
    
    return results


def demo_manual_evaluation():
    """
    Demonstrate manual evaluation of signals.
    
    Shows how to use the strategy's evaluation methods directly
    without running a full backtest.
    """
    
    logger.info("\n" + "=" * 70)
    logger.info("MANUAL EVALUATION DEMO")
    logger.info("=" * 70)
    
    # Create a simple strategy for demonstration
    from src.trading_system.strategies.factory import StrategyFactory
    from src.trading_system.data.yfinance_provider import YFinanceProvider
    from datetime import datetime
    import pandas as pd
    
    # Create a simple dual momentum strategy
    strategy = StrategyFactory.create(
        strategy_type='dual_momentum',
        name='demo_strategy',
        lookback_periods=[60, 120],
        risk_free_symbol='SHY'
    )
    
    # Get some sample data
    data_provider = YFinanceProvider()
    symbols = ['SPY', 'QQQ', 'IWM', 'SHY']
    price_data = data_provider.get_historical_data(
        symbols=symbols,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )
    
    # Generate signals
    logger.info("\nGenerating signals...")
    signals = strategy.generate_signals(
        price_data=price_data,
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2024, 1, 1)
    )
    
    logger.info(f"Generated signals shape: {signals.shape}")
    
    # Now use the evaluation methods manually
    logger.info("\n" + "=" * 70)
    logger.info("Manual Evaluation Results")
    logger.info("=" * 70)
    
    # 1. Evaluate signal quality
    signal_quality = strategy.evaluate_signal_quality()
    logger.info(f"\nSignal Quality: {signal_quality}")
    
    # 2. Analyze positions
    position_metrics = strategy.analyze_positions()
    logger.info(f"\nPosition Metrics: {position_metrics}")
    
    # 3. Calculate concentration risk
    concentration = strategy.calculate_concentration_risk()
    logger.info(f"\nConcentration Risk (HHI): {concentration:.3f}")
    
    # 4. Get diagnostic report
    diagnostic = strategy.get_diagnostic_report()
    logger.info(f"\nDiagnostic Report:")
    for key, value in diagnostic.items():
        if key not in ['signal_quality', 'position_metrics', 'strategy_info']:
            logger.info(f"  {key}: {value}")
    
    # 5. Get health check
    health = strategy.get_health_check()
    logger.info(f"\nHealth Check:")
    logger.info(f"  Is Healthy: {health['is_healthy']}")
    logger.info(f"  Warnings: {health['warnings']}")
    
    # 6. Get current snapshot
    snapshot = strategy.get_current_snapshot()
    logger.info(f"\nCurrent Snapshot: {snapshot}")
    
    logger.info("\n" + "=" * 70)
    logger.info("MANUAL EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STRATEGY EVALUATION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the new strategy evaluation capabilities:")
    print("- Automatic signal quality tracking")
    print("- Position metrics and concentration analysis")
    print("- Health checks and diagnostics")
    print("- Integration with StrategyRunner")
    print("\n" + "=" * 70)
    
    # Run full backtest demo
    logger.info("\n### RUNNING FULL BACKTEST DEMO ###\n")
    results = demo_strategy_evaluation()
    
    # Run manual evaluation demo
    logger.info("\n\n### RUNNING MANUAL EVALUATION DEMO ###\n")
    demo_manual_evaluation()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)

