"""
Portfolio Construction Demo
===========================

This demo shows how to use the new portfolio construction framework
with the ModernSystemOrchestrator to run both quantitative and box-based
portfolio construction methods.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import yaml

from trading_system.orchestration import ModernSystemOrchestrator, ModernSystemConfig
from trading_system.strategies.dual_momentum_strategy import DualMomentumStrategy
from trading_system.strategies.fama_french_strategy import FamaFrenchStrategy
from trading_system.data.yfinance_provider import YFinanceProvider
from trading_system.data.ff5_provider import FF5Provider
from trading_system.data.stock_classifier import StockClassifier
from trading_system.orchestration.meta_model import MetaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def create_strategies(config: dict) -> list:
    """Create trading strategies from configuration."""
    strategies = []

    for strategy_config in config.get('strategies', []):
        if strategy_config['type'] == 'DualMomentumStrategy':
            strategy = DualMomentumStrategy(strategy_config.get('config', {}))
            strategy.name = strategy_config['name']
            strategies.append(strategy)
        elif strategy_config['type'] == 'FamaFrenchStrategy':
            strategy = FamaFrenchStrategy(strategy_config.get('config', {}))
            strategy.name = strategy_config['name']
            strategies.append(strategy)
        else:
            logger.warning(f"Unknown strategy type: {strategy_config['type']}")

    return strategies


def create_meta_model(config: dict) -> MetaModel:
    """Create meta-model from configuration."""
    meta_config = config.get('meta_model', {})
    method = meta_config.get('method', 'weighted_average')

    if method == 'weighted_average':
        from trading_system.orchestration.meta_model import WeightedAverageMetaModel
        return WeightedAverageMetaModel(meta_config.get('config', {}))
    else:
        raise ValueError(f"Unknown meta-model method: {method}")


def get_sample_price_data(universe: list, start_date: datetime, end_date: datetime) -> dict:
    """
    Get sample price data for demonstration.
    In a real scenario, this would come from your data provider.
    """
    logger.info(f"Fetching sample price data for {len(universe)} symbols...")

    # Use YFinance provider for demo
    provider = YFinanceProvider()
    price_data = {}

    for symbol in universe:
        try:
            data = provider.get_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                price_data[symbol] = data
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")

    logger.info(f"Successfully retrieved data for {len(price_data)} symbols")
    return price_data


def run_portfolio_construction_demo(config_path: str):
    """Run the portfolio construction demo."""
    logger.info("=== Portfolio Construction Demo ===")

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Create system configuration
    system_config = ModernSystemConfig(
        initial_capital=config['system']['initial_capital'],
        enable_short_selling=config['system']['enable_short_selling'],
        portfolio_construction=config['portfolio_construction']
    )

    # Create strategies
    strategies = create_strategies(config)
    logger.info(f"Created {len(strategies)} strategies: {[s.name for s in strategies]}")

    # Create meta-model
    meta_model = create_meta_model(config)
    logger.info(f"Created meta-model: {meta_model.__class__.__name__}")

    # Create stock classifier
    classifier_config = config['portfolio_construction'].get('classifier', {})
    stock_classifier = StockClassifier(classifier_config)
    logger.info(f"Created stock classifier: {stock_classifier.__class__.__name__}")

    # Create modern system orchestrator
    orchestrator = ModernSystemOrchestrator(
        system_config=system_config,
        strategies=strategies,
        meta_model=meta_model,
        stock_classifier=stock_classifier,
        custom_configs={
            'max_single_position_weight': config['compliance']['max_single_position_weight'],
            'box_limits': config['compliance'].get('box_exposure_limits', {})
        }
    )

    # Initialize system
    if not orchestrator.initialize_system():
        logger.error("Failed to initialize system")
        return

    # Get sample universe and price data
    sample_universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'PFE', 'KO',
        'CVX', 'LLY', 'ABBV', 'TMO', 'ABT', 'DHR', 'ACN', 'CRM', 'MRK',
        'NFLX', 'AMD', 'COST', 'LIN', 'NKE', 'BMY', 'GE', 'LOW', 'DIS',
        'CAT', 'RTX', 'GS', 'UPS', 'HON', 'SNY', 'T', 'VZ', 'ADBE',
        'IBM', 'INTC', 'BLK', 'WFC', 'CSCO', 'TXN', 'NEE', 'PM', 'QCOM'
    ]

    # Define test date range
    end_date = datetime(2024, 10, 1)
    start_date = datetime(2023, 10, 1)

    # Get price data
    price_data = get_sample_price_data(sample_universe, start_date, end_date)

    if len(price_data) < 20:
        logger.error("Insufficient price data for demo")
        return

    # Run system on test date
    test_date = datetime(2024, 9, 30)
    logger.info(f"Running portfolio construction for {test_date.date()}")

    try:
        # Run with detailed results
        system_result, detailed_results = orchestrator.run_system_with_detailed_result(
            date=test_date,
            price_data=price_data
        )

        # Display results
        logger.info("=== Portfolio Construction Results ===")
        logger.info(f"Status: {system_result.status}")
        logger.info(f"Construction Method: {detailed_results.get('construction_method', 'N/A')}")

        # Portfolio summary
        portfolio_summary = system_result.portfolio_summary
        logger.info(f"Portfolio Summary:")
        logger.info(f"  Total Positions: {system_result.trades_summary['total_positions']}")
        logger.info(f"  Long Positions: {system_result.trades_summary['long_positions']}")
        logger.info(f"  Short Positions: {system_result.trades_summary['short_positions']}")
        logger.info(f"  Total Weight: {portfolio_summary['total_weight']:.4f}")
        logger.info(f"  Max Weight: {portfolio_summary['max_weight']:.4f}")
        logger.info(f"  Weight Concentration (HHI): {portfolio_summary['weight_concentration']:.4f}")

        # Top 10 positions
        top_positions = system_result.trades_summary['top_positions']
        logger.info(f"\nTop 10 Positions:")
        for i, (symbol, weight) in enumerate(list(top_positions.items())[:10], 1):
            logger.info(f"  {i:2d}. {symbol:5s}: {weight:6.4f}")

        # Compliance status
        compliance = system_result.compliance_report
        logger.info(f"\nCompliance Status: {compliance.overall_status.value}")
        if compliance.total_violations > 0:
            logger.info(f"Violations: {compliance.total_violations}")
            for violation in compliance.violations:
                logger.info(f"  - {violation}")

        # Box-based specific information
        if 'box_coverage' in detailed_results:
            logger.info(f"\nBox Coverage Analysis:")
            box_coverage = detailed_results['box_coverage']
            covered_boxes = sum(1 for coverage in box_coverage.values() if coverage > 0)
            total_boxes = len(box_coverage)
            coverage_ratio = covered_boxes / total_boxes if total_boxes > 0 else 0

            logger.info(f"  Coverage Ratio: {coverage_ratio:.1%} ({covered_boxes}/{total_boxes} boxes)")

            # Show covered boxes
            covered = [box for box, coverage in box_coverage.items() if coverage > 0]
            logger.info(f"  Covered Boxes: {covered}")

        # Construction log
        if 'construction_log' in detailed_results:
            logger.info(f"\nConstruction Log:")
            for log_entry in detailed_results['construction_log']:
                logger.info(f"  - {log_entry}")

        logger.info("=== Demo completed successfully ===")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def compare_portfolio_construction_methods():
    """Compare quantitative vs box-based methods."""
    logger.info("=== Method Comparison Demo ===")

    # Configuration files for both methods
    configs = {
        'quantitative': 'configs/quantitative_demo_config.yaml',
        'box_based': 'configs/box_based_demo_config.yaml'
    }

    # Create demo configs if they don't exist
    for method, config_path in configs.items():
        if not os.path.exists(config_path):
            create_demo_config(method, config_path)

    # Run comparison
    results = {}

    for method, config_path in configs.items():
        logger.info(f"\n--- Testing {method.upper()} Method ---")
        try:
            # This would run the full demo for each method
            # For brevity, just show the construction method info
            config = load_config(config_path)
            logger.info(f"Method: {config['portfolio_construction']['method']}")
            logger.info(f"Key Parameters:")

            pc_config = config['portfolio_construction']
            if method == 'quantitative':
                logger.info(f"  Universe Size: {pc_config.get('universe_size', 'N/A')}")
                logger.info(f"  Optimizer: {pc_config.get('optimizer', {}).get('method', 'N/A')}")
                logger.info(f"  Box Sampling: {pc_config.get('use_box_sampling', False)}")
            else:
                logger.info(f"  Stocks per Box: {pc_config.get('stocks_per_box', 'N/A')}")
                logger.info(f"  Allocation Method: {pc_config.get('allocation_method', 'N/A')}")
                logger.info(f"  Box Weight Method: {pc_config.get('box_weights', {}).get('method', 'N/A')}")

            results[method] = config

        except Exception as e:
            logger.error(f"Failed to test {method} method: {e}")

    logger.info("\n=== Comparison Summary ===")
    for method, config in results.items():
        logger.info(f"{method.upper()}: Configuration loaded successfully")

    logger.info("To run full comparison, execute:")
    logger.info("  python portfolio_construction_demo.py quantitative")
    logger.info("  python portfolio_construction_demo.py box_based")


def create_demo_config(method: str, config_path: str):
    """Create a demo configuration file for the specified method."""
    config = {
        'system': {
            'initial_capital': 1000000,
            'enable_short_selling': False
        },
        'portfolio_construction': {
            'method': method,
            'classifier': {
                'method': 'four_factor',
                'cache_enabled': True
            }
        },
        'strategies': [
            {
                'name': 'dual_momentum_core',
                'type': 'DualMomentumStrategy',
                'weight': 0.7,
                'config': {
                    'lookback_period': 60,
                    'cash_proxy': 'SHY'
                }
            }
        ],
        'meta_model': {
            'method': 'weighted_average',
            'config': {
                'normalize_weights': True
            }
        },
        'compliance': {
            'max_single_position_weight': 0.10
        }
    }

    # Add method-specific configuration
    if method == 'quantitative':
        config['portfolio_construction'].update({
            'universe_size': 50,
            'optimizer': {
                'method': 'mean_variance',
                'risk_aversion': 2.0
            },
            'covariance': {
                'lookback_days': 252,
                'method': 'ledoit_wolf'
            }
        })
    else:  # box_based
        config['portfolio_construction'].update({
            'stocks_per_box': 2,
            'min_stocks_per_box': 1,
            'allocation_method': 'equal',
            'box_weights': {
                'method': 'equal',
                'dimensions': {
                    'size': ['large', 'mid'],
                    'style': ['growth', 'value'],
                    'region': ['developed'],
                    'sector': ['Technology', 'Financials', 'Healthcare']
                }
            }
        })

    # Write configuration
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Created demo config: {config_path}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_portfolio_construction_methods()
        elif sys.argv[1] in ["quantitative", "box_based"]:
            # Run specific method demo
            config_path = f"configs/{sys.argv[1]}_demo_config.yaml"
            if not os.path.exists(config_path):
                create_demo_config(sys.argv[1], config_path)
            run_portfolio_construction_demo(config_path)
        else:
            print("Usage: python portfolio_construction_demo.py [compare|quantitative|box_based]")
    else:
        # Run main demo with full configuration
        config_path = "configs/portfolio_construction_config.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please create the configuration file or run with 'compare' to see demo options")
            sys.exit(1)

        run_portfolio_construction_demo(config_path)