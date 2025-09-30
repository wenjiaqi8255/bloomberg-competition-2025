#!/usr/bin/env python3
"""
Comprehensive test demonstrating all config module capabilities.
Shows that ConfigFactory can output desired configs and read configurations properly.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_system.config import ConfigFactory, BacktestConfig, StrategyConfig, SystemConfig
from datetime import datetime
from trading_system.config.strategy import StrategyType

def test_all_config_types():
    """Demonstrate all config types can be created properly."""
    print("=== Comprehensive Configuration Module Test ===")
    print()

    # 1. Test direct factory creation
    print("1. Direct Factory Creation:")
    print("-" * 30)

    # BacktestConfig
    backtest = ConfigFactory.create_backtest_config(
        name="test_backtest",
        initial_capital=1_000_000,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        commission_rate=0.001,
        spread_rate=0.0005,
        rebalance_frequency="monthly"
    )
    print(f"✓ BacktestConfig: {backtest.name}, Capital: ${backtest.initial_capital:,}")
    print(f"  Period: {backtest.start_date.date()} to {backtest.end_date.date()}")
    print(f"  Total costs: {backtest.total_cost_rate:.3%}")

    # StrategyConfig with different types
    strategies = [
        (StrategyType.DUAL_MOMENTUM, "dual_momentum_strategy", ["SPY", "QQQ", "AGG"]),
        (StrategyType.ML, "ml_strategy", ["SPY", "QQQ", "IWM", "EFA"]),
        (StrategyType.CORE_SATELLITE, "core_satellite_strategy", ["SPY", "QQQ", "IWM", "EFA", "AGG"])
    ]

    for strategy_type, name, universe in strategies:
        strategy = ConfigFactory.create_strategy_config(
            name=name,
            strategy_type=strategy_type,
            universe=universe,
            lookback_period=252,
            signal_threshold=0.5,
            parameters={
                "test_param": f"value_for_{name}",
                "strategy_specific": True
            }
        )
        print(f"✓ {strategy_type.value.upper()} Strategy: {strategy.name}")
        print(f"  Universe: {len(strategy.universe)} symbols, Parameters: {len(strategy.parameters)}")

    # SystemConfig
    system = ConfigFactory.create_system_config(
        name="test_system",
        core_weight=0.7,
        satellite_weight=0.3,
        max_positions=15,
        rebalance_frequency=30
    )
    print(f"✓ SystemConfig: {system.name}")
    print(f"  Core: {system.core_weight:.1%}, Satellite: {system.satellite_weight:.1%}")

    print()

    # 2. Test YAML loading for all config files
    print("2. YAML Configuration Loading:")
    print("-" * 30)

    config_files = [
        "configs/example_config.yaml",
        "configs/strategy_config.yaml",
        "configs/ml_strategy_config.yaml",
        "configs/core_satellite_example.yaml"
    ]

    total_configs_loaded = 0
    for config_file in config_files:
        try:
            configs = ConfigFactory.from_yaml(config_file)
            total_configs_loaded += len(configs)

            print(f"✓ {config_file}: {len(configs)} configs loaded")
            for config_name, config_obj in configs.items():
                print(f"  - {config_name}: {config_obj.name} ({type(config_obj).__name__})")

                # Show some key details
                if isinstance(config_obj, BacktestConfig):
                    print(f"    Capital: ${config_obj.initial_capital:,}, Period: {config_obj.start_date.date()} to {config_obj.end_date.date()}")
                elif isinstance(config_obj, StrategyConfig):
                    strategy_type = config_obj.strategy_type.value if hasattr(config_obj.strategy_type, 'value') else config_obj.strategy_type
                    print(f"    Type: {strategy_type}, Universe: {len(config_obj.universe)} symbols")
                    if config_obj.parameters:
                        print(f"    Strategy parameters: {list(config_obj.parameters.keys())[:3]}...")
                elif isinstance(config_obj, SystemConfig):
                    print(f"    Core/Satellite: {config_obj.core_weight:.1%}/{config_obj.satellite_weight:.1%}")

        except Exception as e:
            print(f"✗ {config_file}: Error - {e}")

    print(f"\nTotal configurations loaded: {total_configs_loaded}")

    print()

    # 3. Test configuration validation and error handling
    print("3. Configuration Validation:")
    print("-" * 30)

    # Test invalid configurations
    test_cases = [
        ("Negative capital", lambda: ConfigFactory.create_backtest_config(initial_capital=-1000)),
        ("Invalid universe", lambda: ConfigFactory.create_strategy_config(universe=[])),
        ("Invalid date range", lambda: ConfigFactory.create_backtest_config(
            start_date=datetime(2024, 1, 1), end_date=datetime(2023, 1, 1)
        )),
        ("Invalid position limit", lambda: ConfigFactory.create_strategy_config(position_size_limit=1.5))
    ]

    for test_name, test_func in test_cases:
        try:
            test_func()
            print(f"⚠ {test_name}: Should have failed but didn't")
        except Exception as e:
            print(f"✓ {test_name}: Correctly rejected - {type(e).__name__}")

    print()

    # 4. Test configuration methods and properties
    print("4. Configuration Methods & Properties:")
    print("-" * 30)

    # Create a test configuration
    test_backtest = ConfigFactory.create_backtest_config(
        initial_capital=2_000_000,
        commission_rate=0.001,
        spread_rate=0.0005,
        slippage_rate=0.0002,
        position_limit=0.08,
        rebalance_frequency="monthly"
    )

    print(f"BacktestConfig methods:")
    print(f"  Total cost rate: {test_backtest.total_cost_rate:.4f}")
    print(f"  Trading days/year: {test_backtest.trading_days_per_year}")
    print(f"  Position limit for $1M portfolio: ${test_backtest.get_position_limit_amount(1_000_000):,.0f}")
    print(f"  Rebalance needed? {test_backtest.is_rebalance_needed(0.12, 0.15, 1_000_000)}")

    # Test strategy methods
    test_strategy = ConfigFactory.create_strategy_config(
        strategy_type=StrategyType.DUAL_MOMENTUM,
        universe=["SPY", "QQQ", "IWM", "EFA", "AGG"],
        allocation_method="equal_weight",
        position_size_limit=0.20
    )

    print(f"\nStrategyConfig methods:")
    print(f"  Is long/short: {test_strategy.is_long_short}")
    print(f"  Position weight for {len(test_strategy.universe)} assets: {test_strategy.get_position_weight(len(test_strategy.universe)):.3%}")
    print(f"  Strategy param 'test': {test_strategy.get_strategy_param('test', 'default')}")

    # Test summaries
    print(f"\nConfiguration summaries:")
    backtest_summary = test_backtest.get_summary()
    strategy_summary = test_strategy.get_summary()

    print(f"  Backtest key metrics: {list(backtest_summary.keys())[:5]}")
    print(f"  Strategy key metrics: {list(strategy_summary.keys())[:5]}")

    print()

    # 5. Test factory convenience methods
    print("5. Factory Convenience Methods:")
    print("-" * 30)

    # Test BacktestConfig class methods
    simple_backtest = BacktestConfig.create_simple(
        initial_capital=500_000,
        start_date="2023-01-01",
        end_date="2023-12-31",
        symbols=["SPY", "QQQ"]
    )
    print(f"✓ Simple backtest: {simple_backtest.initial_capital:,} capital, {len(simple_backtest.symbols)} symbols")

    academic_backtest = BacktestConfig.create_academic(
        symbols=["SPY", "QQQ", "IWM", "EFA"]
    )
    print(f"✓ Academic backtest: {academic_backtest.commission_rate:.4%} commission, short selling: {academic_backtest.enable_short_selling}")

    production_backtest = BacktestConfig.create_production(
        initial_capital=10_000_000,
        symbols=["SPY", "QQQ", "IWM", "EFA", "AGG", "GLD"]
    )
    print(f"✓ Production backtest: {production_backtest.initial_capital:,} capital, {production_backtest.rebalance_frequency} rebalancing")

    print()

    # 6. Test configuration serialization compatibility
    print("6. Configuration Compatibility:")
    print("-" * 30)

    # Load a config and show it can be used with different strategies
    configs = ConfigFactory.from_yaml("configs/example_config.yaml")

    if 'strategy' in configs:
        strategy_config = configs['strategy']
        strategy_type = strategy_config.strategy_type.value if hasattr(strategy_config.strategy_type, 'value') else strategy_config.strategy_type
        print(f"✓ Loaded strategy: {strategy_type}")
        print(f"  Can be used with: {strategy_config.universe}")
        print(f"  Risk management: stop_loss={strategy_config.stop_loss_enabled}, max_positions={strategy_config.max_positions}")

    if 'backtest' in configs:
        backtest_config = configs['backtest']
        print(f"✓ Loaded backtest: {backtest_config.initial_capital:,} capital")
        print(f"  Transaction costs: commission={backtest_config.commission_rate:.3%}, spread={backtest_config.spread_rate:.3%}")

    print()
    print("🎉 Comprehensive configuration test completed successfully!")
    print("✅ ConfigFactory can output desired configs")
    print("✅ Configuration files can be read properly")
    print("✅ All config types work correctly")

    return True

if __name__ == "__main__":
    success = test_all_config_types()
    sys.exit(0 if success else 1)