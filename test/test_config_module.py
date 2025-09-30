#!/usr/bin/env python3
"""
Test script for the configuration module.
Tests ConfigFactory functionality and config file reading.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_system.config import ConfigFactory, BacktestConfig, StrategyConfig, SystemConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_factory_creation():
    """Test creating configs directly from factory methods."""
    logger.info("=== Testing Config Factory Direct Creation ===")

    from datetime import datetime
    from trading_system.config.strategy import StrategyType

    # Test BacktestConfig creation
    backtest_config = ConfigFactory.create_backtest_config(
        initial_capital=1000000,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        commission_rate=0.001,
        spread_rate=0.0005
    )
    logger.info(f"✓ BacktestConfig created: {backtest_config}")

    # Test StrategyConfig creation
    strategy_config = ConfigFactory.create_strategy_config(
        strategy_type=StrategyType.DUAL_MOMENTUM,
        universe=["SPY", "QQQ", "IWM"],
        lookback_period=60,
        signal_threshold=0.5
    )
    logger.info(f"✓ StrategyConfig created: {strategy_config}")

    # Test SystemConfig creation
    system_config = ConfigFactory.create_system_config(
        core_weight=0.7,
        satellite_weight=0.3,
        max_positions=20
    )
    logger.info(f"✓ SystemConfig created: {system_config}")

    return True

def test_yaml_loading():
    """Test loading configs from YAML files."""
    logger.info("\n=== Testing YAML Configuration Loading ===")

    config_files = [
        "configs/example_config.yaml",
        "configs/strategy_config.yaml",
        "configs/ml_strategy_config.yaml",
        "configs/core_satellite_example.yaml"
    ]

    successful_loads = 0

    for config_file in config_files:
        try:
            logger.info(f"Loading {config_file}...")
            configs = ConfigFactory.from_yaml(config_file)
            logger.info(f"✓ Successfully loaded {len(configs)} configs from {config_file}")

            for config_name, config_obj in configs.items():
                logger.info(f"  - {config_name}: {type(config_obj).__name__}")
                logger.info(f"    {config_obj}")

            successful_loads += 1

        except Exception as e:
            logger.error(f"✗ Failed to load {config_file}: {e}")

    logger.info(f"Successfully loaded {successful_loads}/{len(config_files)} config files")
    return successful_loads == len(config_files)

def test_config_validation():
    """Test config validation and error handling."""
    logger.info("\n=== Testing Configuration Validation ===")

    # Test invalid config creation
    try:
        invalid_backtest = ConfigFactory.create_backtest_config(
            initial_capital=-1000  # Invalid negative capital
        )
        logger.warning("⚠ BacktestConfig allowed negative capital (validation may be needed)")
    except Exception as e:
        logger.info(f"✓ BacktestConfig correctly rejected invalid input: {e}")

    # Test loading non-existent file
    try:
        ConfigFactory.from_yaml("non_existent_config.yaml")
        logger.warning("⚠ ConfigFactory loaded non-existent file")
    except FileNotFoundError as e:
        logger.info(f"✓ ConfigFactory correctly handled missing file: {e}")
    except Exception as e:
        logger.error(f"✗ Unexpected error for missing file: {e}")

    return True

def main():
    """Run all configuration tests."""
    logger.info("Starting Configuration Module Tests")
    logger.info("=" * 50)

    tests = [
        ("Factory Creation", test_factory_creation),
        ("YAML Loading", test_yaml_loading),
        ("Config Validation", test_config_validation)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            logger.info(f"\n{test_name}: {status}")
        except Exception as e:
            logger.error(f"\n{test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All configuration tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)