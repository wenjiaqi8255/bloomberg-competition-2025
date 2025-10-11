#!/usr/bin/env python3
"""
Unified System Runner (Modern)
==============================

Runs a multi-strategy system using the ModernSystemOrchestrator and the
pluggable portfolio construction framework (quantitative | box_based),
fully configured via a single YAML file.

KISS / YAGNI: This runner focuses on the essentials only.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd

# Ensure src is in path when executing from repo root
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ..portfolio_generation.system_orchestrator import SystemOrchestrator, SystemConfig
from ...trading_system.metamodel.meta_model import MetaModel
from ...trading_system.data.stock_classifier import StockClassifier
from ...trading_system.data.yfinance_provider import YFinanceProvider
from ...trading_system.data.ff5_provider import FF5DataProvider
from ...trading_system.strategies.factory import StrategyFactory


logger = logging.getLogger("run_system_with_metamodel")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_strategies(strategy_cfg_list: list, providers: dict):
    strategies = []
    for cfg in strategy_cfg_list:
        # Expect either already-normalized dict for StrategyFactory
        # or a simplified {name, type, parameters, universe}
        if 'type' in cfg:
            strategy_config_dict = dict(cfg)
        else:
            # Minimal normalization
            strategy_config_dict = {
                'name': cfg.get('name', 'Strategy'),
                'type': cfg.get('strategy_type') or cfg.get('type'),
                'parameters': cfg.get('parameters', {})
            }
            if 'universe' in cfg:
                strategy_config_dict['universe'] = cfg['universe']

        strategy = StrategyFactory.create_from_config(strategy_config_dict, providers=providers)
        strategies.append(strategy)
    return strategies


def build_metamodel(mm_cfg: dict, strategy_names: list) -> MetaModel:
    method = (mm_cfg or {}).get('method', 'equal')
    weights = (mm_cfg or {}).get('weights', None)
    if not weights:
        # Equal weight default if nothing specified
        equal_w = 1.0 / max(1, len(strategy_names))
        weights = {name: equal_w for name in strategy_names}
    meta = MetaModel(method=method)
    # Assign weights if MetaModel supports it
    try:
        meta.strategy_weights = weights
    except Exception:
        logger.warning("MetaModel does not expose strategy_weights assignable property; proceeding with defaults")
    return meta


def build_providers(cfg: dict):
    # Data provider (prices)
    dp_cfg = cfg.get('data_provider', {
        'type': 'YFinanceProvider',
        'parameters': {}
    })
    if dp_cfg.get('type') == 'YFinanceProvider':
        data_provider = YFinanceProvider(**dp_cfg.get('parameters', {}))
    else:
        raise ValueError(f"Unsupported data_provider type: {dp_cfg.get('type')}")

    # Optional factor data provider
    ffp_cfg = cfg.get('factor_data_provider')
    factor_provider = None
    if ffp_cfg:
        if ffp_cfg.get('type') == 'FF5DataProvider':
            factor_provider = FF5DataProvider(**ffp_cfg.get('parameters', {}))
        else:
            raise ValueError(f"Unsupported factor_data_provider type: {ffp_cfg.get('type')}")

    providers = {
        'data_provider': data_provider
    }
    if factor_provider:
        providers['factor_data_provider'] = factor_provider
    return providers


def fetch_price_data(data_provider: YFinanceProvider, symbols: list, start_date: datetime, end_date: datetime, lookback_days: int) -> dict:
    buffer_start = start_date - pd.Timedelta(days=lookback_days)
    logger.info(f"Fetching price data for {len(symbols)} symbols from {buffer_start.date()} to {end_date.date()} (lookback={lookback_days})")
    return data_provider.get_historical_data(symbols=symbols, start_date=buffer_start, end_date=end_date)


def main():
    parser = argparse.ArgumentParser(description='Run modern multi-strategy system with MetaModel')
    parser.add_argument('-c', '--config', required=True, help='Path to system YAML config')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    # System config
    system_cfg = cfg.get('system', {})
    pc_cfg = cfg.get('portfolio_construction', {})
    run_cfg = cfg.get('run', {})
    universe = cfg.get('universe', [])
    if not universe:
        raise ValueError("Config requires a non-empty 'universe' list")

    start_date = datetime.fromisoformat(run_cfg['start_date'])
    end_date = datetime.fromisoformat(run_cfg['end_date'])

    # Providers
    providers = build_providers(cfg)

    # Strategies
    strategies = build_strategies(cfg.get('strategies', []), providers)
    if not strategies:
        raise ValueError("Config requires at least one strategy in 'strategies'")

    # MetaModel
    meta = build_metamodel(cfg.get('metamodel', {}), [s.name for s in strategies])

    # Classifier (used internally by builders where needed)
    classifier_cfg = pc_cfg.get('classifier', {})
    stock_classifier = StockClassifier(classifier_cfg)

    # Modern system config
    modern_cfg = ModernSystemConfig(
        initial_capital=system_cfg.get('initial_capital', 1_000_000),
        enable_short_selling=system_cfg.get('enable_short_selling', False),
        portfolio_construction=pc_cfg
    )

    orchestrator = ModernSystemOrchestrator(
        system_config=modern_cfg,
        strategies=strategies,
        meta_model=meta,
        stock_classifier=stock_classifier,
        custom_configs=cfg.get('custom', {})
    )

    if not orchestrator.initialize_system():
        raise RuntimeError("Failed to initialize system")

    lookback_days = pc_cfg.get('covariance', {}).get('lookback_days', 252)
    price_data = fetch_price_data(
        data_provider=providers['data_provider'],
        symbols=universe,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days
    )

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    for date in dates:
        orchestrator.run_system(date=date, price_data=price_data)

    status = orchestrator.get_system_status()
    logger.info(f"Run completed: {status}")


if __name__ == "__main__":
    main()


