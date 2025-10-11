from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from src.trading_system.data.strategy_data_collector import StrategyDataCollector
from src.trading_system.metamodel.meta_model import MetaModel
from src.trading_system.models.model_persistence import ModelRegistry
from src.trading_system.utils.performance import PerformanceMetrics
from src.trading_system.backtesting.engine import BacktestEngine

try:
    from src.trading_system.config.backtest import BacktestConfig  # type: ignore
except Exception:  # pragma: no cover
    BacktestConfig = object  # type: ignore


@dataclass
class MetaModelRunConfig:
    strategies: List[str]
    start_date: str
    end_date: str
    data_source: str = "backtest_results"  # "backtest_results" or "portfolio_files"
    target_benchmark: Optional[str] = None
    method: str = "ridge"
    alpha: float = 1.0
    model_name: Optional[str] = None
    registry_dir: str = "./models"
    results_dir: str = "./results"
    use_full_backtest: bool = True
    price_data: Optional[Dict[str, pd.DataFrame]] = None
    backtest_config: Optional[BacktestConfig] = None


class MetaModelPipeline:
    def __init__(self, results_dir: str = "./results", registry_dir: str = "./models"):
        self.repo = StrategyDataCollector(results_dir)
        self.registry = ModelRegistry(registry_dir)

    def collect(self, cfg: MetaModelRunConfig) -> Tuple[pd.DataFrame, pd.Series]:
        start = datetime.fromisoformat(cfg.start_date)
        end = datetime.fromisoformat(cfg.end_date)
        if cfg.data_source == "backtest_results":
            return self.repo.collect_from_backtest_results(
                strategy_names=cfg.strategies,
                start_date=start,
                end_date=end,
                target_benchmark=cfg.target_benchmark
            )
        if cfg.data_source == "portfolio_files":
            return self.repo.collect_from_portfolio_files(
                strategy_patterns=cfg.strategies,
                start_date=start,
                end_date=end,
                target_benchmark=cfg.target_benchmark
            )
        raise ValueError(f"Unknown data_source: {cfg.data_source}")

    def train_and_combine(self,
                          strategy_returns: pd.DataFrame,
                          target_returns: pd.Series,
                          cfg: MetaModelRunConfig) -> Tuple[MetaModel, Dict[str, float], pd.Series]:
        model = MetaModel(method=cfg.method, alpha=cfg.alpha)
        model.fit(strategy_returns, target_returns)
        weights = model.strategy_weights
        weight_vec = pd.Series(weights).reindex(strategy_returns.columns).fillna(0.0)
        combined = (strategy_returns * weight_vec).sum(axis=1)
        return model, weights, combined

    def evaluate_light(self,
                       combined_returns: pd.Series,
                       benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        return PerformanceMetrics.calculate_all_metrics(combined_returns, benchmark)

    def evaluate_full_backtest(self,
                               combined_signal: pd.Series,
                               price_data: Dict[str, pd.DataFrame],
                               bt_cfg: BacktestConfig) -> Dict[str, float]:
        symbol = "META_PORT"
        if symbol not in price_data:
            raise ValueError("price_data must include a 'META_PORT' key for representative pricing")

        signal_records = []
        for dt, strength in combined_signal.items():
            signal_records.append({
                'symbol': symbol,
                'strength': float(np.clip(strength, -1, 1)),
                'signal_type': 'weight'
            })

        strategy_signals = {dt.to_pydatetime(): [rec] for dt, rec in zip(combined_signal.index, signal_records)}

        engine = BacktestEngine(bt_cfg)
        results = engine.run_backtest(
            strategy_signals=strategy_signals,
            price_data={symbol: price_data[symbol]}
        )

        out: Dict[str, float] = dict(results.performance_metrics)
        out.update({
            'turnover_rate': results.turnover_rate,
            'final_value': results.final_value
        })
        return out

    def save(self, model: MetaModel, model_name: str, artifacts: Dict) -> str:
        return self.registry.save_model_with_artifacts(
            model=model,
            model_name=model_name,
            artifacts=artifacts,
            tags={'model_type': 'metamodel', 'method': getattr(model, 'method', 'unknown')}
        )



