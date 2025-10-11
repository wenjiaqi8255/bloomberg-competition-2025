"""
Simple Hyperparameter Optimizer - MVP Version

极简的超参数优化器，专注核心功能：
1. TPE优化
2. 基本试验管理
3. 结果保存

No over-engineering, just essential functionality.
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

# Optuna imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Minimal HPO configuration."""
    n_trials: int = 50
    metric: str = "r2"
    direction: str = "maximize"
    save_dir: str = "./hpo_results"


class HyperparameterOptimizer:
    """
    极简超参数优化器，专注核心功能。

    设计原则：
    - 单一职责：只做TPE优化
    - 极简配置：最少参数
    - 一行代码：能用一行解决的绝不用三行
    - 无过度设计：删除所有非必需功能
    """

    def __init__(self, n_trials: int = 50, metric: str = "r2", direction: str = "maximize",
                 model_train_func: Optional[Callable] = None):
        """
        极简初始化。

        Args:
            n_trials: 试验次数
            metric: 优化指标
            direction: 优化方向
            model_train_func: 模型训练函数，用于在最佳参数上重新训练模型
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Install optuna: pip install optuna")

        self.config = HPOConfig(n_trials=n_trials, metric=metric, direction=direction)
        self.study = None
        self.search_spaces = {}
        self.model_train_func = model_train_func
        self.best_model = None
        self.best_params = None
        self.best_score = None

        # 创建结果目录
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"HPO initialized: {n_trials} trials")

    def add_param(self, name: str, param_type: str, low=None, high=None, choices=None) -> 'HyperparameterOptimizer':
        """
        链式添加搜索参数。

        Args:
            name: 参数名
            param_type: 参数类型 (float/int/categorical/log_float)
            low, high: 数值范围
            choices: 分类选项

        Returns:
            self，支持链式调用
        """
        self.search_spaces[name] = {
            'type': param_type,
            'low': low,
            'high': high,
            'choices': choices
        }
        return self

    def optimize(self, eval_func: Callable[[Dict[str, Any], Any, Any], float], 
                 X_train: Any = None, y_train: Any = None) -> Dict[str, Any]:
        """
        执行优化。

        Args:
            eval_func: 评估函数，接收(参数字典, X_train, y_train)，返回CV分数
            X_train: 训练特征数据
            y_train: 训练目标数据

        Returns:
            优化结果字典，包含训练好的最佳模型
        """
        if not self.search_spaces:
            raise ValueError("No search spaces defined")

        # 创建研究
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=optuna.samplers.TPESampler()
        )

        # 优化 - 传入训练数据
        self.study.optimize(
            lambda trial: self._objective(trial, eval_func, X_train, y_train),
            n_trials=self.config.n_trials
        )

        # 保存最佳参数和分数
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        # 🔧 关键修复：训练并保存最佳模型
        logger.info(f"Training best model with params: {self.best_params}")
        if self.model_train_func and X_train is not None and y_train is not None:
            try:
                # 用最佳参数在整个训练集上训练最终模型
                self.best_model = self.model_train_func(self.best_params, X_train, y_train)
                logger.info("✅ Best model trained and saved successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train best model: {e}")
                self.best_model = None
        else:
            logger.warning("⚠️ No model training function or training data provided, best_model will be None")

        # 返回结果
        result = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,  # 🔧 添加最佳模型
            'n_trials': len(self.study.trials),
            'study_name': f"hpo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        }

        # 保存结果
        self._save_results(result)

        logger.info(f"Optimization done. Best {self.config.metric}: {result['best_score']:.4f}")
        if result['best_model'] is not None:
            logger.info("✅ Optimization includes trained model")
        else:
            logger.warning("⚠️ Optimization result missing trained model")
        return result

    def _objective(self, trial: optuna.Trial, eval_func: Callable, X_train: Any, y_train: Any) -> float:
        """单行目标函数实现。"""
        # 建议参数
        params = {}
        for name, space in self.search_spaces.items():
            if space['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, space['choices'])
            elif space['type'] == 'int':
                params[name] = trial.suggest_int(name, int(space['low']), int(space['high']))
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(name, space['low'], space['high'])
            elif space['type'] == 'log_float':
                params[name] = trial.suggest_loguniform(name, space['low'], space['high'])

        # 评估并返回 - 传入训练数据
        try:
            return eval_func(params, X_train, y_train)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('-inf') if self.config.direction == "maximize" else float('inf')

    def _save_results(self, result: Dict[str, Any]) -> None:
        """极简结果保存。"""
        results_dir = Path(self.config.save_dir)

        # 保存最佳参数
        with open(results_dir / f"{result['study_name']}_best.txt", 'w') as f:
            f.write(f"Best {self.config.metric}: {result['best_score']:.4f}\n")
            f.write("Best parameters:\n")
            for k, v in result['best_params'].items():
                f.write(f"  {k}: {v}\n")

        logger.info(f"Results saved to {results_dir}")


# 便捷函数 - 一步创建常用优化器
def create_xgboost_hpo(n_trials: int = 50) -> HyperparameterOptimizer:
    """
    创建XGBoost超参数优化器。

    Args:
        n_trials: 试验次数
    """
    def xgboost_train_func(best_params: Dict[str, Any], X_train: Any = None, y_train: Any = None) -> Any:
        """XGBoost模型训练函数。"""
        try:
            from ...models.implementations.xgboost_model import XGBoostModel

            if X_train is None or y_train is None:
                logger.warning("⚠️ No training data provided for XGBoost model training")
                return None

            # 创建并训练最佳模型
            model = XGBoostModel(config=best_params)
            
            if hasattr(X_train, 'empty') and hasattr(y_train, 'empty'):
                # Pandas DataFrame/Series
                if X_train.empty or y_train.empty:
                    logger.error("❌ Empty training data provided")
                    return None
            elif len(X_train) == 0 or len(y_train) == 0:
                # NumPy arrays or lists
                logger.error("❌ Empty training data provided")
                return None

            model.fit(X_train, y_train)
            logger.info(f"✅ XGBoost model trained with shape: {X_train.shape if hasattr(X_train, 'shape') else len(X_train)}")
            return model

        except Exception as e:
            logger.error(f"❌ XGBoost model training failed: {e}")
            return None

    return (HyperparameterOptimizer(n_trials, model_train_func=xgboost_train_func)
            .add_param('n_estimators', 'int', 50, 500)
            .add_param('max_depth', 'int', 3, 12)
            .add_param('learning_rate', 'log_float', 0.01, 0.3)
            .add_param('subsample', 'float', 0.6, 1.0))


def create_metamodel_hpo(n_trials: int = 50) -> HyperparameterOptimizer:
    """创建MetaModel超参数优化器。"""
    return (HyperparameterOptimizer(n_trials)
            .add_param('method', 'categorical', choices=['equal', 'lasso', 'ridge'])
            .add_param('alpha', 'log_float', 0.01, 10.0)
            .add_param('min_weight', 'float', 0.0, 0.1)
            .add_param('max_weight', 'float', 0.3, 1.0))


# 使用示例
if __name__ == "__main__":
    # 极简使用方式
    def dummy_eval(params):
        return np.random.normal(0.5, 0.1)  # 模拟评估

    # 一行创建优化器
    optimizer = create_xgboost_hpo(20)

    # 一行执行优化
    result = optimizer.optimize(dummy_eval)

    print(f"Best score: {result['best_score']:.4f}")
    print(f"Best params: {result['best_params']}")