"""
Simple Performance Evaluator - MVP Version

极简的性能评估器，专注核心功能：
1. 回归指标计算
2. 分类指标计算
3. 基本数据验证

No over-engineering, just essential functionality.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

from ..base.base_model import BaseModel

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    极简性能评估器。

    设计原则：
    - 单一职责：只做性能评估
    - 极简配置：最少参数
    - 一行代码：能用一行解决的绝不用三行
    - 无过度设计：删除所有非必需功能
    """

    @staticmethod
    def evaluate(model: BaseModel, X: pd.DataFrame, y: pd.Series, task_type: str = 'auto') -> Dict[str, float]:
        """
        一行评估模型性能。

        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标数据
            task_type: 任务类型 ('regression', 'classification', 'auto')

        Returns:
            性能指标字典
        """
        if model.status != "trained":
            raise ValueError("Model must be trained before evaluation")

        # 预测
        predictions = model.predict(X)

        # 数据质量检查
        if len(predictions) != len(y):
            raise ValueError(f"Prediction length ({len(predictions)}) != target length ({len(y)})")

        # 确定任务类型
        if task_type == 'auto':
            task_type = 'regression' if y.dtype in [np.float64, np.float32] else 'classification'

        # 计算指标
        if task_type == 'regression':
            metrics = PerformanceEvaluator._regression_metrics(y, predictions)
        else:
            # 分类：需要类别预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if len(proba.shape) == 2 and proba.shape[1] == 2:
                    proba = proba[:, 1]
                    class_pred = (proba > 0.5).astype(int)
                else:
                    class_pred = predictions
            else:
                class_pred = predictions

            metrics = PerformanceEvaluator._classification_metrics(y, class_pred)

        # 添加模型信息
        metrics.update({
            'model_type': model.model_type,
            'samples': len(X),
            'features': len(X.columns)
        })

        return metrics

    @staticmethod
    def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归指标。"""
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) if (y_true != 0).all() else float('inf')
        }

    @staticmethod
    def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算分类指标。"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    @staticmethod
    def quick_regression_eval(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """快速回归评估。"""
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred)
        }

    @staticmethod
    def quick_classification_eval(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """快速分类评估。"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }


# 便捷函数
def evaluate_model(model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """评估模型性能。"""
    return PerformanceEvaluator.evaluate(model, X, y)


def quick_eval(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """快速评估预测结果。"""
    if y_true.dtype in [np.float64, np.float32]:
        return PerformanceEvaluator.quick_regression_eval(y_true, y_pred)
    else:
        return PerformanceEvaluator.quick_classification_eval(y_true, y_pred)


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    # 一行评估
    metrics = quick_eval(y_true, y_pred)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")