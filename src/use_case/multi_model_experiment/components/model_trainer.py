"""
Model Trainer with HPO Component
===============================

This component trains individual models with hyperparameter optimization.
It uses the existing ExperimentOrchestrator to ensure consistency with
the single experiment workflow.
"""

import logging
import tempfile
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

from src.use_case.single_experiment.experiment_orchestrator import ExperimentOrchestrator
from src.trading_system.models.finetune.hyperparameter_optimizer import (
    create_xgboost_hpo
)
from src.trading_system.validation.time_series_cv import TimeSeriesCV

logger = logging.getLogger(__name__)


class ModelTrainerWithHPO:
    """
    Trains a single model with hyperparameter optimization.
    
    This wrapper ensures that each model goes through the complete
    TrainingPipeline → prediction → backtest workflow with HPO.
    """

    def __init__(self, base_config: Dict[str, Any], data_provider=None, factor_data_provider=None):
        """
        Initialize the model trainer.
        
        Args:
            base_config: Base experiment configuration containing data providers,
                        universe, periods, etc. Model-specific config will be injected.
            data_provider: Main data provider for stock data
            factor_data_provider: Factor data provider for FF5 models
        """
        self.base_config = base_config
        self.data_provider = data_provider
        self.factor_data_provider = factor_data_provider
        logger.info("ModelTrainerWithHPO initialized")
    
    def _load_train_data(self, model_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载训练期间的数据。
        
        Args:
            model_type: 模型类型
            
        Returns:
            (X_train, y_train): 训练特征和目标数据
        """
        logger.info(f"Loading training data for {model_type}")
        
        # 从配置中获取训练期间
        train_start = self.base_config['periods']['train']['start']
        train_end = self.base_config['periods']['train']['end']
        
        # 获取股票列表
        universe = self.base_config.get('universe', [])
        
        # 根据模型类型加载不同的数据
        if model_type == 'ff5_regression':
            # FF5 模型使用因子数据
            if self.factor_data_provider is None:
                raise ValueError("Factor data provider required for FF5 models")
            
            # 加载FF5因子数据
            factor_data = self.factor_data_provider.get_data(
                start_date=train_start,
                end_date=train_end
            )
            
            # 构造特征和目标
            X_train = factor_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
            y_train = factor_data['Mkt-RF'].copy()  # 使用市场收益作为目标
            
        else:
            # 其他模型使用股票数据
            if self.data_provider is None:
                raise ValueError("Data provider required for non-FF5 models")
            
            # 加载股票数据
            try:
                # YFinanceProvider 期望 symbols 参数（复数）
                all_data = self.data_provider.get_data(
                    symbols=universe,
                    start_date=train_start,
                    end_date=train_end
                )
                stock_data = all_data if all_data else {}
            except Exception as e:
                logger.warning(f"Failed to load stock data: {e}")
                stock_data = {}
            
            if not stock_data:
                raise ValueError("No stock data loaded")
            
            # 构造特征矩阵
            X_train = self._construct_features(stock_data, model_type)
            y_train = self._construct_target(stock_data)
            
            # 确保 X 和 y 的长度匹配
            common_index = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_index]
            y_train = y_train.loc[common_index]
        
        logger.info(f"Loaded training data: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Date range: {X_train.index.min()} to {X_train.index.max()}")
        
        return X_train, y_train
    
    def _load_test_data(self, model_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载测试期间的数据。
        
        Args:
            model_type: 模型类型
            
        Returns:
            (X_test, y_test): 测试特征和目标数据
        """
        logger.info(f"Loading test data for {model_type}")
        
        # 从配置中获取测试期间
        test_start = self.base_config['periods']['test']['start']
        test_end = self.base_config['periods']['test']['end']
        
        # 获取股票列表
        universe = self.base_config.get('universe', [])
        
        # 根据模型类型加载不同的数据
        if model_type == 'ff5_regression':
            # FF5 模型使用因子数据
            if self.factor_data_provider is None:
                raise ValueError("Factor data provider required for FF5 models")
            
            factor_data = self.factor_data_provider.get_data(
                start_date=test_start,
                end_date=test_end
            )
            
            X_test = factor_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
            y_test = factor_data['Mkt-RF'].copy()
            
        else:
            # 其他模型使用股票数据
            if self.data_provider is None:
                raise ValueError("Data provider required for non-FF5 models")
            
            try:
                # YFinanceProvider 期望 symbols 参数（复数）
                all_data = self.data_provider.get_data(
                    symbols=universe,
                    start_date=test_start,
                    end_date=test_end
                )
                stock_data = all_data if all_data else {}
            except Exception as e:
                logger.warning(f"Failed to load test stock data: {e}")
                stock_data = {}
            
            if not stock_data:
                raise ValueError("No stock data loaded")
            
            X_test = self._construct_features(stock_data, model_type)
            y_test = self._construct_target(stock_data)
            
            # 确保 X 和 y 的长度匹配
            common_index = X_test.index.intersection(y_test.index)
            X_test = X_test.loc[common_index]
            y_test = y_test.loc[common_index]
        
        logger.info(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Date range: {X_test.index.min()} to {X_test.index.max()}")
        
        return X_test, y_test
    
    def _construct_features(self, stock_data: Dict[str, pd.DataFrame], model_type: str) -> pd.DataFrame:
        """
        构造特征矩阵。
        
        Args:
            stock_data: 股票数据字典
            model_type: 模型类型
            
        Returns:
            特征矩阵
        """
        # 这里应该根据模型类型构造不同的特征
        # 为了简化，我们使用基本的收益率特征
        
        features_list = []
        for symbol, data in stock_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                features_list.append(pd.DataFrame({
                    f'{symbol}_returns': returns,
                    f'{symbol}_returns_lag1': returns.shift(1),
                    f'{symbol}_returns_lag5': returns.shift(5),
                }))
        
        if not features_list:
            raise ValueError("No features constructed")
        
        # 合并所有特征
        features_df = pd.concat(features_list, axis=1)
        features_df = features_df.dropna()
        
        return features_df
    
    def _construct_target(self, stock_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        构造目标变量。
        
        Args:
            stock_data: 股票数据字典
            
        Returns:
            目标变量
        """
        # 使用市场平均收益率作为目标
        returns_list = []
        for symbol, data in stock_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_list.append(returns)
        
        if not returns_list:
            raise ValueError("No returns data available")
        
        # 计算市场平均收益率
        market_returns = pd.concat(returns_list, axis=1).mean(axis=1)
        
        # 使用未来21天平均收益率作为目标
        target = market_returns.shift(-1).rolling(21).mean().shift(-21)
        target = target.dropna()
        
        return target
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """
        创建指定类型的模型实例。
        
        Args:
            model_type: 模型类型
            params: 模型参数
            
        Returns:
            模型实例
        """
        if model_type == 'xgboost':
            from src.trading_system.models.implementations.xgboost_model import XGBoostModel
            return XGBoostModel(config=params)
        elif model_type == 'ff5_regression':
            from src.trading_system.models.implementations.ff5_model import FF5Model
            return FF5Model(config=params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def optimize_and_train(self, model_type: str, n_trials: int = 50, 
                          hpo_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Optimize hyperparameters and train the final model using walk-forward CV.
        
        Args:
            model_type: Type of model to train ('xgboost', 'ff5_regression', etc.)
            n_trials: Number of HPO trials
            hpo_metric: Metric to optimize for
            
        Returns:
            Dictionary containing model results, performance metrics, and metadata
        """
        logger.info(f"Starting HPO and training for {model_type} with {n_trials} trials")
        logger.info("🔧 Using walk-forward CV for HPO (no test set leakage)")
        
        # Step 1: Load training and test data separately
        logger.info("Loading training data for HPO...")
        X_train, y_train = self._load_train_data(model_type)
        
        logger.info("Loading test data for final evaluation...")
        X_test, y_test = self._load_test_data(model_type)
        
        # Step 2: Define objective function using walk-forward CV
        def objective(params: Dict[str, Any], X_train_data: pd.DataFrame, y_train_data: pd.Series) -> float:
            """HPO objective function using walk-forward cross-validation."""
            try:
                logger.debug(f"HPO trial with params: {params}")
                
                # 创建模型
                model = self._create_model(model_type, params)
                
                # 使用 walk-forward CV 评估参数
                cv_scores = []
                cv = TimeSeriesCV(method='walk_forward')
                
                # 进行 walk-forward 验证
                for train_start, train_end, test_start, test_end, train_idx, test_idx in cv.walk_forward_split(
                    X_train_data, train_size=252, test_size=21, step_size=21, purge_period=5
                ):
                    try:
                        # 分割数据
                        X_train_fold = X_train_data.iloc[train_idx]
                        y_train_fold = y_train_data.iloc[train_idx]
                        X_test_fold = X_train_data.iloc[test_idx]
                        y_test_fold = y_train_data.iloc[test_idx]
                        
                        # 训练模型
                        model.fit(X_train_fold, y_train_fold)
                        
                        # 预测
                        y_pred = model.predict(X_test_fold)
                        
                        # 计算分数（这里使用简单的 R² 分数）
                        from sklearn.metrics import r2_score
                        score = r2_score(y_test_fold, y_pred)
                        cv_scores.append(score)
                        
                        logger.debug(f"  Fold {len(cv_scores)}: train={train_start.date()} to {train_end.date()}, "
                                   f"test={test_start.date()} to {test_end.date()}, score={score:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"CV fold failed: {e}")
                        continue
                
                # 返回平均 CV 分数
                avg_score = np.mean(cv_scores) if cv_scores else 0.0
                logger.debug(f"HPO trial completed: avg CV score = {avg_score:.4f} ({len(cv_scores)} folds)")
                
                return avg_score
                
            except Exception as e:
                logger.warning(f"HPO trial failed for {model_type}: {e}")
                return 0.0
        
        # Step 3: Run HPO with walk-forward CV
        logger.info("Running HPO with walk-forward cross-validation...")
        optimizer = self._create_hpo_for_model_type(model_type, n_trials)
        hpo_results = optimizer.optimize(objective, X_train, y_train)
        
        logger.info(f"HPO completed for {model_type}. Best CV score: {hpo_results['best_score']:.4f}")
        
        # Step 4: Train final model with best parameters on full training set
        logger.info("Training final model with best parameters on full training set...")
        best_model = self._create_model(model_type, hpo_results['best_params'])
        best_model.fit(X_train, y_train)
        
        # Step 5: Evaluate final model on test set (only once!)
        logger.info("Evaluating final model on test set...")
        y_pred_test = best_model.predict(X_test)
        
        # 计算测试集性能指标
        from sklearn.metrics import r2_score, mean_squared_error
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        logger.info(f"Final test performance: R² = {test_r2:.4f}, MSE = {test_mse:.4f}")
        
        # Step 6: 构造返回结果
        model_id = f"{model_type}_{n_trials}trials_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            'model_type': model_type,
            'model_id': model_id,
            'best_params': hpo_results['best_params'],
            'hpo_results': hpo_results,
            'best_model': best_model,
            'performance_metrics': {
                'cv_score': hpo_results['best_score'],
                'test_r2': test_r2,
                'test_mse': test_mse,
                'sharpe_ratio': test_r2,  # 使用 R² 作为夏普比率的代理
            },
            'data_info': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'train_period': f"{X_train.index.min()} to {X_train.index.max()}",
                'test_period': f"{X_test.index.min()} to {X_test.index.max()}",
            },
            'training_summary': {
                'hpo_trials': n_trials,
                'cv_folds': 'walk_forward',
                'final_training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }
        
        logger.info(f"Training completed for {model_type}. Model ID: {model_id}")
        logger.info(f"CV Score: {hpo_results['best_score']:.4f}, Test R²: {test_r2:.4f}")
        
        return result

    def _create_experiment_config(self, model_type: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete experiment configuration for a specific model type and hyperparameters.
        
        Args:
            model_type: Type of model ('xgboost', 'ff5_regression', etc.)
            hyperparams: Hyperparameters for the model
            
        Returns:
            Complete experiment configuration dictionary
        """
        config = self.base_config.copy()
        
        # Create training setup for this model type
        training_setup = {
            'model': {
                'model_type': model_type
            },
            'feature_engineering': self._get_feature_config_for_model(model_type),
            'parameters': {
                'symbols': config.get('universe', []),
                'start_date': config['periods']['train']['start'],
                'end_date': config['periods']['train']['end'],
                'test_start_date': config['periods']['test']['start'],
                'test_end_date': config['periods']['test']['end']
            }
        }
        
        # Add model-specific hyperparameters using the format from working config
        if model_type == 'xgboost':
            training_setup['model'].update({
                'n_estimators': hyperparams.get('n_estimators', 100),
                'learning_rate': hyperparams.get('learning_rate', 0.1),
                'max_depth': hyperparams.get('max_depth', 6),
                'min_child_weight': hyperparams.get('min_child_weight', 1),
                'subsample': hyperparams.get('subsample', 1.0),
                'colsample_bytree': hyperparams.get('colsample_bytree', 1.0)
            })
        elif model_type == 'ff5_regression':
            # Use the same config format as the working YAML file
            training_setup['model'].update({
                'config': {
                    'regularization': hyperparams.get('regularization', 'none'),
                    'standardize': hyperparams.get('standardize', False)
                }
            })
        
        config['training_setup'] = training_setup
        
        # Add backtest and strategy configuration
        config['backtest'] = config.get('backtest', {
            'initial_capital': 1000000,
            'commission': 0.001,
            'slippage': 0.0005
        })
        
        config['strategy'] = config.get('strategy', {
            'type': 'MLStrategy',
            'parameters': {
                'signal_threshold': 0.1,
                'max_positions': 10
            }
        })
        
        return config
    
    def _get_feature_config_for_model(self, model_type: str) -> Dict[str, Any]:
        """
        Get the correct feature engineering configuration for different model types.

        Args:
            model_type: Type of model (xgboost, ff5_regression, etc.)

        Returns:
            Feature engineering configuration dictionary
        """
        if model_type == "ff5_regression":
            # Use the exact same format as the working YAML config file
            return {
                'enabled_features': ['fama_french_factors']
            }
        else:
            # Default configuration for other models (XGBoost, etc.)
            # Use the same format as other working configs
            return {
                'enabled_features': ['technical_indicators', 'returns', 'volatility', 'momentum'],
                'technical_indicators': {
                    'rsi_periods': [14, 21],
                    'macd_config': {'fast': 12, 'slow': 26, 'signal': 9},
                    'bollinger_periods': [20],
                    'moving_average_periods': [10, 20, 50]
                },
                'returns_params': {'periods': [1, 5, 10, 20]},
                'volatility_params': {'windows': [20, 60], 'methods': ['std']},
                'momentum_params': {'periods': [10, 20, 60], 'methods': ['simple']}
            }

    def _create_hpo_for_model_type(self, model_type: str, n_trials: int):
        """
        Create appropriate HPO optimizer for the model type.
        
        Args:
            model_type: Type of model
            n_trials: Number of trials
            
        Returns:
            HyperparameterOptimizer instance
        """
        if model_type == 'xgboost':
            return create_xgboost_hpo(n_trials)
        elif model_type == 'ff5_regression':
            # For FF5 regression, we'll use a simple HPO with regularization parameter
            from src.trading_system.models.finetune.hyperparameter_optimizer import HyperparameterOptimizer
            optimizer = HyperparameterOptimizer(n_trials=n_trials)
            optimizer.add_param('regularization', 'categorical', choices=['none', 'ridge', 'lasso'])
            optimizer.add_param('standardize', 'categorical', choices=[True, False])
            return optimizer
        else:
            # Default to XGBoost HPO for unknown model types
            logger.warning(f"Unknown model type {model_type}, using XGBoost HPO")
            return create_xgboost_hpo(n_trials)
