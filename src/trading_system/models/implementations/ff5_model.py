"""
Fama-French 5-Factor Regression Model

Clean implementation of BaseModel that focuses solely on:
- Factor beta estimation using linear regression
- Factor-implied return prediction
- Beta coefficients as feature importance

This model does NOT handle:
- Data loading (handled by DataProvider)
- Training workflow (handled by Trainer)
- Performance evaluation (handled by PerformanceEvaluator)
- Model persistence (handled by ModelRegistry)
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class FF5RegressionModel(BaseModel):
    """
    Fama-French 5-Factor regression model.

    Estimates factor betas for individual stocks and calculates
    factor-implied expected returns using the Fama-French 5-factor model.

    Model Specification:
    R_stock = α + β_MKT * R_MKT + β_SMB * R_SMB + β_HML * R_HML +
              β_RMW * R_RMW + β_CMA * R_CMA + ε

    Where factors are:
    - MKT: Market excess return
    - SMB: Small Minus Big (size)
    - HML: High Minus Low (value)
    - RMW: Robust Minus Weak (profitability)
    - CMA: Conservative Minus Aggressive (investment)
    """

    def __init__(self, model_type: str = "ff5_regression", config: Optional[Dict[str, Any]] = None):
        """
        Initialize FF5 regression model.

        Args:
            model_type: Model identifier
            config: Configuration dictionary with:
                - regularization: 'none' or 'ridge' (default: 'none')
                - alpha: Regularization strength for ridge (default: 1.0)
                - standardize: Whether to standardize factors (default: False)
        """
        super().__init__(model_type, config)

        # Model configuration
        self.regularization = self.config.get('regularization', 'ridge')  # Use ridge by default
        self.alpha = self.config.get('alpha', 1.0)
        self.standardize = self.config.get('standardize', False)

        # Storage for per-symbol betas and alphas
        self.betas = {}  # {symbol: np.ndarray(5,)}
        self.alphas = {}  # {symbol: float}

        # Initialize model components
        if self.regularization == 'ridge':
            # Ensure alpha is always positive
            positive_alpha = max(abs(float(self.alpha)), 1e-6)  # Ensure positive and not zero
            self._model = Ridge(alpha=positive_alpha)
        else:
            self._model = LinearRegression()

        if self.standardize:
            self._scaler = StandardScaler()
        else:
            self._scaler = None

        # Expected factor columns for validation
        self._expected_features = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

        logger.info(f"Initialized FF5 regression model with {self.regularization} regularization")
        logger.info("Model will compute betas for each individual stock")

    @property
    def supports_batch_prediction(self) -> bool:
        """
        FF5 supports batch prediction.
        
        Why: Although trained via time-series regression (each stock gets betas),
             prediction uses shared factor values at each date.
             
             Prediction: returns = betas @ factors
             - betas: (n_stocks, n_factors) - different per stock
             - factors: (n_factors,) - SHARED across all stocks
             - returns: (n_stocks,) - all stocks predicted at once
        """
        return True

    @property  
    def prediction_mode(self) -> str:
        """FF5 uses batch prediction with shared factors."""
        return 'batch'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FF5RegressionModel':
        """
        Fit the factor model to estimate betas for each stock.

        This method implements the "Betas as model parameters" architecture.
        Each symbol gets its own set of betas computed independently.

        Args:
            X: Factor returns DataFrame with MultiIndex (symbol, date) and columns ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            y: Stock excess returns Series with MultiIndex (symbol, date) aligned with X

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data validation fails
        """
        try:
            # Validate input data
            self.validate_data(X, y)

            # Check for required factor columns
            missing_factors = set(self._expected_features) - set(X.columns)
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            # Verify MultiIndex structure
            if not isinstance(X.index, pd.MultiIndex) or not isinstance(y.index, pd.MultiIndex):
                raise ValueError("X and y must have MultiIndex with 'symbol' and 'date' levels")

            if not all(level in X.index.names for level in ['symbol', 'date']):
                raise ValueError("X must have MultiIndex with 'symbol' and 'date' levels")

            if not all(level in y.index.names for level in ['symbol', 'date']):
                raise ValueError("y must have MultiIndex with 'symbol' and 'date' levels")

            # Get unique symbols
            if 'symbol' in X.index.names:
                symbols = X.index.get_level_values('symbol').unique()
            else:
                raise ValueError("X MultiIndex must have 'symbol' level")

            logger.info(f"Fitting FF5 model for {len(symbols)} symbols: {list(symbols)}")

            # Clear previous betas
            self.betas = {}
            self.alphas = {}

            total_samples = 0
            successful_symbols = 0

            # Fit a separate regression for each symbol
            for symbol in symbols:
                try:
                    # Extract data for this symbol
                    symbol_X = X.xs(symbol, level='symbol')
                    symbol_y = y.xs(symbol, level='symbol')

                    # Align data by index (date)
                    aligned_data = pd.concat([symbol_y, symbol_X[self._expected_features]], axis=1, join='inner').dropna()

                    if len(aligned_data) < 50:  # Need at least 50 observations
                        logger.warning(f"Insufficient data for {symbol}: {len(aligned_data)} observations, skipping")
                        continue

                    symbol_y_clean = aligned_data.iloc[:, 0]
                    symbol_X_clean = aligned_data.iloc[:, 1:]

                    # Standardize factors if requested (per symbol)
                    if self.standardize:
                        symbol_scaler = StandardScaler()
                        symbol_X_clean = pd.DataFrame(
                            symbol_scaler.fit_transform(symbol_X_clean),
                            index=symbol_X_clean.index,
                            columns=symbol_X_clean.columns
                        )
                    else:
                        symbol_scaler = None

                    # Create and fit regression model for this symbol
                    if self.regularization == 'ridge':
                        symbol_model = Ridge(alpha=max(abs(float(self.alpha)), 1e-6))
                    else:
                        symbol_model = LinearRegression()

                    symbol_model.fit(symbol_X_clean, symbol_y_clean)

                    # Store betas and alpha for this symbol
                    self.betas[symbol] = symbol_model.coef_
                    self.alphas[symbol] = symbol_model.intercept_

                    total_samples += len(symbol_y_clean)
                    successful_symbols += 1

                    logger.debug(f"{symbol}: fitted with {len(symbol_y_clean)} samples, "
                                f"alpha={symbol_model.intercept_:.4f}, "
                                f"betas=[{symbol_model.coef_[0]:.3f}, {symbol_model.coef_[1]:.3f}, {symbol_model.coef_[2]:.3f}, {symbol_model.coef_[3]:.3f}, {symbol_model.coef_[4]:.3f}]")

                except Exception as e:
                    logger.error(f"Failed to fit {symbol}: {e}")
                    continue

            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = total_samples
            self.metadata.features = list(self._expected_features)

            # Store summary statistics in hyperparameters
            if self.betas:
                beta_arrays = np.array(list(self.betas.values()))
                mean_betas = dict(zip(self._expected_features, np.mean(beta_arrays, axis=0)))
                std_betas = dict(zip(self._expected_features, np.std(beta_arrays, axis=0)))

                self.metadata.hyperparameters.update({
                    'symbols_trained': successful_symbols,
                    'total_samples': total_samples,
                    'mean_betas': mean_betas,
                    'std_betas': std_betas,
                    'regularization': self.regularization,
                    'regularization_alpha': self.alpha,
                    'standardize': self.standardize
                })

                logger.info(f"Successfully fitted FF5 model for {successful_symbols}/{len(symbols)} symbols")
                logger.info(f"Total samples used: {total_samples}")
                logger.info(f"Mean betas: {mean_betas}")
                logger.info(f"Beta std dev: {std_betas}")

                # Mark model as trained for registration purposes
                self.is_trained = True

            else:
                logger.error("No symbols were successfully fitted")
                self.status = ModelStatus.FAILED
                raise ValueError("Failed to fit any symbols")

            return self

        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit FF5 regression model: {e}")
            raise

    def predict(self, X: pd.DataFrame, symbols: Optional[List[str]] = None) -> np.ndarray:
        """
        统一预测接口 - 自动检测场景
        
        支持两种场景：
        1. 单日横截面预测（回测）：X shape (1, 5) 或 (n_symbols, 5)
        2. 多日时间序列预测（训练验证）：X shape (n_observations, 5) with MultiIndex
        """
        logger.info(f"🔍 FF5 Prediction - Input: shape={X.shape}, index_type={type(X.index)}")
        
        if self.status != ModelStatus.TRAINED or not self.betas:
            raise ValueError("Model must be trained before predictions")
        
        # 验证因子列
        missing_factors = set(self._expected_features) - set(X.columns)
        if missing_factors:
            raise ValueError(f"Missing factors: {missing_factors}")
        
        # ====================================================
        # 场景检测与路由
        # ====================================================
        
        # 场景 1: MultiIndex - 训练/验证场景
        if isinstance(X.index, pd.MultiIndex):
            if 'symbol' in X.index.names and 'date' in X.index.names:
                logger.info("📊 Scenario: Training/Validation (time-series format)")
                return self._predict_time_series(X, symbols)
        
        # 场景 2: 单日横截面 - 回测场景
        logger.info("🎯 Scenario: Backtesting (cross-sectional format)")
        result = self._predict_batch(X, symbols)
        
        # 转换为 ndarray
        if isinstance(result, pd.Series):
            return result.values
        elif isinstance(result, pd.DataFrame):
            return result.values.flatten()
        else:
            return result

    def _predict_batch(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> Union[pd.Series, pd.DataFrame]:
        """
        批量预测逻辑 - 始终为当前时间点生成横截面预测
        
        ⚠️ 重要：此方法只处理单个时间点（横截面）的预测
        多时间点的循环由 BaseStrategy._get_predictions() 处理
        
        Args:
            X: 因子数据，应该只包含**单个日期**的因子值
            - 对于 FF5：所有股票共享相同的因子值 → shape (1, 5) 或 (5,)
            symbols: 要预测的股票列表
        
        Returns:
            pd.Series: 当前日期的横截面预测，index为股票symbols
        """
        logger.info("Using new batch prediction logic")
        
        # 确定要预测的股票
        if symbols is None:
            symbols = list(self.betas.keys())
        
        valid_symbols = [s for s in symbols if s in self.betas]
        missing_symbols = [s for s in symbols if s not in self.betas]
        
        if missing_symbols:
            logger.warning(f"⚠️  Missing symbols: {missing_symbols}")
        
        # ⚠️ 关键修复：提取因子值（应该只有一行或一个向量）
        factor_values = X[self._expected_features].values
        
        # 验证输入维度
        if factor_values.ndim == 1:
            # 已经是一维向量 (5,)
            factor_vector = factor_values
        elif factor_values.shape[0] == 1:
            # 单行矩阵 (1, 5) → 提取为向量
            factor_vector = factor_values[0]
        else:
            # ⚠️ 错误：不应该有多行
            logger.error(f"❌ Expected single date factors, got shape {factor_values.shape}")
            logger.error(f"❌ This indicates incorrect data preparation in upstream code")
            # 降级处理：取第一行
            factor_vector = factor_values[0]
            logger.warning(f"⚠️  Using first row as fallback: {factor_vector}")
        
        logger.info(f"Factor vector shape: {factor_vector.shape}")  # 应该是 (5,)
        
        # 批量预测所有股票（向量化）
        predictions = {}
        for symbol in valid_symbols:
            symbol_betas = self.betas[symbol]  # shape: (5,)
            symbol_alpha = self.alphas[symbol]
            
            # 向量化预测：r = α + β @ f
            symbol_prediction = symbol_alpha + factor_vector @ symbol_betas  # scalar
            predictions[symbol] = symbol_prediction
        
        # 处理缺失股票（零预测）
        for symbol in missing_symbols:
            predictions[symbol] = 0.0
        
        # 始终返回 Series（横截面预测）
        result = pd.Series(predictions, name='ff5_prediction')
        logger.info(f"Generated cross-sectional predictions for {len(result)} symbols")
        
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get mean factor betas as feature importance scores.

        Returns:
            Dictionary mapping factor names to their mean beta coefficients across all symbols
        """
        if self.status != ModelStatus.TRAINED or not self.betas:
            return {}

        # Calculate mean betas across all symbols
        beta_arrays = np.array(list(self.betas.values()))
        mean_betas = dict(zip(self._expected_features, np.mean(beta_arrays, axis=0)))

        return mean_betas

    def get_factor_exposures(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get factor betas (exposures) for a specific symbol or mean across all symbols.

        Args:
            symbol: Optional symbol name. If None, returns mean betas.

        Returns:
            Dictionary of factor betas
        """
        if self.status != ModelStatus.TRAINED or not self.betas:
            return {}

        if symbol and symbol in self.betas:
            # Return betas for specific symbol
            return dict(zip(self._expected_features, self.betas[symbol]))
        else:
            # Return mean betas across all symbols
            return self.get_feature_importance()

    def get_alpha(self, symbol: Optional[str] = None) -> float:
        """
        Get the model's alpha (intercept) for a specific symbol or mean across all symbols.

        Args:
            symbol: Optional symbol name. If None, returns mean alpha.

        Returns:
            Alpha value
        """
        if self.status != ModelStatus.TRAINED or not self.alphas:
            return 0.0

        if symbol and symbol in self.alphas:
            return self.alphas[symbol]
        else:
            # Return mean alpha across all symbols
            return np.mean(list(self.alphas.values()))

    def get_symbol_betas(self) -> Dict[str, np.ndarray]:
        """
        Get all symbol betas.

        Returns:
            Dictionary mapping symbol names to their beta coefficient arrays
        """
        return self.betas.copy()

    def get_symbol_alphas(self) -> Dict[str, float]:
        """
        Get all symbol alphas.

        Returns:
            Dictionary mapping symbol names to their alpha values
        """
        return self.alphas.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        return {
            'model_type': self.model_type,
            'status': self.status,
            'regularization': self.regularization,
            'regularization_alpha': self.alpha,
            'standardize': self.standardize,
            'factors': self._expected_features,
            'training_samples': self.metadata.training_samples,
            'factor_betas': self.get_feature_importance(),
            'alpha': self.get_alpha()
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the FF5 regression model to disk.

        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Create model dictionary with all components
        model_dict = {
            'model': self._model,
            'scaler': self._scaler,
            'expected_features': self._expected_features,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'standardize': self.standardize
        }

        # Save the model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"FF5RegressionModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FF5RegressionModel':
        """
        Load a FF5 regression model from disk.

        Args:
            path: Directory path where model is saved

        Returns:
            Loaded FF5RegressionModel instance
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")

        # Load metadata
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)

        # Load config
        config_path = path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Load the model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Create instance with proper configuration
        instance = cls(config=config)
        instance.metadata = metadata
        instance.status = ModelStatus.TRAINED

        # Restore model components and attributes
        instance._model = model_dict['model']
        instance._scaler = model_dict['scaler']
        instance._expected_features = model_dict['expected_features']
        instance.regularization = model_dict['regularization']
        instance.alpha = model_dict['alpha']
        instance.standardize = model_dict['standardize']

        logger.info(f"FF5RegressionModel loaded from {path}")
        return instance

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Get hyperparameter search space for FF5 model optimization (MVP - simple dict).

        Returns:
            Simple dictionary defining parameter ranges for optimization
        """
        return {
            'regularization': ['none', 'ridge'],
            'alpha': (0.01, 10.0),
            'standardize': [True, False]
        }

    def get_tunable_hyperparameters(self) -> List[str]:
        """
        Get list of tunable hyperparameter names.

        Returns:
            List of hyperparameter names that can be optimized
        """
        return [
            'regularization',
            'alpha',
            'standardize'
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model metadata and current configuration
        """
        return {
            'model_type': self.model_type,
            'status': self.status.value,
            'hyperparameters': self.get_model_params(),
            'expected_features': self._expected_features,
            'tunable_parameters': self.get_tunable_hyperparameters(),
            'capabilities': {
                'factor_modeling': True,
                'beta_estimation': True,
                'regularization': self.regularization != 'none',
                'feature_standardization': self.standardize,
                'interpretable': True
            }
        }