"""
Fama-French 3-Factor Regression Model

Lightweight implementation mirroring FF5 model, restricted to three factors:
- MKT (market excess return)
- SMB (size)
- HML (value)

Reuses BaseModel lifecycle. Train per-symbol linear regressions to estimate betas
and predict expected returns from factor realizations.
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


class FF3RegressionModel(BaseModel):
    """
    Fama-French 3-Factor regression model.

    R_stock = α + β_MKT * R_MKT + β_SMB * R_SMB + β_HML * R_HML + ε
    """

    def __init__(self, model_type: str = "ff3_regression", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, config)

        # Model configuration
        self.regularization = self.config.get('regularization', 'ridge')
        self.alpha = self.config.get('alpha', 1.0)
        self.standardize = self.config.get('standardize', False)

        # Storage for per-symbol betas and alphas
        self.betas: Dict[str, np.ndarray] = {}
        self.alphas: Dict[str, float] = {}

        # Initialize model components
        if self.regularization == 'ridge':
            positive_alpha = max(abs(float(self.alpha)), 1e-6)
            self._model = Ridge(alpha=positive_alpha)
        else:
            self._model = LinearRegression()

        self._scaler = StandardScaler() if self.standardize else None

        # Expected factor columns
        self._expected_features = ['MKT', 'SMB', 'HML']

        logger.info("Initialized FF3 regression model")

    @property
    def supports_batch_prediction(self) -> bool:
        return True

    @property
    def prediction_mode(self) -> str:
        return 'batch'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FF3RegressionModel':
        try:
            self.validate_data(X, y)

            missing_factors = set(self._expected_features) - set(X.columns)
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            if not isinstance(X.index, pd.MultiIndex) or not isinstance(y.index, pd.MultiIndex):
                raise ValueError("X and y must have MultiIndex with 'symbol' and 'date' levels")

            if not all(level in X.index.names for level in ['symbol', 'date']):
                raise ValueError("X must have MultiIndex with 'symbol' and 'date' levels")

            if not all(level in y.index.names for level in ['symbol', 'date']):
                raise ValueError("y must have MultiIndex with 'symbol' and 'date' levels")

            if 'symbol' not in X.index.names:
                raise ValueError("X MultiIndex must have 'symbol' level")

            symbols = X.index.get_level_values('symbol').unique()

            logger.info(f"Fitting FF3 model for {len(symbols)} symbols: {list(symbols)}")

            self.betas = {}
            self.alphas = {}

            total_samples = 0
            successful_symbols = 0

            for symbol in symbols:
                try:
                    symbol_X = X.xs(symbol, level='symbol')
                    symbol_y = y.xs(symbol, level='symbol')

                    aligned = pd.concat([symbol_y, symbol_X[self._expected_features]], axis=1, join='inner').dropna()
                    if len(aligned) < 50:
                        logger.warning(f"Insufficient data for {symbol}: {len(aligned)} observations, skipping")
                        continue

                    y_clean = aligned.iloc[:, 0]
                    X_clean = aligned.iloc[:, 1:]

                    if self.standardize:
                        local_scaler = StandardScaler()
                        X_clean = pd.DataFrame(
                            local_scaler.fit_transform(X_clean),
                            index=X_clean.index,
                            columns=X_clean.columns
                        )

                    model = Ridge(alpha=max(abs(float(self.alpha)), 1e-6)) if self.regularization == 'ridge' else LinearRegression()
                    model.fit(X_clean, y_clean)

                    self.betas[symbol] = model.coef_
                    self.alphas[symbol] = model.intercept_

                    total_samples += len(y_clean)
                    successful_symbols += 1
                except Exception as e:
                    logger.error(f"Failed to fit {symbol}: {e}")
                    continue

            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = total_samples
            self.metadata.features = list(self._expected_features)

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

                self.is_trained = True
            else:
                self.status = ModelStatus.FAILED
                raise ValueError("Failed to fit any symbols")

            return self
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit FF3 regression model: {e}")
            raise

    def predict(self, X: pd.DataFrame, symbols: Optional[List[str]] = None) -> np.ndarray:
        logger.info(f"FF3 Prediction - Input: shape={X.shape}, index_type={type(X.index)}")

        if self.status != ModelStatus.TRAINED or not self.betas:
            raise ValueError("Model must be trained before predictions")

        missing_factors = set(self._expected_features) - set(X.columns)
        if missing_factors:
            raise ValueError(f"Missing factors: {missing_factors}")

        if isinstance(X.index, pd.MultiIndex) and 'symbol' in X.index.names and 'date' in X.index.names:
            return self._predict_time_series(X, symbols)

        result = self._predict_batch(X, symbols)
        if isinstance(result, pd.Series):
            return result.values
        elif isinstance(result, pd.DataFrame):
            return result.values.flatten()
        else:
            return result

    def _predict_time_series(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> np.ndarray:
        predictions: List[float] = []
        for (symbol, date), row in X.iterrows():
            if symbol in self.betas:
                factor_values = row[self._expected_features].values
                beta = self.betas[symbol]
                alpha = self.alphas[symbol]
                predictions.append(alpha + np.dot(beta, factor_values))
            else:
                predictions.append(0.0)
        return np.array(predictions)

    def _predict_batch(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> Union[pd.Series, pd.DataFrame]:
        if symbols is None:
            symbols = list(self.betas.keys())

        valid_symbols = [s for s in symbols if s in self.betas]
        missing_symbols = [s for s in symbols if s not in self.betas]

        factor_values = X[self._expected_features].values
        if factor_values.ndim == 1:
            factor_vector = factor_values
        elif factor_values.shape[0] == 1:
            factor_vector = factor_values[0]
        else:
            factor_vector = factor_values[0]

        predictions: Dict[str, float] = {}
        for symbol in valid_symbols:
            beta = self.betas[symbol]
            alpha = self.alphas[symbol]
            predictions[symbol] = float(alpha + factor_vector @ beta)

        for symbol in missing_symbols:
            predictions[symbol] = 0.0

        return pd.Series(predictions, name='ff3_prediction')

    def get_feature_importance(self) -> Dict[str, float]:
        if self.status != ModelStatus.TRAINED or not self.betas:
            return {}
        beta_arrays = np.array(list(self.betas.values()))
        return dict(zip(self._expected_features, np.mean(beta_arrays, axis=0)))

    def get_symbol_alphas(self) -> Dict[str, float]:
        """Return all symbol alphas for strategy usage."""
        if self.status != ModelStatus.TRAINED or not self.alphas:
            return {}
        return self.alphas.copy()

    def get_alpha(self, symbol: Optional[str] = None) -> float:
        """Return alpha for a symbol or mean alpha if None."""
        if self.status != ModelStatus.TRAINED or not self.alphas:
            return 0.0
        if symbol and symbol in self.alphas:
            return float(self.alphas[symbol])
        return float(np.mean(list(self.alphas.values())))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'status': self.status,
            'regularization': self.regularization,
            'regularization_alpha': self.alpha,
            'standardize': self.standardize,
            'factors': self._expected_features,
            'training_samples': self.metadata.training_samples,
            'factor_betas': self.get_feature_importance(),
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_dict = {
            'model': self._model,
            'scaler': self._scaler,
            'expected_features': self._expected_features,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'standardize': self.standardize
        }
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump(model_dict, f)
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        with open(path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"FF3RegressionModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FF3RegressionModel':
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        with open(path / "metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        config = {}
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        with open(path / "model.pkl", 'rb') as f:
            model_dict = pickle.load(f)

        instance = cls(config=config)
        instance.metadata = metadata
        instance.status = ModelStatus.TRAINED

        instance._model = model_dict['model']
        instance._scaler = model_dict['scaler']
        instance._expected_features = model_dict['expected_features']
        instance.regularization = model_dict['regularization']
        instance.alpha = model_dict['alpha']
        instance.standardize = model_dict['standardize']

        logger.info(f"FF3RegressionModel loaded from {path}")
        return instance

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            'regularization': ['none', 'ridge'],
            'alpha': (0.01, 10.0),
            'standardize': [True, False]
        }

    def get_tunable_hyperparameters(self) -> List[str]:
        return ['regularization', 'alpha', 'standardize']


