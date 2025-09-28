"""
Machine Learning Strategy implementation.

This strategy implements a comprehensive ML-based trading approach:
- Feature engineering with technical indicators
- Time series cross-validation to prevent look-ahead bias
- Model training with multiple algorithms (XGBoost, LightGBM, Random Forest)
- Hyperparameter optimization with Optuna
- Ensemble methods for robust predictions
- Model interpretability and feature importance analysis
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

from .base_strategy import BaseStrategy
from ..feature_engineering import FeatureEngine
from ..utils.secrets_manager import SecretsManager

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine Learning Trading Strategy.

    Strategy Workflow:
    1. Feature Engineering: Compute technical indicators and theoretical features
    2. Target Definition: Predict future returns or return direction
    3. Model Training: Use time series CV to prevent overfitting
    4. Prediction Generation: Generate daily trading signals
    5. Portfolio Construction: Convert predictions to portfolio weights
    6. Performance Monitoring: Track model performance and retrain as needed

    Features:
    - Multiple model types (XGBoost, LightGBM, Random Forest)
    - Automated feature selection and normalization
    - Ensemble predictions for robustness
    - Model interpretability with feature importance
    - Continuous learning with incremental updates
    """

    def __init__(self, name: str = "MLStrategy",
                 # Data parameters
                 lookback_days: int = 252,
                 feature_periods: List[int] = None,
                 # Model parameters
                 model_type: str = "xgboost",
                 prediction_horizon: int = 20,
                 target_type: str = "returns",  # "returns" or "direction"
                 # Training parameters
                 train_test_split: float = 0.8,
                 cv_folds: int = 5,
                 min_train_samples: int = 500,
                 # Feature parameters
                 max_features: int = 50,
                 feature_selection_method: str = "importance",
                 # Portfolio parameters
                 top_n_assets: int = 5,
                 min_prediction_score: float = 0.1,
                 max_position_size: float = 0.2,
                 # Optimization parameters
                 use_optuna: bool = False,
                 optuna_trials: int = 50,
                 # Model management
                 retrain_frequency: str = "monthly",
                 model_save_path: str = "./models/",
                 **kwargs):
        """
        Initialize ML Strategy.

        Args:
            name: Strategy name
            lookback_days: Lookback period for feature calculation
            feature_periods: List of periods for feature engineering
            model_type: Type of ML model ("xgboost", "lightgbm", "random_forest")
            prediction_horizon: Number of days ahead to predict
            target_type: Type of target variable ("returns" or "direction")
            train_test_split: Train/test split ratio
            cv_folds: Number of cross-validation folds
            min_train_samples: Minimum samples required for training
            max_features: Maximum number of features to use
            feature_selection_method: Method for feature selection
            top_n_assets: Number of top assets to select
            min_prediction_score: Minimum prediction score for asset selection
            max_position_size: Maximum position size for risk management
            use_optuna: Whether to use Optuna for hyperparameter optimization
            optuna_trials: Number of Optuna optimization trials
            retrain_frequency: How often to retrain the model
            model_save_path: Path to save trained models
        """
        # Data parameters
        self.lookback_days = lookback_days
        self.feature_periods = feature_periods or [21, 63, 126, 252]

        # Model parameters
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.target_type = target_type

        # Training parameters
        self.train_test_split = train_test_split
        self.cv_folds = cv_folds
        self.min_train_samples = min_train_samples

        # Feature parameters
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method

        # Portfolio parameters
        self.top_n_assets = top_n_assets
        self.min_prediction_score = min_prediction_score
        self.max_position_size = max_position_size

        # Optimization parameters
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials

        # Model management
        self.retrain_frequency = retrain_frequency
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.feature_engine = FeatureEngine(
            lookback_periods=[20, 50, 200],
            momentum_periods=self.feature_periods,
            volatility_windows=[20, 60],
            include_technical=True,
            include_theoretical=True
        )

        # Model storage
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.model_metadata = {}

        # Performance tracking
        self.prediction_history = []
        self.performance_history = []

        super().__init__(name=name, **kwargs)

    def validate_parameters(self):
        """Validate strategy parameters."""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")

        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")

        if self.cv_folds <= 0:
            raise ValueError("cv_folds must be positive")

        if self.max_features <= 0:
            raise ValueError("max_features must be positive")

        if self.top_n_assets <= 0:
            raise ValueError("top_n_assets must be positive")

        if not 0 <= self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")

        if self.model_type not in ["xgboost", "lightgbm", "random_forest"]:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.target_type not in ["returns", "direction"]:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def generate_signals(self, price_data: Dict[str, pd.DataFrame],
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals using ML predictions.

        Args:
            price_data: Dictionary of price DataFrames for each symbol
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with trading signals (weight allocations for each symbol)
        """
        logger.info(f"Generating ML signals from {start_date} to {end_date}")

        # Create date range for rebalancing
        if self.retrain_frequency == "monthly":
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')

        signals = pd.DataFrame(index=date_range, columns=list(price_data.keys()))

        # Train model if not already trained
        if self.model is None:
            logger.info("Training initial ML model...")
            self._train_model(price_data, start_date, end_date)

        for date in date_range:
            try:
                signal = self._generate_signal_for_date(price_data, date)
                signals.loc[date] = signal
            except Exception as e:
                logger.warning(f"Failed to generate signal for {date}: {e}")
                # Use neutral position
                signals.loc[date] = 0

        return signals

    def _generate_signal_for_date(self, price_data: Dict[str, pd.DataFrame],
                                 date: datetime) -> pd.Series:
        """
        Generate trading signal for a specific date.

        Args:
            price_data: Dictionary of price DataFrames
            date: Date for signal generation

        Returns:
            Series with weight allocations
        """
        # Get prediction for each asset
        predictions = {}

        for symbol, data in price_data.items():
            try:
                prediction = self._predict_asset_returns(data, date, symbol)
                if prediction is not None:
                    predictions[symbol] = prediction
            except Exception as e:
                logger.debug(f"Failed to predict for {symbol}: {e}")

        if not predictions:
            logger.warning(f"No predictions generated for {date}")
            return pd.Series(0, index=list(price_data.keys()))

        # Convert predictions to DataFrame for easier manipulation
        pred_df = pd.DataFrame(list(predictions.items()),
                               columns=['symbol', 'prediction']).set_index('symbol')

        # Filter by minimum prediction score
        if self.target_type == "returns":
            qualified_assets = pred_df[pred_df['prediction'] >= self.min_prediction_score]
        else:  # direction
            qualified_assets = pred_df[abs(pred_df['prediction']) >= self.min_prediction_score]

        if len(qualified_assets) < self.top_n_assets:
            logger.debug(f"Only {len(qualified_assets)} assets qualified (need {self.top_n_assets})")
            if len(qualified_assets) == 0:
                return pd.Series(0, index=list(price_data.keys()))
            top_assets = qualified_assets
        else:
            # Select top N assets by prediction score
            top_assets = qualified_assets.nlargest(self.top_n_assets, 'prediction')

        # Equal weight selected assets with size constraints
        weight_per_asset = min(1.0 / len(top_assets), self.max_position_size)
        allocation = pd.Series(0.0, index=list(price_data.keys()), dtype=float)

        for symbol in top_assets.index:
            allocation[symbol] = weight_per_asset

        logger.debug(f"Selected {len(top_assets)} assets: {list(top_assets.index)}")

        return allocation

    def _train_model(self, price_data: Dict[str, pd.DataFrame],
                    start_date: datetime, end_date: datetime):
        """Train ML model with time series cross-validation."""
        logger.info("Training ML model with time series cross-validation")

        try:
            # 1. Compute features
            features = self.feature_engine.compute_features(price_data, start_date, end_date)

            # 2. Create target variable
            targets = self._create_targets(price_data, self.prediction_horizon)

            # 3. Prepare training data
            X, y, feature_names = self._prepare_training_data(features, targets)

            if len(X) < self.min_train_samples:
                logger.warning(f"Insufficient training data: {len(X)} < {self.min_train_samples}")
                return

            # 4. Feature selection
            if len(feature_names) > self.max_features:
                selected_features = self.feature_engine.select_features(
                    features, y, method=self.feature_selection_method, max_features=self.max_features
                )
                X = X[selected_features]
                feature_names = selected_features

            # 5. Split data (time series aware)
            split_idx = int(len(X) * self.train_test_split)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # 6. Normalize features
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 7. Train model
            if self.use_optuna:
                self._optimize_with_optuna(X_train_scaled, y_train, feature_names)
            else:
                self._train_default_model(X_train_scaled, y_train, feature_names)

            # 8. Evaluate model
            self._evaluate_model(X_test_scaled, y_test)

            # 9. Save model
            self._save_model()

            # 10. Store feature names
            self.feature_names = feature_names

            logger.info(f"Model training completed. Features: {len(feature_names)}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def _create_targets(self, price_data: Dict[str, pd.DataFrame],
                       horizon: int) -> Dict[str, pd.Series]:
        """Create target variables for ML training."""
        targets = {}

        for symbol, data in price_data.items():
            try:
                if self.target_type == "returns":
                    # Future returns
                    future_returns = data['Close'].pct_change(horizon).shift(-horizon)
                    targets[symbol] = future_returns
                elif self.target_type == "direction":
                    # Return direction (binary classification)
                    future_returns = data['Close'].pct_change(horizon).shift(-horizon)
                    targets[symbol] = (future_returns > 0).astype(int)
                else:
                    raise ValueError(f"Unknown target type: {self.target_type}")

            except Exception as e:
                logger.debug(f"Failed to create target for {symbol}: {e}")

        return targets

    def _prepare_training_data(self, features: Dict[str, pd.DataFrame],
                              targets: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for ML training."""
        # Combine all features and targets
        all_features = []
        all_targets = []

        for symbol in features.keys():
            if symbol in targets:
                # Align features and targets
                symbol_features = features[symbol]
                symbol_targets = targets[symbol]

                # Remove rows with NaN in targets
                valid_mask = symbol_targets.notna()
                if valid_mask.sum() > 0:
                    symbol_features = symbol_features[valid_mask]
                    symbol_targets = symbol_targets[valid_mask]

                    all_features.append(symbol_features)
                    all_targets.append(symbol_targets)

        if not all_features:
            raise ValueError("No valid training data available")

        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)

        # Remove features with too many NaN values
        nan_threshold = 0.3
        valid_features = X.columns[X.isna().mean() < nan_threshold]
        X = X[valid_features].fillna(0)

        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        return X, y, list(X.columns)

    def _train_default_model(self, X: np.ndarray, y: pd.Series, feature_names: List[str]):
        """Train default model based on type."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X, y)

    def _optimize_with_optuna(self, X: np.ndarray, y: pd.Series, feature_names: List[str]):
        """Optimize hyperparameters using Optuna."""
        try:
            import optuna

            def objective(trial):
                if self.model_type == "xgboost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = xgb.XGBRegressor(**params)
                elif self.model_type == "lightgbm":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    model = lgb.LGBMRegressor(**params)
                elif self.model_type == "random_forest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 5, 15),
                        'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = RandomForestRegressor(**params)

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = []

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)  # Negative MSE for maximization
                    scores.append(score)

                return np.mean(scores)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params['random_state'] = 42
            if self.model_type in ["xgboost", "lightgbm"]:
                best_params['n_jobs'] = -1
            if self.model_type == "lightgbm":
                best_params['verbose'] = -1

            # Train final model with best parameters
            if self.model_type == "xgboost":
                self.model = xgb.XGBRegressor(**best_params)
            elif self.model_type == "lightgbm":
                self.model = lgb.LGBMRegressor(**best_params)
            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(**best_params)

            self.model.fit(X, y)

            logger.info(f"Optuna optimization completed. Best score: {study.best_value:.4f}")

        except ImportError:
            logger.warning("Optuna not available, using default parameters")
            self._train_default_model(X, y, feature_names)
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}, using default parameters")
            self._train_default_model(X, y, feature_names)

    def _evaluate_model(self, X_test: np.ndarray, y_test: pd.Series):
        """Evaluate model performance."""
        if self.model is None:
            return

        y_pred = self.model.predict(X_test)

        if self.target_type == "returns":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"Model performance - MSE: {mse:.6f}, R²: {r2:.6f}")
            self.model_metadata['mse'] = mse
            self.model_metadata['r2'] = r2
        else:  # direction
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            precision = precision_score(y_test, (y_pred > 0.5).astype(int), average='weighted')
            recall = recall_score(y_test, (y_pred > 0.5).astype(int), average='weighted')
            logger.info(f"Model performance - Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}")
            self.model_metadata['accuracy'] = accuracy
            self.model_metadata['precision'] = precision
            self.model_metadata['recall'] = recall

    def _predict_asset_returns(self, data: pd.DataFrame, date: datetime,
                               symbol: str) -> Optional[float]:
        """Predict returns for a specific asset."""
        if self.model is None or self.feature_names is None or self.scaler is None:
            return None

        try:
            # Get data up to the prediction date
            data_up_to_date = data[data.index <= date]

            if len(data_up_to_date) < self.lookback_days:
                logger.debug(f"Insufficient data for prediction: {len(data_up_to_date)} < {self.lookback_days}")
                return None

            # Compute features
            features = self.feature_engine._compute_symbol_features(data_up_to_date, symbol)

            # Select and order features
            available_features = [f for f in self.feature_names if f in features.columns]
            if len(available_features) < len(self.feature_names) * 0.5:  # Need at least 50% of features
                logger.debug(f"Insufficient features available: {len(available_features)} < {len(self.feature_names)}")
                return None

            feature_data = features[available_features].fillna(0).iloc[-1:]  # Use most recent data

            # Scale features
            feature_scaled = self.scaler.transform(feature_data)

            # Make prediction
            prediction = self.model.predict(feature_scaled)[0]

            return prediction

        except Exception as e:
            logger.debug(f"Prediction failed for {symbol}: {e}")
            return None

    def _save_model(self):
        """Save trained model and metadata."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_metadata': self.model_metadata,
                'feature_engine_config': self.feature_engine.get_feature_info(),
                'training_date': datetime.now().isoformat()
            }

            model_path = self.model_save_path / f"{self.name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, model_path: str = None):
        """Load trained model from file."""
        if model_path is None:
            model_path = self.model_save_path / f"{self.name}_model.pkl"

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_metadata = model_data.get('model_metadata', {})

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from trained model."""
        if self.model is None or self.feature_names is None:
            return None

        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                logger.warning("Model does not have feature importances")
                return None

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'target_type': self.target_type,
            'prediction_horizon': self.prediction_horizon,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_metadata': self.model_metadata,
            'training_date': self.model_metadata.get('training_date', 'Not trained'),
            'performance_metrics': {
                k: v for k, v in self.model_metadata.items()
                if k in ['mse', 'r2', 'accuracy', 'precision', 'recall']
            }
        }

    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information."""
        info = super().get_info()
        info.update({
            'description': 'Machine Learning-based trading strategy',
            'lookback_days': self.lookback_days,
            'model_type': self.model_type,
            'target_type': self.target_type,
            'prediction_horizon': self.prediction_horizon,
            'top_n_assets': self.top_n_assets,
            'min_prediction_score': self.min_prediction_score,
            'max_position_size': self.max_position_size,
            'retrain_frequency': self.retrain_frequency,
            'use_optuna': self.use_optuna,
            'feature_engine': self.feature_engine.get_feature_info(),
            'model_info': self.get_model_info(),
            'risk_management': 'Feature-based selection with position size constraints'
        })
        return info

    def calculate_risk_metrics(self, price_data: Dict[str, pd.DataFrame],
                             strategy_signals: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ML strategy."""
        try:
            metrics = {}

            # Calculate portfolio volatility
            if not strategy_signals.empty:
                # Equal-weighted portfolio returns
                portfolio_weights = strategy_signals.div(strategy_signals.sum(axis=1), axis=0)

                # Calculate asset returns
                asset_returns = {}
                for symbol in strategy_signals.columns:
                    if symbol in price_data and price_data[symbol] is not None:
                        close_col = 'Close' if 'Close' in price_data[symbol].columns else 'close'
                        asset_returns[symbol] = price_data[symbol][close_col].pct_change().dropna()

                # Portfolio returns
                if asset_returns:
                    returns_df = pd.DataFrame(asset_returns)
                    portfolio_returns = (returns_df * portfolio_weights.reindex(returns_df.index, method='ffill')).sum(axis=1)

                    # Risk metrics
                    metrics['portfolio_volatility'] = portfolio_returns.std() * np.sqrt(252)
                    metrics['max_drawdown'] = (portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()).max()
                    metrics['sharpe_ratio'] = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
                    metrics['var_95'] = portfolio_returns.quantile(0.05)
                    metrics['expected_shortfall'] = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()

            # Model-based risk metrics
            if hasattr(self, 'model_metadata') and self.model_metadata:
                metrics.update({
                    'model_r2': self.model_metadata.get('r2', 0),
                    'model_mse': self.model_metadata.get('mse', 0),
                    'model_accuracy': self.model_metadata.get('accuracy', 0),
                    'prediction_confidence': self._calculate_prediction_confidence()
                })

            # Position concentration risk
            if not strategy_signals.empty:
                position_concentration = strategy_signals.div(strategy_signals.sum(axis=1), axis=0)
                metrics['max_position_concentration'] = position_concentration.max().max()
                metrics['herfindahl_index'] = (position_concentration ** 2).sum(axis=1).mean()

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_prediction_confidence(self) -> float:
        """Calculate model prediction confidence."""
        try:
            if hasattr(self, 'model') and hasattr(self.model, 'predict_proba'):
                # For classification models
                return 0.5  # Placeholder - would need actual predictions
            elif hasattr(self, 'model_metadata'):
                # Use model performance as proxy for confidence
                r2 = self.model_metadata.get('r2', 0)
                return max(0, min(1, (r2 + 1) / 2))  # Scale R² to [0,1]
            return 0.5
        except:
            return 0.5