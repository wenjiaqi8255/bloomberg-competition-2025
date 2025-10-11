"""
Model Trainer

This module provides a unified training interface for all ML models.
It handles the complete training workflow while keeping model logic
separate from training orchestration.

Key Features:
- Separation of training logic from model implementation
- Built-in cross-validation support
- Performance evaluation and reporting
- Early stopping and model selection
- Comprehensive logging and tracking
"""

import logging
import time
import copy
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np

from ..base.base_model import BaseModel
from ...validation.time_series_cv import TimeSeriesCV
from ..utils.performance_evaluator import PerformanceEvaluator as PerformanceEvaluator
from .types import TrainingConfig, TrainingResult
# ExperimentLogger已删除 - 简化版本不需要

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for all ML models.

    This trainer handles the complete training workflow while keeping
    concerns properly separated:
    - Model logic stays in the model
    - Training orchestration stays here
    - Validation uses TimeSeriesCV
    - Performance evaluation uses a dedicated evaluator
    """

    def __init__(self,
                 config: Optional[TrainingConfig] = None,
                 cv: Optional[TimeSeriesCV] = None,
                 experiment_logger: Optional[Any] = None,  # 简化版本，不再依赖ExperimentLogger
                 performance_evaluator: Optional[PerformanceEvaluator] = None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            cv: Cross-validation instance (created if None)
            logger: Logger for experiment tracking (created if None)
            performance_evaluator: Evaluator for model performance (created if None)
        """
        self.config = config or TrainingConfig()
        self.cv = cv or TimeSeriesCV(
            method=self.config.cv_method,
            n_splits=self.config.cv_folds,
            purge_period=self.config.purge_period,
            embargo_period=self.config.embargo_period
        )
        self.evaluator = performance_evaluator or PerformanceEvaluator()
        # 简化版本 - 不再使用ExperimentLogger
        self.experiment_logger = None
        
        # Log CV configuration
        logger.info(f"Cross-validation configured: method={self.config.cv_method}, "
                   f"folds={self.config.cv_folds}, purge_period={self.config.purge_period}, "
                   f"embargo_period={self.config.embargo_period}")

    def train(self,
              model: BaseModel,
              X: pd.DataFrame,
              y: pd.Series,
              X_test: Optional[pd.DataFrame] = None,
              y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train a model with the specified configuration.

        This is a simplified entry point that delegates to the main training
        loop and optionally logs the results if configured.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information
        """
        logger.info(f"Starting training for {model.model_type}")
        start_time = time.time()

        result = self._perform_training(model, X, y, X_test, y_test)
        
        training_time = time.time() - start_time
        result.training_time = training_time

        # 简化版本 - 不再使用ExperimentLogger
        # 实验跟踪功能已移除，专注于核心训练逻辑

        logger.info(f"Training completed in {result.training_time:.2f} seconds")
        return result

    def train_with_tracking(self,
                           model: BaseModel,
                           X: pd.DataFrame,
                           y: pd.Series,
                           experiment_config: Optional[Dict[str, Any]] = None,
                           X_test: Optional[pd.DataFrame] = None,
                           y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train a model with comprehensive experiment tracking.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            experiment_config: Configuration for the experiment
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information
        """
        logger.info(f"Starting tracked training for {model.model_type}")
        total_start_time = time.time()

        self.logger.init_run(model, X, self.config, experiment_config)

        try:
            self.logger.log_data_statistics(X, y)

            result = self._perform_training(model, X, y, X_test, y_test, track_cv=True)

            total_training_time = time.time() - total_start_time
            result.training_time = total_training_time

            self.logger.log_metrics(result.validation_metrics, split_name="training")
            if result.test_metrics:
                self.logger.log_metrics(result.test_metrics, split_name="test")

            self.logger.log_model_information(model)
            self.logger.finish_run(total_training_time)

            logger.info(f"Tracked training completed in {total_training_time:.2f} seconds")
            return result

        except Exception as e:
            self.logger.log_failure(model, e)
            raise RuntimeError(f"Training failed: {e}")

    def _perform_training(self,
                          model: BaseModel,
                          X: pd.DataFrame,
                          y: pd.Series,
                          X_test: Optional[pd.DataFrame] = None,
                          y_test: Optional[pd.Series] = None,
                          track_cv: bool = False) -> TrainingResult:
        """
        Core internal training loop.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets
            track_cv: Whether to log detailed CV progress

        Returns:
            TrainingResult object (without training_time set)
        """
        self._validate_training_data(X, y)
        model.validate_data(X, y)

        training_history = []
        cv_results = None

        if self.config.use_cross_validation:
            logger.info("Performing cross-validation...")
            cv_method = self._cross_validate_with_tracking if track_cv else self._cross_validate
            cv_results = cv_method(model, X, y)
            training_history.append({'stage': 'cross_validation', **cv_results})

        logger.info("Training final model...")
        try:
            model.fit(X, y)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

        validation_metrics = self._calculate_metrics(model, X, y, split_name="training")

        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(model, X_test, y_test, split_name="test")

        return TrainingResult(
            model=model,
            training_time=0.0,  # Will be set by the caller
            cv_results=cv_results,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            training_history=training_history
        )

    def _log_data_statistics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Log data statistics to experiment tracker."""
        self.logger.log_data_statistics(X, y)

    def _cross_validate_with_tracking(self,
                                     model: BaseModel,
                                     X: pd.DataFrame,
                                     y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation with experiment tracking for each fold.
        """
        cv_scores = []
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = self._create_model_copy(model)
            fold_model.fit(X_train_fold, y_train_fold)

            fold_metrics = self._calculate_metrics(
                fold_model, X_val_fold, y_val_fold, split_name=f"fold_{fold_idx}"
            )
            fold_results.append(fold_metrics)
            cv_scores.append(fold_metrics.get('r2', 0.0))

            self.logger.log_cv_fold(fold_idx, fold_metrics)

        cv_result = {
            'mean_r2': np.mean(cv_scores) if cv_scores else 0.0,
            'std_r2': np.std(cv_scores) if cv_scores else 0.0,
            'fold_results': fold_results,
            'cv_scores': cv_scores
        }

        self.logger.log_cv_summary(cv_result)

        logger.info(f"Cross-validation R²: {cv_result['mean_r2']:.4f} ± {cv_result['std_r2']:.4f}")
        return cv_result

    def _log_model_information(self, model: BaseModel) -> None:
        """Log model-specific information to experiment tracker."""
        self.logger.log_model_information(model)

    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate training data.

        Args:
            X: Feature DataFrame
            y: Target Series

        Raises:
            ValueError: If data is invalid
        """
        # Handle both DataFrame and numpy array validation
        if isinstance(X, pd.DataFrame):
            if X.empty or y.empty:
                raise ValueError("Training data cannot be empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0 or y.size == 0:
                raise ValueError("Training data cannot be empty")
        else:
            # Fallback for other data types
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if len(X) < self.config.cv_folds:
            raise ValueError(f"Not enough data for {self.config.cv_folds}-fold CV")

        # Check for sufficient data after CV splits
        min_samples_per_split = len(X) // self.config.cv_folds
        required_samples = self.config.purge_period + self.config.embargo_period + 10

        if min_samples_per_split < required_samples:
            logger.warning(
                f"Cross-validation may be unreliable. "
                f"Only {min_samples_per_split} samples per split, "
                f"but need at least {required_samples}"
            )

    def _cross_validate(self,
                        model: BaseModel,
                        X: pd.DataFrame,
                        y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Targets

        Returns:
            Cross-validation results
        """
        cv_scores = []
        fold_results = []

        # Perform time series cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            # Handle both DataFrame and numpy array indexing
            if isinstance(X, pd.DataFrame):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            elif isinstance(X, np.ndarray):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            else:
                # Fallback for other types
                X_train_fold, X_val_fold = [X[i] for i in train_idx], [X[i] for i in val_idx]
                y_train_fold, y_val_fold = [y[i] for i in train_idx], [y[i] for i in val_idx]

            # Create a copy of the model for this fold
            fold_model = self._create_model_copy(model)

            # Train the fold model
            fold_model.fit(X_train_fold, y_train_fold)

            # Evaluate on validation set
            fold_metrics = self._calculate_metrics(
                fold_model, X_val_fold, y_val_fold, split_name=f"fold_{fold_idx}"
            )
            fold_results.append(fold_metrics)
            cv_scores.append(fold_metrics.get('r2', 0.0))

        # Aggregate CV results
        cv_result = {
            'mean_r2': np.mean(cv_scores),
            'std_r2': np.std(cv_scores),
            'fold_results': fold_results,
            'cv_scores': cv_scores
        }

        logger.info(f"Cross-validation R²: {cv_result['mean_r2']:.4f} ± {cv_result['std_r2']:.4f}")
        return cv_result

    def _calculate_metrics(self,
                          model: BaseModel,
                          X: pd.DataFrame,
                          y: pd.Series,
                          split_name: str = "default") -> Dict[str, float]:
        """
        Calculate performance metrics using the dedicated evaluator.

        Args:
            model: Trained model
            X: Features
            y: True targets
            split_name: Name for this data split

        Returns:
            Dictionary of metrics
        """
        try:
            logger.info(f"Calculating metrics for {split_name} split...")
            logger.info(f"DEBUG: Metrics calculation - X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"DEBUG: Metrics calculation - X has NaN: {X.isnull().any().any()}, y has NaN: {y.isnull().any()}")
            logger.info(f"DEBUG: Metrics calculation - y stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")

            # Make predictions first to debug them
            predictions = model.predict(X)
            logger.info(f"DEBUG: Predictions shape: {predictions.shape}, predictions have NaN: {np.isnan(predictions).any()}")
            logger.info(f"DEBUG: Prediction stats: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}, min={np.min(predictions):.6f}, max={np.max(predictions):.6f}")
            logger.info(f"DEBUG: Sample predictions (first 10): {predictions[:10]}")
            logger.info(f"DEBUG: Sample y_true (first 10): {y.values[:10]}")

            # Calculate correlation to understand relationship
            if len(y) > 1:
                correlation = np.corrcoef(y, predictions)[0, 1]
                logger.info(f"DEBUG: Correlation between y_true and predictions: {correlation:.6f}")

            # Note: evaluate_model returns more metrics than before.
            # We can filter them here if needed, or just return all of them.
            # For now, returning all seems fine.
            metrics = self.evaluator.evaluate(model, X, y)

            # Filter to only include metrics relevant to the old implementation if necessary
            # For example: metrics = {k: v for k, v in metrics.items() if k in ['r2', 'mse', 'mae', 'rmse', 'ic']}

            logger.info(f"{split_name} metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics for {split_name}: {e}")
            return {}

    def _create_model_copy(self, model: BaseModel) -> BaseModel:
        """
        Create a copy of the model for cross-validation.

        Args:
            model: Original model

        Returns:
            Model copy
        """
        # Use the factory to create a new instance of the same type
        from ..base.model_factory import ModelFactory

        return ModelFactory.create(
            model.model_type,
            config=model.config.copy()
        )

    def train_with_cv(self,
                     model: BaseModel,
                     data: Dict[str, Any],
                     feature_pipeline: Any,
                     date_range: Tuple[datetime, datetime]) -> TrainingResult:
        """
        Train model with cross-validation, fitting pipeline independently for each fold.

        This method implements the key architectural principle: "Who evaluates, who splits."
        The trainer is responsible for CV splitting and ensuring data independence.

        Args:
            model: Model to train
            data: Raw data dictionary containing price_data, factor_data, target_data
            feature_pipeline: Unfitted feature engineering pipeline
            date_range: Tuple of (start_date, end_date) for actual training period

        Returns:
            TrainingResult with fitted model and pipeline
        """
        logger.info("Starting CV training with independent pipeline fitting per fold")
        start_time = time.time()
        
        start_date, end_date = date_range
        
        # 1. Extract all available dates from data
        all_available_dates = self._extract_all_dates(data)
        
        # 2. Filter to training period
        train_period_dates = [d for d in all_available_dates 
                             if start_date <= d <= end_date]
        
        if len(train_period_dates) < self.config.cv_folds * 20:
            raise ValueError(
                f"Insufficient dates: {len(train_period_dates)} dates "
                f"for {self.config.cv_folds}-fold CV (need at least {self.config.cv_folds * 20})"
            )
        
        logger.info(f"CV on {len(train_period_dates)} dates from {train_period_dates[0]} to {train_period_dates[-1]}")
        
        # 3. Generate CV splits using the new date-range method
        try:
            cv_splits = list(self.cv.split_by_date_range(
                train_period_dates[0], 
                train_period_dates[-1]
            ))
        except Exception as e:
            logger.error(f"CV split generation failed: {e}")
            raise
        
        if not cv_splits:
            raise ValueError("No valid CV splits generated")
        
        logger.info(f"Generated {len(cv_splits)} CV splits")
        
        # 4. Process each fold with error handling
        cv_results = []
        successful_folds = 0
        
        for fold_idx, (train_dates_fold, val_dates_fold) in enumerate(cv_splits):
            try:
                logger.info(f"Fold {fold_idx}:")
                logger.info(f"  Train dates: {len(train_dates_fold)} ({train_dates_fold[0]} to {train_dates_fold[-1]})")
                logger.info(f"  Val dates: {len(val_dates_fold)} ({val_dates_fold[0]} to {val_dates_fold[-1]})")
                
                # ** CRITICAL: Create independent pipeline copy for this fold
                fold_pipeline = self._clone_pipeline(feature_pipeline)
                
                # Filter data for this fold
                train_data = self._filter_data_by_dates(data, train_dates_fold)
                val_data = self._filter_data_by_dates(data, val_dates_fold)
                
                logger.info(f"  Train data shapes: price={len(train_data['price_data'])}, target={len(train_data['target_data'])}")
                
                # ** CRITICAL: Fit pipeline on full data to enable proper feature calculation
                logger.info(f"Fitting pipeline for fold {fold_idx} with full price history")
                fold_pipeline.fit({
                    'price_data': train_data['price_data'],  # Full history for feature calculation
                    'factor_data': train_data.get('factor_data', {})
                })
                
                # ** CRITICAL: Transform with full data, then filter by target dates
                logger.info(f"Computing features with full price history")
                X_train_full = fold_pipeline.transform({
                    'price_data': train_data['price_data'],  # Full history
                    'factor_data': train_data.get('factor_data', {})
                })
                X_val_full = fold_pipeline.transform({
                    'price_data': val_data['price_data'],  # Full history
                    'factor_data': val_data.get('factor_data', {})
                })
                
                logger.info(f"  Full features computed: train={X_train_full.shape}, val={X_val_full.shape}")
                
                # Prepare targets for this fold (only the fold's date range)
                y_train = self._prepare_targets(train_data['target_data'], train_dates_fold)
                y_val = self._prepare_targets(val_data['target_data'], val_dates_fold)
                
                logger.info(f"  Targets prepared: train={y_train.shape}, val={y_val.shape}")
                
                # ** CRITICAL: Standardize data formats before alignment
                logger.info(f"  Before format standardization: X_train_full={X_train_full.shape}, y_train={y_train.shape}")
                logger.info(f"  X_train_full index: {X_train_full.index.names}")
                logger.info(f"  y_train index: {y_train.index.names}")
                
                # Apply format standardization to ensure consistent index order
                try:
                    from ...feature_engineering.utils.panel_data_transformer import PanelDataTransformer
                    
                    # Standardize training data format
                    X_train_standardized, y_train_standardized = PanelDataTransformer.to_panel_format(
                        X_train_full, y_train
                    )
                    logger.info(f"  Training data standardized: X={X_train_standardized.shape}, y={y_train_standardized.shape}")
                    logger.info(f"  Standardized X index: {X_train_standardized.index.names}")
                    logger.info(f"  Standardized y index: {y_train_standardized.index.names}")
                    
                    # Standardize validation data format
                    X_val_standardized, y_val_standardized = PanelDataTransformer.to_panel_format(
                        X_val_full, y_val
                    )
                    logger.info(f"  Validation data standardized: X={X_val_standardized.shape}, y={y_val_standardized.shape}")
                    
                except Exception as e:
                    logger.warning(f"Format standardization failed: {e}, using original data")
                    X_train_standardized, y_train_standardized = X_train_full, y_train
                    X_val_standardized, y_val_standardized = X_val_full, y_val
                
                # ** CRITICAL: Filter features to match target dates using intersection
                logger.info(f"  Before alignment: X_train_standardized={X_train_standardized.shape}, y_train_standardized={y_train_standardized.shape}")
                
                # Find common index between standardized features and targets
                common_train_index = X_train_standardized.index.intersection(y_train_standardized.index)
                common_val_index = X_val_standardized.index.intersection(y_val_standardized.index)
                
                if len(common_train_index) == 0:
                    logger.error(f"No common index between standardized X_train and y_train!")
                    logger.error(f"X_train_standardized index sample: {X_train_standardized.index[:5]}")
                    logger.error(f"y_train_standardized index sample: {y_train_standardized.index[:5]}")
                    raise ValueError("No common index between features and targets in training set")
                
                if len(common_val_index) == 0:
                    logger.error(f"No common index between standardized X_val and y_val!")
                    logger.error(f"X_val_standardized index sample: {X_val_standardized.index[:5]}")
                    logger.error(f"y_val_standardized index sample: {y_val_standardized.index[:5]}")
                    raise ValueError("No common index between features and targets in validation set")
                
                # Filter features to match target dates using standardized data
                X_train = X_train_standardized.loc[common_train_index]
                y_train = y_train_standardized.loc[common_train_index]
                X_val = X_val_standardized.loc[common_val_index]
                y_val = y_val_standardized.loc[common_val_index]
                
                logger.info(f"  After alignment: X_train={X_train.shape}, y_train={y_train.shape}")
                logger.info(f"  Validation alignment: X_val={X_val.shape}, y_val={y_val.shape}")
                
                # Final validation
                assert len(X_train) == len(y_train), f"Training data mismatch: X={len(X_train)}, y={len(y_train)}"
                assert len(X_val) == len(y_val), f"Validation data mismatch: X={len(X_val)}, y={len(y_val)}"
                
                # Create and train model for this fold
                fold_model = self._create_model_copy(model)
                fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_metrics = self._calculate_metrics(fold_model, X_val, y_val, split_name=f"fold_{fold_idx}")
                cv_results.append(val_metrics)
                successful_folds += 1
                
                logger.info(f"✅ Fold {fold_idx} completed successfully: {val_metrics}")
                
            except Exception as e:
                logger.error(f"❌ Fold {fold_idx} FAILED: {e}")
                logger.error(f"Fold {fold_idx} traceback:", exc_info=True)
                
                # Add empty result for failed fold
                cv_results.append({})
                logger.warning(f"Continuing with remaining folds...")
                continue
        
        # ** CRITICAL: Train final model on full training data
        logger.info("Training final model on full training period")
        
        # Use the original date_range for final training (not from CV splits)
        final_train_dates = [d for d in all_available_dates 
                            if start_date <= d <= end_date]
        
        # Ensure dates are sorted and unique
        final_train_dates = sorted(list(set(final_train_dates)))
        
        logger.info(f"Final model training on {len(final_train_dates)} dates: "
                   f"{final_train_dates[0]} to {final_train_dates[-1]}")
        
        # Filter data for full training period
        full_train_data = self._filter_data_by_dates(data, final_train_dates)
        
        # Fit final pipeline on full training data
        final_pipeline = self._clone_pipeline(feature_pipeline)
        final_pipeline.fit({
            'price_data': full_train_data['price_data'],
            'factor_data': full_train_data.get('factor_data', {})
        })
        
        # Transform full training data
        X_full = final_pipeline.transform({
            'price_data': full_train_data['price_data'],
            'factor_data': full_train_data.get('factor_data', {})
        })
        y_full = self._prepare_targets(full_train_data['target_data'], final_train_dates)
        
        # ** CRITICAL: Standardize formats for final model training
        logger.info(f"Final model data before format standardization: X_full={X_full.shape}, y_full={y_full.shape}")
        logger.info(f"X_full index: {X_full.index.names}")
        logger.info(f"y_full index: {y_full.index.names}")
        
        # Apply format standardization for final model training
        try:
            from ...feature_engineering.utils.panel_data_transformer import PanelDataTransformer
            
            X_full_standardized, y_full_standardized = PanelDataTransformer.to_panel_format(
                X_full, y_full
            )
            logger.info(f"Final model data standardized: X={X_full_standardized.shape}, y={y_full_standardized.shape}")
            logger.info(f"Standardized X index: {X_full_standardized.index.names}")
            logger.info(f"Standardized y index: {y_full_standardized.index.names}")
            
        except Exception as e:
            logger.warning(f"Final model format standardization failed: {e}, using original data")
            X_full_standardized, y_full_standardized = X_full, y_full
        
        # ** CRITICAL: Force alignment for final model training
        logger.info(f"Final model data before alignment: X_full_standardized={X_full_standardized.shape}, y_full_standardized={y_full_standardized.shape}")
        
        common_full_index = X_full_standardized.index.intersection(y_full_standardized.index)
        if len(common_full_index) == 0:
            logger.error(f"No common index between standardized X_full and y_full!")
            logger.error(f"X_full_standardized index sample: {X_full_standardized.index[:5]}")
            logger.error(f"y_full_standardized index sample: {y_full_standardized.index[:5]}")
            raise ValueError("No common index between features and targets in final training set")
        
        X_full = X_full_standardized.loc[common_full_index]
        y_full = y_full_standardized.loc[common_full_index]
        
        logger.info(f"Final model data after alignment: X_full={X_full.shape}, y_full={y_full.shape}")
        
        # Final validation
        assert len(X_full) == len(y_full), f"Final training data mismatch: X={len(X_full)}, y={len(y_full)}"
        
        # Train final model
        model.fit(X_full, y_full)
        
        # Calculate final validation metrics
        final_metrics = self._calculate_metrics(model, X_full, y_full, split_name="final_training")
        
        training_time = time.time() - start_time
        
        # Aggregate CV results
        successful_cv_results = [r for r in cv_results if r]  # Filter out empty results
        failed_folds = len(cv_results) - successful_folds
        
        if successful_cv_results:
            cv_summary = {
                'mean_r2': np.mean([r.get('r2', 0.0) for r in successful_cv_results]),
                'std_r2': np.std([r.get('r2', 0.0) for r in successful_cv_results]),
                'fold_results': cv_results,
                'cv_scores': [r.get('r2', 0.0) for r in cv_results],
                'successful_folds': successful_folds,
                'failed_folds': failed_folds,
                'total_folds': len(cv_splits)
            }
        else:
            cv_summary = {
                'mean_r2': 0.0,
                'std_r2': 0.0,
                'fold_results': cv_results,
                'cv_scores': [0.0] * len(cv_results),
                'successful_folds': 0,
                'failed_folds': failed_folds,
                'total_folds': len(cv_splits)
            }
        
        logger.info(f"CV training completed in {training_time:.2f} seconds")
        logger.info(f"Successful folds: {successful_folds}/{len(cv_splits)}")
        if successful_cv_results:
            logger.info(f"CV R²: {cv_summary['mean_r2']:.4f} ± {cv_summary['std_r2']:.4f}")
        else:
            logger.warning("No successful CV folds - all folds failed!")
        
        return TrainingResult(
            model=model,
            training_time=training_time,
            cv_results=cv_summary,
            validation_metrics=final_metrics,
            feature_pipeline=final_pipeline  # ** CRITICAL: Return fitted pipeline
        )

    def _clone_pipeline(self, pipeline: Any) -> Any:
        """
        Create an independent copy of the feature pipeline for CV.

        Args:
            pipeline: Original pipeline

        Returns:
            Independent pipeline copy
        """
        # Method 1: Deep copy (preferred if pipeline supports it)
        try:
            return copy.deepcopy(pipeline)
        except Exception as e:
            logger.warning(f"Deep copy failed: {e}, trying shallow copy")
        
        # Method 2: Create new instance from config
        try:
            # This assumes the pipeline has a from_config method
            if hasattr(pipeline, 'config') and hasattr(pipeline.__class__, 'from_config'):
                return pipeline.__class__.from_config(pipeline.config, model_type=pipeline.model_type)
        except Exception as e:
            logger.warning(f"Config-based copy failed: {e}")
        
        # Method 3: Manual copy of essential attributes
        try:
            new_pipeline = pipeline.__class__()
            for attr in ['config', 'model_type', 'feature_engineering_steps']:
                if hasattr(pipeline, attr):
                    setattr(new_pipeline, attr, getattr(pipeline, attr))
            return new_pipeline
        except Exception as e:
            logger.error(f"All pipeline cloning methods failed: {e}")
            raise RuntimeError(f"Cannot clone pipeline: {e}")

    def _filter_data_by_dates(self, data: Dict[str, Any], target_dates: List[datetime]) -> Dict[str, Any]:
        """
        Filter data dictionary, keeping price and factor data intact for feature calculation.
        Only filters target data to prevent leakage.

        Args:
            data: Original data dictionary
            target_dates: List of target dates (used only for filtering targets)

        Returns:
            Filtered data dictionary with intact price/factor data, filtered target data
        """
        filtered_data = {}
        
        # ** CRITICAL: Keep price_data intact - needed for feature lookback
        # Price data must include lookback period for computing cross-sectional features
        filtered_data['price_data'] = data['price_data']
        logger.debug(f"Kept {len(data['price_data'])} symbols with full price history")
        
        # ** CRITICAL: Keep factor_data intact if present
        # Factor data also needs full history for feature calculation
        if 'factor_data' in data:
            filtered_data['factor_data'] = data['factor_data']
            logger.debug(f"Kept factor data intact")
        
        # ** ONLY filter target_data to match the fold's date range
        # This prevents data leakage while preserving feature calculation capability
        target_dates_set = set(pd.to_datetime(d).date() for d in target_dates)
        logger.debug(f"Filtering targets for {len(target_dates_set)} target dates")
        
        if 'target_data' in data:
            filtered_target_data = {}
            for symbol, series in data['target_data'].items():
                if hasattr(series, 'index'):
                    series_dates = pd.to_datetime(series.index).date
                    mask = np.array([d in target_dates_set for d in series_dates])
                    filtered_target_data[symbol] = series[mask]
                    logger.debug(f"Filtered {symbol} targets: {len(series)} -> {len(series[mask])} dates")
                else:
                    filtered_target_data[symbol] = series
            filtered_data['target_data'] = filtered_target_data
            logger.debug(f"Filtered targets to {len(target_dates_set)} target dates")
        
        return filtered_data

    def _prepare_targets(self, target_data: Dict[str, pd.Series], target_dates: List[datetime]) -> pd.Series:
        """
        Prepare target data for training with proper MultiIndex alignment.

        This method constructs a MultiIndex Series (symbol, date) that matches
        the feature structure, ensuring we don't lose data due to duplicate dates.

        Args:
            target_data: Dictionary of target series by symbol
            target_dates: List of target dates

        Returns:
            MultiIndex Series with (symbol, date) index
        """
        # Convert target_dates to date-only for robust matching
        target_dates_set = set(pd.to_datetime(d).date() for d in target_dates)
        all_target_records = []
        
        for symbol, series in target_data.items():
            if not hasattr(series, 'index'):
                continue
                
            # Filter by dates using date-only comparison
            series_dates = pd.to_datetime(series.index).date
            mask = np.array([d in target_dates_set for d in series_dates])
            filtered_series = series[mask]
            
            if len(filtered_series) > 0:
                # ** CRITICAL: Create (symbol, date) records to avoid data loss
                for date, value in filtered_series.items():
                    all_target_records.append({
                        'symbol': symbol,
                        'date': pd.to_datetime(date),
                        'target': value
                    })
                logger.debug(f"Prepared {len(filtered_series)} targets for {symbol}")
        
        if not all_target_records:
            raise ValueError(f"No target data available for dates: {target_dates[0]} to {target_dates[-1]}")
        
        # ** CRITICAL: Build MultiIndex DataFrame to preserve all (symbol, date) combinations
        target_df = pd.DataFrame(all_target_records)
        target_df = target_df.set_index(['symbol', 'date'])
        target_series = target_df['target'].sort_index()
        
        # Remove NaN values
        target_series = target_series.dropna()
        
        logger.info(f"Prepared {len(target_series)} targets from {len(target_data)} symbols")
        logger.info(f"Target index structure: {target_series.index.names}")
        logger.info(f"Target date range: {target_series.index.get_level_values('date').min()} to {target_series.index.get_level_values('date').max()}")
        
        return target_series

    def _extract_all_dates(self, data: Dict[str, Any]) -> List[datetime]:
        """
        Extract all unique dates from data.

        Args:
            data: Data dictionary containing price_data

        Returns:
            List of unique dates sorted chronologically
        """
        all_dates = set()
        
        for symbol_data in data['price_data'].values():
            if hasattr(symbol_data, 'index'):
                # Convert to date-only to avoid timezone/time precision issues
                dates = pd.to_datetime(symbol_data.index).date
                all_dates.update(dates)
        
        # Convert back to datetime and sort
        sorted_dates = sorted([pd.to_datetime(d) for d in all_dates])
        
        logger.debug(f"Extracted {len(sorted_dates)} unique dates from data")
        return sorted_dates
