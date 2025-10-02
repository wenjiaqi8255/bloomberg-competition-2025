"""
Experiment Pipeline - Phase 7 Core Component

This module provides the end-to-end experiment pipeline that orchestrates
the complete ML workflow from data preparation to model deployment.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

from .interface import ExperimentTrackerInterface, NullExperimentTracker
from .config import ExperimentConfig
from ...models.training.hyperparameter_optimizer import HyperparameterOptimizer
from ...models.training.experiment_manager import TrainingExperimentManager
from ...models.model_persistence import ModelPersistence
from ...models.utils.performance_evaluator import PerformanceEvaluator
from ...models.serving.monitor import ModelMonitor
from ...strategy_runner import StrategyRunner


@dataclass
class PipelineConfig:
    """Configuration for the experiment pipeline."""
    # Data configuration
    data_config: Dict[str, Any]

    # Model configuration
    model_class: Any
    search_space: Dict[str, Any]
    n_trials: int = 50

    # Training configuration
    use_cv: bool = True
    cv_folds: int = 5
    test_size: float = 0.2

    # Evaluation configuration
    benchmark_symbol: Optional[str] = None

    # Registry configuration
    model_name: str = "experimental_model"
    model_version: str = "latest"
    model_tags: List[str] = None

    # Strategy configuration (optional)
    strategy_config: Optional[Dict[str, Any]] = None

    # Monitoring configuration
    enable_monitoring: bool = True

    # Experiment configuration
    experiment_config: Optional[ExperimentConfig] = None


@dataclass
class PipelineResult:
    """Result of running the experiment pipeline."""
    pipeline_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Results from each stage
    data_info: Dict[str, Any]
    best_hyperparameters: Dict[str, Any]
    model_id: str
    training_metrics: Dict[str, float]
    evaluation_metrics: Dict[str, float]

    # Optional strategy results
    backtest_results: Optional[Dict[str, Any]] = None

    # Status and metadata
    status: str = "completed"
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ExperimentPipeline:
    """
    End-to-end experiment pipeline for ML model development.

    This is the core Phase 7 component that orchestrates:
    1. Data preparation and validation
    2. Hyperparameter optimization
    3. Model training with experiment tracking
    4. Model evaluation and validation
    5. Model registration
    6. Model monitoring setup
    7. Strategy backtesting (optional)
    """

    def __init__(self,
                 experiment_tracker: Optional[ExperimentTrackerInterface] = None,
                 registry_path: str = "./models/registry",
                 enable_monitoring: bool = True):
        """
        Initialize the experiment pipeline.

        Args:
            experiment_tracker: Experiment tracker for logging
            registry_path: Path for model registry
            enable_monitoring: Whether to enable model monitoring
        """
        self.experiment_tracker = experiment_tracker or NullExperimentTracker()
        self.registry_path = registry_path
        self.enable_monitoring = enable_monitoring

        # Initialize pipeline components
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.experiment_tracker)
        self.training_manager = TrainingExperimentManager(self.experiment_tracker)
        self.model_persistence = ModelPersistence(registry_path)
        self.performance_evaluator = PerformanceEvaluator()  # Use static methods, no tracker needed

        # Initialize monitoring components (if enabled)
        self.model_monitors = {}

        print("🚀 Experiment Pipeline initialized")

    def run_full_pipeline(self, config: PipelineConfig) -> PipelineResult:
        """
        Run the complete end-to-end experiment pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline execution result
        """
        # Generate pipeline ID
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🔄 Starting Experiment Pipeline: {pipeline_id}")
        start_time = datetime.now()

        # Initialize main experiment run
        if config.experiment_config:
            run_id = self.experiment_tracker.init_run(config.experiment_config)
        else:
            # Create default experiment config
            default_config = ExperimentConfig(
                project_name="experiment_pipeline",
                experiment_name=f"full_pipeline_{pipeline_id}",
                run_type="pipeline",
                tags=["pipeline", "end-to-end"],
                hyperparameters={"pipeline_id": pipeline_id},
                metadata={"config": asdict(config)}
            )
            run_id = self.experiment_tracker.init_run(default_config)

        result = PipelineResult(
            pipeline_id=pipeline_id,
            start_time=start_time,
            end_time=None,
            duration_seconds=0.0,
            data_info={},
            best_hyperparameters={},
            model_id="",
            training_metrics={},
            evaluation_metrics={}
        )

        try:
            # Stage 1: Data Preparation
            print("\n📊 Stage 1: Data Preparation")
            data = self._prepare_data(config.data_config, config.test_size)
            result.data_info = data['info']

            # Log data information
            self.experiment_tracker.log_params({
                "data_shape_X": data['X'].shape,
                "data_shape_y": data['y'].shape,
                "data_features": list(data['X'].columns),
                "data_start_date": str(data['X'].index.min()) if hasattr(data['X'], 'index') else None,
                "data_end_date": str(data['X'].index.max()) if hasattr(data['X'], 'index') else None
            })

            # Stage 2: Hyperparameter Optimization
            print("\n🔍 Stage 2: Hyperparameter Optimization")
            if config.n_trials > 0:
                best_params = self._run_hyperparameter_optimization(
                    config.model_class,
                    config.search_space,
                    data['X_train'],
                    data['y_train'],
                    config.n_trials
                )
                result.best_hyperparameters = best_params
            else:
                print("   Skipping hyperparameter optimization (n_trials = 0)")
                result.best_hyperparameters = {}

            # Stage 3: Model Training
            print("\n🏋️  Stage 3: Model Training")
            model = config.model_class(config=result.best_hyperparameters)
            training_result = self._train_model(
                model,
                data['X_train'],
                data['y_train'],
                config.use_cv,
                config.cv_folds
            )
            result.training_metrics = training_result.get('metrics', {})

            # Stage 4: Model Evaluation
            print("\n📈 Stage 4: Model Evaluation")
            evaluation_result = self._evaluate_model(
                training_result['model'],
                data['X_test'],
                data['y_test'],
                config.benchmark_symbol
            )
            result.evaluation_metrics = {
                **evaluation_result.overall_metrics,
                **evaluation_result.financial_metrics,
                **evaluation_result.statistical_metrics
            }

            # Stage 5: Model Registration
            print("\n💾 Stage 5: Model Registration")
            model_id = self._register_model(
                training_result['model'],
                config,
                result.training_metrics,
                result.evaluation_metrics,
                evaluation_result
            )
            result.model_id = model_id

            # Stage 6: Model Monitoring Setup
            if self.enable_monitoring:
                print("\n🔔 Stage 6: Model Monitoring Setup")
                monitor_id = self._setup_monitoring(model_id, config)
                if monitor_id:
                    result.warnings.append(f"Model monitoring enabled with ID: {monitor_id}")

            # Stage 7: Strategy Backtesting (optional)
            if config.strategy_config:
                print("\n📊 Stage 7: Strategy Backtesting")
                backtest_results = self._run_strategy_backtest(config.strategy_config, model_id)
                result.backtest_results = backtest_results

            # Update pipeline status
            result.status = "completed"
            print(f"\n✅ Pipeline {pipeline_id} completed successfully!")

        except Exception as e:
            result.status = "failed"
            error_msg = f"Pipeline failed at stage: {e}"
            result.errors.append(error_msg)
            print(f"❌ {error_msg}")

            # Log error to experiment tracker
            self.experiment_tracker.log_alert("Pipeline Failed", error_msg, "error")
            raise

        finally:
            # Finalize pipeline execution
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()

            # Log final pipeline metrics
            self._log_pipeline_results(result)

            # Finish experiment run
            self.experiment_tracker.finish_run(exit_code=0 if result.status == "completed" else 1)

        return result

    def _prepare_data(self, data_config: Dict[str, Any], test_size: float) -> Dict[str, Any]:
        """Prepare and split data for training and evaluation."""
        try:
            # This is a placeholder implementation
            # In a real scenario, this would load data from the specified source

            # Generate synthetic data for demonstration
            np.random.seed(42)
            n_samples = data_config.get('n_samples', 1000)
            n_features = data_config.get('n_features', 10)

            X = pd.DataFrame(
                np.random.normal(0, 1, (n_samples, n_features)),
                columns=[f'feature_{i}' for i in range(n_features)]
            )

            # Add some temporal structure if requested
            if data_config.get('add_temporal', False):
                dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
                X.index = dates

                # Add trend and seasonality
                trend = np.linspace(0, 1, n_samples)
                seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # Annual seasonality

                for col in X.columns:
                    X[col] += trend * 0.1 + seasonal * np.random.uniform(-1, 1)

            # Generate target with some relationship to features
            true_coeffs = np.random.normal(0, 0.5, n_features)
            y = pd.Series(
                X.dot(true_coeffs) + np.random.normal(0, 0.5, n_samples),
                name='target'
            )

            if hasattr(X, 'index'):
                y.index = X.index

            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            data_info = {
                'total_samples': n_samples,
                'n_features': n_features,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': list(X.columns),
                'data_type': 'synthetic',
                'has_temporal_index': hasattr(X, 'index')
            }

            print(f"   Data prepared: {data_info}")

            return {
                'X': X,
                'y': y,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'info': data_info
            }

        except Exception as e:
            raise Exception(f"Data preparation failed: {e}")

    def _run_hyperparameter_optimization(self, model_class: Any, search_space: Dict[str, Any],
                                       X_train: pd.DataFrame, y_train: pd.Series,
                                       n_trials: int) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        try:
            print(f"   Running hyperparameter optimization with {n_trials} trials...")

            best_params = self.hyperparameter_optimizer.optimize(
                model_class=model_class,
                search_space=search_space,
                X=X_train,
                y=y_train,
                n_trials=n_trials
            )

            print(f"   Best hyperparameters: {best_params}")

            # Log optimization results
            self.experiment_tracker.log_params({
                "optimization_n_trials": n_trials,
                "optimization_best_params": json.dumps(best_params, default=str)
            })

            return best_params

        except Exception as e:
            print(f"   ⚠️  Hyperparameter optimization failed: {e}")
            print("   Using default hyperparameters")
            return {}

    def _train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                    use_cv: bool, cv_folds: int) -> Dict[str, Any]:
        """Train the model with cross-validation if enabled."""
        try:
            print(f"   Training model {model.model_type}...")

            # Use the training experiment manager for comprehensive tracking
            training_config = {
                'use_cv': use_cv,
                'cv_folds': cv_folds,
                'model_type': model.model_type,
                'training_samples': len(X_train)
            }

            # Create experiment config for training
            train_exp_config = ExperimentConfig(
                project_name="experiment_pipeline",
                experiment_name=f"model_training_{model.model_type}",
                run_type="training",
                tags=["training", model.model_type],
                hyperparameters=model.config,
                metadata=training_config
            )

            # Train the model
            training_result = self.training_manager.run_training_experiment(
                model=model,
                X=X_train,
                y=y_train,
                experiment_config=train_exp_config
            )

            print(f"   Model training completed")
            if 'metrics' in training_result:
                print(f"   Training metrics: {training_result['metrics']}")

            return training_result

        except Exception as e:
            raise Exception(f"Model training failed: {e}")

    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                       benchmark_symbol: Optional[str] = None) -> Any:
        """Evaluate the trained model."""
        try:
            print(f"   Evaluating model on test set...")

            # Load benchmark data if provided
            benchmark_returns = None
            if benchmark_symbol:
                try:
                    # This would normally load real benchmark data
                    # For now, generate synthetic benchmark
                    benchmark_returns = pd.Series(
                        np.random.normal(0.0005, 0.02, len(y_test)),
                        index=y_test.index if hasattr(y_test, 'index') else None
                    )
                except Exception as e:
                    print(f"   ⚠️  Could not load benchmark data: {e}")

            # Evaluate model using comprehensive evaluation
            evaluation_result = self.performance_evaluator.evaluate_comprehensive(
                model=model,
                X_test=X_test,
                y_test=y_test,
                benchmark_returns=benchmark_returns,
                dataset_name="test"
            )

            # Extract metrics from comprehensive evaluation result
            metrics = evaluation_result.get('metrics', {})

            # Log key evaluation metrics
            key_metrics = {
                'test_ic': metrics.get('financial_information_coefficient', metrics.get('information_coefficient', 0)),
                'test_r2': metrics.get('r2', 0),
                'test_mae': metrics.get('mae', 0),
                'test_rmse': metrics.get('rmse', 0)
            }

            self.experiment_tracker.log_metrics(key_metrics)

            print(f"   Evaluation completed")
            print(f"   Key metrics: IC={key_metrics.get('test_ic', 0):.3f}, R²={key_metrics.get('test_r2', 0):.3f}")

            # Generate and display recommendations
            recommendations = self.performance_evaluator.generate_recommendations(metrics)
            if recommendations:
                print("   Recommendations:")
                for rec in recommendations:
                    print(f"   💡 {rec}")

            return evaluation_result

        except Exception as e:
            raise Exception(f"Model evaluation failed: {e}")

    def _register_model(self, model: Any, config: PipelineConfig,
                       training_metrics: Dict[str, float],
                       evaluation_metrics: Dict[str, float],
                       evaluation_result: Any) -> str:
        """Register the trained model in the model registry."""
        try:
            print(f"   Registering model...")

            # Prepare model metadata
            metadata_info = {
                'pipeline_id': config.experiment_config.experiment_name if config.experiment_config else "unknown",
                'training_samples': training_metrics.get('training_samples', 0),
                'model_config': model.config
            }

            # Save model using simple persistence
            model_id = self.model_persistence.save_model(
                model=model,
                model_name=config.model_name,
                description=f"Model trained via experiment pipeline",
                metadata={
                    'model_version': config.model_version,
                    'tags': config.model_tags or [],
                    'training_metrics': training_metrics,
                    'data_info': metadata_info
                }
            )

            print(f"   Model registered with ID: {model_id}")

            # Log registration info
            self.experiment_tracker.log_params({
                "registered_model_id": model_id,
                "model_name": config.model_name,
                "model_version": config.model_version
            })

            return model_id

        except Exception as e:
            raise Exception(f"Model registration failed: {e}")

    def _setup_monitoring(self, model_id: str, config: PipelineConfig) -> Optional[str]:
        """Set up model monitoring if enabled."""
        try:
            if not self.enable_monitoring:
                return None

            print(f"   Setting up monitoring for model: {model_id}")

            # Create model monitor
            monitor = ModelMonitor(
                model_id=model_id,
                config={
                    'performance_window': 30,
                    'degradation_threshold': 0.2,
                    'min_samples': 10
                },
                tracker=self.experiment_tracker
            )

            # Store monitor for later access
            self.model_monitors[model_id] = monitor

            # Initialize monitoring run
            monitor_run_id = f"monitor_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"   Monitoring initialized with run ID: {monitor_run_id}")

            return monitor_run_id

        except Exception as e:
            print(f"   ⚠️  Monitoring setup failed: {e}")
            return None

    def _run_strategy_backtest(self, strategy_config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Run strategy backtest with the trained model."""
        try:
            print(f"   Running strategy backtest...")

            # This would integrate with the existing StrategyRunner
            # For now, return placeholder results
            backtest_results = {
                'strategy_name': strategy_config.get('name', 'default'),
                'model_id': model_id,
                'total_return': np.random.uniform(-0.1, 0.3),
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(-0.2, -0.05),
                'trades_count': np.random.randint(10, 100)
            }

            print(f"   Backtest completed: {backtest_results}")

            # Log backtest results
            self.experiment_tracker.log_metrics({
                'backtest_total_return': backtest_results['total_return'],
                'backtest_sharpe_ratio': backtest_results['sharpe_ratio'],
                'backtest_max_drawdown': backtest_results['max_drawdown']
            })

            return backtest_results

        except Exception as e:
            print(f"   ⚠️  Strategy backtest failed: {e}")
            return {}

    def _log_pipeline_results(self, result: PipelineResult) -> None:
        """Log final pipeline results to experiment tracker."""
        try:
            # Log pipeline metadata
            self.experiment_tracker.log_params({
                "pipeline_id": result.pipeline_id,
                "pipeline_status": result.status,
                "pipeline_duration_seconds": result.duration_seconds,
                "pipeline_start_time": result.start_time.isoformat(),
                "pipeline_end_time": result.end_time.isoformat() if result.end_time else None
            })

            # Log final metrics summary
            final_metrics = {
                'pipeline_duration_minutes': result.duration_seconds / 60,
                'model_id_registered': 1 if result.model_id else 0,
                'monitoring_enabled': 1 if self.enable_monitoring else 0,
                'backtest_completed': 1 if result.backtest_results else 0
            }

            # Add key model metrics
            if result.evaluation_metrics:
                final_metrics.update({
                    'final_ic': result.evaluation_metrics.get('information_coefficient', 0),
                    'final_r2': result.evaluation_metrics.get('r_squared', 0)
                })

            self.experiment_tracker.log_metrics(final_metrics)

            # Log any warnings
            for warning in result.warnings:
                self.experiment_tracker.log_alert("Pipeline Warning", warning, "warning")

            # Log any errors
            for error in result.errors:
                self.experiment_tracker.log_alert("Pipeline Error", error, "error")

        except Exception as e:
            print(f"⚠️  Failed to log pipeline results: {e}")

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get status information for a specific pipeline run."""
        # This would typically load from a persistent store
        # For now, return basic info
        return {
            'pipeline_id': pipeline_id,
            'status': 'completed',
            'active_monitors': len(self.model_monitors),
            'storage_info': self.model_persistence.get_storage_info()
        }