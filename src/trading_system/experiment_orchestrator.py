"""
End-to-End Experiment Orchestrator
====================================

This module provides the `ExperimentOrchestrator`, a top-level controller
designed to automate the complete research workflow, from model training to
strategy backtesting.

It addresses the need for a seamless, repeatable process for evaluating how a
newly trained model performs within a given strategy.

Core Workflow:
--------------
1.  **Train Model**: It initializes and runs a `TrainingPipeline` based on a
    provided training configuration. This step:
    - Creates and configures data providers (price + factor data)
    - Fits the FeatureEngineeringPipeline on training data
    - Trains the model and outputs a versioned `model_id`
    
2.  **Configure Backtest**: It programmatically takes the `model_id` and the
    fitted feature pipeline from training and injects them into the backtest
    configuration. This ensures:
    - The backtest uses the *exact* model that was just trained
    - Feature computation is consistent between training and prediction
    - Factor data flows correctly for factor-based models (e.g., FF5)
    
3.  **Run Backtest**: It initializes and runs a `StrategyRunner` with:
    - The trained model
    - The fitted feature pipeline (reused from training)
    - Data providers (for automatic data acquisition)
    
4.  **Link and Report**: It captures results from both stages, links them
    together in an experiment tracking system (like WandB), and provides a
    consolidated summary.

Architecture Integration:
-------------------------
This orchestrator properly implements the symmetric architecture:

Training Flow:
    TrainingPipeline → FeatureEngineeringPipeline.fit() → Model.train()
    
Prediction Flow:
    PredictionPipeline → FeatureEngineeringPipeline.transform() → Model.predict()
    
By reusing the fitted FeatureEngineeringPipeline and data providers, we ensure
perfect consistency between training and prediction phases.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import yaml

from .models.training.training_pipeline import TrainingPipeline
from .feature_engineering.pipeline import FeatureEngineeringPipeline
from .strategy_backtest.strategy_runner import create_strategy_runner
from .config.factory import ConfigFactory
from .data.base_data_provider import BaseDataProvider

logger = logging.getLogger(__name__)


def _create_data_provider(config: Dict[str, Any]) -> BaseDataProvider:
    """Dynamically creates a data provider based on configuration."""
    provider_type = config.get('type')
    params = config.get('parameters', {})

    if provider_type == "YFinanceProvider":
        from .data.yfinance_provider import YFinanceProvider
        logger.info(f"Creating YFinanceProvider with params: {params}")
        return YFinanceProvider(**params)
    # Add other providers here as elif blocks
    # elif provider_type == "AnotherProvider":
    #     from .data.another_provider import AnotherProvider
    #     return AnotherProvider(**params)
    else:
        raise ValueError(f"Unsupported data provider type: {provider_type}")


def _create_factor_data_provider(config: Dict[str, Any]):
    """Dynamically creates a factor data provider based on configuration."""
    provider_type = config.get('type')
    params = config.get('parameters', {})

    if provider_type == "FF5DataProvider":
        from .data.ff5_provider import FF5DataProvider
        logger.info(f"Creating FF5DataProvider with params: {params}")
        return FF5DataProvider(**params)
    # Add other factor providers here as elif blocks
    else:
        raise ValueError(f"Unsupported factor data provider type: {provider_type}")


class ExperimentOrchestrator:
    """
    Automates the end-to-end process of training a model and backtesting it.
    """

    def __init__(self, experiment_config_path: str):
        """
        Initialize the orchestrator with the path to a unified experiment config.

        Args:
            experiment_config_path: Path to the unified YAML experiment config file.
        """
        self.experiment_config_path = Path(experiment_config_path)
        if not self.experiment_config_path.exists():
            raise FileNotFoundError(f"Experiment config not found at: {self.experiment_config_path}")

        logger.info(f"Loading experiment configuration from {self.experiment_config_path}")
        with open(self.experiment_config_path, 'r') as f:
            self.full_config = yaml.safe_load(f)

    def run_experiment(self) -> Dict[str, Any]:
        """
        Executes the full training-then-backtesting experiment.

        Returns:
            A dictionary containing the consolidated results from both stages.
        """
        logger.info("Starting end-to-end experiment...")

        # --- Part 1: Run the Training Pipeline ---
        training_setup = self.full_config.get('training_setup', {})
        if not training_setup:
            raise ValueError("Config must contain a 'training_setup' section.")

        # Dynamically create data providers
        data_provider_config = self.full_config.get('data_provider')
        if not data_provider_config:
            raise ValueError("Config must contain a 'data_provider' section.")
        data_provider = _create_data_provider(data_provider_config)

        # Create factor data provider if specified
        factor_data_provider = None
        factor_data_config = self.full_config.get('factor_data_provider')
        if factor_data_config:
            logger.info("Creating factor data provider...")
            factor_data_provider = _create_factor_data_provider(factor_data_config)
        else:
            logger.info("No factor data provider configured")

        # Setup FeatureEngineering and Training Pipelines
        model_config = training_setup.get('model', {})
        feature_config = training_setup.get('feature_engineering', {})
        training_params = training_setup.get('parameters', {})

        feature_pipeline = FeatureEngineeringPipeline.from_config(feature_config)

        train_pipeline = TrainingPipeline(
            model_type=model_config.get('model_type'),
            feature_pipeline=feature_pipeline,
            registry_path="./models/"  # Explicitly set the model registry path
        )
        train_pipeline.configure_data(data_provider=data_provider, factor_data_provider=factor_data_provider)
        
        logger.info("Executing training pipeline...")
        training_results = train_pipeline.run_pipeline(**training_params)
        
        model_id = training_results.get('pipeline_info', {}).get('model_id')
        if not model_id:
            raise RuntimeError("Training pipeline did not return a valid model_id.")
        
        logger.info(f"Training complete. New model_id: {model_id}")

        # --- Part 2: Run the Backtest with the new model ---
        # IMPORTANT: Reuse the fitted feature_pipeline from training
        # This ensures feature consistency between training and prediction
        logger.info("Using fitted feature pipeline from training for backtest")
        
        # Create provider instances to be passed down to the backtesting components
        # Include the fitted feature pipeline in providers
        providers = {
            'data_provider': data_provider,
            'feature_pipeline': feature_pipeline  # Pass the fitted pipeline
        }
        if factor_data_provider:
            providers['factor_data_provider'] = factor_data_provider

        # Load backtest and strategy configs using the factory
        backtest_configs = {
            'backtest': self.full_config.get('backtest', {}),
            'strategy': self.full_config.get('strategy', {})
        }

        # The ConfigFactory needs a file path, so we'll simulate one.
        # A better long-term solution is to allow it to parse a dict.
        # For now, let's dump the relevant sections to a string and load from there.
        import io
        temp_yaml_string = yaml.dump(backtest_configs)
        
        # We need to load from a file-like object for the factory.
        with io.StringIO(temp_yaml_string) as temp_yaml_file:
            # We can't directly use from_yaml, let's create objects manually
            # This avoids file I/O completely.
            backtest_config_obj = ConfigFactory.create_backtest_config(
                **ConfigFactory._process_backtest_params(backtest_configs['backtest'])
            )
            strategy_config_obj = ConfigFactory.create_strategy_config(
                **ConfigFactory._process_strategy_params(backtest_configs['strategy'])
            )
        
        backtest_config_objects = {
            'backtest': backtest_config_obj,
            'strategy': strategy_config_obj
        }
        
        logger.info(f"Updating backtest config to use model_id: {model_id}")
        # The 'parameters' dict is directly on the dataclass
        backtest_config_objects['strategy'].parameters['model_id'] = model_id
        backtest_config_objects['strategy'].parameters['model_registry_path'] = "./models/"
        
        # Pass the config objects and providers directly to the refactored runner
        logger.info("Executing backtest pipeline with the newly trained model...")
        backtest_runner = create_strategy_runner(
            config_obj=backtest_config_objects,
            providers=providers,
            use_wandb=True
        )
        
        experiment_name = f"e2e_{model_id}"
        backtest_results = backtest_runner.run_strategy(experiment_name=experiment_name)
        
        logger.info("Backtest complete.")

        # --- Part 3: Consolidate and Link Results ---
        # The linking would happen via the experiment tracker (e.g., WandB API)
        # For simplicity, we just return a combined result dictionary here.

        final_results = {
            "experiment_name": experiment_name,
            "status": "SUCCESS",
            "trained_model_id": model_id,
            "training_summary": training_results.get('pipeline_info'),
            "backtest_summary": backtest_results.get('performance_metrics')
        }
        
        logger.info("End-to-end experiment finished successfully.")
        return final_results
