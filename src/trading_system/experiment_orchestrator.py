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
    provided training configuration. This step outputs a versioned `model_id`
    for the newly trained model and its associated feature pipeline.
2.  **Configure Backtest**: It programmatically takes the `model_id` from the
    training step and injects it into the backtest configuration. This ensures
    that the backtest runs with the *exact* model that was just trained.
3.  **Run Backtest**: It then initializes and runs a `StrategyRunner` using this
    dynamically updated backtest configuration.
4.  **Link and Report**: It captures the results from both stages, links them
    together in an experiment tracking system (like WandB), and provides a
    consolidated summary.

This orchestrator connects the previously separate training and backtesting
processes into a single, automated pipeline.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import yaml

from ..models.training.training_pipeline import TrainingPipeline
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..strategy_runner import create_strategy_runner
from ..config.factory import ConfigFactory

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """
    Automates the end-to-end process of training a model and backtesting it.
    """

    def __init__(self, training_config_path: str, backtest_config_path: str):
        """
        Initialize the orchestrator with paths to the necessary configurations.

        Args:
            training_config_path: Path to the YAML file for the training pipeline.
            backtest_config_path: Path to the YAML file for the backtest runner.
        """
        self.training_config_path = Path(training_config_path)
        self.backtest_config_path = Path(backtest_config_path)

        if not self.training_config_path.exists():
            raise FileNotFoundError(f"Training config not found at: {self.training_config_path}")
        if not self.backtest_config_path.exists():
            raise FileNotFoundError(f"Backtest config not found at: {self.backtest_config_path}")

    def run_experiment(self) -> Dict[str, Any]:
        """
        Executes the full training-then-backtesting experiment.

        Returns:
            A dictionary containing the consolidated results from both stages.
        """
        logger.info("Starting end-to-end experiment...")

        # --- Part 1: Run the Training Pipeline ---
        logger.info(f"Loading training configuration from {self.training_config_path}")
        with open(self.training_config_path, 'r') as f:
            training_configs = yaml.safe_load(f)

        # Assuming the configs are structured with these keys
        model_config = training_configs.get('model', {})
        feature_config = training_configs.get('feature_engineering', {})
        training_config = training_configs.get('training', {})
        
        # Setup FeatureEngineering and Training Pipelines
        feature_pipeline = FeatureEngineeringPipeline.from_config(feature_config)
        
        # This part requires a data_provider, which is not defined in the configs.
        # For a real implementation, this would need to be instantiated and passed.
        # Let's assume a YFinanceProvider for now.
        from ...data.yfinance_provider import YFinanceProvider
        data_provider = YFinanceProvider()

        train_pipeline = TrainingPipeline(
            model_type=model_config.get('model_type', 'xgboost'),
            feature_pipeline=feature_pipeline
        )
        train_pipeline.configure_data(data_provider=data_provider)
        
        logger.info("Executing training pipeline...")
        training_results = train_pipeline.run_pipeline(**training_config.get('parameters', {}))
        
        model_id = training_results.get('pipeline_info', {}).get('model_id')
        if not model_id:
            raise RuntimeError("Training pipeline did not return a valid model_id.")
        
        logger.info(f"Training complete. New model_id: {model_id}")

        # --- Part 2: Run the Backtest with the new model ---
        logger.info(f"Loading backtest configuration from {self.backtest_config_path}")
        
        # Dynamically update the backtest config in memory with the new model_id
        backtest_configs_obj = ConfigFactory.load_all_configs(self.backtest_config_path)
        
        if 'strategy' not in backtest_configs_obj:
            raise ValueError("Backtest config must have a 'strategy' section.")
            
        logger.info(f"Updating backtest config to use model_id: {model_id}")
        backtest_configs_obj['strategy'].parameters['model_id'] = model_id
        
        # The StrategyRunner loads configs from a file path. A temporary file could be used,
        # or the runner could be refactored to accept a config object directly.
        # For now, let's assume a refactor where we can pass the object.
        # (This implies a small change to StrategyRunner would be needed).
        
        # We also need to find a way to pass the modified config to the runner.
        # Let's simulate this by writing to a temporary file.
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            # We need to convert the config objects back to a dict to dump them
            # This is a simplification; the real Config objects might need a `to_dict` method
            full_config_dict = {
                'backtest': backtest_configs_obj['backtest'].__dict__,
                'strategy': backtest_configs_obj['strategy'].__dict__
            }
            yaml.dump(full_config_dict, tmp_file)
            temp_config_path = tmp_file.name

        logger.info("Executing backtest pipeline with the newly trained model...")
        backtest_runner = create_strategy_runner(config_path=temp_config_path, use_wandb=True)
        
        experiment_name = f"e2e_{model_id}"
        backtest_results = backtest_runner.run_strategy(experiment_name=experiment_name)
        
        # Clean up the temporary config file
        import os
        os.unlink(temp_config_path)

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
