# -*- coding: utf-8 -*-
"""
End-to-End Experiment Starter Script
====================================

This script serves as the main entry point for running a complete
experiment, from model training to strategy backtesting, using the
`ExperimentOrchestrator`.

To run an experiment:
1. Ensure you have a unified experiment configuration file in the `configs`
   directory (e.g., `e2e_ff5_experiment.yaml`).
2. Update the `EXPERIMENT_CONFIG_FILE` variable below to point to your
   config file.
3. Run the script from the root of the project:
   python run_experiment.py
"""

import logging
import json
from src.trading_system.experiment_orchestrator import ExperimentOrchestrator

# --- Configuration ---
# Specify the configuration file for the experiment you want to run.
EXPERIMENT_CONFIG_FILE = 'configs/e2e_ff5_experiment.yaml'

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Silence overly verbose libraries
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def main():
    """
    Initializes and runs the ExperimentOrchestrator.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting experiment using config: {EXPERIMENT_CONFIG_FILE}")

    try:
        # 1. Initialize the orchestrator with the config file path
        orchestrator = ExperimentOrchestrator(
            experiment_config_path=EXPERIMENT_CONFIG_FILE
        )

        # 2. Run the entire experiment pipeline
        results = orchestrator.run_experiment()

        # 3. Print the final consolidated report
        logger.info("\n" + "="*50 + "\n--- EXPERIMENT FINISHED ---\n" + "="*50)
        
        # Use pretty-printing for the final results dictionary
        print(json.dumps(results, indent=2, default=str))
        
        logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)

    except FileNotFoundError:
        logger.error(
            f"Configuration file not found at '{EXPERIMENT_CONFIG_FILE}'. "
            "Please ensure the file exists in the 'configs' directory."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    main()
