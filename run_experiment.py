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
2. Run the script from the root of the project:
   python run_experiment.py                           # Uses default config
   python run_experiment.py --config configs/my_experiment.yaml  # Custom config
   python run_experiment.py -c configs/test.yaml     # Short form
"""

import argparse
import logging
import json
import os
from src.trading_system.experiment_orchestrator import ExperimentOrchestrator

# --- Configuration ---
# Default configuration file for the experiment
DEFAULT_EXPERIMENT_CONFIG_FILE = 'configs/e2e_ff5_experiment.yaml'

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run end-to-end trading experiment with model training and backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py                                    # Use default config
  python run_experiment.py --config configs/my_exp.yaml       # Custom config
  python run_experiment.py -c configs/test.yaml              # Short form
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default=DEFAULT_EXPERIMENT_CONFIG_FILE,
        help=f'Path to experiment configuration file (default: {DEFAULT_EXPERIMENT_CONFIG_FILE})'
    )

    return parser.parse_args()


def main():
    """
    Initializes and runs the ExperimentOrchestrator.
    """
    # Parse command line arguments
    args = parse_arguments()
    config_file = args.config

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting experiment using config: {config_file}")

    try:
        # Validate config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at '{config_file}'")

        # 1. Initialize the orchestrator with the config file path
        orchestrator = ExperimentOrchestrator(experiment_config_path=config_file)

        # 2. Run the entire experiment pipeline
        results = orchestrator.run_experiment()

        # 3. Print the final consolidated report
        logger.info("\n" + "="*50 + "\n--- EXPERIMENT FINISHED ---\n" + "="*50)

        # Use pretty-printing for the final results dictionary
        print(json.dumps(results, indent=2, default=str))

        logger.info("\n" + "="*50 + "\n--- END OF REPORT ---\n" + "="*50)

    except FileNotFoundError:
        logger.error(
            f"Configuration file not found at '{config_file}'. "
            "Please ensure the file exists and the path is correct."
        )
        logger.info(f"Available config files in 'configs' directory:")
        if os.path.exists('configs'):
            for file in os.listdir('configs'):
                if file.endswith(('.yaml', '.yml')):
                    logger.info(f"  - configs/{file}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    main()
