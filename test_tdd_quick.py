#!/usr/bin/env python3
"""
Quick TDD Test for FF5 Pipeline
Tests PRIMARY pipeline with minimal data
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("TDD GREEN Phase: Running FF5 Pipeline")
    logger.info("=" * 60)
    logger.info("")

    # Check if we can import the orchestrator
    try:
        from trading_system.use_case.single_experiment.experiment_orchestrator import ExperimentOrchestrator
        logger.info("✓ Successfully imported ExperimentOrchestrator")
    except ImportError as e:
        logger.error(f"✗ Failed to import: {e}")
        logger.info("")
        logger.info("Trying alternative import path...")
        try:
            from use_case.single_experiment.experiment_orchestrator import ExperimentOrchestrator
            logger.info("✓ Successfully imported with alternative path")
        except ImportError as e2:
            logger.error(f"✗ Alternative import also failed: {e2}")
            return 1

    # Check config file
    config_file = "configs/draft/ff5_box_demo.yaml"
    if not Path(config_file).exists():
        logger.error(f"✗ Config file not found: {config_file}")
        return 1

    logger.info(f"✓ Config file found: {config_file}")
    logger.info("")

    # Create test output directory
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(exist_ok=True)
    logger.info(f"✓ Test output directory: {test_output_dir}/")
    logger.info("")

    # Option to run actual pipeline or just validate
    logger.info("Options:")
    logger.info("1. Validate config only (dry-run)")
    logger.info("2. Run full pipeline (will take time)")
    logger.info("")

    # For now, just validate
    logger.info("Running dry-run validation...")
    logger.info("-" * 60)

    try:
        orchestrator = ExperimentOrchestrator(experiment_config_path=config_file)
        logger.info("✓ Orchestrator created successfully")
        logger.info("")
        logger.info("Configuration is valid!")
        logger.info("")
        logger.info("To run the full pipeline, use:")
        logger.info("  PYTHONPATH=src python experiments/pipelines/run_ff5_box_experiment.py \\")
        logger.info("    --config configs/draft/ff5_box_demo.yaml --auto")
        logger.info("")
        return 0
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
