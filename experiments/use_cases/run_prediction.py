"""
Prediction Service CLI Entry Point
==================================

Command-line interface for the prediction service, supporting both single
and meta-model predictions. Follows the same patterns as other CLI tools
in the system.

Usage:
    python run_prediction.py --config configs/prediction_config.yaml
    python run_prediction.py --config configs/prediction_meta_config.yaml --output-dir ./results
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to Python path (script is now in experiments/use_cases/)
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from use_case.prediction.prediction_orchestrator import PredictionOrchestrator
from use_case.prediction.formatters import PredictionResultFormatter


def setup_logging():
    """Configure logging for the application."""
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
        description='Generate investment predictions from trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single model prediction
  python run_prediction.py --config configs/prediction_config.yaml
  
  # Run meta-model prediction
  python run_prediction.py --config configs/prediction_meta_config.yaml
  
  # Custom output directory
  python run_prediction.py --config configs/prediction_config.yaml --output-dir ./my_results
  
  # Verbose output
  python run_prediction.py --config configs/prediction_config.yaml --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/prediction_config.yaml',
        help='Path to prediction configuration file (default: configs/prediction_config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./prediction_results',
        help='Directory to save prediction results (default: ./prediction_results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['console', 'json', 'csv', 'all'],
        default='console',
        help='Output format (default: console)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save results to files (default: True)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the prediction service."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting prediction service...")
    
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Using configuration: {config_path}")
        
        # Initialize orchestrator
        orchestrator = PredictionOrchestrator(str(config_path))
        
        # Run prediction
        logger.info("Running prediction workflow...")
        result = orchestrator.run_prediction()
        
        # Initialize formatter
        formatter = PredictionResultFormatter()
        
        # Display results based on format
        if args.format in ['console', 'all']:
            print("\n" + "="*80)
            print("INVESTMENT PREDICTION RESULTS")
            print("="*80)
            print(formatter.format_console_report(result))
            print("="*80)
        
        # Save results if requested
        if args.save_results:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving results to {output_dir}")
            
            # Save JSON
            if args.format in ['json', 'all']:
                json_path = output_dir / 'prediction_result.json'
                formatter.save_to_json(result, json_path)
                logger.info(f"JSON results saved to {json_path}")
            
            # Save CSV
            if args.format in ['csv', 'all']:
                csv_path = output_dir / 'recommendations.csv'
                formatter.save_to_csv(result, csv_path)
                logger.info(f"CSV results saved to {csv_path}")
            
            # Always save summary
            summary_path = output_dir / 'prediction_summary.txt'
            formatter.save_summary_report(result, summary_path)
            logger.info(f"Summary report saved to {summary_path}")
        
        logger.info("Prediction service completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction service failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
