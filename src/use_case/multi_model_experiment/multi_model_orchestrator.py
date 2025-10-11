"""
Multi-Model Orchestrator
========================

Main orchestrator for the multi-model experiment framework. This orchestrator:
1. Trains multiple base models with HPO
2. Trains a metamodel to combine the base models with HPO
3. Evaluates the complete system with real backtesting
4. Generates comprehensive reports

All training, prediction, and evaluation uses real data and proper pipelines.
No mock or simulated data anywhere in the system.
"""

import logging
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .components.model_trainer import ModelTrainerWithHPO
from .components.metamodel_trainer import MetaModelTrainerWithHPO

logger = logging.getLogger(__name__)


class MultiModelOrchestrator:
    """
    Orchestrates the complete multi-model experiment workflow.
    
    This orchestrator ensures that:
    - Each base model goes through proper training → prediction → backtest
    - Metamodel training uses real prediction signals from backtests
    - All performance metrics are derived from actual trading simulations
    - No mock or simulated data is used anywhere
    """

    def __init__(self, config_path: str):
        """
        Initialize the multi-model orchestrator.
        
        Args:
            config_path: Path to the multi-model experiment configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize results storage
        self.base_model_results = []
        self.metamodel_result = {}
        self.final_results = {}
        
        # Setup output directory
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data providers
        self.data_provider = self._create_data_provider(self.config.get('data_provider', {}))
        self.factor_data_provider = self._create_factor_data_provider(self.config.get('factor_data_provider', {}))
        
        logger.info(f"MultiModelOrchestrator initialized with config: {config_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete multi-model experiment workflow.
        
        Returns:
            Comprehensive results dictionary containing all experiment outcomes
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING MULTI-MODEL EXPERIMENT")
        logger.info("="*60)
        
        try:
            # Step 1: Train and backtest each base model with HPO
            logger.info("STEP 1: Training base models with HPO")
            self._train_base_models()
            
            # Step 2: Train and backtest metamodel with HPO
            logger.info("STEP 2: Training metamodel with HPO")
            self._train_metamodel()
            
            # Step 3: Generate comprehensive report
            logger.info("STEP 3: Generating comprehensive report")
            self._generate_comprehensive_report()
            
            # Step 4: Save results
            self._save_results()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.final_results['execution_time'] = execution_time
            self.final_results['status'] = 'SUCCESS'
            
            logger.info("="*60)
            logger.info("MULTI-MODEL EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info("="*60)
            
            return self.final_results
            
        except Exception as e:
            logger.error(f"Multi-model experiment failed: {e}")
            self.final_results['status'] = 'FAILED'
            self.final_results['error'] = str(e)
            raise

    def _create_data_provider(self, config: Dict[str, Any]):
        """Create data provider from configuration."""
        provider_type = config.get('type')
        params = config.get('parameters', {})
        
        if provider_type == "YFinanceProvider":
            from src.trading_system.data.yfinance_provider import YFinanceProvider
            return YFinanceProvider(**params)
        else:
            raise ValueError(f"Unsupported data provider type: {provider_type}")

    def _create_factor_data_provider(self, config: Dict[str, Any]):
        """Create factor data provider from configuration."""
        provider_type = config.get('type')
        params = config.get('parameters', {})
        
        if provider_type == "FF5DataProvider":
            from src.trading_system.data.ff5_provider import FF5DataProvider
            return FF5DataProvider(**params)
        else:
            return None  # Allow no factor data provider

    def _train_base_models(self):
        """Train all base models with hyperparameter optimization."""
        base_models_config = self.config.get('base_models', [])
        
        if not base_models_config:
            raise ValueError("No base models configured in experiment config")
        
        logger.info(f"Training {len(base_models_config)} base models")
        
        model_trainer = ModelTrainerWithHPO(
            self.config, 
            data_provider=self.data_provider, 
            factor_data_provider=self.factor_data_provider
        )
        
        for i, model_config in enumerate(base_models_config):
            model_type = model_config['model_type']
            n_trials = model_config.get('hpo_trials', 50)
            hpo_metric = model_config.get('hpo_metric', 'sharpe_ratio')
            
            logger.info(f"Training base model {i+1}/{len(base_models_config)}: {model_type}")
            logger.info(f"HPO trials: {n_trials}, metric: {hpo_metric}")
            
            try:
                model_result = model_trainer.optimize_and_train(
                    model_type=model_type,
                    n_trials=n_trials,
                    hpo_metric=hpo_metric
                )
                
                self.base_model_results.append(model_result)
                
                logger.info(f"✓ {model_type} completed successfully")
                logger.info(f"  Model ID: {model_result['model_id']}")
                logger.info(f"  Best {hpo_metric}: {model_result['performance_metrics'].get(hpo_metric, 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"✗ {model_type} failed: {e}")
                # Continue with other models
                continue
        
        if not self.base_model_results:
            raise RuntimeError("All base models failed to train")
        
        logger.info(f"Successfully trained {len(self.base_model_results)} base models")

    def _train_metamodel(self):
        """Train the metamodel with hyperparameter optimization."""
        if not self.base_model_results:
            raise ValueError("No base model results available for metamodel training")
        
        metamodel_config = self.config.get('metamodel', {})
        n_trials = metamodel_config.get('hpo_trials', 50)
        hpo_metric = metamodel_config.get('hpo_metric', 'sharpe_ratio')
        methods_to_try = metamodel_config.get('methods_to_try', ['ridge', 'lasso', 'equal'])
        
        logger.info(f"Training metamodel with {n_trials} HPO trials")
        logger.info(f"Methods to try: {methods_to_try}")
        
        metamodel_trainer = MetaModelTrainerWithHPO(
            model_results=self.base_model_results,
            data_config=self.config
        )
        
        try:
            self.metamodel_result = metamodel_trainer.optimize_and_train(
                n_trials=n_trials,
                hpo_metric=hpo_metric,
                methods_to_try=methods_to_try
            )
            
            logger.info("✓ Metamodel completed successfully")
            logger.info(f"  Model ID: {self.metamodel_result['model_id']}")
            logger.info(f"  Best method: {self.metamodel_result['best_params']['method']}")
            logger.info(f"  Best {hpo_metric}: {self.metamodel_result['performance_metrics'].get(hpo_metric, 0.0):.4f}")
            logger.info(f"  Strategy weights: {self.metamodel_result['weights']}")
            
        except Exception as e:
            logger.error(f"✗ Metamodel training failed: {e}")
            raise

    def _generate_comprehensive_report(self):
        """Generate comprehensive experiment report."""
        logger.info("Generating comprehensive experiment report...")
        
        # Calculate summary statistics
        base_model_summary = self._calculate_base_model_summary()
        metamodel_summary = self._calculate_metamodel_summary()
        system_summary = self._calculate_system_summary()
        
        self.final_results = {
            'experiment_info': {
                'name': self.config['experiment']['name'],
                'config_path': str(self.config_path),
                'output_dir': str(self.output_dir),
                'timestamp': datetime.now().isoformat()
            },
            'base_models': {
                'results': self.base_model_results,
                'summary': base_model_summary
            },
            'metamodel': {
                'result': self.metamodel_result,
                'summary': metamodel_summary
            },
            'system_summary': system_summary,
            'configuration': self.config
        }
        
        logger.info("Comprehensive report generated successfully")

    def _calculate_base_model_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for base models."""
        if not self.base_model_results:
            return {}
        
        sharpe_ratios = [m['performance_metrics'].get('sharpe_ratio', 0.0) for m in self.base_model_results]
        total_returns = [m['performance_metrics'].get('total_return', 0.0) for m in self.base_model_results]
        
        return {
            'total_models': len(self.base_model_results),
            'model_types': [m['model_type'] for m in self.base_model_results],
            'best_model': self.base_model_results[np.argmax(sharpe_ratios)]['model_type'],
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'best_sharpe_ratio': np.max(sharpe_ratios),
            'avg_total_return': np.mean(total_returns),
            'best_total_return': np.max(total_returns),
            'model_ids': [m['model_id'] for m in self.base_model_results]
        }

    def _calculate_metamodel_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for metamodel."""
        if not self.metamodel_result:
            return {}
        
        return {
            'model_id': self.metamodel_result.get('model_id', ''),
            'method': self.metamodel_result.get('best_params', {}).get('method', ''),
            'alpha': self.metamodel_result.get('best_params', {}).get('alpha', 1.0),
            'sharpe_ratio': self.metamodel_result.get('performance_metrics', {}).get('sharpe_ratio', 0.0),
            'total_return': self.metamodel_result.get('performance_metrics', {}).get('total_return', 0.0),
            'max_drawdown': self.metamodel_result.get('performance_metrics', {}).get('max_drawdown', 0.0),
            'weights': self.metamodel_result.get('weights', {}),
            'base_models_used': self.metamodel_result.get('base_models', [])
        }

    def _calculate_system_summary(self) -> Dict[str, Any]:
        """Calculate overall system summary."""
        base_summary = self._calculate_base_model_summary()
        meta_summary = self._calculate_metamodel_summary()
        
        # Determine if system meets requirements
        system_requirements = self.config.get('system_requirements', {})
        min_sharpe = system_requirements.get('min_sharpe_ratio', 0.5)
        max_drawdown_threshold = system_requirements.get('max_drawdown_threshold', -0.3)
        
        metamodel_sharpe = meta_summary.get('sharpe_ratio', 0.0)
        metamodel_drawdown = meta_summary.get('max_drawdown', 0.0)
        
        meets_sharpe = metamodel_sharpe >= min_sharpe
        meets_drawdown = metamodel_drawdown >= max_drawdown_threshold
        overall_valid = meets_sharpe and meets_drawdown
        
        return {
            'overall_performance': f"Sharpe: {metamodel_sharpe:.3f}, Return: {meta_summary.get('total_return', 0.0):.2%}, DD: {metamodel_drawdown:.2%}",
            'validation_status': "✓ Valid" if overall_valid else "✗ Invalid",
            'meets_requirements': {
                'sharpe_ratio': meets_sharpe,
                'max_drawdown': meets_drawdown,
                'overall_valid': overall_valid
            },
            'system_improvement': {
                'vs_best_base_model': metamodel_sharpe - base_summary.get('best_sharpe_ratio', 0.0),
                'vs_avg_base_model': metamodel_sharpe - base_summary.get('avg_sharpe_ratio', 0.0)
            },
            'diversification_benefit': len(base_summary.get('model_types', [])) > 1
        }

    def _save_results(self):
        """Save all experiment results to files."""
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save main results as JSON
        results_file = self.output_dir / 'multi_model_experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.final_results, f, indent=2, default=str)
        
        # Save summary report as text
        summary_file = self.output_dir / 'experiment_summary.txt'
        with open(summary_file, 'w') as f:
            self._write_summary_report(f)
        
        # Save configuration
        config_file = self.output_dir / 'experiment_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Results saved to {self.output_dir}")

    def _write_summary_report(self, file_handle):
        """Write a human-readable summary report."""
        file_handle.write("="*60 + "\n")
        file_handle.write("MULTI-MODEL EXPERIMENT SUMMARY\n")
        file_handle.write("="*60 + "\n\n")
        
        # Experiment info
        file_handle.write(f"Experiment: {self.final_results['experiment_info']['name']}\n")
        file_handle.write(f"Timestamp: {self.final_results['experiment_info']['timestamp']}\n")
        file_handle.write(f"Status: {self.final_results.get('status', 'UNKNOWN')}\n\n")
        
        # Base models summary
        base_summary = self.final_results['base_models']['summary']
        file_handle.write("BASE MODELS:\n")
        file_handle.write("-" * 20 + "\n")
        file_handle.write(f"Total models trained: {base_summary.get('total_models', 0)}\n")
        file_handle.write(f"Model types: {', '.join(base_summary.get('model_types', []))}\n")
        file_handle.write(f"Best model: {base_summary.get('best_model', 'N/A')}\n")
        file_handle.write(f"Best Sharpe ratio: {base_summary.get('best_sharpe_ratio', 0.0):.3f}\n")
        file_handle.write(f"Average Sharpe ratio: {base_summary.get('avg_sharpe_ratio', 0.0):.3f}\n\n")
        
        # Metamodel summary
        meta_summary = self.final_results['metamodel']['summary']
        file_handle.write("METAMODEL:\n")
        file_handle.write("-" * 20 + "\n")
        file_handle.write(f"Model ID: {meta_summary.get('model_id', 'N/A')}\n")
        file_handle.write(f"Method: {meta_summary.get('method', 'N/A')}\n")
        file_handle.write(f"Alpha: {meta_summary.get('alpha', 1.0):.3f}\n")
        file_handle.write(f"Sharpe ratio: {meta_summary.get('sharpe_ratio', 0.0):.3f}\n")
        file_handle.write(f"Total return: {meta_summary.get('total_return', 0.0):.2%}\n")
        file_handle.write(f"Max drawdown: {meta_summary.get('max_drawdown', 0.0):.2%}\n")
        file_handle.write(f"Strategy weights: {meta_summary.get('weights', {})}\n\n")
        
        # System summary
        system_summary = self.final_results['system_summary']
        file_handle.write("SYSTEM SUMMARY:\n")
        file_handle.write("-" * 20 + "\n")
        file_handle.write(f"Overall performance: {system_summary.get('overall_performance', 'N/A')}\n")
        file_handle.write(f"Validation status: {system_summary.get('validation_status', 'N/A')}\n")
        
        validation = system_summary.get('meets_requirements', {})
        file_handle.write(f"Meets Sharpe requirement: {validation.get('sharpe_ratio', False)}\n")
        file_handle.write(f"Meets drawdown requirement: {validation.get('max_drawdown', False)}\n")
        file_handle.write(f"Overall valid: {validation.get('overall_valid', False)}\n")
        
        improvement = system_summary.get('system_improvement', {})
        file_handle.write(f"Improvement vs best base model: {improvement.get('vs_best_base_model', 0.0):.3f}\n")
        file_handle.write(f"Improvement vs average base model: {improvement.get('vs_avg_base_model', 0.0):.3f}\n")
        
        file_handle.write("\n" + "="*60 + "\n")
