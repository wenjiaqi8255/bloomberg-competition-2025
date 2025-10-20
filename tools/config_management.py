#!/usr/bin/env python3
"""
Configuration Management Tool
============================

Command-line tool for managing trading system configurations.
Provides validation, listing, migration, and template generation functionality.
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_system.validation.config import (
    PortfolioConfigValidator,
    SchemaValidator
)
from trading_system.validation import ValidationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Main configuration management class."""
    
    def __init__(self, configs_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            configs_dir: Path to the configurations directory
        """
        self.configs_dir = Path(configs_dir)
        self.registry_path = self.configs_dir / "CONFIG_REGISTRY.yaml"
        self.schemas_dir = self.configs_dir / "schemas"
        
        # Load registry
        self.registry = self._load_registry()
        
        # Initialize validators
        self.portfolio_validator = PortfolioConfigValidator()
        self.schema_validator = SchemaValidator()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the configuration registry."""
        if not self.registry_path.exists():
            logger.error(f"Configuration registry not found: {self.registry_path}")
            return {}
        
        try:
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}
    
    def validate_config(self, config_path: str, schema_name: Optional[str] = None) -> ValidationResult:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            schema_name: Optional schema name for validation
            
        Returns:
            ValidationResult with validation outcome
        """
        config_path = Path(config_path)
        if not config_path.exists():
            result = ValidationResult()
            result.add_error(f"Configuration file not found: {config_path}")
            return result
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # For now, just do basic validation
            result = ValidationResult()
            
            # Check for required top-level fields
            if not isinstance(config_data, dict):
                result.add_error("Configuration must be a dictionary")
            else:
                # Basic validation - check for common required fields
                if 'data_provider' not in config_data:
                    result.add_warning("Missing 'data_provider' configuration")
                if 'strategy' not in config_data:
                    result.add_warning("Missing 'strategy' configuration")
            
            # Additional schema validation if requested
            if schema_name and result.is_valid:
                schema_result = self.schema_validator.validate(config_data, schema_name)
                if not schema_result.is_valid:
                    result.issues.extend(schema_result.issues)
                    result.is_valid = False
            
            return result
            
        except Exception as e:
            result = ValidationResult()
            result.add_error(f"Failed to validate configuration: {e}")
            return result
    
    def _detect_config_type(self, config_data: Dict[str, Any]) -> str:
        """Detect the type of configuration."""
        # Check for multi-model specific keys
        if 'base_models' in config_data or 'metamodel' in config_data:
            return "multi_model"
        
        # Check for prediction specific keys
        if 'prediction' in config_data or config_data.get('strategy', {}).get('type') == 'meta':
            return "prediction"
        
        # Check for single experiment specific keys
        if 'training_setup' in config_data or 'backtest' in config_data:
            return "single_experiment"
        
        # Default to single experiment
        return "single_experiment"
    
    def list_configs(self, config_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available configurations.
        
        Args:
            config_type: Filter by configuration type (single_experiment, multi_model, prediction)
            status: Filter by status (active, template, archived)
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        if not self.registry:
            logger.error("Registry not loaded")
            return configs
        
        # Get active configs
        active_configs = self.registry.get('active_configs', {})
        for category, configs_dict in active_configs.items():
            if config_type and category != config_type:
                continue
            
            for name, config_info in configs_dict.items():
                if status and config_info.get('status') != status:
                    continue
                
                configs.append({
                    'name': name,
                    'category': category,
                    'file': config_info.get('file'),
                    'description': config_info.get('description'),
                    'status': config_info.get('status', 'active'),
                    'strategy_type': config_info.get('strategy_type'),
                    'portfolio_method': config_info.get('portfolio_method'),
                    'tested': config_info.get('tested'),
                    'entry_point': config_info.get('entry_point'),
                    'tags': config_info.get('tags', [])
                })
        
        # Get template configs
        template_configs = self.registry.get('template_configs', {})
        for name, config_info in template_configs.items():
            if status and status != 'template':
                continue
            
            configs.append({
                'name': name,
                'category': 'template',
                'file': config_info.get('file'),
                'description': config_info.get('description'),
                'status': 'template',
                'strategy_type': config_info.get('strategy_type'),
                'portfolio_method': config_info.get('portfolio_method'),
                'tested': None,
                'entry_point': None,
                'tags': config_info.get('tags', [])
            })
        
        return configs
    
    def migrate_config(self, source_path: str, target_path: str, 
                      config_type: str, description: str) -> bool:
        """
        Migrate a legacy configuration to the new format.
        
        Args:
            source_path: Path to the source configuration file
            target_path: Path to the target configuration file
            config_type: Type of configuration (single_experiment, multi_model, prediction)
            description: Description of the configuration
            
        Returns:
            True if migration successful, False otherwise
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        if not source_path.exists():
            logger.error(f"Source configuration not found: {source_path}")
            return False
        
        try:
            # Load source configuration
            with open(source_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate source configuration
            result = self.validate_config(str(source_path))
            if not result.is_valid:
                logger.error(f"Source configuration validation failed: {result.get_summary()}")
                return False
            
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save migrated configuration
            with open(target_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Successfully migrated configuration: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def generate_template(self, config_type: str, output_path: str, 
                         strategy_type: str = "fama_french_5",
                         portfolio_method: str = "box_based") -> bool:
        """
        Generate a configuration template.
        
        Args:
            config_type: Type of configuration to generate
            output_path: Path to save the template
            strategy_type: Strategy type for the template
            portfolio_method: Portfolio construction method
            
        Returns:
            True if generation successful, False otherwise
        """
        output_path = Path(output_path)
        
        try:
            # Generate template based on type
            if config_type == "single_experiment":
                template = self._generate_single_experiment_template(strategy_type, portfolio_method)
            elif config_type == "multi_model":
                template = self._generate_multi_model_template()
            elif config_type == "prediction":
                template = self._generate_prediction_template(strategy_type, portfolio_method)
            else:
                logger.error(f"Unknown configuration type: {config_type}")
                return False
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save template
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Successfully generated template: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return False
    
    def _generate_single_experiment_template(self, strategy_type: str, portfolio_method: str) -> Dict[str, Any]:
        """Generate a single experiment configuration template."""
        template = {
            "data_provider": {
                "type": "YFinanceProvider",
                "parameters": {
                    "max_retries": 3,
                    "retry_delay": 1.0
                }
            },
            "training_setup": {
                "model": {
                    "model_type": strategy_type,
                    "config": {
                        "regularization": "ridge",
                        "alpha": 1.0,
                        "standardize": True
                    }
                },
                "feature_engineering": {
                    "include_technical": False,
                    "include_cross_sectional": True,
                    "include_theoretical": False,
                    "normalize_features": True,
                    "normalization_method": "minmax",
                    "handle_missing": "forward_fill"
                },
                "parameters": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                }
            },
            "strategy": {
                "type": strategy_type,
                "name": f"{strategy_type}_strategy",
                "parameters": {
                    "model_id": "placeholder_model_id",
                    "lookback_days": 252,
                    "risk_free_rate": 0.02,
                    "portfolio_construction": {
                        "method": portfolio_method,
                        "stocks_per_box": 3,
                        "allocation_method": "equal"
                    }
                }
            },
            "backtest": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "initial_capital": 1000000,
                "benchmark_symbol": "SPY",
                "commission_rate": 0.001,
                "slippage_rate": 0.0005,
                "rebalance_frequency": "weekly"
            }
        }
        
        # Add factor data provider for FF5 strategies
        if strategy_type in ["fama_french_5", "ff5_regression"]:
            template["factor_data_provider"] = {
                "type": "FF5DataProvider",
                "parameters": {
                    "data_frequency": "daily"
                }
            }
        
        return template
    
    def _generate_multi_model_template(self) -> Dict[str, Any]:
        """Generate a multi-model configuration template."""
        return {
            "experiment": {
                "name": "multi_model_experiment",
                "output_dir": "./results/multi_model_experiment"
            },
            "data_provider": {
                "type": "YFinanceProvider",
                "parameters": {
                    "max_retries": 3,
                    "retry_delay": 1.0
                }
            },
            "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "periods": {
                "train": {
                    "start": "2024-01-01",
                    "end": "2024-06-30"
                },
                "test": {
                    "start": "2024-07-01",
                    "end": "2024-12-31"
                }
            },
            "base_models": [
                {
                    "model_type": "ff5_regression",
                    "hpo_trials": 10,
                    "hpo_metric": "sharpe_ratio"
                },
                {
                    "model_type": "xgboost",
                    "hpo_trials": 10,
                    "hpo_metric": "sharpe_ratio"
                }
            ],
            "metamodel": {
                "hpo_trials": 10,
                "hpo_metric": "sharpe_ratio",
                "methods_to_try": ["ridge", "equal"]
            }
        }
    
    def _generate_prediction_template(self, strategy_type: str, portfolio_method: str) -> Dict[str, Any]:
        """Generate a prediction configuration template."""
        template = {
            "prediction": {
                "prediction_date": "2024-01-15"
            },
            "strategy": {
                "type": strategy_type,
                "name": f"{strategy_type}_prediction_strategy",
                "model_id": "trained_model_id",
                "model_registry_path": "./models/",
                "min_signal_strength": 0.00001,
                "enable_normalization": True,
                "normalization_method": "minmax"
            },
            "data_provider": {
                "type": "YFinanceProvider",
                "parameters": {
                    "max_retries": 3,
                    "retry_delay": 1.0
                }
            },
            "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "portfolio_construction": {
                "method": portfolio_method,
                "stocks_per_box": 3,
                "allocation_method": "equal"
            },
            "output": {
                "format": "detailed",
                "include_risk_analysis": True,
                "save_results": True,
                "output_path": "./prediction_results/"
            }
        }
        
        # Add meta strategy specific fields
        if strategy_type == "meta":
            template["strategy"]["base_model_ids"] = ["model_1", "model_2"]
            template["strategy"]["meta_weights"] = {
                "model_1": 0.6,
                "model_2": 0.4
            }
        
        return template
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """Get all available configuration options."""
        return self.registry.get('available_options', {})
    
    def get_config_info(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific configuration."""
        active_configs = self.registry.get('active_configs', {})
        for category, configs_dict in active_configs.items():
            if config_name in configs_dict:
                return configs_dict[config_name]
        
        template_configs = self.registry.get('template_configs', {})
        if config_name in template_configs:
            return template_configs[config_name]
        
        return None


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description="Trading System Configuration Management Tool")
    parser.add_argument("--configs-dir", default="configs", help="Path to configurations directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config_path", help="Path to configuration file")
    validate_parser.add_argument("--schema", help="Schema name for validation")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available configurations")
    list_parser.add_argument("--type", choices=["single_experiment", "multi_model", "prediction"], 
                           help="Filter by configuration type")
    list_parser.add_argument("--status", choices=["active", "template", "archived"], 
                           help="Filter by status")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate a legacy configuration")
    migrate_parser.add_argument("source_path", help="Path to source configuration")
    migrate_parser.add_argument("target_path", help="Path to target configuration")
    migrate_parser.add_argument("--type", required=True, 
                              choices=["single_experiment", "multi_model", "prediction"],
                              help="Configuration type")
    migrate_parser.add_argument("--description", required=True, help="Configuration description")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a configuration template")
    generate_parser.add_argument("config_type", 
                               choices=["single_experiment", "multi_model", "prediction"],
                               help="Type of configuration to generate")
    generate_parser.add_argument("output_path", help="Path to save the template")
    generate_parser.add_argument("--strategy-type", default="fama_french_5",
                               choices=["ml", "fama_macbeth", "fama_french_5", "ff5_regression", "meta"],
                               help="Strategy type for the template")
    generate_parser.add_argument("--portfolio-method", default="box_based",
                               choices=["quantitative", "box_based"],
                               help="Portfolio construction method")
    
    # Options command
    options_parser = subparsers.add_parser("options", help="Show available configuration options")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about a configuration")
    info_parser.add_argument("config_name", help="Name of the configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize config manager
    config_manager = ConfigManager(args.configs_dir)
    
    # Execute command
    if args.command == "validate":
        result = config_manager.validate_config(args.config_path, args.schema)
        print(result.get_summary())
        if not result.is_valid:
            for issue in result.get_errors():
                print(f"  ❌ {issue}")
        for issue in result.get_warnings():
            print(f"  ⚠️  {issue}")
    
    elif args.command == "list":
        configs = config_manager.list_configs(args.type, args.status)
        if not configs:
            print("No configurations found matching the criteria.")
            return
        
        print(f"Found {len(configs)} configurations:")
        for config in configs:
            print(f"\n📁 {config['name']} ({config['category']})")
            print(f"   File: {config['file']}")
            print(f"   Description: {config['description']}")
            print(f"   Status: {config['status']}")
            if config['strategy_type']:
                print(f"   Strategy: {config['strategy_type']}")
            if config['portfolio_method']:
                print(f"   Portfolio: {config['portfolio_method']}")
            if config['tested']:
                print(f"   Last tested: {config['tested']}")
            if config['tags']:
                print(f"   Tags: {', '.join(config['tags'])}")
    
    elif args.command == "migrate":
        success = config_manager.migrate_config(
            args.source_path, args.target_path, args.type, args.description
        )
        if success:
            print("✅ Migration completed successfully")
        else:
            print("❌ Migration failed")
            sys.exit(1)
    
    elif args.command == "generate":
        success = config_manager.generate_template(
            args.config_type, args.output_path, args.strategy_type, args.portfolio_method
        )
        if success:
            print("✅ Template generated successfully")
        else:
            print("❌ Template generation failed")
            sys.exit(1)
    
    elif args.command == "options":
        options = config_manager.get_available_options()
        print("Available Configuration Options:")
        for category, values in options.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for value in values:
                print(f"  - {value}")
    
    elif args.command == "info":
        info = config_manager.get_config_info(args.config_name)
        if info:
            print(f"Configuration: {args.config_name}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Configuration '{args.config_name}' not found")


if __name__ == "__main__":
    main()
