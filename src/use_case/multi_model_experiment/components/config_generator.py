"""
Model Configuration Generator
=============================

This component generates single-model experiment configurations from multi-model configurations.
It ensures that each base model gets a complete, valid configuration that can be used
with the ExperimentOrchestrator.
"""

import logging
import copy
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelConfigGenerator:
    """
    Generates single-model experiment configurations from multi-model configurations.
    
    This class extracts the shared configuration (data providers, universe, periods, etc.)
    and injects model-specific parameters to create complete experiment configurations
    that can be used with ExperimentOrchestrator.
    """

    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize the config generator.
        
        Args:
            base_config: The multi-model configuration dictionary
        """
        self.base_config = base_config
        logger.debug("ModelConfigGenerator initialized")

    def generate_for_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete experiment configuration for a single model.
        
        Args:
            model_config: Model-specific configuration containing model_type, parameters, etc.
            
        Returns:
            Complete experiment configuration dictionary
        """
        logger.debug(f"Generating config for model: {model_config.get('model_type', 'unknown')}")
        
        # Start with a deep copy of the base config
        exp_config = copy.deepcopy(self.base_config)
        
        # Remove multi-model specific sections
        exp_config.pop('base_models', None)
        exp_config.pop('metamodel', None)
        exp_config.pop('fail_fast', None)
        
        # Create training_setup section
        exp_config['training_setup'] = self._create_training_setup(model_config)
        
        # Ensure all required sections exist
        self._validate_required_sections(exp_config)
        
        logger.debug(f"Generated config for {model_config.get('model_type')} with {len(exp_config)} sections")
        return exp_config

    def _create_training_setup(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the training_setup section for a single model.
        
        Args:
            model_config: Model-specific configuration
            
        Returns:
            training_setup configuration dictionary
        """
        model_type = model_config.get('model_type')
        if not model_type:
            raise ValueError("model_config must contain 'model_type'")
        
        # Extract HPO configuration
        hpo_trials = model_config.get('hpo_trials', 10)
        hpo_metric = model_config.get('hpo_metric', 'sharpe_ratio')
        
        # Extract model parameters (excluding HPO-specific ones)
        model_parameters = {k: v for k, v in model_config.get('parameters', {}).items()}
        
        training_setup = {
            'model': {
                'model_type': model_type,
                **model_parameters
            },
            'feature_engineering': self.base_config.get('feature_engineering', {}),
            'hyperparameter_optimization': {
                'enabled': True,
                'n_trials': hpo_trials,
                'metric': hpo_metric,
                'cv_folds': 5,
                'time_series_cv': True
            },
            'parameters': {
                'validation_split': 0.2,
                'early_stopping_rounds': 10
            }
        }
        
        return training_setup

    def _validate_required_sections(self, config: Dict[str, Any]):
        """
        Validate that all required sections are present in the configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If required sections are missing
        """
        required_sections = [
            'universe',
            'periods', 
            'data_provider',
            'training_setup'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate training_setup structure
        training_setup = config['training_setup']
        required_training_sections = ['model']
        for section in required_training_sections:
            if section not in training_setup:
                raise ValueError(f"Missing required training_setup section: {section}")
        
        # Validate model configuration
        model_config = training_setup['model']
        if 'model_type' not in model_config:
            raise ValueError("training_setup.model must contain 'model_type'")

    def save_config_to_file(self, config: Dict[str, Any], file_path: str) -> str:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            file_path: Path where to save the configuration
            
        Returns:
            Path to the saved file
        """
        import yaml
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.debug(f"Saved configuration to {path}")
        return str(path)

    @staticmethod
    def create_temp_config_path(model_type: str, suffix: str = "yaml") -> str:
        """
        Create a temporary configuration file path.
        
        Args:
            model_type: Type of model (for naming)
            suffix: File suffix (default: yaml)
            
        Returns:
            Temporary file path
        """
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        filename = f"exp_config_{model_type}_{os.getpid()}.{suffix}"
        return os.path.join(temp_dir, filename)
