"""
Result Validator
===============

Utility class for validating experiment results and data integrity.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class ResultValidator:
    """
    Validates experiment results and data integrity.
    
    This class provides comprehensive validation for:
    - Experiment result dictionaries
    - Strategy returns files
    - Returns matrices
    - File existence and format
    """

    @staticmethod
    def validate_experiment_result(result: Dict[str, Any]) -> bool:
        """
        Validate a single experiment result dictionary.
        
        Args:
            result: Experiment result dictionary
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        logger.debug("Validating experiment result")
        
        # Check required keys
        required_keys = ['trained_model_id', 'performance_metrics']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")
        
        # Validate model ID
        model_id = result.get('trained_model_id')
        if not model_id or not isinstance(model_id, str):
            raise ValidationError("trained_model_id must be a non-empty string")
        
        # Validate performance metrics
        metrics = result.get('performance_metrics', {})
        if not isinstance(metrics, dict):
            raise ValidationError("performance_metrics must be a dictionary")
        
        # Check for at least one metric
        if not metrics:
            logger.warning("No performance metrics found")
        
        # Validate returns_path if present
        returns_path = result.get('returns_path')
        if returns_path:
            if not isinstance(returns_path, (str, Path)):
                raise ValidationError("returns_path must be a string or Path")
            
            # Check if file exists
            path = Path(returns_path)
            if not path.exists():
                raise ValidationError(f"Returns file does not exist: {returns_path}")
            
            # Validate file format
            if not ResultValidator.validate_returns_file(str(path)):
                raise ValidationError(f"Invalid returns file format: {returns_path}")
        
        logger.debug("✓ Experiment result validation passed")
        return True

    @staticmethod
    def validate_returns_file(file_path: str) -> bool:
        """
        Validate a strategy returns file format.
        
        Args:
            file_path: Path to the returns file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            logger.debug(f"Validating returns file: {file_path}")
            
            # Check file exists
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Load and check format
            returns_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check required columns
            if 'daily_return' not in returns_df.columns:
                logger.error(f"Missing 'daily_return' column in {file_path}")
                return False
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(returns_df['daily_return']):
                logger.error(f"'daily_return' column is not numeric in {file_path}")
                return False
            
            # Check for reasonable values
            returns = returns_df['daily_return']
            extreme_returns = returns.abs() > 1.0  # > 100% daily return
            if extreme_returns.any():
                extreme_count = extreme_returns.sum()
                logger.warning(
                    f"{extreme_count} extreme returns (>100%) found in {file_path}"
                )
            
            # Check for sufficient data
            if len(returns) < 10:
                logger.warning(f"Very few observations ({len(returns)}) in {file_path}")
            
            # Check date continuity
            date_diff = returns.index.to_series().diff()
            max_gap = date_diff.max()
            if max_gap > pd.Timedelta(days=7):
                logger.warning(f"Large time gaps (up to {max_gap}) in {file_path}")
            
            logger.debug(f"✓ Returns file validation passed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate returns file {file_path}: {e}")
            return False

    @staticmethod
    def validate_returns_matrix(returns_df: pd.DataFrame) -> bool:
        """
        Validate a strategy returns matrix.
        
        Args:
            returns_df: DataFrame with strategies as columns and dates as index
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        logger.debug("Validating returns matrix")
        
        # Check if DataFrame is empty
        if returns_df.empty:
            raise ValidationError("Returns matrix is empty")
        
        # Check minimum dimensions
        if len(returns_df.columns) < 2:
            raise ValidationError(
                f"Need at least 2 strategies, got {len(returns_df.columns)}"
            )
        
        if len(returns_df) < 20:
            logger.warning(
                f"Few observations ({len(returns_df)}), recommend at least 20"
            )
        
        # Check for missing values
        missing_pct = returns_df.isnull().sum() / len(returns_df)
        if missing_pct.max() > 0.1:  # More than 10% missing
            problematic = missing_pct[missing_pct > 0.1]
            logger.warning(
                f"High missing data rate:\n{problematic}"
            )
        
        # Check for extreme returns
        extreme_mask = returns_df.abs() > 0.5  # Daily returns > 50%
        if extreme_mask.any().any():
            extreme_counts = extreme_mask.sum()
            logger.warning(
                f"Extreme returns (>50%) detected:\n"
                f"{extreme_counts[extreme_counts > 0]}"
            )
        
        # Check for zero variance strategies
        zero_variance = returns_df.std() == 0
        if zero_variance.any():
            zero_var_strategies = returns_df.columns[zero_variance].tolist()
            logger.warning(f"Strategies with zero variance: {zero_var_strategies}")
        
        # Check time series continuity
        date_diff = returns_df.index.to_series().diff()
        max_gap = date_diff.max()
        if max_gap > pd.Timedelta(days=5):
            logger.warning(f"Time series has gaps up to {max_gap}")
        
        logger.debug("✓ Returns matrix validation passed")
        return True

    @staticmethod
    def validate_model_directory(model_id: str, base_dir: str = "./results") -> bool:
        """
        Validate that a model directory contains all required files.
        
        Args:
            model_id: Model identifier
            base_dir: Base results directory
            
        Returns:
            True if directory is valid, False otherwise
        """
        logger.debug(f"Validating model directory: {model_id}")
        
        model_dir = Path(base_dir) / model_id
        
        if not model_dir.exists():
            logger.error(f"Model directory does not exist: {model_dir}")
            return False
        
        # Check for required files
        required_files = [
            "strategy_returns.csv"
        ]
        
        missing_files = []
        for filename in required_files:
            file_path = model_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.error(
                f"Missing required files in {model_dir}: {missing_files}"
            )
            return False
        
        # Validate the returns file
        returns_file = model_dir / "strategy_returns.csv"
        if not ResultValidator.validate_returns_file(str(returns_file)):
            return False
        
        logger.debug(f"✓ Model directory validation passed: {model_id}")
        return True

    @staticmethod
    def get_validation_summary(strategy_names: List[str], 
                             base_dir: str = "./results") -> Dict[str, Any]:
        """
        Get a validation summary for multiple strategies.
        
        Args:
            strategy_names: List of strategy names to validate
            base_dir: Base results directory
            
        Returns:
            Summary dictionary with validation results
        """
        summary = {
            'total_strategies': len(strategy_names),
            'valid_strategies': [],
            'invalid_strategies': [],
            'missing_strategies': [],
            'validation_details': {}
        }
        
        for strategy_name in strategy_names:
            try:
                if ResultValidator.validate_model_directory(strategy_name, base_dir):
                    summary['valid_strategies'].append(strategy_name)
                else:
                    summary['invalid_strategies'].append(strategy_name)
            except Exception as e:
                summary['missing_strategies'].append(strategy_name)
                summary['validation_details'][strategy_name] = str(e)
        
        return summary

    @staticmethod
    def validate_experiment_config(config: Dict[str, Any]) -> bool:
        """
        Validate an experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        logger.debug("Validating experiment configuration")
        
        # Check required sections
        required_sections = ['universe', 'periods', 'data_provider']
        missing_sections = [section for section in required_sections 
                          if section not in config]
        
        if missing_sections:
            raise ValidationError(f"Missing required sections: {missing_sections}")
        
        # Validate universe
        universe = config.get('universe', [])
        if not isinstance(universe, list) or len(universe) == 0:
            raise ValidationError("universe must be a non-empty list")
        
        # Validate periods
        periods = config.get('periods', {})
        required_periods = ['train', 'test']
        for period in required_periods:
            if period not in periods:
                raise ValidationError(f"Missing {period} period")
            
            period_config = periods[period]
            if 'start' not in period_config or 'end' not in period_config:
                raise ValidationError(f"Missing start/end in {period} period")
        
        # Validate data provider
        data_provider = config.get('data_provider', {})
        if 'type' not in data_provider:
            raise ValidationError("data_provider must have 'type' field")
        
        logger.debug("✓ Experiment configuration validation passed")
        return True




