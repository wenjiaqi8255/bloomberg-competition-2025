"""
Experiment Result Validator
==========================

Validates experiment result dictionaries and files.
Enhanced version of the original ResultValidator with better error reporting.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class ExperimentResultValidator(BaseValidator):
    """
    Validates experiment result dictionaries and files.
    
    Enhanced version with better error reporting and validation result structure.
    Maintains backward compatibility with the original ResultValidator.
    """
    
    def __init__(self):
        super().__init__("ExperimentResultValidator")
    
    def validate(self, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate an experiment result dictionary.
        
        Args:
            result: Experiment result dictionary to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        self.log_validation_start("experiment result")
        validation_result = ValidationResult()
        
        # Check required keys
        required_keys = ['trained_model_id', 'performance_metrics']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            validation_result.add_error(f"Missing required keys: {missing_keys}",
                                      field="required_keys", suggestion="Ensure all required result fields are present")
            return validation_result
        
        # Validate model ID
        model_id = result.get('trained_model_id')
        if not model_id or not isinstance(model_id, str):
            validation_result.add_error("trained_model_id must be a non-empty string",
                                      field="trained_model_id")
        
        # Validate performance metrics
        metrics = result.get('performance_metrics', {})
        if not isinstance(metrics, dict):
            validation_result.add_error("performance_metrics must be a dictionary",
                                      field="performance_metrics")
        else:
            self._validate_performance_metrics(metrics, validation_result)
        
        # Validate returns_path if present
        returns_path = result.get('returns_path')
        if returns_path:
            if not isinstance(returns_path, (str, Path)):
                validation_result.add_error("returns_path must be a string or Path",
                                          field="returns_path")
            else:
                # Check if file exists
                path = Path(returns_path)
                if not path.exists():
                    validation_result.add_error(f"Returns file does not exist: {returns_path}",
                                              field="returns_path", suggestion="Check file path and ensure file was created")
                else:
                    # Validate file format
                    file_validation = self._validate_returns_file(str(path))
                    if not file_validation.is_valid:
                        validation_result.add_error(f"Invalid returns file format: {returns_path}",
                                                  field="returns_path", suggestion="Check file format and content")
                        validation_result.issues.extend(file_validation.issues)
        
        # Validate experiment metadata
        experiment_name = result.get('experiment_name')
        if experiment_name and not isinstance(experiment_name, str):
            validation_result.add_error("experiment_name must be a string",
                                      field="experiment_name")
        
        status = result.get('status')
        if status and status not in ['SUCCESS', 'FAILED', 'PARTIAL']:
            validation_result.add_warning(f"Unknown experiment status: {status}",
                                        field="status", suggestion="Expected SUCCESS, FAILED, or PARTIAL")
        
        # Validate training summary if present
        training_summary = result.get('training_summary')
        if training_summary and not isinstance(training_summary, dict):
            validation_result.add_error("training_summary must be a dictionary",
                                      field="training_summary")
        
        # Validate component stats if present
        component_stats = result.get('component_stats')
        if component_stats and not isinstance(component_stats, dict):
            validation_result.add_error("component_stats must be a dictionary",
                                      field="component_stats")
        
        self.log_validation_complete(validation_result)
        return validation_result
    
    def _validate_performance_metrics(self, metrics: Dict[str, Any], result: ValidationResult):
        """Validate performance metrics dictionary."""
        # Check for at least one metric
        if not metrics:
            result.add_warning("No performance metrics found",
                             field="performance_metrics", suggestion="Ensure performance metrics are calculated")
            return
        
        # Validate common performance metrics
        expected_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'volatility']
        for metric in expected_metrics:
            if metric in metrics:
                value = metrics[metric]
                if not isinstance(value, (int, float)):
                    result.add_error(f"Performance metric '{metric}' must be numeric",
                                   field=f"performance_metrics.{metric}")
                else:
                    # Validate reasonable ranges
                    if metric == 'sharpe_ratio':
                        if value < -5 or value > 5:
                            result.add_warning(f"Extreme Sharpe ratio: {value:.4f}",
                                             field=f"performance_metrics.{metric}", 
                                             suggestion="Verify calculation and data quality")
                    elif metric == 'max_drawdown':
                        if value > 0 or value < -1:
                            result.add_error(f"Invalid max drawdown: {value:.4f} (should be negative and >= -1)",
                                           field=f"performance_metrics.{metric}")
                    elif metric == 'volatility':
                        if value < 0 or value > 1:
                            result.add_error(f"Invalid volatility: {value:.4f} (should be between 0 and 1)",
                                           field=f"performance_metrics.{metric}")
        
        # Check for suspicious performance metrics
        if 'sharpe_ratio' in metrics and 'volatility' in metrics:
            sharpe = metrics['sharpe_ratio']
            volatility = metrics['volatility']
            if volatility > 0:
                implied_return = sharpe * volatility
                if abs(implied_return) > 0.5:  # 50% annual return
                    result.add_warning(f"High implied return: {implied_return:.2%} (Sharpe: {sharpe:.2f}, Vol: {volatility:.2%})",
                                     field="performance_metrics", suggestion="Verify return and volatility calculations")
    
    def _validate_returns_file(self, file_path: str) -> ValidationResult:
        """
        Validate a strategy returns file format.
        
        Args:
            file_path: Path to the returns file
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        try:
            self.logger.debug(f"Validating returns file: {file_path}")
            
            # Try to load as CSV
            try:
                returns_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            except Exception as e:
                result.add_error(f"Failed to load returns file as CSV: {str(e)}",
                               field="file_format", suggestion="Ensure file is valid CSV format")
                return result
            
            # Validate DataFrame structure
            if returns_df.empty:
                result.add_error("Returns file is empty",
                               field="file_content", suggestion="Ensure returns data was properly saved")
                return result
            
            # Check for required columns
            expected_columns = ['daily_return']
            missing_columns = [col for col in expected_columns if col not in returns_df.columns]
            if missing_columns:
                result.add_error(f"Missing required columns in returns file: {missing_columns}",
                               field="columns", suggestion="Ensure returns file has 'daily_return' column")
            
            # Validate return values
            if 'daily_return' in returns_df.columns:
                returns = returns_df['daily_return'].dropna()
                if len(returns) == 0:
                    result.add_error("No valid return values in file",
                                   field="returns_data", suggestion="Check for data quality issues")
                else:
                    # Check for extreme returns
                    extreme_returns = (abs(returns) > 0.2)  # 20% daily return
                    if extreme_returns.any():
                        extreme_count = extreme_returns.sum()
                        max_return = returns.max()
                        min_return = returns.min()
                        result.add_warning(f"Extreme daily returns detected ({extreme_count} occurrences, range: [{min_return:.2%}, {max_return:.2%}])",
                                         field="returns_data", suggestion="Verify these are legitimate returns")
                    
                    # Check for constant returns
                    if len(returns.unique()) == 1:
                        result.add_warning("Constant returns detected - this may indicate a data issue",
                                         field="returns_data", suggestion="Check if returns calculation is working correctly")
            
            # Validate date index
            if hasattr(returns_df.index, 'to_pydatetime'):
                date_index = returns_df.index
                if len(date_index) > 1:
                    # Check for date gaps
                    time_diffs = date_index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        max_gap = time_diffs.max()
                        if hasattr(max_gap, 'days') and max_gap.days > 7:
                            result.add_warning(f"Large time gap in returns data: {max_gap}",
                                             field="date_index", suggestion="Check if gaps are expected (e.g., market holidays)")
            
            self.logger.debug("✓ Returns file validation passed")
            
        except Exception as e:
            result.add_error(f"Returns file validation failed: {str(e)}")
            self.logger.error(f"Returns file validation error: {e}")
        
        return result
    
    def validate_experiment_result(self, result: Dict[str, Any]) -> bool:
        """
        Legacy method for backward compatibility.
        
        Args:
            result: Experiment result dictionary
            
        Returns:
            True if valid, raises ValidationError otherwise
        """
        validation_result = self.validate(result)
        if validation_result.has_errors():
            from .experiment_result_validator import ValidationError
            error_messages = [issue.message for issue in validation_result.get_errors()]
            raise ValidationError(f"Experiment result validation failed: {'; '.join(error_messages)}")
        return True
    
    def validate_returns_file(self, file_path: str) -> bool:
        """
        Legacy method for backward compatibility.
        
        Args:
            file_path: Path to the returns file
            
        Returns:
            True if file is valid, False otherwise
        """
        result = self._validate_returns_file(file_path)
        return result.is_valid
    
    def validate_multiple_results(self, results: List[Dict[str, Any]]) -> Dict[int, ValidationResult]:
        """
        Validate multiple experiment results.
        
        Args:
            results: List of experiment result dictionaries
            
        Returns:
            Dictionary mapping result index to ValidationResult
        """
        validation_results = {}
        for i, result in enumerate(results):
            validation_results[i] = self.validate(result)
        
        return validation_results
    
    def get_validation_summary(self, results: Dict[int, ValidationResult]) -> str:
        """
        Get a summary of validation results for multiple experiments.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Summary string
        """
        total_results = len(results)
        valid_results = sum(1 for result in results.values() if result.is_valid)
        error_results = sum(1 for result in results.values() if result.has_errors())
        warning_results = sum(1 for result in results.values() if result.has_warnings())
        
        summary = f"Experiment result validation summary: {valid_results}/{total_results} results valid"
        if error_results > 0:
            summary += f", {error_results} with errors"
        if warning_results > 0:
            summary += f", {warning_results} with warnings"
        
        return summary


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass
