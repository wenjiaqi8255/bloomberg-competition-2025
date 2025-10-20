"""
JSON Schema Validator
====================

Validates configuration files against JSON schemas.
Provides schema-based validation for all configuration types.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class SchemaValidator(BaseValidator):
    """
    Validates configuration files against JSON schemas.
    
    Supports both JSON and YAML configuration files.
    """
    
    def __init__(self):
        super().__init__("SchemaValidator")
        self.schemas_dir = Path(__file__).parent.parent.parent.parent / "configs" / "schemas"
        self._schemas_cache = {}
    
    def validate(self, config_data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """
        Validate configuration data against a JSON schema.
        
        Args:
            config_data: Configuration data to validate
            schema_name: Name of the schema file (without .json extension)
            
        Returns:
            ValidationResult with validation outcome
        """
        self.log_validation_start(f"schema '{schema_name}'")
        result = ValidationResult()
        
        try:
            # Load schema
            schema = self._load_schema(schema_name)
            if not schema:
                result.add_error(f"Schema '{schema_name}' not found", suggestion="Check schema file exists")
                return result
            
            # Basic schema validation (simplified - in production would use jsonschema library)
            result = self._validate_against_schema(config_data, schema, result)
            
        except Exception as e:
            result.add_error(f"Schema validation failed: {str(e)}")
            logger.error(f"Schema validation error: {e}")
        
        self.log_validation_complete(result)
        return result
    
    def validate_file(self, config_path: str, schema_name: str) -> ValidationResult:
        """
        Validate a configuration file against a JSON schema.
        
        Args:
            config_path: Path to the configuration file
            schema_name: Name of the schema file (without .json extension)
            
        Returns:
            ValidationResult with validation outcome
        """
        self.log_validation_start(f"file '{config_path}' against schema '{schema_name}'")
        result = ValidationResult()
        
        try:
            # Load configuration file
            config_data = self._load_config_file(config_path)
            if not config_data:
                result.add_error(f"Failed to load configuration file: {config_path}")
                return result
            
            # Validate against schema
            result = self.validate(config_data, schema_name)
            
        except Exception as e:
            result.add_error(f"File validation failed: {str(e)}")
            logger.error(f"File validation error: {e}")
        
        self.log_validation_complete(result)
        return result
    
    def _load_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Load a JSON schema from the schemas directory."""
        if schema_name in self._schemas_cache:
            return self._schemas_cache[schema_name]
        
        schema_path = self.schemas_dir / f"{schema_name}.json"
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return None
        
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                self._schemas_cache[schema_name] = schema
                return schema
        except Exception as e:
            logger.error(f"Failed to load schema {schema_name}: {e}")
            return None
    
    def _load_config_file(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load a configuration file (JSON or YAML)."""
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return None
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    logger.error(f"Unsupported file format: {path.suffix}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load configuration file {config_path}: {e}")
            return None
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any], 
                               result: ValidationResult) -> ValidationResult:
        """
        Validate data against schema using jsonschema library.
        
        Falls back to simple validation if jsonschema is not available.
        """
        try:
            import jsonschema
            from jsonschema import RefResolver
            
            # Create resolver for $ref references
            schema_store = {}
            base_uri = self.schemas_dir.as_uri() + '/'
            
            # Load base_schemas.json for references
            base_schema_path = self.schemas_dir / 'base_schemas.json'
            if base_schema_path.exists():
                with open(base_schema_path, 'r') as f:
                    base_schema = json.load(f)
                    schema_store[base_uri + 'base_schemas.json'] = base_schema
            
            resolver = RefResolver(base_uri, schema, store=schema_store)
            
            # Validate
            validator = jsonschema.Draft7Validator(schema, resolver=resolver)
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            
            for error in errors:
                field_path = '.'.join(str(p) for p in error.path) if error.path else 'root'
                result.add_error(
                    f"{error.message}",
                    field=field_path,
                    suggestion=f"Check field '{field_path}' matches schema requirements"
                )
            
        except ImportError:
            logger.warning("jsonschema library not available, using basic validation")
            # Fall back to existing simple validation
            return self._simple_validate(data, schema, result)
        except Exception as e:
            result.add_error(f"Schema validation error: {str(e)}")
        
        return result
    
    def _simple_validate(self, data: Dict[str, Any], schema: Dict[str, Any], 
                        result: ValidationResult) -> ValidationResult:
        """
        Simple validation fallback when jsonschema is not available.
        
        Provides basic validation for common schema patterns.
        """
        # Validate required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        # Validate field types
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in data:
                field_type = field_schema.get('type')
                if field_type and not self._validate_field_type(data[field], field_type):
                    result.add_error(f"Field '{field}' must be of type {field_type}")
                
                # Validate enum values
                enum_values = field_schema.get('enum')
                if enum_values and data[field] not in enum_values:
                    result.add_error(f"Field '{field}' must be one of {enum_values}")
                
                # Validate numeric ranges
                if field_type in ['number', 'integer']:
                    minimum = field_schema.get('minimum')
                    maximum = field_schema.get('maximum')
                    if minimum is not None and data[field] < minimum:
                        result.add_error(f"Field '{field}' must be >= {minimum}")
                    if maximum is not None and data[field] > maximum:
                        result.add_error(f"Field '{field}' must be <= {maximum}")
        
        return result
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type."""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected_python_type)
    
    def list_available_schemas(self) -> list[str]:
        """List all available schema files."""
        if not self.schemas_dir.exists():
            return []
        
        schemas = []
        for schema_file in self.schemas_dir.glob("*.json"):
            schemas.append(schema_file.stem)
        
        return sorted(schemas)
