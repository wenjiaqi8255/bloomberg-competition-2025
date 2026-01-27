# Archived Configuration Files

This directory contains legacy configuration files that are no longer actively maintained but are kept for reference and potential migration.

## Archived Configurations

### Legacy Strategy Configurations
- `fama_macbeth_strategy_config.yaml` - **Replaced by**: `active/single_experiment/fama_macbeth_box_based_config.yaml`
  - **Reason**: Added box_based portfolio construction
  - **Migration**: Use the new configuration with enhanced portfolio construction options

- `fama_macbeth_with_country_risk.yaml` - **Replaced by**: `active/single_experiment/fama_macbeth_box_based_config.yaml`
  - **Reason**: Country risk functionality integrated into main configuration
  - **Migration**: Country risk is now available as an optional configuration option

- `fama_macbeth_country_risk_simple.yaml` - **Replaced by**: `active/single_experiment/fama_macbeth_box_based_config.yaml`
  - **Reason**: Simplified version integrated into main configuration
  - **Migration**: Use the main configuration with simplified country risk settings

### Legacy System Configurations
- `system_config.yaml` - **Replaced by**: `active/system/optimal_system_config.yaml`
  - **Reason**: More comprehensive system configuration options
  - **Migration**: Use the optimal system configuration for better system management

- `system_backtest_config.yaml` - **Replaced by**: Integrated into individual experiment configurations
  - **Reason**: Backtest configuration is now part of each experiment configuration
  - **Migration**: Use the backtest section in individual experiment configurations

### Legacy Experiment Configurations
- `metamodel_experiment_config.yaml` - **Replaced by**: `active/multi_model/multi_model_experiment.yaml`
  - **Reason**: Renamed and restructured for clarity
  - **Migration**: Use the new multi-model experiment configuration

- `e2e_refactoring_test.yaml` - **Replaced by**: `active/single_experiment/e2e_ff5_experiment.yaml`
  - **Reason**: Test configuration replaced by production configuration
  - **Migration**: Use the production end-to-end FF5 experiment configuration

### Legacy Feature Configurations
- `country_risk_config.yaml` - **Replaced by**: Integrated into factor data provider configurations
  - **Reason**: Country risk is now handled as a factor data provider
  - **Migration**: Use the CountryRiskProvider in factor_data_provider section

## Migration Guidelines

### How to Migrate from Legacy Configurations

1. **Identify the replacement**: Check this README to find the current equivalent
2. **Use the migration tool**: 
   ```bash
   python tools/config_management.py migrate <legacy_file> <new_file> --type <config_type> --description "Migrated from legacy"
   ```
3. **Validate the migrated configuration**:
   ```bash
   python tools/config_management.py validate <new_file>
   ```
4. **Test thoroughly**: Run experiments to ensure functionality is preserved

### Key Changes in New Configurations

1. **Unified Structure**: All configurations now follow a consistent structure
2. **Enhanced Validation**: JSON Schema validation ensures configuration correctness
3. **Better Documentation**: Each configuration includes comprehensive documentation
4. **Improved Portfolio Construction**: Box-based portfolio construction is now the default
5. **Integrated Features**: Previously separate features are now integrated options

### Backward Compatibility

- Legacy configurations may still work but are not guaranteed
- New features and improvements are only available in active configurations
- Validation and error checking is more comprehensive in new configurations
- Performance optimizations are only available in active configurations

## When to Use Archived Configurations

**Do NOT use archived configurations for**:
- New experiments
- Production deployments
- Performance-critical applications
- When you need the latest features

**You may reference archived configurations for**:
- Understanding historical approaches
- Migration reference
- Learning about configuration evolution
- Troubleshooting legacy systems

## Getting Help

If you need to migrate from a legacy configuration:

1. Check this README for the replacement configuration
2. Use the configuration management tools for migration
3. Validate the migrated configuration
4. Test thoroughly before production use
5. Refer to the main configuration documentation in `../README.md`

## Last Updated

This archive was created on 2024-01-15 as part of the configuration system reorganization.
