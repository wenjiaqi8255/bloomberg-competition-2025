#!/usr/bin/env python3
"""
配置验证脚本
检查所有配置文件是否符合 schema
"""

import sys
from pathlib import Path

# Add both src and project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from trading_system.validation.config.schema_validator import SchemaValidator
import yaml

def validate_config(config_path, schema_name):
    """验证单个配置文件"""
    try:
        validator = SchemaValidator()
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        result = validator.validate(config_data, schema_name)
        return result
    except Exception as e:
        return None

def main():
    configs_dir = Path("../configs/active/single_experiment")
    validator = SchemaValidator()

    print("=== 单一实验配置验证 ===\n")
    print(f"Scanning directory: {configs_dir.absolute()}")
    print(f"Found {len(list(configs_dir.glob('*.yaml')))} YAML files\n")

    for config_file in sorted(configs_dir.glob("*.yaml")):
        print(f"📄 {config_file.name}")

        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            result = validator.validate(config_data, 'single_experiment_schema')

            if result.is_valid:
                print("  ✅ 验证通过")
            else:
                errors = result.get_errors()
                warnings = result.get_warnings()
                print(f"  ❌ {len(errors)} 个错误, {len(warnings)} 个警告")

                if errors:
                    print("  错误:")
                    for error in errors[:3]:  # 只显示前3个
                        print(f"    - {error.message}")

                if warnings:
                    print(f"  ⚠️  {len(warnings)} 个警告")

        except Exception as e:
            print(f"  ❌ 验证失败: {e}")

        print()

if __name__ == "__main__":
    main()
