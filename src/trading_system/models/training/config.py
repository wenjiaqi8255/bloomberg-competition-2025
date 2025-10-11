"""
Unified Configuration Management - MVP Version

极简的配置管理，专注核心功能：
1. YAML配置加载
2. 配置验证
3. 配置转换

No over-engineering, just essential functionality.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """统一训练配置。"""
    # 基本设置
    model_type: str = "xgboost"
    experiment_name: str = "training"

    # 数据设置
    data_period: str = "2022-01-01:2023-12-31"
    train_split: float = 0.8

    # 模型参数
    model_params: Dict[str, Any] = None

    # MetaModel特定配置
    metamodel_method: str = "ridge"
    metamodel_alpha: float = 1.0
    strategies: list = None

    # 超参数优化
    enable_hpo: bool = False
    hpo_trials: int = 50
    hpo_metric: str = "r2"

    # 保存设置
    save_dir: str = "./models"

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        if self.strategies is None:
            self.strategies = ["DualMomentumStrategy", "MLStrategy", "FF5Strategy"]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """从YAML文件加载配置。"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # 提取训练配置
            training_config = config_data.get('training', config_data)

            # 转换字段
            return cls(
                model_type=training_config.get('model_type', 'xgboost'),
                experiment_name=training_config.get('experiment_name', 'training'),
                data_period=training_config.get('data_period', '2022-01-01:2023-12-31'),
                train_split=training_config.get('train_split', 0.8),
                model_params=training_config.get('model_params', {}),
                metamodel_method=training_config.get('metamodel_method', 'ridge'),
                metamodel_alpha=training_config.get('metamodel_alpha', 1.0),
                strategies=training_config.get('strategies', ["DualMomentumStrategy", "MLStrategy", "FF5Strategy"]),
                enable_hpo=training_config.get('enable_hpo', False),
                hpo_trials=training_config.get('hpo_trials', 50),
                hpo_metric=training_config.get('hpo_metric', 'r2'),
                save_dir=training_config.get('save_dir', './models')
            )

        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            'model_type': self.model_type,
            'experiment_name': self.experiment_name,
            'data_period': self.data_period,
            'train_split': self.train_split,
            'model_params': self.model_params,
            'metamodel_method': self.metamodel_method,
            'metamodel_alpha': self.metamodel_alpha,
            'strategies': self.strategies,
            'enable_hpo': self.enable_hpo,
            'hpo_trials': self.hpo_trials,
            'hpo_metric': self.hpo_metric,
            'save_dir': self.save_dir
        }

    def save(self, config_path: str) -> None:
        """保存配置到YAML文件。"""
        config_data = {'training': self.to_dict()}

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Config saved to {config_path}")


# 便捷函数
def load_config(config_path: str) -> TrainingConfig:
    """加载配置文件。"""
    return TrainingConfig.from_yaml(config_path)


def create_default_config() -> TrainingConfig:
    """创建默认配置。"""
    return TrainingConfig()


# 配置验证
def validate_config(config: TrainingConfig) -> bool:
    """验证配置有效性。"""
    try:
        # 基本验证
        if not config.model_type:
            raise ValueError("model_type is required")

        if config.train_split <= 0 or config.train_split >= 1:
            raise ValueError("train_split must be between 0 and 1")

        if config.hpo_trials <= 0:
            raise ValueError("hpo_trials must be positive")

        # MetaModel特定验证
        if config.model_type == "metamodel":
            if not config.strategies:
                raise ValueError("strategies required for metamodel")
            if config.metamodel_method not in ["equal", "lasso", "ridge"]:
                raise ValueError("Invalid metamodel method")

        return True

    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 创建默认配置
    config = create_default_config()

    # 保存配置
    config.save("test_config.yaml")

    # 加载配置
    loaded_config = load_config("test_config.yaml")

    # 验证配置
    if validate_config(loaded_config):
        print("Config is valid!")
    else:
        print("Config is invalid!")