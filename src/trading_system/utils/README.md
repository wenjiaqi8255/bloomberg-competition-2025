# Trading System Utils Module

## 概述
工具模块，提供交易系统中常用的工具函数、实用类和辅助功能。

## 主要组件

### 1. Risk Utils (`risk.py`)
提供风险管理相关的计算工具。

#### RiskMetrics (风险指标)
```python
@dataclass
class RiskMetrics:
    var_95: float          # 95% VaR
    var_99: float          # 99% VaR
    expected_shortfall: float  # 期望缺口
    max_drawdown: float    # 最大回撤
    volatility: float      # 波动率
    sharpe_ratio: float    # 夏普比率
    sortino_ratio: float   # 索提诺比率
    beta: float           # Beta系数
    alpha: float          # Alpha值
    information_ratio: float  # 信息比率
```

#### RiskCalculator (风险计算器)
```python
class RiskCalculator:
    """风险指标计算器"""

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算VaR"""
        pass

    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算期望缺口"""
        pass

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """计算最大回撤"""
        pass

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        pass

    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """计算Beta系数"""
        pass
```

### 2. Performance Utils (`performance.py`)
提供性能分析和评估工具。

#### 主要函数
```python
def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """计算收益率"""

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """计算累积收益率"""

def calculate_information_coefficient(predictions: np.ndarray, actual: np.ndarray) -> float:
    """计算信息系数"""

def calculate_hit_rate(predictions: np.ndarray, actual: np.ndarray, threshold: float = 0.0) -> float:
    """计算命中率"""

def calculate_turnover(weights: pd.DataFrame) -> float:
    """计算换手率"""

def performance_attribution(portfolio_returns: pd.Series,
                          factor_returns: pd.DataFrame) -> Dict[str, float]:
    """业绩归因分析"""
```

### 3. Data Utils (`data_utils.py`)
提供数据处理和转换工具。

#### 主要函数
```python
def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗价格数据"""

def align_data_sources(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """对齐多个数据源"""

def resample_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """重采样数据"""

def handle_missing_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """处理缺失数据"""

def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """检测异常值"""

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
```

### 4. Validation Utils (`validation.py`)
提供数据验证和规则检查工具。

#### 主要函数
```python
def validate_price_data(df: pd.DataFrame) -> bool:
    """验证价格数据的有效性"""

def validate_portfolio_weights(weights: pd.Series, tolerance: float = 1e-6) -> bool:
    """验证投资组合权重"""

def validate_trading_signals(signals: pd.DataFrame) -> bool:
    """验证交易信号"""

def validate_factor_data(df: pd.DataFrame) -> bool:
    """验证因子数据"""

def check_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """检查数据一致性"""
```

### 5. Secrets Manager (`secrets_manager.py`)
安全的密钥和敏感信息管理。

#### SecretsManager
```python
class SecretsManager:
    """密钥管理器"""

    def __init__(self, storage_path: str = ".secrets"):
        self.storage_path = storage_path

    def store_secret(self, key: str, value: str) -> None:
        """存储密钥"""

    def retrieve_secret(self, key: str) -> Optional[str]:
        """检索密钥"""

    def delete_secret(self, key: str) -> bool:
        """删除密钥"""

    def list_secrets(self) -> List[str]:
        """列出所有密钥"""
```

### 6. WandB Logger (`wandb_logger.py`)
实验跟踪和日志记录工具。

#### WandBLogger
```python
class WandBLogger:
    """WandB日志记录器"""

    def __init__(self, project_name: str, config: Optional[Dict] = None):
        self.project_name = project_name
        self.config = config or {}

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """记录参数"""

    def log_model(self, model_path: str, name: str) -> None:
        """记录模型"""

    def log_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """记录预测结果"""

    def finish(self) -> None:
        """结束实验"""
```

## Experiment Tracking Components

实验跟踪子模块 (`experiment_tracking/`) 提供完整的实验管理功能。

### 1. Interface (`experiment_tracking/interface.py`)
```python
class ExperimentTracker:
    """实验跟踪器接口"""

    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """开始实验"""

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """记录指标"""

    def log_parameter(self, name: str, value: Any) -> None:
        """记录参数"""

    def log_artifact(self, path: str, name: str, type: str) -> None:
        """记录产物"""

    def end_experiment(self, status: str = "completed") -> None:
        """结束实验"""
```

### 2. WandB Adapter (`experiment_tracking/wandb_adapter.py`)
```python
class WandBAdapter(ExperimentTracker):
    """WandB适配器"""

    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        wandb.log({name: value}, step=step)

    def log_parameter(self, name: str, value: Any) -> None:
        wandb.config.update({name: value})
```

### 3. Pipeline (`experiment_tracking/pipeline.py`)
```python
class ExperimentPipeline:
    """实验流水线"""

    def __init__(self, tracker: ExperimentTracker, config: Dict[str, Any]):
        self.tracker = tracker
        self.config = config

    def run_experiment(self, data: pd.DataFrame, model: BaseModel) -> Dict[str, Any]:
        """运行实验"""
        pass

    def log_experiment_summary(self, results: Dict[str, Any]) -> None:
        """记录实验总结"""
        pass
```

### 4. Visualizer (`experiment_tracking/visualizer.py`)
```python
class ExperimentVisualizer:
    """实验可视化工具"""

    @staticmethod
    def plot_training_curves(metrics: Dict[str, List[float]]) -> None:
        """绘制训练曲线"""

    @staticmethod
    def plot_feature_importance(importance: Dict[str, float]) -> None:
        """绘制特征重要性"""

    @staticmethod
    def plot_prediction_distribution(predictions: np.ndarray, targets: np.ndarray) -> None:
        """绘制预测分布"""

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame) -> None:
        """绘制相关性矩阵"""
```

## 使用示例

### 风险计算
```python
from trading_system.utils.risk import RiskCalculator, RiskMetrics
import pandas as pd
import numpy as np

# 计算风险指标
returns = pd.Series(np.random.normal(0.001, 0.02, 252))

var_95 = RiskCalculator.calculate_var(returns, 0.95)
max_dd = RiskCalculator.calculate_max_drawdown(returns.cumsum())
sharpe = RiskCalculator.calculate_sharpe_ratio(returns)

print(f"95% VaR: {var_95:.4f}")
print(f"最大回撤: {max_dd:.4f}")
print(f"夏普比率: {sharpe:.4f}")
```

### 性能分析
```python
from trading_system.utils.performance import calculate_information_coefficient

# 计算信息系数
predictions = np.random.normal(0, 1, 100)
actual_returns = np.random.normal(0.001, 0.02, 100)

ic = calculate_information_coefficient(predictions, actual_returns)
print(f"信息系数: {ic:.4f}")
```

### 实验跟踪
```python
from trading_system.utils.experiment_tracking.wandb_adapter import WandBAdapter

# 初始化实验跟踪
tracker = WandBAdapter(project="trading-experiments")
experiment_id = tracker.start_experiment("ff5-model-test", {
    "model_type": "ff5_regression",
    "training_period": "2020-2023"
})

# 记录指标
tracker.log_metric("train_loss", 0.123, step=1)
tracker.log_metric("val_loss", 0.145, step=1)
tracker.log_metric("sharpe_ratio", 1.25, step=1)

# 结束实验
tracker.end_experiment("completed")
```

### 数据验证
```python
from trading_system.utils.validation import validate_price_data, validate_portfolio_weights
import pandas as pd

# 验证价格数据
price_data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [101, 102, 103],
    'low': [99, 100, 101],
    'close': [101, 102, 102],
    'volume': [1000, 1100, 1200]
})

is_valid = validate_price_data(price_data)
print(f"价格数据有效: {is_valid}")

# 验证投资组合权重
weights = pd.Series([0.3, 0.4, 0.3], index=['AAPL', 'MSFT', 'GOOGL'])
weights_valid = validate_portfolio_weights(weights)
print(f"权重有效: {weights_valid}")
```

## 配置选项

### 风险管理配置
```yaml
risk:
  var_confidence: 0.95
  var_method: "historical"
  drawdown_window: 252
  sharpe_risk_free_rate: 0.02
  beta_market_index: "SPY"
```

### 实验跟踪配置
```yaml
experiment_tracking:
  backend: "wandb"
  project: "trading-system"
  entity: "team-name"
  log_model: true
  log_predictions: true
  save_frequency: 10
```

## 依赖项

### 核心依赖
- `numpy` - 数值计算
- `pandas` - 数据处理
- `scipy` - 科学计算
- `scikit-learn` - 机器学习

### 可选依赖
- `wandb` - 实验跟踪
- `plotly` - 交互式可视化
- `matplotlib` - 静态可视化
- `seaborn` - 统计可视化

## 最佳实践

1. **风险管理**: 总是计算和监控关键风险指标
2. **数据验证**: 在处理数据前进行验证
3. **实验跟踪**: 记录所有实验参数和结果
4. **性能监控**: 定期评估模型和策略性能
5. **错误处理**: 使用适当的异常处理和日志记录

## 注意事项

1. **数值稳定性**: 在金融计算中注意数值精度
2. **时间序列**: 正确处理时间序列数据的时序性
3. **内存管理**: 处理大量数据时注意内存使用
4. **并发安全**: 在多线程环境中使用适当的锁机制