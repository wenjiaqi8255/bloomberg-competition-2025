# 卖空控制功能使用指南

本文档介绍了系统中新增的 卖空控制功能，包括配置方法、实现原理和使用示例。

## 功能概述

系统现在支持在多个层面控制 卖空行为：

1. **策略层面** (`BaseStrategy`): 在信号生成时过滤负值信号
2. **投资组合优化层面** (`PortfolioOptimizer`): 在优化时设置权重边界
3. **系统层面** (`SystemOrchestrator`): 统一的系统级配置
4. **回测层面** (`BacktestConfig`): 回测时的 卖空成本和控制

## 配置方法

### 1. 系统级配置

在 `SystemConfig` 或系统配置文件中：

```yaml
system:
  enable_short_selling: false  # 默认禁用，设为 true 启用
  max_leverage: 1.0           # 最大杠杆倍数
```

### 2. 策略配置

在策略初始化参数中：

```python
strategy = MLStrategy(
    name="MyStrategy",
    enable_short_selling=False  # 禁用卖空
)
```

或在配置文件中：

```yaml
strategy:
  enable_short_signals: false  # 策略级信号控制
```

### 3. 投资组合优化配置

在 `portfolio_optimization` 部分：

```yaml
portfolio_optimization:
  method: "mean_variance"
  enable_short_selling: false  # 优化器级控制
  risk_aversion: 2.0
```

### 4. 回测配置

在 `backtest` 部分：

```yaml
backtest:
  enable_short_selling: false  # 回测级控制
  short_borrow_rate: 0.03     # 卖空成本率
```

## 实现原理

### BaseStrategy 信号过滤

当 `enable_short_selling=False` 时：
- 负值预测信号被设为 0
- 保留正值信号的相对强度
- 记录被过滤的负值信号数量

```python
def _apply_short_selling_restrictions(self, predictions):
    if not self.enable_short_selling:
        # 过滤负值信号
        filtered_predictions = predictions.copy()
        filtered_predictions[filtered_predictions < 0] = 0
        return filtered_predictions
    else:
        return predictions  # 保持原样
```

### PortfolioOptimizer 边界控制

根据配置设置不同的优化边界：

```python
if self.enable_short_selling:
    # 允许 卖空: -1 <= weight <= 1
    bounds = Bounds(-1, 1)
else:
    # 仅多头: 0 <= weight <= 1
    bounds = Bounds(0, 1)
```

### 系统级配置传递

`SystemOrchestrator` 将系统配置传递给所有子组件：

```python
# 将系统配置合并到 custom_configs 中
self.custom_configs['enable_short_selling'] = system_config.enable_short_selling

# 传递给优化器
optimizer_config['enable_short_selling'] = self.custom_configs.get('enable_short_selling', False)
```

## 使用示例

### 示例 1: 长多头策略

```python
# 配置
config = {
    'enable_short_selling': False,
    'portfolio_optimization': {
        'method': 'equal_weight',
        'enable_short_selling': False
    }
}

# 结果：所有权重为正或零
weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}
```

### 示例 2: 多头-空头策略

```python
# 配置
config = {
    'enable_short_selling': True,
    'portfolio_optimization': {
        'method': 'mean_variance',
        'enable_short_selling': True,
        'risk_aversion': 2.0
    }
}

# 结果：可包含负值权重（空头头寸）
weights = {'AAPL': 0.5, 'MSFT': -0.3, 'GOOGL': 0.8, 'AMZN': -0.2}
```

## 配置文件示例

### 长多头配置 (`ml_strategy_config_new.yaml`)

```yaml
portfolio_optimization:
  method: "equal_weight"
  enable_short_selling: false  # 禁用 卖空

backtest:
  enable_short_selling: false
  short_borrow_rate: 0.02
```

### 多头-空头配置 (`ml_strategy_short_selling_example.yaml`)

```yaml
portfolio_optimization:
  method: "mean_variance"  # 推荐用于 卖空策略
  enable_short_selling: true   # 启用 卖空
  risk_aversion: 2.0

backtest:
  enable_short_selling: true
  short_borrow_rate: 0.03     # 卖空成本
```

## 注意事项

1. **默认安全**: 系统默认禁用 卖空，确保意外情况下不会产生空头头寸

2. **一致性**: 建议在所有层级使用一致的 卖空配置，避免行为冲突

3. **方法选择**:
   - `equal_weight` 和 `top_n` 方法只产生正权重
   - `mean_variance` 方法在启用 卖空时可产生负权重

4. **成本考虑**: 启用 卖空时，系统会自动计算 卖空成本（short_borrow_rate）

5. **风险控制**: 卖空头寸会增加投资组合风险，建议谨慎使用并设置适当的风险限制

## 测试验证

运行测试脚本验证 卖空控制功能：

```bash
poetry run python test_short_selling_controls.py
```

测试包括：
- PortfolioOptimizer 的边界控制
- BaseStrategy 的信号过滤
- 配置系统的集成测试

## 故障排除

### 常见问题

1. **权重全为零**: 检查是否有有效的正值信号
2. **优化失败**: 确保协方差矩阵正定，预期收益有合理差异
3. **配置冲突**: 确保各层级的 卖空配置一致

### 调试日志

启用 DEBUG 级别日志查看详细信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

关键日志信息：
- "Applying long-only constraint"
- "Using short selling bounds: [-1, 1]"
- "Using long-only bounds: [0, 1]"