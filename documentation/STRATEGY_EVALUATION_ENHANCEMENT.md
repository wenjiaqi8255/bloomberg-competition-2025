# Strategy Evaluation Enhancement

## 概述

这次改进为交易策略系统添加了全面的信号质量评估和诊断功能，充分利用了 `PortfolioCalculator` 中的分析方法。

## 核心改进

### 1. BaseStrategy 新增评估能力

策略类现在可以自我评估和诊断，提供对其生成的信号的深入洞察。

#### 新增的实例变量
```python
self._last_signals = None              # 缓存最新生成的信号
self._last_price_data = None           # 缓存价格数据
self._last_signal_quality = None       # 最新的信号质量指标
self._last_position_metrics = None     # 最新的持仓指标
self._signal_generation_count = 0      # 信号生成次数计数器
```

#### 新增的核心方法

##### 1.1 自动评估 (内部方法)
```python
_evaluate_and_cache_signals(signals, price_data)
```
- 在每次 `generate_signals()` 后自动调用
- 提供当前时刻的信号质量"快照"
- 自动记录关键指标到日志

##### 1.2 信号质量评估
```python
evaluate_signal_quality(signals=None) -> Dict[str, Any]
```
返回指标：
- `avg_signal_intensity`: 平均信号强度
- `max_signal_intensity`: 最大信号强度
- `avg_signal_consistency`: 信号生成的一致性
- `signal_frequency`: 信号变化频率
- `total_signal_changes`: 总信号变化次数

##### 1.3 持仓特征分析
```python
analyze_positions(signals=None) -> Dict[str, Any]
```
返回指标：
- `avg_number_of_positions`: 平均持仓数量
- `max_number_of_positions`: 最大持仓数量
- `min_number_of_positions`: 最小持仓数量
- `avg_position_weight`: 平均持仓权重
- `max_position_weight`: 最大持仓权重
- `avg_concentration`: 平均集中度（最大持仓）

##### 1.4 集中度风险
```python
calculate_concentration_risk(signals=None) -> float
```
- 返回 Herfindahl-Hirschman Index (HHI)
- 范围：0 到 1
- 值越高表示越集中

##### 1.5 诊断报告
```python
get_diagnostic_report() -> Dict[str, Any]
```
提供全面的策略状态报告：
- 状态（是否已生成信号）
- 信号生成次数
- 信号质量指标
- 持仓指标
- 集中度风险
- 策略配置信息

##### 1.6 健康检查
```python
get_health_check() -> Dict[str, Any]
```
检查潜在问题：
- 是否生成了信号
- 集中度是否过高 (> 0.8 高风险, > 0.6 中等风险)
- 持仓数量是否异常 (< 1 太少, > 50 太多)
- 信号强度是否过低 (< 0.01)

返回：
- `is_healthy`: 布尔值
- `warnings`: 警告列表
- `checks_performed`: 执行的检查项目
- 当前指标快照

##### 1.7 当前快照
```python
get_current_snapshot() -> Dict[str, Any]
```
轻量级的当前状态视图：
- 时间戳
- 资产数量
- 平均持仓数
- 集中度
- 信号强度

### 2. StrategyRunner 增强

`StrategyRunner` 现在会调用策略的评估方法并汇总整个回测期间的指标。

#### 增强的 `_calculate_strategy_specific_metrics()` 方法

现在包含：

1. **信号质量指标** - 调用 `strategy.evaluate_signal_quality()`
2. **持仓指标** - 调用 `strategy.analyze_positions()`
3. **集中度风险** - 调用 `strategy.calculate_concentration_risk()`
4. **换手率分析** - 使用 `PortfolioCalculator.calculate_turnover()`
5. **诊断报告** - 调用 `strategy.get_diagnostic_report()`
6. **健康检查** - 调用 `strategy.get_health_check()`
7. **组合分析** - 使用 `PortfolioCalculator.analyze_portfolio_composition()`
   - 返回最佳/最差贡献资产
8. **遗留指标** - 保持向后兼容

#### 输出格式

所有指标都被扁平化并记录到实验追踪系统：

```python
{
    'signal_quality': {...},
    'signal_avg_signal_intensity': 0.45,
    'signal_signal_frequency': 0.12,
    ...
    'position_metrics': {...},
    'position_avg_number_of_positions': 5.2,
    'position_avg_position_weight': 0.19,
    ...
    'concentration_risk_hhi': 0.35,
    'portfolio_turnover': 0.15,
    'strategy_diagnostic': {...},
    'strategy_health': {
        'is_healthy': True,
        'warnings': []
    },
    'top_contributors': [('SPY', 0.045), ...],
    'worst_contributors': [('IWM', -0.012), ...]
}
```

#### 控制台输出

回测完成后会输出清晰的评估摘要：

```
============================================================
STRATEGY EVALUATION SUMMARY
============================================================
Signal Quality: {'avg_signal_intensity': 0.45, ...}
Position Metrics: {'avg_number_of_positions': 5.2, ...}
Concentration Risk (HHI): 0.350
Portfolio Turnover: 0.150
Health Status: ✓ Healthy
============================================================
```

## 架构分层

### 层级 1: 单策略层 (BaseStrategy)
**职责：**
- ✅ 生成信号
- ✅ 评估自己的信号质量（实时快照）
- ✅ 分析持仓特征
- ✅ 提供诊断信息和健康检查
- ❌ 不计算回测性能（留给 StrategyRunner）

**关键特性：**
- 每次调用时提供当前状态的"切片"
- 缓存最新信号和指标
- 可以手动调用评估方法
- 自动在信号生成后评估

### 层级 2: 单策略回测层 (StrategyRunner)
**职责：**
- ✅ 运行回测
- ✅ 计算性能指标（Sharpe、Drawdown等）
- ✅ 调用策略的诊断方法
- ✅ 汇总整个回测期间的指标
- ✅ 记录到实验追踪系统

**关键特性：**
- 从单次快照聚合到整体统计
- 提供时间序列视角
- 完整的实验追踪集成

### 层级 3: 多策略协调层 (SystemOrchestrator)
**职责：**
- ✅ 比较不同策略的表现
- ✅ 选择和组合策略
- ✅ 整体风险管理

**未来扩展：**
- 可以使用策略的评估方法来选择最佳策略
- 基于健康检查动态调整资本分配
- 跨策略的风险分析

## 使用示例

### 示例 1: 自动评估（回测中）

```python
from src.trading_system.strategy_backtest.strategy_runner import create_strategy_runner

# 创建 runner
runner = create_strategy_runner(
    config_path="configs/dual_momentum_config.yaml",
    use_wandb=True
)

# 运行回测 - 自动评估信号
results = runner.run_strategy(experiment_name="my_backtest")

# 查看策略特定指标
print(results['strategy_metrics']['signal_quality'])
print(results['strategy_metrics']['position_metrics'])
print(results['strategy_metrics']['strategy_health'])
```

### 示例 2: 手动评估

```python
from src.trading_system.strategies.factory import StrategyFactory

# 创建策略
strategy = StrategyFactory.create(
    strategy_type='dual_momentum',
    name='my_strategy'
)

# 生成信号
signals = strategy.generate_signals(price_data, start_date, end_date)

# 手动评估
signal_quality = strategy.evaluate_signal_quality()
position_metrics = strategy.analyze_positions()
concentration = strategy.calculate_concentration_risk()

# 获取健康检查
health = strategy.get_health_check()
if not health['is_healthy']:
    print(f"Warnings: {health['warnings']}")

# 获取诊断报告
diagnostic = strategy.get_diagnostic_report()
print(diagnostic)
```

### 示例 3: 实时监控

```python
# 在策略运行期间
snapshot = strategy.get_current_snapshot()
print(f"Current state: {snapshot}")

# 定期健康检查
health = strategy.get_health_check()
if not health['is_healthy']:
    logger.warning(f"Strategy health issues: {health['warnings']}")
    # 可能触发调整或警报
```

## 集成的 PortfolioCalculator 方法

现在完全集成到策略评估流程中的方法：

1. ✅ `calculate_signal_quality()` - 信号质量指标
2. ✅ `calculate_position_metrics()` - 持仓特征
3. ✅ `calculate_concentration_risk()` - HHI 集中度
4. ✅ `calculate_turnover()` - 换手率
5. ✅ `analyze_portfolio_composition()` - 组合构成分析
6. ✅ `calculate_portfolio_returns()` - 组合收益（在 BacktestEngine 中使用）
7. ✅ `calculate_portfolio_metrics()` - 综合指标（可选）

## 日志输出示例

### 信号生成时的自动评估
```
2025-10-02 10:15:23 - INFO - [dual_momentum] Signal Quality Snapshot:
2025-10-02 10:15:23 - INFO -   - Avg positions: 4.5
2025-10-02 10:15:23 - INFO -   - Avg position weight: 0.222
2025-10-02 10:15:23 - INFO -   - Signal intensity: 0.412
2025-10-02 10:15:23 - INFO -   - Concentration risk: 0.285
```

### 回测结束时的综合评估
```
============================================================
STRATEGY EVALUATION SUMMARY
============================================================
Signal Quality: {
    'avg_signal_intensity': 0.412,
    'max_signal_intensity': 0.825,
    'avg_signal_consistency': 0.95,
    'signal_frequency': 0.08
}
Position Metrics: {
    'avg_number_of_positions': 4.5,
    'max_number_of_positions': 6,
    'avg_position_weight': 0.222,
    'max_position_weight': 0.35
}
Concentration Risk (HHI): 0.285
Portfolio Turnover: 0.125
Health Status: ✓ Healthy
============================================================
```

## 优势

### 1. 实时可见性
- 每次信号生成后立即了解策略状态
- 不需要等到回测结束

### 2. 问题早期检测
- 健康检查可以识别异常情况
- 警告系统帮助快速诊断

### 3. 全面的指标
- 不仅是性能指标
- 包括信号特征、持仓行为、风险指标

### 4. 分层清晰
- 策略层：当前状态快照
- 回测层：聚合和时间序列视图
- 协调层：跨策略比较

### 5. 可扩展性
- 易于添加新的评估指标
- 可以自定义健康检查阈值
- 支持自定义诊断逻辑

### 6. 向后兼容
- 保留所有旧的指标计算
- 新功能是增量添加
- 不破坏现有代码

## 未来扩展

### 短期
- [ ] 添加信号质量时间序列追踪
- [ ] 自定义健康检查阈值配置
- [ ] 更丰富的可视化图表

### 中期
- [ ] 策略之间的比较分析
- [ ] 基于健康状态的自动调整
- [ ] 异常检测和警报

### 长期
- [ ] 机器学习驱动的策略评估
- [ ] 预测性健康监控
- [ ] 自适应风险管理

## 演示文件

运行演示查看完整功能：
```bash
python examples/strategy_evaluation_demo.py
```

这将展示：
1. 自动评估在回测中的工作方式
2. 如何手动调用评估方法
3. 所有新功能的实际输出

## 总结

这次增强使得策略评估从被动变为主动：

**之前：**
- ✗ 只在回测结束时看到性能
- ✗ 信号特征不可见
- ✗ 问题难以诊断
- ✗ PortfolioCalculator 未充分利用

**现在：**
- ✓ 实时信号质量快照
- ✓ 全面的诊断和健康检查
- ✓ 清晰的指标分层
- ✓ PortfolioCalculator 完全集成
- ✓ 易于扩展和自定义

这为构建更稳健、可观察、可维护的交易系统奠定了坚实基础！

