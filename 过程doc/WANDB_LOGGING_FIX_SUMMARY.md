# WandB 日志记录修复总结

## 问题描述

在 WandB 日志中，只从 "run strategy" 开始记录日志，初始化阶段的日志（如 "Initializing StrategyRunner from configuration object"）只在 terminal 中出现，不在 WandB 中。

## 根本原因分析

1. **WandB run 创建时机太晚**：WandB run 在 `run_strategy()` 方法中才创建，而不是在 `initialize()` 中
2. **初始化失败太早**：在 WandB run 创建之前就失败了（`strategy_type` 属性访问错误）
3. **日志记录时机不对**：早期日志无法被 WandB 捕获

## 修复方案

### 1. 调整 WandB 初始化时机

在 `StrategyRunner.initialize()` 方法开始时就创建 WandB run：

```python
def initialize(self):
    """Initialize all components based on configuration."""
    try:
        # 先创建 WandB run，这样初始化日志就能被记录
        self._initialize_wandb_run()
        
        logger.info("Initializing strategy runner components...")
        # ... 后续初始化逻辑
```

### 2. 添加 strategy_type 容错处理

```python
# 添加容错处理 strategy_type 访问
try:
    strategy_config_dict['type'] = self.configs['strategy'].strategy_type.value
except AttributeError:
    # 如果 strategy_type 不存在，使用 type 字段
    strategy_config_dict['type'] = self.configs['strategy'].type
    logger.warning("strategy_type property not available, using type field directly")
```

### 3. 避免重复创建 WandB run

在 `run_strategy()` 方法中检查是否已经有 WandB run，避免重复创建。

## 修复的文件

- `src/trading_system/strategy_backtest/strategy_runner.py`

## 主要修改

1. **新增 `_initialize_wandb_run()` 方法**：
   - 在初始化开始时就创建 WandB run
   - 使用临时配置进行初始化日志记录
   - 添加容错处理，失败时不中断初始化过程

2. **修改 `initialize()` 方法**：
   - 在方法开始就调用 `_initialize_wandb_run()`
   - 添加 `strategy_type` 访问的容错处理

3. **修改 `run_strategy()` 方法**：
   - 检查是否已经有 WandB run，避免重复创建
   - 添加 `strategy_type` 访问的容错处理

## 测试验证

创建了测试脚本验证修复效果：

1. **`test_strategy_type_fix.py`**：测试 strategy_type 容错处理
2. **`test_wandb_logging_fix.py`**：测试完整的 WandB 日志记录修复

## 预期效果

修复后，WandB 日志应该包含：

1. ✅ "Started initialization run: xxx" - 初始化开始
2. ✅ "Initializing strategy runner components..." - 组件初始化
3. ✅ "Data provider not pre-initialized..." - 数据提供者初始化
4. ✅ "Started experiment run: xxx" - 正式实验开始
5. ✅ 所有后续的日志记录

## 遵循的设计原则

- **KISS**：保持修复简单，不引入复杂逻辑
- **SOLID**：单一职责，WandB 初始化独立于其他组件
- **YAGNI**：只修复必要的问题，不添加额外功能

## 向后兼容性

修复保持了向后兼容性：
- 如果 WandB 初始化失败，会回退到 NullExperimentTracker
- 如果 strategy_type 不存在，会使用 type 字段
- 不会中断现有的工作流程
