# Residual Momentum 验证指南

## 概述

这个验证脚本用于快速验证 **Residual Momentum** 方法是否比当前的 **Alpha** 方法更有效。

## 快速开始

### 1. 运行验证脚本

```bash
python validate_residual_momentum.py
```

### 2. 查看结果

脚本会输出一个对比表格，显示：
- Mean IC（平均信息系数）
- IC Std（IC标准差）
- IC Sharpe（IC的Sharpe比率）
- Positive IC Ratio（IC为正的比例）

结果会保存在 `validation_results/` 目录下：
- `ic_comparison.csv`: 每个rebalance date的IC对比
- `summary_comparison.csv`: 汇总统计对比

## 输出解读

### 成功指标

如果 **Residual Momentum** 的以下指标显著优于 **Alpha**：
- ✅ Mean IC 提升 > 0.01（1%）
- ✅ IC Sharpe 提升 > 0.2
- ✅ Positive IC Ratio 提升 > 5%

**建议**：继续到 Stage 1 实现

### 边际改善

如果改善较小：
- Mean IC 提升 0.005-0.01
- IC Sharpe 提升 0.1-0.2

**建议**：可以考虑实现，但需要进一步优化

### 无改善或更差

如果 Residual Momentum 没有明显改善：
- Mean IC 提升 < 0.005 或为负
- IC Sharpe 没有提升

**建议**：
1. 检查数据质量
2. 调整参数（formation_period, skip_recent_days）
3. 检查universe选择是否合适

## 参数配置

可以在脚本中修改以下参数：

```python
validator = ResidualMomentumValidator(
    formation_period=252,      # Residual momentum formation period (天)
    skip_recent_days=21,       # 跳过最近N天（避免短期反转）
    forward_lookback_days=21   # Forward return窗口（天）
)
```

### 参数说明

- **formation_period**: 
  - 默认 252（12个月）
  - 可以尝试 126（6个月）或 504（24个月）
  
- **skip_recent_days**:
  - 默认 21（1个月）
  - 学术研究中通常跳过最近1-3个月
  
- **forward_lookback_days**:
  - 默认 21（1个月forward return）
  - 可以尝试 63（3个月）或 126（6个月）

## 自定义股票列表

修改 `main()` 函数中的 `symbols` 列表：

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', ...]
```

建议：
- 至少10只股票（保证统计显著性）
- 选择流动性好的股票
- 覆盖不同行业

## 自定义日期范围

```python
end_date = datetime.now()
validation_start = end_date - timedelta(days=365)  # 验证期开始
validation_end = end_date - timedelta(days=30)     # 验证期结束（留出forward return空间）
```

## 故障排除

### 1. 数据加载失败

**错误**：`Failed to load factor data`

**解决**：
- 检查 `data/ff5_factors_processed.csv` 是否存在
- 如果不存在，脚本会尝试从网络获取（需要网络连接）

### 2. 股票数据加载失败

**错误**：`Failed to load stock returns`

**解决**：
- 检查网络连接（使用yfinance需要网络）
- 确认股票代码正确
- 可以修改代码使用本地CSV数据

### 3. 回归失败

**错误**：`Insufficient data for {symbol}`

**解决**：
- 某些股票可能数据不足
- 脚本会自动跳过这些股票
- 确保至少有几只股票成功回归

### 4. IC计算失败

**错误**：`No valid IC calculations`

**解决**：
- 检查日期范围是否合理
- 确保有足够的rebalance dates
- 检查forward returns是否计算成功

## 下一步

### 如果验证通过（Residual Momentum更好）

1. **Stage 1**: 修改 `FF5RegressionModel.fit()` 存储residuals
2. **Stage 1**: 修改 `FamaFrench5Strategy._get_predictions()` 使用residual momentum
3. **Stage 2**: 优化参数和实现细节

### 如果验证未通过

1. 检查数据质量
2. 尝试不同的参数组合
3. 考虑universe选择（residual momentum可能在某些universe更有效）
4. 查看学术论文中的实施细节是否有遗漏

## 参考

- Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506-521.
- 当前实现基于time-series regression，符合Fama-French模型标准


