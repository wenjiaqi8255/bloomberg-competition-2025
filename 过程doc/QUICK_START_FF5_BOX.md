# 🚀 FF5 + Box-Based 快速开始指南

这个指南让你在5分钟内体验 FF5 模型与 Box-First 组合构建的强大功能。

## ⚡ 一键运行

### 方法1：快速演示 (推荐新手)
```bash
python run_ff5_box_experiment.py --demo
```

### 方法2：使用统一实验运行器
```bash
poetry run python run_experiment.py experiment -c configs/ff5_box_demo.yaml
```

## 📊 预期运行时间

| 配置 | 运行时间 | 说明 |
|------|----------|------|
| `--demo` | 5-10分钟 | 20只股票，2年回测 |
| `ff5_box_demo.yaml` | 5-10分钟 | 快速演示配置 |
| `ff5_box_based_experiment.yaml` | 15-30分钟 | 完整实验配置 |

## 🎯 你将看到什么

### 1. 配置摘要
```
📊 Training Period: 2019-01-01 to 2020-12-31
📈 Model Type: ff5_regression
🎯 Symbols: 21 stocks
📦 Box-Based Construction:
   Method: box_based
   Stocks per box: 2
   Total target boxes: 32
```

### 2. 训练进度
```
🔄 Training FF5 model...
📊 Hyperparameter optimization: Trial 15/15
✅ Model training completed
```

### 3. 组合构建过程
```
📦 Step 1/4: Classifying stocks into boxes...
📦 Step 2/4: Analyzing box coverage...
📦 Step 3/4: Selecting stocks and allocating weights...
📦 Step 4/4: Normalizing final weights...
```

### 4. 最终结果
```
🎉 EXPERIMENT COMPLETED SUCCESSFULLY
📊 Final Portfolio Value: $125,000
📈 Total Return: 25.00%
🎯 Sharpe Ratio: 1.25
📉 Max Drawdown: -8.50%
📦 Construction Method: BoxBased
🏗️ Box Coverage: 75.0%
📊 Boxes Covered: 24/32
```

## 🔧 如果遇到问题

### 问题1：配置验证失败
```bash
# 检查配置文件
python run_ff5_box_experiment.py --config configs/ff5_box_demo.yaml --dry-run
```

### 问题2：运行时间过长
```bash
# 使用快速演示配置
python run_ff5_box_experiment.py --demo
```

### 问题3：内存不足
减少股票池大小：
```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  # 只保留前10只股票
```

## 📈 结果解读

### 关键指标说明

| 指标 | 好的范围 | 说明 |
|------|----------|------|
| **Sharpe Ratio** | > 1.0 | 风险调整后收益 |
| **Max Drawdown** | < 15% | 最大损失 |
| **Box Coverage** | > 60% | 盒子覆盖率 |
| **Total Return** | > 10% | 总收益率 |

### Box-First 特有指标

- **Box Coverage**: 目标覆盖的盒子比例
- **Boxes Covered**: 实际覆盖的盒子数量
- **Concentration Risk**: 集中度风险指标

## 🎯 下一步尝试

### 1. 自定义股票池
编辑 `configs/ff5_box_demo.yaml`：
```yaml
symbols:
  - TSLA  # 电动车
  - NVDA  # 芯片
  - AMD   # 半导体
  # 添加你感兴趣的股票
```

### 2. 调整盒子参数
```yaml
portfolio_construction:
  stocks_per_box: 3        # 每个盒子3只股票
  allocation_method: "equal" # 等权重分配
```

### 3. 改变盒子维度
```yaml
box_weights:
  dimensions:
    size: ["large"]        # 只关注大盘股
    style: ["growth", "value"]
    region: ["developed"]
    sector: ["Technology", "Financials"]  # 只关注2个行业
```

### 4. 运行完整实验
```bash
python run_ff5_box_experiment.py --config configs/ff5_box_based_experiment.yaml
```

## 📚 深入学习

### 理解 Box-First 方法
1. 阅读 `FF5_BOX_README.md` 详细文档
2. 查看 `src/trading_system/portfolio_construction/` 源码
3. 运行完整测试套件：
   ```bash
   poetry run python tests/test_portfolio_construction.py
   ```

### 性能优化技巧
1. **增加盒子覆盖率**: 扩大股票池
2. **降低风险**: 减少 `stocks_per_box`
3. **提高收益**: 使用 `signal_proportional` 分配
4. **稳定性能**: 增加 `min_stocks_per_box`

### 高级功能
1. **动态盒子权重**: 根据市场条件调整
2. **多因子模型**: 集成更多因子
3. **风险预算**: 盒子层级风险控制
4. **回测分析**: 深度性能归因

## 🆘 获取帮助

### 常见错误解决

```bash
# 检查依赖
poetry install

# 验证配置
python run_ff5_box_experiment.py --dry-run

# 查看日志
python run_ff5_box_experiment.py --demo --verbose
```

### 技术支持
- 检查 `logs/` 目录中的详细日志
- 查看 `results/` 目录中的输出结果
- 参考 `README.md` 主项目文档

---

**🎉 恭喜！你已经成功体验了 FF5 + Box-First 的强大功能！**

现在你可以：
1. 自定义配置来测试不同策略
2. 比较传统优化与 Box-First 的差异
3. 集成到你的投资研究流程中

**开始你的量化投资之旅吧！** 🚀