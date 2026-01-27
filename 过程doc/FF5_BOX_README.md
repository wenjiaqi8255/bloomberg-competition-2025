# FF5 + Box-Based Portfolio Construction Integration

这个配置演示了如何将 Fama-French 5因子模型与我们刚实现的 Box-First 组合构建方法结合使用。

## 🎯 核心功能

### 问题解决
传统的量化优化往往将投资集中在少数几个投资风格盒子中（例如80%投资在[大盘/成长/美国/科技]），而 **Box-First 方法** 确保在每个目标盒子中都有代表性投资。

### 整合方案
1. **FF5模型训练**: 使用Fama-French 5因子模型预测股票收益
2. **Box分类**: 将股票按4个维度分类：规模(大/中/小)、风格(成长/价值)、地区(发达/新兴)、行业
3. **Box覆盖**: 确保系统性地覆盖所有目标盒子
4. **权重分配**: 根据FF5信号强度在盒子内分配权重

## 🚀 快速开始

### 1. 快速演示 (推荐第一次使用)
```bash
# 使用简化的演示配置
python run_ff5_box_experiment.py --demo
```

### 2. 完整实验
```bash
# 使用完整的配置文件
python run_ff5_box_experiment.py --config configs/ff5_box_based_experiment.yaml
```

### 3. 验证配置
```bash
# 只验证配置，不运行实验
python run_ff5_box_experiment.py --config configs/ff5_box_demo.yaml --dry-run
```

### 4. 使用统一实验运行器
```bash
# 也可以使用主实验运行器
poetry run python run_experiment.py experiment -c configs/ff5_box_demo.yaml
```

## 📁 配置文件说明

### 核心配置项

```yaml
strategy:
  parameters:
    portfolio_construction:
      method: "box_based"                    # 使用Box-First方法
      stocks_per_box: 2                     # 每个盒子选择2只股票
      allocation_method: "signal_proportional" # 根据FF5信号分配权重

      box_weights:
        method: "equal"                     # 所有盒子等权重
        dimensions:
          size: ["large", "mid", "small"]   # 规模维度
          style: ["growth", "value"]        # 风格维度
          region: ["developed"]             # 地区维度
          sector: ["Technology", "Financials", ...] # 行业维度
```

### 关键参数解释

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `stocks_per_box` | 每个盒子选择的股票数量 | 2-3 |
| `min_stocks_per_box` | 每个盒子最少股票数量 | 1 |
| `allocation_method` | 盒子内权重分配方法 | signal_proportional |
| `box_weights.method` | 盒子权重分配方法 | equal (MVP) |

## 📊 预期结果

### Box覆盖分析
- **覆盖比例**: 目标覆盖60%以上的盒子
- **分散度**: 避免集中在少数几个盒子
- **行业分布**: 系统性地覆盖多个行业

### 性能指标
- **夏普比率**: 与传统方法对比
- **最大回撤**: 控制在15%以内
- **信息比率**: 相对于基准的表现

### 风险控制
- **单只股票最大权重**: 8%
- **单个盒子最大权重**: 15%
- **行业集中度**: 不超过25%

## 🔧 自定义配置

### 修改股票池
编辑配置文件中的 `symbols` 列表：
```yaml
training_setup:
  parameters:
    symbols:
      - AAPL  # 科技大盘成长股
      - JPM   # 金融大盘价值股
      - JNJ   # 医疗大盘价值股
      # 添加更多股票...
```

### 调整盒子维度
```yaml
box_weights:
  dimensions:
    size: ["large"]           # 只关注大盘股
    style: ["growth", "value"] # 成长+价值
    region: ["developed"]      # 只关注发达市场
    sector: ["Technology", "Financials", "Healthcare"] # 3个主要行业
```

### 优化参数调整
```yaml
hyperparameter_optimization:
  n_trials: 50        # 增加试验次数以找到更好参数
  objective: "sharpe_ratio"  # 优化目标改为夏普比率
```

## 📈 性能对比

### Box-Based vs 传统量化优化
| 指标 | Box-Based | 传统量化 |
|------|-----------|----------|
| 盒子覆盖率 | 60-80% | 10-30% |
| 集中度风险 | 低 | 高 |
| 行业分散度 | 高 | 低 |
| 夏普比率 | 预期更稳定 | 可能较高但波动大 |

### 分析输出
实验会生成以下分析：
- **Box覆盖热力图**: 显示每个盒子的覆盖情况
- **权重分布图**: 展示权重在盒子间的分布
- **因子暴露分析**: FF5因子的时间序列分析
- **贡献度分析**: 每个盒子对收益的贡献

## 🐛 常见问题

### Q: 为什么某些盒子没有被覆盖？
A: 可能原因：
1. 股票池中没有对应类别的股票
2. 盒子内股票数量少于 `min_stocks_per_box`
3. 所有股票的FF5信号都为负

### Q: 如何增加盒子覆盖率？
A: 方法：
1. 扩大股票池，增加不同类别的股票
2. 降低 `stocks_per_box` 要求
3. 调整盒子维度定义

### Q: 实验运行时间太长？
A: 优化方法：
1. 使用 `--demo` 配置进行快速测试
2. 减少 `n_trials` 超参数优化次数
3. 缩短回测时间段

### Q: 如何调试配置问题？
A: 调试步骤：
1. 使用 `--dry-run` 验证配置
2. 检查日志输出的配置摘要
3. 确认所有必需的配置项都存在

## 📚 技术细节

### Box-First 算法流程
1. **分类**: 使用4因子模型将股票分类到盒子
2. **信号计算**: 使用训练好的FF5模型计算预期收益
3. **盒子选择**: 确定目标盒子集合
4. **股票筛选**: 在每个盒子内按信号强度选择股票
5. **权重分配**: 根据信号强度分配盒子内权重
6. **归一化**: 确保总权重为1

### FF5模型集成
- **训练阶段**: 使用历史数据训练5因子回归模型
- **预测阶段**: 为每只股票生成预期收益信号
- **信号应用**: 信号用于盒内股票排序和权重分配

### 风险管理
- **盒子层级风险**: 限制单个盒子的最大权重
- **股票层级风险**: 限制单只股票的最大权重
- **行业风险**: 控制行业集中度
- **流动性风险**: 最小持仓要求

## 🎯 下一步优化

1. **动态盒子权重**: 根据市场条件调整盒子权重
2. **多因子模型**: 集成更多因子到FF5模型
3. **时间序列优化**: 考虑因子的时间序列特性
4. **机器学习**: 使用ML模型优化盒子权重分配
5. **替代数据**: 集成另类数据改善信号质量

---

## 📞 支持

如果遇到问题或需要帮助：
1. 检查日志输出的错误信息
2. 验证配置文件的格式和内容
3. 使用 `--dry-run` 选项验证配置
4. 参考主项目的README文档

**开始体验Box-First + FF5的强大组合吧！** 🚀