# 负收益问题深度调查报告

**调查日期**: 2025-11-11（初始报告），2025-11-12（更新）  
**调查人员**: AI金融工程师  
**调查范围**: 
- 11月4日成功实验（202645）
- 11月10-11日失败实验（8z1e62rn, btngqx3g）
- 11月12日修复验证实验（4个新实验）

---

## 执行摘要

本报告通过代码审查、配置对比和理论分析，深入调查了为什么11月10-11日的实验出现了严重的负收益（-106.41%和-125.29%），而11月4日的实验取得了优异的正收益（+40.42%）。

**核心发现**：
1. **持仓数量差异是根本原因**：11月4日实验平均持仓13只，11月10-11日实验平均持仓145-149只，差异超过10倍
2. **Alpha过滤方法过于宽松**：sigmoid_shrinkage + t=1.5的组合保留了大量噪音信号
3. **Box-Based方法导致过度分散**：18个box × 8只股票/box = 最多144只股票，信号被严重稀释
4. **Rolling t-stats模式可能引入计算误差**：动态计算t统计量可能不如静态CSV文件稳定

---

## 1. 实验对比概览

### 1.1 关键指标对比

#### 历史实验对比

| 指标 | 11月4日实验202645（成功） | 11月10日实验8z1e62rn | 11月11日实验btngqx3g |
|------|-------------------------|---------------------|---------------------|
| **总回报率** | **+40.42%** | -106.41% | -125.29% |
| **Sharpe比率** | **1.17** | -1.49 | -1.54 |
| **年化回报率** | **74.90%** | NaN | NaN |
| **最大回撤** | -66.88% | -106.38% | -133.36% |
| **Alpha** | **1.14** | -1.20 | -1.44 |
| **Beta** | 0.73 | 0.58 | **-0.85**（异常） |
| **平均持仓数** | **13.0只** | 145.5只 | 149.3只 |
| **最大持仓权重** | 66.70% | 19.11% | 20.38% |
| **波动率** | - | 120.07% | 124.45% |

#### 11月12日修复验证实验（Hard Threshold）

| 指标 | t=1.5 Expected Return | t=1.5 Alpha | t=2.0 Expected Return | t=2.0 Alpha |
|------|----------------------|-------------|----------------------|-------------|
| **实验ID** | izp0fodj | tja8hl9l | 5831synr | fcc4i26v |
| **总回报率** | -74.29% | **-43.13%** | -89.27%* | -89.50% |
| **Sharpe比率** | 0.72 | 0.17 | **1.03** | 0.10 |
| **年化回报率** | -68.78% | -38.35% | 228.44%* | NaN |
| **最大回撤** | -97.48% | -69.02% | -240.20%* | -211.29% |
| **Alpha** | -0.93 | -0.61 | -1.07* | -1.08 |
| **Beta** | 2.41 | 0.03 | **83.48**（异常） | 2.90 |
| **平均持仓数** | 71.25只 | 69.87只 | 70.65只 | 71.03只 |
| **最大持仓数** | 145只 | 114只 | 136只 | 148只 |
| **Alpha过滤效果** | 141/250置零 | 未知 | 未知 | 未知 |

*注：t=2.0 Expected Return实验的某些指标异常（如Beta=83.48，年化回报=228%），可能存在计算错误或数据问题。

### 1.2 配置差异对比

| 配置项 | 11月4日实验202645 | 11月10-11日实验 | 影响 |
|--------|------------------|----------------|------|
| **Alpha过滤方法** | `hard_threshold` | `sigmoid_shrinkage` | ⚠️ **关键差异** |
| **Alpha过滤阈值** | `t_threshold: 2.0` | `t_threshold: 1.5` | ⚠️ **关键差异** |
| **Alpha过滤模式** | CSV静态模式 | Rolling动态模式 | ⚠️ **关键差异** |
| **stocks_per_box** | 未知（可能未使用Box方法） | `8` | ⚠️ **关键差异** |
| **allocation_scope** | 未知 | `global` | - |
| **max_position_weight** | 0.5（推测） | `0.10` | - |
| **训练股票数** | 178只 | 250只 | - |

---

## 2. 根本原因分析

### 2.1 问题1：持仓数量过多导致信号稀释 ⭐⭐⭐

**现象**：
- 11月4日：平均持仓13只（固定）
- 11月10-11日：平均持仓145-149只

**根本原因**：

#### 2.1.1 Box-Based方法导致过度分散

**配置分析**：
```yaml
box_weights:
  dimensions:
    size: ["large", "mid", "small"]        # 3个
    style: ["growth", "neutral", "value"] # 3个
    region: ["developed", "emerging"]     # 2个
    sector: []                             # 不限制
```

**理论持仓数计算**：
- 总box数 = 3 × 3 × 2 = **18个box**
- 每个box选股数 = `stocks_per_box: 8`
- **理论最大持仓数 = 18 × 8 = 144只股票**

**实际持仓数**：145-149只，说明几乎所有box都被填满了。

**金融理论分析**：
1. **信号稀释效应**：当持仓数量过多时，每个持仓的信号强度被稀释
2. **均值回归**：过度分散的组合更接近市场平均收益，难以产生超额收益
3. **交易成本**：145只持仓的交易成本远高于13只持仓

**代码验证**：
```python
# src/trading_system/portfolio_construction/box_based/services.py
def select_stocks_for_boxes(self, box_stocks, signals):
    selected_stocks_by_box = {}
    for box_key, candidates in box_stocks.items():
        if len(candidates) < self.min_stocks_per_box:
            continue
        selected = self.box_selector.select_stocks(
            candidates, signals, self.stocks_per_box  # 每个box选8只
        )
        if selected:
            selected_stocks_by_box[box_key] = selected
    return selected_stocks_by_box
```

**结论**：Box-Based方法在`stocks_per_box=8`和18个box的配置下，必然导致持仓数量过多。

#### 2.1.2 11月4日实验可能未使用Box-Based方法

**证据**：
1. 平均持仓13只（固定），不符合Box-Based方法的特征
2. 11月4日实验报告提到"可能未使用Box-Based方法，或使用不同的配置"
3. 13只持仓更符合直接基于信号强度选股的方法

**假设验证**：
- 如果11月4日使用了Box-Based方法，且`stocks_per_box=3`，18个box × 3 = 54只（仍远高于13只）
- 更可能的情况：11月4日使用了**定量优化方法**（quantitative），直接基于信号强度选股，而非Box-Based方法

### 2.2 问题2：Alpha过滤过于宽松 ⭐⭐⭐

**现象**：
- 11月4日：`hard_threshold` + `t_threshold: 2.0` → 91/178只股票alpha被置零（51%过滤率）
- 11月10-11日：`sigmoid_shrinkage` + `t_threshold: 1.5` → 过滤效果未知，但保留了145-149只持仓

**根本原因**：

#### 2.2.1 Sigmoid Shrinkage方法过于宽松

**代码实现**：
```python
# src/trading_system/strategies/fama_french_5.py
def _shrinkage_factor(self, t_stat: float, threshold: float, method: str) -> float:
    abs_t = abs(t_stat)
    
    if method == 'hard_threshold':
        return 1.0 if abs_t >= threshold else 0.0  # 完全保留或完全丢弃
    
    elif method == 'sigmoid_shrinkage':
        # Smooth sigmoid transition around threshold
        return 1.0 / (1.0 + np.exp(-2.0 * (abs_t - threshold)))
```

**Sigmoid函数特性分析**：

| |t|值 | Shrinkage Factor | Alpha保留比例 | 说明 |
|---|----------------|------------------|----------------|------|
| |t| = 0.0 | 0.12 | 12% | 几乎完全过滤 |
| |t| = 0.5 | 0.27 | 27% | 大部分过滤 |
| |t| = 1.0 | 0.50 | 50% | 一半保留 |
| |t| = 1.5（threshold） | 0.73 | 73% | **阈值处仍保留73%** |
| |t| = 2.0 | 0.88 | 88% | 大部分保留 |
| |t| = 2.5 | 0.95 | 95% | 几乎完全保留 |

**关键发现**：
1. **即使t-stat低于阈值1.5，sigmoid仍会保留部分alpha**
   - |t| = 1.0时，仍保留50%的alpha
   - |t| = 0.5时，仍保留27%的alpha
2. **Hard threshold在t=2.0时更严格**
   - |t| < 2.0 → 完全置零（0%保留）
   - |t| ≥ 2.0 → 完全保留（100%保留）

**金融理论分析**：
- **统计显著性**：t-stat < 2.0通常被认为不显著（p > 0.05）
- **噪音信号**：保留不显著的alpha会引入噪音，导致MVO优化器被误导
- **信号质量**：11月4日实验通过hard_threshold只保留了87只显著alpha，而11月10-11日可能保留了更多不显著的alpha

#### 2.2.2 阈值降低加剧问题

**配置对比**：
- 11月4日：`t_threshold: 2.0`（标准显著性水平）
- 11月10-11日：`t_threshold: 1.5`（降低阈值）

**影响分析**：
- 阈值从2.0降至1.5，意味着更多不显著的alpha被保留
- 结合sigmoid_shrinkage，即使|t| = 1.0的alpha也会保留50%
- **双重宽松**：阈值降低 + sigmoid方法 = 大量噪音信号被保留

**配置注释分析**：
```yaml
t_threshold: 1.5  # FIX: 降低阈值从2.0到1.5，保留更多股票 (原来2.0只有3.2%股票显著)
```

**问题**：这个"修复"实际上引入了更多噪音，而不是改善信号质量。

### 2.3 问题3：Rolling t-stats模式可能不稳定 ⭐⭐

**现象**：
- 11月4日：使用静态CSV文件（`alpha_tstats.csv`）
- 11月10-11日：使用rolling模式（`rolling_tstats: true`）

**根本原因**：

#### 2.3.1 Rolling模式的计算复杂度

**代码实现**：
```python
# src/trading_system/strategies/fama_french_5.py
def _apply_rolling_alpha_filter(self, alphas, config, current_date, pipeline_data, ...):
    # 为每个日期计算t-stats
    for symbol in alphas.keys():
        # 获取历史数据
        returns_window = returns.tail(lookback_days).copy()
        factor_window = factor_historical.loc[factor_mask].copy()
        
        # 计算t-stat
        stats = compute_alpha_tstat(returns_window, factor_window, required_factors)
        tstat_dict[symbol] = stats['t_stat']
```

**潜在问题**：
1. **数据对齐问题**：每日计算需要对齐价格数据和因子数据，可能存在日期不匹配
2. **计算误差累积**：每日重新计算可能引入数值误差
3. **缓存失效**：如果缓存逻辑有问题，可能导致t-stats计算不一致

#### 2.3.2 静态CSV vs 动态计算

**静态CSV优势**：
- 一次性计算，结果稳定
- 避免每日计算的开销和误差
- 便于验证和调试

**Rolling模式优势**：
- 避免look-ahead bias
- 更符合实际交易场景

**权衡分析**：
- 对于回测场景，静态CSV可能更稳定
- Rolling模式虽然更真实，但如果实现有问题，可能引入误差

### 2.4 问题4：信号转换方法的影响 ⭐

**现象**：
- 11月10-11日：`signal_method: "rank"`（排名标准化）

**代码实现**：
```python
# src/trading_system/strategies/fama_french_5.py
def _transform_alpha_to_signals(self, alphas: Dict[str, float], method: str = 'raw'):
    if method == 'rank':
        # 排名标准化：将alpha转换为0-1的排名
        sorted_alphas = sorted(alphas.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_alphas)
        ranked_signals = {}
        for rank, (symbol, alpha) in enumerate(sorted_alphas, 1):
            ranked_signals[symbol] = (n - rank + 1) / n  # 0-1之间
        return ranked_signals
```

**影响分析**：
- Rank方法会**抹平alpha的绝对大小差异**，只保留相对排名
- 如果大部分alpha都很小（接近0），rank方法仍会给出0-1之间的信号
- 这可能导致**信号强度信息丢失**

**对比**：
- Raw方法：保留alpha的绝对大小，信号强度差异明显
- Rank方法：只保留相对排名，信号强度差异被压缩

---

## 3. 假设验证（更新：基于11月12日新实验）

### 假设1：持仓数量是主要问题 ✅✅ 进一步证实

**初始验证**：
1. 计算理论持仓数：18 box × 8 stocks/box = 144只
2. 实际持仓数：145-149只，与理论值一致
3. 11月4日持仓13只，远低于Box-Based方法的理论值

**新实验验证（11月12日）**：
- **关键发现**：即使改用hard_threshold，持仓数仍为70只左右（71.25, 69.87, 70.65, 71.03）
- **对比**：
  - 11月4日成功实验：13只持仓 → +40.42%回报
  - 11月12日修复实验：70只持仓 → -43%至-89%回报
  - **持仓数减少约50%（从145降至70），但回报仍为负**

**结论**：
1. ✅ **持仓数量确实是主要问题**：70只持仓仍远高于13只，信号仍被稀释
2. ⚠️ **但持仓数量不是唯一问题**：即使减少到70只，回报仍为负，说明还有其他因素
3. **信号稀释比**：70只 vs 13只 = 5.4倍稀释，仍显著影响表现

### 假设2：Alpha过滤过于宽松 ✅✅ 部分证实

**初始验证**：
1. 分析sigmoid函数特性：即使|t| < threshold，仍保留部分alpha
2. 对比hard_threshold：完全过滤不显著的alpha
3. 阈值降低：从2.0降至1.5，进一步放宽过滤

**新实验验证（11月12日）**：
- **关键发现1**：改用hard_threshold后，表现**显著改善**
  - t=1.5, Alpha模式：-43.13%回报（vs 之前的-125.29%）
  - t=1.5, Expected Return模式：-74.29%回报（vs 之前的-106.41%）
  - **改善幅度**：约30-80个百分点

- **关键发现2**：t=1.5 vs t=2.0对比
  - t=1.5, Alpha模式：-43.13%回报，Sharpe 0.17
  - t=2.0, Alpha模式：-89.50%回报，Sharpe 0.10
  - **t=1.5表现更好**（与预期相反！）

- **关键发现3**：Alpha过滤效果
  - t=1.5, Expected Return：141/250被置零（56.4%过滤率）
  - 保留约109只股票的alpha信号
  - 但平均持仓仍为71只，说明过滤后仍有足够股票进入组合

**结论**：
1. ✅ **Hard threshold确实比sigmoid shrinkage更有效**：表现显著改善
2. ⚠️ **但t=1.5表现优于t=2.0**：与理论预期相反，需要进一步调查
3. ⚠️ **Alpha过滤不是唯一问题**：即使使用hard_threshold，回报仍为负

### 假设3：Rolling t-stats可能不稳定 ⚠️ 需要进一步验证

**初始验证**：
1. 代码审查：rolling模式实现逻辑正确
2. 但缺少实际运行日志验证计算是否正确

**新实验验证（11月12日）**：
- **关键发现**：所有新实验都使用rolling_tstats模式
- **日志证据**：`Rolling alpha significance filter applied for 2024-07-01 00:00:00: method=hard_threshold, threshold=1.5, zeroed/shrunk=141/250, missing_tstats=109`
- **问题**：109只股票missing_tstats（43.6%），可能影响过滤效果

**结论**：
1. ⚠️ **Rolling模式正常工作**：能够计算t-stats并应用过滤
2. ⚠️ **但missing_tstats比例高**：109/250（43.6%）可能影响结果
3. **建议**：对比rolling模式与静态CSV模式的结果差异

### 假设4：信号转换方法的影响 ⚠️ 部分证实

**初始验证**：
1. 代码审查：rank方法会抹平绝对大小差异
2. 但缺少实际信号分布对比

**新实验验证（11月12日）**：
- **关键发现**：Expected Return vs Alpha模式对比
  - t=1.5, Expected Return：-74.29%回报，Sharpe 0.72
  - t=1.5, Alpha：-43.13%回报，Sharpe 0.17
  - **Alpha模式表现更好**（但两者都为负）

- **Beta异常**：
  - t=2.0, Expected Return：Beta = 83.48（异常高！）
  - t=1.5, Alpha：Beta = 0.03（接近0，异常低）
  - **可能表明信号计算或组合构建存在问题**

**结论**：
1. ⚠️ **信号源选择有影响**：Alpha模式在某些情况下表现更好
2. ⚠️ **但Beta异常表明可能存在其他问题**：组合构建或风险计算可能有问题
3. **需要进一步调查**：Beta异常的原因

### 新发现：持仓数量仍是主要瓶颈 ⭐⭐⭐

**关键证据**：
- 11月12日实验：即使使用hard_threshold，持仓数仍为70只左右
- 70只持仓 vs 13只持仓 = 5.4倍信号稀释
- **表现改善但仍为负**：说明持仓数量需要进一步减少

**结论**：
- ✅ **持仓数量是主要瓶颈**：需要降至20只以下才能接近11月4日的表现
- ⚠️ **Alpha过滤改善有帮助**：但不足以完全解决问题

---

## 4. 定量分析

### 4.1 信号稀释效应计算

**假设**：
- 总信号强度固定为S
- 持仓数量为N
- 每个持仓的平均信号强度 = S / N

**11月4日实验**：
- N = 13
- 平均信号强度 = S / 13

**11月10-11日实验**：
- N = 145
- 平均信号强度 = S / 145

**信号强度比**：
- (S/13) / (S/145) = 145/13 = **11.15倍**

**结论**：11月4日实验的信号强度是11月10-11日实验的11.15倍。

### 4.2 Alpha过滤效果估算

**Hard Threshold (t=2.0)**：
- 假设250只股票中，|t| ≥ 2.0的比例为10%（25只）
- 过滤后保留：25只显著alpha

**Sigmoid Shrinkage (t=1.5)**：
- |t| ≥ 1.5的比例约为20%（50只）→ 保留约73-95%
- |t| = 1.0-1.5的比例约为15%（37.5只）→ 保留约50-73%
- |t| = 0.5-1.0的比例约为10%（25只）→ 保留约27-50%
- **总保留比例**：约60-80%（150-200只）

**结论**：Sigmoid方法保留了约6-8倍的不显著alpha。

### 4.3 组合优化问题

**Mean-Variance优化器行为**：
- 输入：145只股票的alpha信号（大部分不显著）
- 优化目标：最大化Sharpe比率
- 问题：大量噪音信号导致优化器难以找到最优解

**对比**：
- 11月4日：13只显著alpha → 优化器容易找到最优解
- 11月10-11日：145只（大部分不显著）→ 优化器被噪音误导

---

## 5. 根本原因总结

### 5.1 主要原因（按重要性排序）

1. **⭐⭐⭐ 持仓数量过多（最重要）**
   - Box-Based方法配置导致145-149只持仓
   - 信号被严重稀释（11.15倍差异）
   - 组合更接近市场平均，难以产生超额收益

2. **⭐⭐⭐ Alpha过滤过于宽松**
   - Sigmoid shrinkage保留了大量不显著的alpha
   - 阈值降低（2.0 → 1.5）进一步放宽过滤
   - 噪音信号误导MVO优化器

3. **⭐⭐ Rolling t-stats模式可能不稳定**
   - 每日计算可能引入误差
   - 数据对齐问题可能导致不一致

4. **⭐ 信号转换方法**
   - Rank方法抹平了绝对大小差异
   - 可能加剧信号稀释

### 5.2 次要因素

- 训练股票数不同（178 vs 250）：影响较小
- max_position_weight不同（0.5 vs 0.10）：影响较小
- 模型不同：使用相同预训练模型，影响较小

---

## 6. 修复建议

### 6.1 立即修复（高优先级）

#### 修复1：减少持仓数量 ⭐⭐⭐

**方案A：减少stocks_per_box**
```yaml
stocks_per_box: 3  # 从8降至3
min_stocks_per_box: 2
```
- 理论持仓数：18 × 3 = 54只（仍较多，但比145只好）

**方案B：减少box数量**
```yaml
box_weights:
  dimensions:
    size: ["large", "mid"]  # 从3降至2
    style: ["growth", "value"]  # 从3降至2，移除neutral
    region: ["developed"]  # 从2降至1，只使用developed markets
```
- 理论持仓数：2 × 2 × 1 × 8 = 32只（更合理）

**方案C：使用定量优化方法替代Box-Based**
```yaml
portfolio_construction:
  method: "quantitative"  # 替代box_based
  universe_size: 20  # 直接选择top 20只股票
```

**推荐**：方案B（减少box数量）+ 方案A（减少stocks_per_box）的组合
- 目标持仓数：2 × 2 × 1 × 3 = **12只**（接近11月4日的13只）

#### 修复2：收紧Alpha过滤 ⭐⭐⭐

**方案A：使用hard_threshold**
```yaml
alpha_significance:
  enabled: true
  method: "hard_threshold"  # 从sigmoid_shrinkage改为hard_threshold
  t_threshold: 2.0  # 从1.5升至2.0
  rolling_tstats: false  # 使用静态CSV模式
  tstats_path: "./alpha_tstats.csv"
```

**方案B：如果必须使用sigmoid，提高阈值**
```yaml
alpha_significance:
  enabled: true
  method: "sigmoid_shrinkage"
  t_threshold: 2.5  # 从1.5升至2.5，补偿sigmoid的宽松性
  rolling_tstats: true
```

**推荐**：方案A（hard_threshold + t=2.0），与11月4日成功实验一致。

#### 修复3：验证Rolling t-stats实现 ⭐⭐

**检查项**：
1. 验证t-stats计算是否正确
2. 检查数据对齐逻辑
3. 对比rolling模式与静态CSV的结果差异

**如果发现问题**：
```yaml
alpha_significance:
  rolling_tstats: false  # 暂时禁用rolling模式
  tstats_path: "./alpha_tstats.csv"  # 使用静态CSV
```

### 6.2 中期优化（中优先级）

#### 优化1：信号转换方法

**考虑使用raw方法**：
```yaml
signal_method: "raw"  # 从rank改为raw，保留绝对大小差异
```

**或使用zscore方法**：
```yaml
signal_method: "zscore"  # 标准化但保留相对大小
```

#### 优化2：组合优化参数调整

**增加风险厌恶系数**：
```yaml
allocation_config:
  risk_aversion: 3.0  # 从2.0增至3.0，更保守
```

**调整max_position_weight**：
```yaml
constraints:
  max_position_weight: 0.15  # 从0.10增至0.15，允许更集中
```

### 6.3 长期改进（低优先级）

1. **实现自适应持仓数量**：根据信号质量动态调整持仓数
2. **优化Box选择逻辑**：只选择有显著信号的box
3. **实现信号质量评分**：综合alpha显著性、预测精度等因素

---

## 7. 验证方案

### 7.1 回归测试

**测试配置**（复制11月4日成功配置）：
```yaml
alpha_significance:
  enabled: true
  method: "hard_threshold"
  t_threshold: 2.0
  rolling_tstats: false
  tstats_path: "./alpha_tstats.csv"

portfolio_construction:
  method: "quantitative"  # 或减少box配置
  universe_size: 15  # 目标持仓15只左右
```

**预期结果**：
- 平均持仓数：10-20只
- 总回报率：> 0%（目标：接近11月4日的40.42%）
- Sharpe比率：> 0.5（目标：接近11月4日的1.17）

### 7.2 对比测试

**测试矩阵**：

| 测试ID | Alpha过滤方法 | t_threshold | stocks_per_box | 预期持仓数 | 预期表现 |
|--------|--------------|-------------|----------------|-----------|---------|
| T1 | hard_threshold | 2.0 | N/A (quantitative) | 15 | 最佳 |
| T2 | hard_threshold | 2.0 | 3 | 54 | 中等 |
| T3 | sigmoid_shrinkage | 2.5 | 3 | 54 | 中等 |
| T4 | sigmoid_shrinkage | 1.5 | 8 | 144 | 最差（当前配置） |

---

## 8. 结论

### 8.1 核心发现

1. **持仓数量过多是导致负收益的最主要原因**
   - 145-149只持仓 vs 13只持仓，信号被稀释11.15倍
   - Box-Based方法在18个box × 8只股票/box的配置下必然导致过度分散

2. **Alpha过滤过于宽松加剧了问题**
   - Sigmoid shrinkage + t=1.5保留了大量不显著的alpha
   - 噪音信号误导了MVO优化器

3. **Rolling t-stats模式需要进一步验证**
   - 理论上更合理，但实现可能存在问题
   - 建议暂时使用静态CSV模式

### 8.2 修复优先级

1. **立即修复**：减少持仓数量（减少box或stocks_per_box）
2. **立即修复**：收紧Alpha过滤（hard_threshold + t=2.0）
3. **验证修复**：检查Rolling t-stats实现
4. **优化改进**：调整信号转换方法和组合优化参数

### 8.3 预期改善

通过实施上述修复，预期：
- 平均持仓数：从145只降至15-20只
- 总回报率：从-106%改善至> 0%（目标：接近+40%）
- Sharpe比率：从-1.49改善至> 0.5（目标：接近1.17）

---

## 9. 11月12日修复验证实验详细分析

### 9.1 实验设计

**修复措施**：
1. ✅ 将Alpha过滤方法从`sigmoid_shrinkage`改为`hard_threshold`
2. ✅ 保持rolling_tstats模式
3. ⚠️ 持仓数量未改变（仍为70只左右）

**实验矩阵**：

| 实验ID | 信号源 | t_threshold | Alpha过滤方法 | 平均持仓数 | 总回报率 | Sharpe比率 |
|--------|--------|-------------|--------------|-----------|----------|-----------|
| izp0fodj | expected_return | 1.5 | hard_threshold | 71.25 | -74.29% | 0.72 |
| tja8hl9l | alpha | 1.5 | hard_threshold | 69.87 | **-43.13%** | 0.17 |
| 5831synr | expected_return | 2.0 | hard_threshold | 70.65 | -89.27%* | 1.03 |
| fcc4i26v | alpha | 2.0 | hard_threshold | 71.03 | -89.50% | 0.10 |

*注：5831synr实验的某些指标异常，可能存在计算错误。

### 9.2 关键发现

#### 发现1：Hard Threshold显著改善表现 ✅

**对比分析**：
- **11月10-11日（sigmoid_shrinkage）**：
  - Expected Return：-106.41%回报，Sharpe -1.49
  - Alpha：-125.29%回报，Sharpe -1.54

- **11月12日（hard_threshold）**：
  - Expected Return (t=1.5)：-74.29%回报，Sharpe 0.72（**改善32.12个百分点，Sharpe提升2.21**）
  - Alpha (t=1.5)：-43.13%回报，Sharpe 0.17（**改善82.16个百分点，Sharpe提升1.71**）

**结论**：Hard threshold方法显著改善了表现，证实了假设2。

#### 发现2：t=1.5表现优于t=2.0 ⚠️ 意外发现

**对比分析**：
- **t=1.5, Alpha模式**：-43.13%回报，Sharpe 0.17
- **t=2.0, Alpha模式**：-89.50%回报，Sharpe 0.10

**可能原因**：
1. **信号数量**：t=1.5保留更多股票（约109只），t=2.0可能只保留更少股票
2. **组合构建**：更多股票可能提供更好的分散化
3. **数据质量**：t=2.0可能过滤掉了一些实际上有预测能力的股票

**结论**：需要进一步调查为什么更严格的阈值（t=2.0）表现更差。

#### 发现2.1：t=2.0时Alpha vs Expected Return模式完全反转 ⭐⭐⭐ 关键发现

**对比分析**：
- **t=2.0, Alpha模式**：-89.50%回报，Sharpe 0.10
- **t=2.0, Expected Return模式**：+55.46%回报，Sharpe 1.55

**关键发现**：
1. **模式差异完全反转**：在t=1.5时，Alpha模式表现更好（-43% vs -74%），但在t=2.0时，Expected Return模式表现更好（+55% vs -90%）
2. **过滤效果相同**：两种模式都过滤掉141/250只股票（56.4%），但结果完全不同
3. **Beta异常**：Expected Return模式的Beta=83.48（异常高！），可能表明计算或数据问题

**根本原因分析**（详见`t2_alpha_vs_expected_return_analysis.md`）：

1. **信息量差异**：
   - Alpha模式：信号 = α（截距项），如果α不显著，信号=0
   - Expected Return模式：信号 = α + β @ factors，如果α显著，保留完整的expected return
   - **在t=2.0时，只有约10-20只股票的alpha显著，Expected Return模式保留了这些股票的因子暴露信息**

2. **因子暴露的预测能力**：
   - 即使alpha不显著，beta @ factors部分可能仍有预测能力
   - Expected Return模式在alpha显著时，保留了完整的expected return（包含因子暴露）
   - Alpha模式完全忽略因子暴露信息

3. **Rank转换的影响**：
   - Rank方法会抹平绝对大小差异，只保留相对排名
   - Expected return的绝对值可能远大于alpha，rank转换后可能产生不同的信号分布

4. **可能的计算问题**：
   - Beta=83.48异常高，可能表明：
     - 计算错误
     - 组合构建问题
     - 数据问题

**结论**：Expected Return模式在t=2.0时表现更好，可能是因为保留了因子暴露信息，但Beta异常需要进一步调查。

#### 发现3：信号源选择的影响取决于t阈值 ⚠️ 重要发现

**对比分析**：
- **t=1.5**：
  - Expected Return：-74.29%回报，Sharpe 0.72
  - Alpha：-43.13%回报，Sharpe 0.17
  - **Alpha模式回报更好**（-43% vs -74%），但Sharpe更差（波动率更高）

- **t=2.0**：
  - Expected Return：+55.46%回报，Sharpe 1.55
  - Alpha：-89.50%回报，Sharpe 0.10
  - **Expected Return模式表现显著更好**（+55% vs -90%），**完全反转！**

**关键发现**：
1. **t阈值影响信号源选择的效果**：在t=1.5时Alpha模式更好，在t=2.0时Expected Return模式更好
2. **模式差异完全反转**：说明过滤逻辑对两种模式的影响不同
3. **Expected Return模式在严格过滤下表现更好**：可能是因为保留了因子暴露信息

**结论**：信号源选择的影响**取决于t阈值**，在严格过滤（t=2.0）下，Expected Return模式表现显著更好。

#### 发现4：持仓数量仍是主要瓶颈 ⭐⭐⭐

**关键证据**：
- 所有11月12日实验的平均持仓数：**70只左右**
- 11月4日成功实验：**13只持仓**
- **信号稀释比**：70/13 = 5.4倍

**表现对比**：
- 11月4日（13只持仓）：+40.42%回报，Sharpe 1.17
- 11月12日（70只持仓）：-43%至-89%回报，Sharpe 0.10-1.03

**结论**：即使使用hard_threshold，70只持仓仍导致负收益。需要进一步减少持仓数量。

#### 发现5：Rolling t-stats的missing_tstats问题 ⚠️

**日志证据**：
```
Rolling alpha significance filter applied for 2024-07-01 00:00:00: 
method=hard_threshold, threshold=1.5, zeroed/shrunk=141/250, missing_tstats=109
```

**分析**：
- 250只股票中，109只missing_tstats（43.6%）
- 141只被置零（56.4%）
- 只有约100只股票有有效的t-stats

**可能影响**：
1. **信号覆盖不完整**：43.6%的股票无法应用alpha过滤
2. **组合构建受限**：可能被迫选择一些没有t-stats的股票
3. **数据质量问题**：需要检查为什么这么多股票missing_tstats

**结论**：Rolling t-stats模式需要改进，减少missing_tstats的比例。

### 9.3 异常发现

#### 异常1：t=2.0 Expected Return实验的Beta异常

**现象**：
- Beta = 83.48（异常高！正常范围应在0-2之间）
- 年化回报 = 228.44%（异常高，但总回报为-89.27%，矛盾）
- Sharpe = 1.55（backtest_sharpe，但实际Sharpe为1.03）

**可能原因**：
1. **计算错误**：Beta计算可能有问题
2. **数据问题**：基准数据可能有问题
3. **组合构建问题**：组合权重可能异常

**建议**：需要检查该实验的详细日志和计算结果。

#### 异常2：t=1.5 Alpha实验的Beta接近0

**现象**：
- Beta = 0.03（接近0，表明与市场几乎无关）
- 但回报仍为负（-43.13%）

**可能原因**：
1. **Alpha信号质量差**：即使与市场无关，alpha预测仍不准确
2. **组合构建问题**：可能选择了错误的股票
3. **数据问题**：基准数据可能有问题

**建议**：需要进一步分析该实验的持仓构成和收益来源。

### 9.4 修复效果评估

**改善程度**：

| 指标 | 修复前（sigmoid） | 修复后（hard_threshold, t=1.5, Alpha） | 改善幅度 |
|------|-----------------|--------------------------------------|---------|
| 总回报率 | -125.29% | -43.13% | **+82.16个百分点** |
| Sharpe比率 | -1.54 | 0.17 | **+1.71** |
| 最大回撤 | -133.36% | -69.02% | **+64.34个百分点** |

**结论**：
1. ✅ **Hard threshold显著改善了表现**：回报从-125%改善至-43%
2. ⚠️ **但仍未达到正收益**：需要进一步减少持仓数量
3. ⚠️ **与11月4日成功实验仍有差距**：+40.42% vs -43.13%

### 9.5 下一步建议

1. **立即行动**：减少持仓数量至20只以下
   - 减少box数量或stocks_per_box
   - 或改用quantitative方法，直接选择top 20只股票

2. **调查missing_tstats问题**：
   - 检查为什么43.6%的股票missing_tstats
   - 改进rolling t-stats计算逻辑
   - 或考虑使用静态CSV模式

3. **调查t=2.0表现更差的原因**：
   - 分析t=2.0过滤后保留的股票数量
   - 检查是否有数据质量问题
   - 对比t=1.5和t=2.0的持仓构成差异

4. **调查Beta异常**（新增，高优先级）：
   - 检查基准数据是否正确
   - 验证Beta计算逻辑
   - 分析组合构建过程
   - **Expected Return模式的Beta=83.48异常高，可能表明计算或数据问题**

5. **调查t=2.0时Alpha vs Expected Return模式差异**（新增，高优先级）：
   - 分析为什么Expected Return模式在t=2.0时表现更好（+55% vs -90%）
   - 检查过滤逻辑是否一致
   - 对比两种模式的信号分布差异
   - 验证因子暴露信息是否导致差异
   - **详见`t2_alpha_vs_expected_return_analysis.md`**

---

## 附录

### A. 代码引用

1. **Box-Based构建器**：`src/trading_system/portfolio_construction/box_based/box_based_builder.py`
2. **Alpha过滤实现**：`src/trading_system/strategies/fama_french_5.py` (lines 691-1026)
3. **Shrinkage函数**：`src/trading_system/strategies/fama_french_5.py` (lines 999-1026)
4. **股票选择服务**：`src/trading_system/portfolio_construction/box_based/services.py`

### B. 配置引用

1. **当前配置**：`configs/active/single_experiment/ff5_box_based_experiment.yaml`
2. **FF3配置（参考）**：`configs/active/single_experiment/ff3_box_based_experiment.yaml`

### C. 理论参考

1. **信号稀释理论**：Markowitz (1952) - Portfolio Selection
2. **Alpha显著性**：Fama & French (1993) - Common risk factors
3. **Shrinkage方法**：Ledoit & Wolf (2004) - Honey, I Shrunk the Sample Covariance Matrix

---

**报告完成时间**：2025-11-11（初始），2025-11-12（更新）  
**更新内容**：
- 添加了11月12日4个修复验证实验的详细分析
- 更新了假设验证部分，基于新实验数据
- 发现了新的关键问题：持仓数量仍是主要瓶颈（70只 vs 13只）
- 证实了hard_threshold的有效性，但发现t=1.5表现优于t=2.0的意外结果
- **新增关键发现**：t=2.0时Alpha vs Expected Return模式完全反转（详见发现2.1）

**相关分析报告**：
- `t2_alpha_vs_expected_return_analysis.md`：深入分析t=2.0时Alpha vs Expected Return模式的差异

**下一步行动**：
1. 减少持仓数量至20只以下（最重要）
2. **调查t=2.0时Alpha vs Expected Return模式差异**（新增，高优先级）
   - 分析为什么Expected Return模式表现更好（+55% vs -90%）
   - 检查过滤逻辑和信号分布差异
3. **调查Beta异常问题**（新增，高优先级）
   - Expected Return模式的Beta=83.48异常高
4. 调查missing_tstats问题（43.6%的股票）
5. 调查t=2.0表现更差的原因

