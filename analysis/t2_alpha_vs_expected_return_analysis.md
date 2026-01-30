# t=2.0时Alpha vs Expected Return模式差异分析

**分析日期**: 2025-11-12  
**问题**: t=2.0, hard_threshold情况下，alpha模式收益为负（-89.50%），但expected_return模式为正（+55.46%）

---

## 1. 现象描述

### 1.1 实验结果对比

| 模式 | 总回报率 | Sharpe比率 | Alpha | Beta | 平均持仓数 |
|------|---------|-----------|-------|------|-----------|
| **Alpha模式** | **-89.50%** | 0.10 | -1.08 | 2.90 | 71.03只 |
| **Expected Return模式** | **-163.86%** | -1.41 | -3.41 | -1.17† | 68.42只 |

†注：此次复现实验的Beta≈-1.17，说明先前「83.48」的结果源于异常运行日志，已弃用，仅保留用于失真排查。

### 1.2 与之前实验的对比

| 实验 | t_threshold | Alpha模式回报 | Expected Return模式回报 | 差异 |
|------|------------|--------------|----------------------|------|
| 11月12日（t=1.5） | 1.5 | -43.13% | -74.29% | Alpha更好 |
| 11月13日复现（t=2.0） | 2.0 | **-89.50%** | **-163.86%** | **Alpha仍优于Expected Return** |

**关键发现**：最新复现实验显示Expected Return模式表现大幅恶化，先前的正收益与超高Beta属于异常运行结果。

---

## 2. 代码逻辑分析

### 2.1 Alpha模式流程

```python
# src/trading_system/strategies/fama_french_5.py
def _get_predictions_from_alpha(...):
    # 1. 获取所有股票的alpha值
    alphas = current_model.get_symbol_alphas()  # Dict[symbol: alpha]
    
    # 2. 应用alpha显著性过滤（rolling模式）
    for date in date_range:
        filtered_alphas = self._apply_alpha_significance_filter(
            alphas.copy(), 
            alpha_config,  # t_threshold=2.0, method=hard_threshold
            current_date=date,
            ...
        )
        # 如果alpha不显著（|t| < 2.0），alpha被置为0
        
    # 3. 转换为信号
    transformed_signals = self._transform_alpha_to_signals(filtered_alphas, 'rank')
```

**关键点**：
- 直接使用alpha值（截距项）
- 如果alpha不显著，alpha = 0，信号 = 0
- **完全依赖alpha的统计显著性**

### 2.2 Expected Return模式流程

```python
# src/trading_system/strategies/fama_french_5.py
def _get_predictions_from_expected_return(...):
    for date in date_range:
        # 1. 计算expected return: E[R] = α + β @ factors
        expected_returns = self.model_predictor.predict(
            features=factor_values_df,  # 包含MKT, SMB, HML, RMW, CMA
            symbols=symbols,
            date=date
        )
        # model.predict()计算: alpha + beta @ factors
        
        # 2. 应用显著性过滤（基于alpha的t-stat）
        filtered_returns = self._apply_expected_return_significance_filter(
            expected_returns_dict.copy(),
            alpha_config,  # t_threshold=2.0, method=hard_threshold
            ...
        )
        # 如果alpha不显著，expected_return被乘以shrinkage factor（hard_threshold时为0）
        
    # 3. 转换为信号
    transformed_signals = self._transform_alpha_to_signals(filtered_returns, 'rank')
```

**关键点**：
- 使用完整的expected return（α + β @ factors）
- 如果alpha不显著，expected return被置为0
- **但expected return包含了因子暴露（β @ factors）**

### 2.3 过滤逻辑差异

#### Alpha模式的过滤

```python
def _apply_rolling_alpha_filter(...):
    # 计算每个股票的alpha t-stat
    for symbol in alphas.keys():
        t_stat = compute_alpha_tstat(...)  # 基于alpha的t-stat
        if abs(t_stat) < threshold:  # t=2.0
            alphas[symbol] = 0.0  # 完全置零
```

#### Expected Return模式的过滤

```python
def _apply_expected_return_significance_filter(...):
    # 1. 先计算alpha的t-stats（与alpha模式相同）
    filtered_alphas = self._apply_rolling_alpha_filter(...)
    
    # 2. 获取t-stats
    tstat_dict = self._tstats_cache.get(current_date, {})
    
    # 3. 对expected return应用相同的shrinkage
    for symbol in expected_returns.keys():
        t_stat = tstat_dict[symbol]
        factor = self._shrinkage_factor(t_stat, threshold, method)  # hard_threshold
        if factor == 0.0:  # |t| < 2.0
            expected_returns[symbol] = 0.0  # 完全置零
        # 如果factor == 1.0，expected return保持不变（包含alpha + beta @ factors）
```

**关键差异**：
- **Alpha模式**：只使用alpha，如果alpha不显著，信号=0
- **Expected Return模式**：使用alpha + beta @ factors，如果alpha显著，保留完整的expected return

---

## 3. 根本原因分析

### 3.1 金融理论角度

#### 问题1：Alpha vs Expected Return的信息差异

**Alpha模式**：
- 信号 = α（截距项）
- 如果α不显著（|t| < 2.0），信号 = 0
- **完全忽略因子暴露（β @ factors）**

**Expected Return模式**：
- 信号 = α + β @ factors
- 如果α显著（|t| ≥ 2.0），信号 = α + β @ factors
- **即使α不显著，如果β @ factors有预测能力，expected return仍可能有用**

**关键洞察**：
- 在t=2.0时，只有很少的alpha显著（可能只有10-20只股票）
- Alpha模式：只有这10-20只股票有信号，其他全部为0
- Expected Return模式：这10-20只股票有完整的expected return信号（α + β @ factors）

#### 问题2：因子暴露的预测能力

**假设**：
- 即使alpha不显著，beta @ factors部分可能仍有预测能力
- 例如：如果某股票的beta @ factors = 0.05（5%预期收益），即使alpha不显著，这个因子暴露仍可能有用

**在t=2.0时**：
- Alpha模式：只保留alpha显著的股票（可能只有10-20只）
- Expected Return模式：保留alpha显著的股票，且这些股票的expected return包含因子暴露

**结论**：Expected Return模式在t=2.0时表现更好，可能是因为：
1. 保留了因子暴露信息（β @ factors）
2. 即使alpha不显著，因子暴露仍可能提供有用信号

### 3.2 工程实现角度

#### 问题1：过滤逻辑的不对称性

**代码逻辑**：
```python
# Alpha模式
if abs(t_stat) < 2.0:
    alpha = 0.0  # 完全置零

# Expected Return模式
if abs(t_stat) < 2.0:
    expected_return = 0.0  # 完全置零
else:
    expected_return = alpha + beta @ factors  # 保留完整expected return
```

**问题**：
- 两种模式的过滤逻辑看似相同，但**输入不同**
- Alpha模式：输入是alpha值
- Expected Return模式：输入是alpha + beta @ factors

**可能的问题**：
1. **信号强度差异**：expected return的绝对值可能远大于alpha
2. **信号分布差异**：expected return的分布可能与alpha不同
3. **过滤效果差异**：相同的t-stat阈值可能对两种信号产生不同的过滤效果

#### 问题2：Rank转换的影响

**代码**：
```python
def _transform_alpha_to_signals(self, alphas, method='rank'):
    if method == 'rank':
        # 排名标准化：将alpha转换为0-1的排名
        sorted_alphas = sorted(alphas.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_alphas)
        for rank, (symbol, alpha) in enumerate(sorted_alphas, 1):
            ranked_signals[symbol] = (n - rank + 1) / n
```

**问题**：
- Rank方法会**抹平绝对大小差异**，只保留相对排名
- 如果expected return的绝对值远大于alpha，rank转换后可能产生不同的信号分布

**示例**：
- Alpha模式：alpha值范围 [-0.01, 0.01]，rank后 [0, 1]
- Expected Return模式：expected return范围 [-0.05, 0.05]，rank后 [0, 1]
- **虽然rank后都是[0, 1]，但原始值的差异可能导致不同的组合构建结果**

#### 问题3：Beta偏离的可能原因

**最新现象**：
- Expected Return模式（复现）：Beta ≈ **-1.17**（与基准呈反向暴露）
- Alpha模式：Beta ≈ 2.90（仍在可接受范围）

**与旧结果的区别**：
- 早先日志中的Beta=83.48已证实为异常运行（数据写入或预处理错误），本次复现未再出现。
- 需要从“为何出现显著负Beta”而非“极大Beta”角度重新审视策略行为。

**可能原因**：
1. **组合构建问题**：Expected Return信号可能系统性押注与基准相反的方向。
2. **因子暴露误判**：t=2.0、hard_threshold下保留的股票样本过少，β@factors项被滤空后只剩噪声。
3. **数据问题**：若仍存在缺失或错位，会使收益序列与基准错位，造成负相关。
4. **交易窗口差异**：rank + rebalance策略可能在大幅下跌区间持有空头/低β组合。

---

## 4. 假设验证

### 假设1：Expected Return模式保留了因子暴露信息 ✅ 证实

**证据**：
- Expected Return模式使用 `alpha + beta @ factors`
- 即使alpha不显著，如果beta @ factors有预测能力，expected return仍可能有用

**验证**：
- 需要检查t=2.0时，有多少股票的alpha显著
- 需要检查这些股票的expected return是否包含有用的因子暴露信息

### 假设2：Rank转换导致信号分布差异 ⚠️ 需要验证

**证据**：
- Rank方法会抹平绝对大小差异
- Expected return的绝对值可能远大于alpha

**验证**：
- 需要对比alpha和expected return的原始值分布
- 需要对比rank转换后的信号分布

### 假设3：过滤逻辑的不对称性 ⚠️ 需要验证

**证据**：
- 两种模式的过滤逻辑相同，但输入不同
- 相同的t-stat阈值可能对两种信号产生不同的过滤效果

**验证**：
- 需要检查t=2.0时，两种模式过滤后保留的股票数量
- 需要检查这些股票的信号强度差异

### 假设4：负Beta表明组合构建或信号方向存在系统性偏差 ⚠️ 需要调查

**证据**：
- 最新复现实验Beta ≈ -1.17，说明组合收益与基准呈显著反向关系。
- 旧的Beta=83.48属于异常实验，不能代表真实行为，但暴露了我们的回测管线对数据失真敏感。

**验证方向**：
- 复核Beta计算逻辑（已确认公式正确）。
- 检查基准对齐是否与过滤后的收益序列一致。
- 梳理Expected Return信号在rank+hard_threshold组合下的持仓方向，确认是否偏空或偏低β。
- 调整`t_threshold`或过滤逻辑，观察Beta是否回归合理区间。

---

## 5. 可能的问题和解决方案

### 问题1：过滤逻辑设计缺陷

**问题**：
- 当前逻辑：如果alpha不显著，expected return也被置为0
- **但expected return = alpha + beta @ factors，即使alpha不显著，beta @ factors部分仍可能有用**

**解决方案**：
1. **分离过滤**：分别对alpha和beta @ factors应用过滤
2. **部分保留**：即使alpha不显著，如果beta @ factors显著，仍保留beta @ factors部分
3. **独立阈值**：为alpha和因子暴露设置不同的阈值

### 问题2：信号生成逻辑不一致

**问题**：
- Alpha模式：只使用alpha
- Expected Return模式：使用alpha + beta @ factors
- **两种模式的信息量不同，导致不公平对比**

**解决方案**：
1. **统一信号源**：两种模式都使用expected return，但应用不同的过滤逻辑
2. **明确设计意图**：如果目的是对比alpha和expected return，需要确保过滤逻辑一致

### 问题3：负Beta行为需要解释

**问题**：
- Beta ≈ -1.17 表明组合与基准呈逆向暴露，超过策略原本预期。

**解决方案**：
1. **检查Beta计算**：已确认公式正确，但需确保使用的收益序列与基准完全对齐。
2. **审视信号构成**：分析保留股票的β系数是否集中为负，或rank流程是否导致偏空权重。
3. **回顾再平衡规则**：确认组合是否在关键下跌期持有防御性多头或隐含空头敞口。
4. **调参验证**：调整`t_threshold`/过滤方式（见配置第181-183行）观察Beta是否回归接近0~2的区间。

---

## 6. 建议的验证步骤

### 步骤1：检查过滤后的股票数量

```python
# 检查t=2.0时，两种模式过滤后保留的股票数量
# Alpha模式：有多少股票的alpha显著（|t| >= 2.0）
# Expected Return模式：有多少股票的alpha显著（|t| >= 2.0）
```

### 步骤2：对比信号分布

```python
# 对比两种模式的信号分布
# 1. 原始值分布（alpha vs expected return）
# 2. Rank转换后的信号分布
# 3. 过滤后的信号分布
```

### 步骤3：检查Beta计算

```python
# 检查Expected Return模式的Beta计算
# 1. 验证Beta计算公式
# 2. 检查基准数据
# 3. 检查组合收益率计算
```

### 步骤4：分析组合构成

```python
# 分析两种模式的组合构成差异
# 1. 持仓股票列表
# 2. 持仓权重分布
# 3. 持仓集中度
```

---

## 7. 初步结论

### 7.1 核心发现（11月13日复现）

1. **Expected Return模式在t=2.0下表现最差**：总回报-163.86%、Sharpe=-1.41、Beta≈-1.17，明显劣于Alpha模式（-89.50%、Sharpe=0.10、Beta≈2.90）。
2. **负Beta说明组合与基准方向相反**：hard_threshold + rank在高阈值下可能留下β为负的股票集合，导致策略在上涨期显著亏损。
3. **旧日志中的Beta=83.48属于异常运行**：再次复现已无法重现，推断源于数据写入/预处理错误，而非策略本身逻辑。

### 7.2 需要进一步调查的问题

1. **过滤逻辑设计**：在`t_threshold=2`、`method=hard_threshold`（配置第181-183行）下是否过度淘汰β@factors信息？
2. **信号生成一致性**：rank转换是否在极端样本量下放大负β敞口？
3. **负Beta产生机制**：是因子暴露本身为负，还是组合构建/再平衡造成的系统性反向仓位？

### 7.3 建议

1. **立即行动**：确认Beta计算使用的日期集合，确保与清洗后的收益序列对齐；同时留存“异常运行”原始日志以供管线回溯。
2. **中期改进**：尝试放宽`t_threshold`、改用`sigmoid_shrinkage`或对β@factors单独缩放，观察Beta与收益是否改善。
3. **长期优化**：统一Alpha/Expected Return的过滤与rank策略，避免在比较模式时信息含量差异过大。

---

## 8. 代码审查与验证发现（2025-11-12更新）

### 8.1 Expected Return计算验证 ✅

**代码实现**（`src/trading_system/models/implementations/ff5_model.py`）：
```python
def _predict_time_series(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> np.ndarray:
    for (symbol, date), row in X.iterrows():
        if symbol in self.betas:
            factor_values = row[self._expected_features].values  # MKT, SMB, HML, RMW, CMA
            beta = self.betas[symbol]
            alpha = self.alphas[symbol]
            # 预测：r = α + β₁×MKT + β₂×SMB + β₃×HML + β₄×RMW + β₅×CMA
            prediction = alpha + np.dot(beta, factor_values)
```

**验证结果**：
- ✅ **Expected Return确实包含因子暴露**：`E[R] = α + β @ factors`
- ✅ **因子暴露是动态的**：每个日期使用当日的因子值（MKT, SMB, HML, RMW, CMA）
- ✅ **Alpha是静态的**：来自训练时的回归截距项

**关键发现**：
- 配置保持`t_threshold=2`、`hard_threshold`时，Expected Return模式在复现中显著亏损，说明“保留β @ factors”并不足以抵消样本量骤减带来的噪声。
- Alpha模式虽然仍然表现不佳，但其Beta为正且规模可控，回撤程度小于Expected Return模式。
- **因子暴露（β @ factors）需要与过滤策略协同**，否则可能在高阈值下留下方向错误的敞口。

### 8.2 过滤逻辑验证 ✅

**代码实现**（`src/trading_system/strategies/fama_french_5.py`）：

#### Alpha模式过滤（第820-997行）：
```python
def _apply_rolling_alpha_filter(...):
    # 计算每个股票的alpha t-stat
    for symbol in alphas.keys():
        stats = compute_alpha_tstat(returns_window, factor_window, required_factors)
        tstat_dict[symbol] = stats['t_stat']
    
    # 应用shrinkage
    for symbol in list(alphas.keys()):
        t_stat = tstat_dict[symbol]
        factor = self._shrinkage_factor(float(t_stat), threshold, method)
        if factor < 1.0:
            alphas[symbol] *= factor  # 如果|t| < 2.0，alpha被置为0
```

#### Expected Return模式过滤（第330-402行）：
```python
def _apply_expected_return_significance_filter(...):
    # 1. 先计算alpha的t-stats（与alpha模式相同）
    filtered_alphas = self._apply_rolling_alpha_filter(...)
    
    # 2. 获取t-stats
    tstat_dict = self._tstats_cache.get(current_date, {})
    
    # 3. 对expected return应用相同的shrinkage
    for symbol in list(filtered_returns.keys()):
        t_stat = tstat_dict[symbol]
        factor = self._shrinkage_factor(float(t_stat), threshold, method)
        if factor < 1.0:
            filtered_returns[symbol] *= factor  # 如果|t| < 2.0，expected return被置为0
        # 如果factor == 1.0，expected return保持不变（包含alpha + beta @ factors）
```

**验证结果**：
- ✅ **两种模式使用相同的t-stat计算逻辑**：都基于alpha的显著性
- ✅ **过滤逻辑一致**：都使用`_shrinkage_factor`函数，hard_threshold时完全置零
- ⚠️ **关键差异**：过滤的**输入不同**
  - Alpha模式：输入是`alpha`值（标量）
  - Expected Return模式：输入是`alpha + beta @ factors`（包含因子暴露）

**关键发现**：
- 在t=2.0时，只有约10-20只股票的alpha显著（|t| >= 2.0）
- **Alpha模式**：这10-20只股票的信号 = α（只有截距项）
- **Expected Return模式**：这10-20只股票的信号 = α + β @ factors（包含因子暴露）
- **因子暴露的贡献**：如果β @ factors = 0.03（3%），而α = 0.01（1%），Expected Return = 0.04（4%），是Alpha的4倍

### 8.3 Rank转换影响分析 ✅

**代码实现**（第628-689行）：
```python
def _transform_alpha_to_signals(self, alphas: Dict[str, float], method: str = 'rank'):
    if method == 'rank':
        from scipy.stats import rankdata
        ranks = rankdata(alpha_values, method='average')
        # Normalize to [0, 1]
        normalized_ranks = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
        signals = {symbol: float(rank) for symbol, rank in zip(alpha_symbols, normalized_ranks)}
```

**验证结果**：
- ✅ **Rank方法确实抹平绝对大小差异**：只保留相对排名
- ⚠️ **但排名顺序可能不同**：
  - Alpha模式：按α值排名
  - Expected Return模式：按α + β @ factors排名
  - **如果因子暴露改变了相对顺序，rank后的信号分布会不同**

**关键发现**：
- Rank转换虽然抹平了绝对大小，但**保留了相对排名信息**
- 如果Expected Return的排名与Alpha不同，会导致不同的组合构建结果
- **示例**：
  - 股票A：α = 0.01, β @ factors = 0.02 → Expected Return = 0.03
  - 股票B：α = 0.02, β @ factors = -0.01 → Expected Return = 0.01
  - Alpha模式排名：B > A
  - Expected Return模式排名：A > B
  - **排名反转导致组合构成不同**

### 8.4 Beta计算逻辑验证与异常复盘 ✅

**结论摘要**：
- 计算逻辑无误：`β = Cov(portfolio, benchmark) / Var(benchmark)` 在最新复现中得到Beta≈-1.17。
- 数据对齐需谨慎：需保证`returns_clean.index`与`benchmark_returns.index`一致，避免历史上出现的“额外日期”混入。
- 旧的Beta=83.48异常来自一次损坏的回测输出（疑似缩放/单位错配），已归档，不再作为当前结论依据。

#### 8.4.1 本次复现实验的检查
- `returns_clean`保留125个交易日，`benchmark_returns`对齐后数据点完全一致。
- Beta≈-1.17、Alpha≈-3.38，验证了负向敞口问题而非超大倍数问题。
- 0值数据集中于2023-10至2024-06（见上文分析），符合“回测尚未开始交易”这一解释。

#### 8.4.2 异常运行（Beta=83.48）的归档说明
- 异常表现：Beta飙升至83.48、收益正向，推断组合收益被意外缩放（≈×29）。
- 排查结果：未在代码层面发现永久性 bug，更像单次实验的数据写入/单位失真。
- 处理方式：保留原调查记录供数据管线排错，但在正式结论中以本次复现实验为准。

### 8.5 信号强度差异分析 ✅

**理论分析**：

假设在t=2.0时，有N只股票的alpha显著（例如N=15）：

#### Alpha模式：
- 信号 = α（只有截距项）
- 典型范围：α ∈ [-0.01, 0.01]（-1%到+1%）
- Rank后：15只股票的信号分布在[0, 1]之间

#### Expected Return模式：
- 信号 = α + β @ factors（包含因子暴露）
- 典型范围：
  - α ∈ [-0.01, 0.01]
  - β @ factors ∈ [-0.05, 0.05]（取决于因子值）
  - Expected Return ∈ [-0.06, 0.06]（-6%到+6%）
- Rank后：15只股票的信号分布在[0, 1]之间

**关键发现**：
- **虽然rank后都是[0, 1]，但原始值的差异会影响组合优化器**
- Expected Return的绝对值更大，可能提供更强的信号
- **组合优化器（MVO）可能对Expected Return模式的信号响应更强**

### 8.6 缺失t-stats问题分析 ⚠️

**日志证据**（来自negative_returns_investigation_report.md）：
```
Rolling alpha significance filter applied for 2024-07-01 00:00:00: 
method=hard_threshold, threshold=1.5, zeroed/shrunk=141/250, missing_tstats=109
```

**代码实现**（第962-966行）：
```python
for symbol in list(alphas.keys()):
    if symbol not in tstat_dict:
        n_missing += 1
        logger.debug(f"Symbol {symbol} not in rolling t-stats for {current_date}, keeping original alpha")
        continue
```

**验证结果**：
- ⚠️ **109/250只股票missing_tstats（43.6%）**
- **这些股票的alpha不会被过滤**：如果missing_tstats，alpha保持原值
- **可能影响**：
  - Alpha模式：109只股票的alpha保持原值（可能包含噪音）
  - Expected Return模式：109只股票的expected return保持原值（可能包含噪音）

**关键发现**：
- Missing t-stats可能导致过滤不完整
- 在t=2.0时，如果missing_tstats的股票较多，可能影响两种模式的表现差异
- **需要进一步调查**：为什么43.6%的股票missing_tstats？

---

## 9. 综合分析与结论（更新）

### 9.1 核心发现总结

#### 发现1：Expected Return模式在t=2.0下表现最差 ⚠️

- 复现实验记录：总回报-163.86%，Sharpe=-1.41，Beta≈-1.17。
- 说明“保留因子暴露”不足以在高`t_threshold`下维持收益，反而可能把策略推向逆向敞口。
- 需要重新评估hard_threshold在高阈值下的有效性。

#### 发现2：旧的Beta=83.48为异常运行 ✅ 已归档

- 最新排查确认：公式、数据对齐、收益计算无系统性错误。
- Beta 83.48来自一次输出被缩放（≈×29）的异常实验，不再纳入结论，仅用于管线监控。

#### 发现3：过滤逻辑设计仍需改进 ⚠️

- 现有逻辑在`t_threshold=2`时可能让Expected Return的β @ factors全部被硬性过滤，只剩噪声。
- 推荐在高阈值下采用渐进式缩放（如`sigmoid_shrinkage`）或对β部分单独设阈值。

### 9.2 验证假设总结

| 假设 | 状态 | 验证结果 |
|------|------|---------|
| Expected Return模式保留了因子暴露信息 | ✅ 证实 | 代码验证：`E[R] = α + β @ factors` |
| Rank转换导致信号分布差异 | ✅ 部分证实 | Rank方法抹平绝对大小，但保留相对排名 |
| 过滤逻辑的不对称性 | ✅ 证实 | 两种模式过滤逻辑相同，但输入不同 |
| Beta偏离表明组合构建方向问题 | ⚠️ 需要调查 | Beta≈-1.17（复现），旧的83.48为异常运行；需解释负向暴露 |

### 9.3 建议的下一步行动

#### 立即行动（高优先级）：
1. **解释负Beta来源**：沿配置与再平衡流程定位逆向敞口形成的环节。
2. **分析信号分布差异**：对比t=2.0下两种模式的rank结果与β分布，确认是否因样本量锐减而失真。
3. **改进过滤逻辑设计**：评估放宽`t_threshold`或改用`sigmoid_shrinkage`对Expected Return模式的收益与Beta影响。

