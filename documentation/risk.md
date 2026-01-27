# Risk management (portfolio volatility, box allocation and meta model)

好的，让我系统地梳理整个流程，从策略到最终投资组合的完整过程。

---

## 完整流程图示

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 0: Universe Definition (投资域定义)                     │
│ - 基本面筛选：市值、流动性、数据质量                            │
│ - IC筛选：保留策略有预测能力的股票                             │
│ 输出：Investment Universe (300-800只股票)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Strategy Signal Generation (策略信号生成)            │
│                                                              │
│  Strategy A (e.g., FF5因子模型)                              │
│    输入：Universe内的价格+因子数据                             │
│    输出：Expected Returns_A (预期收益率)                      │
│                                                              │
│  Strategy B (e.g., 动量策略)                                 │
│    输入：Universe内的价格数据                                 │
│    输出：Expected Returns_B (预期收益率)                      │
│                                                              │
│  Strategy C (e.g., ML策略)                                   │
│    输入：Universe内的价格+特征数据                             │
│    输出：Expected Returns_C (预期收益率)                      │
│                                                              │
│ 关键：每个策略只输出预期收益，不做风险调整                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Meta-Model (策略组合)                               │
│                                                              │
│ 训练阶段：                                                    │
│   X = [历史Expected Returns_A, _B, _C]                      │
│   y = 真实收益                                               │
│   Lasso回归学习权重：w_A, w_B, w_C                           │
│                                                              │
│ 预测阶段：                                                    │
│   Combined Signal = w_A × Returns_A + w_B × Returns_B       │
│                     + w_C × Returns_C                        │
│                                                              │
│ 输出：Combined Expected Returns (所有Universe内的股票)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Dimensional Reduction (可选降维)                     │
│                                                              │
│ 目的：降低后续协方差估计和优化的计算复杂度                      │
│                                                              │
│ 方法1：保留Top N (如N=200)                                   │
│   按 |Combined Signal| 排序，取前200只                       │
│                                                              │
│ 方法2：宽松threshold                                         │
│   保留 Combined Signal > 某个低阈值的股票                     │
│                                                              │
│ 注意：这不是"去噪声"，只是为了让后续计算可行                    │
│                                                              │
│ 输出：Reduced Universe (100-200只股票) + 对应的signals       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Risk Model (风险估计)                                │
│                                                              │
│ 输入：Reduced Universe的历史价格数据                          │
│                                                              │
│ 协方差矩阵估计：                                              │
│   方法A：Ledoit-Wolf收缩估计                                 │
│   方法B：DCC-GARCH (时变协方差)                              │
│   方法C：因子模型 (Σ = BFB^T + D)                            │
│                                                              │
│ 输出：Covariance Matrix Σ (N×N，N为Reduced Universe大小)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Stock Classification (Box分类)                      │
│                                                              │
│ 对Reduced Universe中的每只股票分类：                          │
│   - Sector: Tech, Finance, Healthcare, ...                  │
│   - Region: US, Europe, Asia, ...                           │
│   - Size: Large-cap, Mid-cap, Small-cap                     │
│   - Style: Growth, Value, ...                               │
│                                                              │
│ 输出：每只股票的Box归属                                       │
│   例如：AAPL → {Sector: Tech, Region: US, Size: Large}      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Portfolio Optimization with Box Constraints          │
│                                                              │
│ 优化问题：                                                    │
│                                                              │
│   maximize: Sharpe Ratio = (w^T μ) / sqrt(w^T Σ w)         │
│                                                              │
│   subject to:                                                │
│     1. 全仓约束：Σ w_i = 1                                   │
│     2. 非负约束：w_i ≥ 0                                     │
│     3. 个股上限：w_i ≤ 15%                                   │
│     4. Box上限（关键！）：                                    │
│        Σ_{i in Tech} w_i ≤ 30%                              │
│        Σ_{i in Finance} w_i ≤ 25%                           │
│        Σ_{i in US} w_i ≤ 70%                                │
│        ...                                                   │
│     5. (可选) Turnover约束：|w_new - w_old|_1 ≤ 20%         │
│                                                              │
│ 其中：                                                        │
│   μ = Combined Expected Returns (来自Stage 2/3)             │
│   Σ = Covariance Matrix (来自Stage 4)                       │
│                                                              │
│ 求解方法：SLSQP / Interior Point / CLA                        │
│                                                              │
│ 输出：Final Portfolio Weights w* (N维向量)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: Compliance Check (合规检查)                          │
│                                                              │
│ 验证w*是否满足所有约束：                                       │
│   - Box暴露检查                                              │
│   - 风险指标检查 (VaR, CVaR, Max Drawdown)                   │
│   - 流动性检查                                               │
│                                                              │
│ 如果违规 → 返回Stage 6，调整约束重新优化                       │
│ 如果合规 → 输出最终权重                                       │
│                                                              │
│ 输出：Validated Portfolio Weights                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 各阶段详细说明

### Stage 0: Universe Definition

**职责：**
- 定义"可投资"的股票范围
- 这是整个系统的基础

**当前方案：**
```
手动定义：从配置文件中手动输入
```

**改进方案：**
```
1. 基础筛选：
   - 市值 > $1B
   - 日均成交量 > $10M
   - 价格 > $5
   - 至少2年历史数据
   
2. IC筛选（可选）：
   对每只股票计算rolling IC
   保留IC > 0.02的股票
   
结果：从全市场筛选到300-800只
```

---

### Stage 1: Strategy Signal Generation

**职责：**
- 各策略独立预测
- 输出预期收益率，不做风险调整

**关键设计：**
```python
# BaseStrategy的generate_signals_single_date()应该返回：
{
    'expected_returns': pd.DataFrame,  # 预期收益率
    'confidence': pd.DataFrame,        # 可选：预测置信度
    'metadata': {...}
}

# 不应该返回：
{
    'weights': ...  # 这是错的！策略不应该直接给权重
}
```

**当前问题：**
- BaseStrategy在`apply_risk_adjustment()`中使用了协方差
- 这应该去除，策略层不做风险调整

**改进方案：**
- 策略只做预测
- 移除`apply_risk_adjustment()`和`_apply_constraints()`
- 这些留给后续Stage 6处理

---

### Stage 2: Meta-Model

**职责：**
- 学习各策略的可靠性
- 组合策略信号

**数学形式：**
```
训练：
  min ||y - Xw||² + λ||w||₁
  
  X = [Strategy_A_predictions, Strategy_B_predictions, ...]
  y = realized_returns
  w = 策略权重

预测：
  combined_signal = w_A × signal_A + w_B × signal_B + ...
```

**当前状态：**
- 完成，但是没有被整合进当前系统中

**实现要点：**
```python
# 伪代码
class MetaModel:
    def fit(self, strategy_predictions_history, realized_returns):
        # 用Lasso学习权重
        pass
    
    def combine(self, strategy_signals_current):
        # 加权组合当前信号
        return combined_signal
```

---

### Stage 3: Dimensional Reduction

**职责：**
- 降低计算复杂度
- 不是"去噪声"

**Rationale：**
```
协方差矩阵参数数量 = N(N-1)/2

500只股票 → 124,750个参数
200只股票 → 19,900个参数
100只股票 → 4,950个参数

如果历史数据只有250天，500只股票的协方差矩阵极不稳定
```

**方法：**
```
选项1：Top N （推荐，采用）
  按|Combined Signal|排序，取前N只

选项2：Soft threshold
  Combined Signal > 某个宽松阈值（如0.5%）

选项3：跳过此阶段
  如果N本来就不大（<200），且计算资源充足
```

---

### Stage 4: Risk Model

**职责：**
- 估计协方差矩阵
- 这是整个风险管理的核心

**当前实现：**
```python
# 已有三个estimator
SimpleCovarianceEstimator
LedoitWolfCovarianceEstimator
FactorModelCovarianceEstimator
```

**选择建议：**
```
股票数量 < 100：Simple或Ledoit-Wolf
股票数量 100-300：Ledoit-Wolf （采用）
股票数量 > 300：Factor Model

如果有高质量因子数据：优先Factor Model
```

---

### Stage 5: Stock Classification

**职责：**
- 为Box约束做准备
- 将股票分类到各个维度

**当前实现：**
```python
StockClassifier.classify_stocks()
```

**分类维度：**
```
Sector: Tech, Finance, Healthcare, Consumer, Industrial, ...
Region: US, Europe, Asia, Emerging Markets, ...
Size: Large (>$10B), Mid ($2-10B), Small (<$2B)
Style: Growth, Value, Blend
```

**输出示例：**
```python
{
    'AAPL': {
        'sector': 'Tech',
        'region': 'US',
        'size': 'Large',
        'style': 'Growth'
    },
    'JPM': {
        'sector': 'Finance',
        'region': 'US',
        'size': 'Large',
        'style': 'Value'
    }
}
```

---

### Stage 6: Portfolio Optimization

**职责：**
- 在约束下找到最优权重
- 这里整合了Box约束和协方差优化

**优化问题的完整形式：**

```
输入：
  μ: Combined Expected Returns (N×1)
  Σ: Covariance Matrix (N×N)
  Box分类: 每只股票属于哪些box

目标函数：
  max  (w^T μ) / sqrt(w^T Σ w)  
  
  或等价地（二次规划形式）：
  max  w^T μ - (λ/2) w^T Σ w

约束：
  1. Σw = 1 (全仓)
  2. w ≥ 0 (只做多)
  3. w_i ≤ 0.15 (单股上限)
  4. Σ_{i in Tech} w_i ≤ 0.30 (Tech sector上限)
  5. Σ_{i in Finance} w_i ≤ 0.25
  6. ... (其他box约束)
```

**Box约束的实现：**
```python
# 伪代码
def build_box_constraints(stocks, classifications, box_limits):
    constraints = []
    
    for box_dimension in ['sector', 'region', 'size']:
        for box_value in unique_values(box_dimension):
            # 找到属于这个box的所有股票
            stocks_in_box = [s for s in stocks 
                           if classifications[s][box_dimension] == box_value]
            
            # 创建约束矩阵
            mask = [1 if s in stocks_in_box else 0 for s in stocks]
            
            # 添加约束: mask @ w <= limit
            limit = box_limits[box_dimension][box_value]
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: limit - mask @ w
            })
    
    return constraints
```

**关键：Box约束是不等式约束（上限），不是等式约束**

---

### Stage 7: Compliance Check

**职责：**
- 最后验证
- 不修改权重，只检查

**检查项：**
```
1. Box暴露是否在限制内
2. VaR-95 < 5%
3. Expected Shortfall < 8%
4. Beta绝对值 < 1.5
5. 最大回撤预期 < 20%
6. 单股流动性占比 < ADV的10%
```

**如果违规：**
```
返回Stage 6，收紧约束重新优化
不是在这一层修正权重
```

---

## 当前方案 vs 改进方案对比

### 当前方案的主要问题：

1. **Strategy层做了风险调整**
   ```
   BaseStrategy.apply_risk_adjustment() 使用协方差
   → 应该删除，策略只做预测
   ```

2. **缺少Meta-Model集成**
   ```
   多策略直接送给Coordinator
   → 应该先用Meta-Model组合
   ```

3. **信号筛选的意义不清**
   ```
   文档说是"去噪声"
   → 应该改为"降维"，或干脆去掉
   ```

4. **Box Allocator的定位模糊**
   ```
   BoxAllocator做equal weight分配
   → 应该改为约束条件，不是独立的allocator
   ```

### 改进方案的核心改动：

| 组件 | 当前 | 改进 |
|------|------|------|
| **BaseStrategy** | 输出风险调整后的weights | 只输出expected returns |
| **Meta-Model** | 不存在 | 新增Lasso组合层 |
| **信号筛选** | "去噪声" | 改为"降维"或删除 |
| **Box处理** | BoxAllocator独立分配 | 改为优化约束 |
| **协方差优化** | Strategy层和Portfolio层都做 | 只在Portfolio层做一次 |

---

## 伪代码示意

```python
# 完整流程
def construct_portfolio(date, universe):
    # Stage 1: 各策略生成信号
    signals_A = strategy_A.generate_expected_returns(date, universe)
    signals_B = strategy_B.generate_expected_returns(date, universe)
    signals_C = strategy_C.generate_expected_returns(date, universe)
    
    # Stage 2: Meta-Model组合
    combined_signal = meta_model.combine({
        'A': signals_A,
        'B': signals_B,
        'C': signals_C
    })
    
    # Stage 3: (可选) 降维
    if len(combined_signal) > 200:
        top_stocks = combined_signal.abs().nlargest(200).index
        combined_signal = combined_signal[top_stocks]
    
    # Stage 4: 估计协方差
    cov_matrix = risk_estimator.estimate(price_data, date)
    
    # Stage 5: 股票分类
    classifications = stock_classifier.classify(combined_signal.index)
    
    # Stage 6: 优化
    box_constraints = build_box_constraints(
        stocks=combined_signal.index,
        classifications=classifications,
        box_limits={'Tech': 0.30, 'Finance': 0.25, ...}
    )
    
    weights = optimize_portfolio(
        expected_returns=combined_signal,
        cov_matrix=cov_matrix,
        box_constraints=box_constraints
    )
    
    # Stage 7: 合规检查
    if not compliance_check(weights):
        # 重新优化with更严格约束
        weights = optimize_portfolio(..., tighter_constraints)
    
    return weights
```

这个流程清楚地分离了各层职责，避免了重复的风险调整，整合了Meta-Model和Box约束。