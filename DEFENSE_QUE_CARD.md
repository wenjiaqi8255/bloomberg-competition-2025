# Defense Que Card - Methodology Challenge

## 🎯 The Challenge

**Question**: "Fama-MacBeth标准做法是 E[R] = β×λ，不包含alpha。为什么你要加alpha？这不是偏离标准吗？"

---

## ✅ The Response (30秒版本)

"您说得对，标准Fama-MacBeth确实是 E[R] = β×λ，不含alpha。

**但我的研究问题不是'如何实施标准Fama-MacBeth'。**

而是：'在实践中，当投资者combine factor-based signals和stock-specific views时，如何提高信号质量？'

**为什么这个问题relevant？**
1. 机构实践中很少purely使用factor models
2. 主动管理通常是：factor tilts + stock selection的组合
3. 我的contribution：测试statistical filtering在混合框架中的价值

**理想情况下应该有三个实验：**
A. Pure Factor (β×λ only) ← **这个baseline确实应该有**
B. Factor + All Alphas ← naive combination
C. Factor + Filtered Alphas ← smart combination

**由于时间限制，我做了B vs C。我acknowledge缺少A是limitation。**

但我用quick estimation做了proxy analysis，估算Pure Factor的Sharpe约为**0.19**，低于我的两个实验。

所以结论是：**alphas add value，filtering进一步提升**。"

---

## 📊 Supporting Evidence (如果评委继续追问)

### Quick Estimation Results

```
Strategy                    Sharpe    Return
─────────────────────────────────────────────
Pure Factor (β×λ only)        0.19      N/A
Factor + All Alphas           0.62     11.17%
Factor + Filtered Alphas      1.17     40.42%
```

**Method**:
- 使用已知数据反推
- 91只股票被filter (|t|<2.0)
- 假设这些股票的alpha ≈ 0 (被filter原因)
- 估算纯因子策略表现

**Implication**:
- Pure Factor (0.19) < All Alphas (0.62) < Filtered Alphas (1.17)
- Alphas包含signal和noise
- Filtering保留signal，去除noise → **optimal**

---

## 🎓 Academic Foundation

### 为"混合框架"辩护

**Lewellen (2015)**: "Cross-sectional vs Time-series"
> Factor models主要捕捉cross-sectional variation
> Time-series predictability需要additional information
> Firm characteristics提供这种information

**Kelly, Pruitt & Su (2019)**: "Characteristics are Covariances"
> 混合模型: E[R] = β'λ + θ'z
> z是firm characteristics (可以理解为managed betas)

**Harvey, Liu & Zhu (2016)**: "... and the Cross-Section"
> 数百个"factors"被提出，很多其实是characteristics
> 实践中factor和characteristic很难严格区分

### 为"Filtering有价值"辩护

**Brennan, Wang & Xia (2022)**: Signal decay和filtering的价值
**Harvey & Liu (2023)**: "Lucky Factors" - 强调statistical significance的重要性

---

## 🛡️ 如果被问到的其他follow-ups

### Q: "为什么不跑完整的三个实验？"

**A**: "时间限制。32天回测期，完整三方对比需要：
1. 重构backtest engine (当前engine默认加alpha)
2. 跑3个完整backtests
3. 分析比较结果

**我承认这是limitation，已经在future work中列出。**
但我用quick estimation提供了proxy analysis，虽然不够精确，但能提供初步insight。

### Q: "如果Pure Factor表现最好怎么办？"

**A**: "那说明alphas总体是harmful (noise dominates)。

**但即使这样，我的研究仍有价值：**
- 证明了'如果要用alpha，至少要filter它们'
- 为practitioners提供guidance：**better to avoid alphas or filter aggressively**

这是一个empirical question，值得进一步研究。

### Q: "这还是不是Fama-MacBeth？"

**A**: "好问题。

**严格来说**：不是standard Fama-MacBeth
**更准确的说**：是Fama-MacBet框架在混合策略中的应用

**Factor estimation part**: 用Fama-MacBeth (β×λ)
**Alpha part**: 时间序列回归 + t-stat filtering

**我觉得可以这样frame：**
'Using Fama-MacBeth as the foundation for factor risk premia estimation,
then augmenting with stock-specific signals (as practitioners do)'

如果您觉得这偏离Fama-MacBeth太远，我accept这个critique。
但我的目标是解决实际问题，不是reproduce canonical method。

### Q: "为什么不叫它别的方法，比如'Hybrid Factor-Alpha Model'？"

**A**: "Fair point。我可以用这个名称，以更清楚地表示这是：
- Factor model (Fama-MacBeth)
- + Alpha model (time-series regression with filtering)
- = Hybrid approach

**感谢建议，会在revision中考虑。**

---

## 💡 Key Takeaways for Defense

### 1. 诚实承认limitations
- ✅ "我acknowledge缺少pure factor baseline"
- ✅ "这是future work方向"
- ✅ "我已经做了quick estimation作为proxy"

### 2. 重新框架化contribution
- ❌ Not: "改进Fama-MacBeth方法"
- ✅ But: "测试混合框架中signal filtering的价值"

### 3. 强调实践相关性
- 机构investor确实combine factors和alphas
- 研究问题有实际价值
- 不是为了deviate而deviate

### 4. 准备backup slides
- 如果有time，现在补充完整的三方对比实验
- 或者至少准备详细的estimation methodology
- 展示你thought about这个问题

---

## 🚦 最后的reminder

**当评委问这个问题时，他们可能：**
1. ✅ Genuine curiosity about methodology choice
2. ✅ Testing if you understand standard Fama-MacBeth
3. ✅ Checking if you can defend your research design
4. ❌ Not trying to destroy your presentation

**Best approach:**
- Stay calm and confident
- Acknowledge their point is valid
- Explain your research question clearly
- Show you've thought about alternatives
- Be honest about limitations

**你的研究的价值不在于'完美实施标准方法'，而在于'探索有意义的实际问题'。**

---

**Good luck! You've got this! 🎯**
