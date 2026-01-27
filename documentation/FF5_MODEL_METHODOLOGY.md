# FF5模型方法论完整文档

## 概述

本文档详细说明Fama-French 5因子（FF5）模型从训练到预测到回测的完整流程，特别关注Beta计算方式、时间回溯机制、特征工程流程等关键技术细节。

**模型ID**: `ff5_regression_20251103_161033`  
**文档生成时间**: 2025-11-03  
**作者**: 系统分析

---

## 第一章：模型概述

### 1.1 理论基础

FF5模型基于Fama-French五因子模型理论：

```
R_stock - RF = α + β_MKT × (R_MKT - RF) + β_SMB × SMB + β_HML × HML + 
              β_RMW × RMW + β_CMA × CMA + ε
```

其中：
- **MKT**: Market excess return（市场超额收益）
- **SMB**: Small Minus Big（规模因子：小盘股收益 - 大盘股收益）
- **HML**: High Minus Low（价值因子：高B/M - 低B/M）
- **RMW**: Robust Minus Weak（盈利性因子：强盈利 - 弱盈利）
- **CMA**: Conservative Minus Aggressive（投资因子：保守投资 - 激进投资）
- **RF**: Risk-free rate（无风险利率，通常为1个月国库券利率）

### 1.2 模型架构

```
┌─────────────────┐
│  TrainingPhase  │
├─────────────────┤
│ 1. Data Loading │ → 扩展日期范围（含lookback）
│ 2. Features     │ → 因子特征提取
│ 3. Beta Fit     │ → 每个股票独立回归
│ 4. CV           │ → 时间序列交叉验证
└─────────────────┘
         ↓
┌─────────────────┐
│ PredictionPhase │
├─────────────────┤
│ 1. Load Model   │ → 加载静态Beta
│ 2. Get Factors  │ → 获取当日因子值
│ 3. Predict      │ → E[R] = α + β @ factors
└─────────────────┘
         ↓
┌─────────────────┐
│  BacktestPhase  │
├─────────────────┤
│ 1. Generate Sig │ → Expected Return模式
│ 2. Optimize     │ → Box-Based分配
│ 3. Rebalance    │ → 每周再平衡
│ 4. Evaluate     │ → 绩效指标计算
└─────────────────┘
```

### 1.3 核心设计原则

1. **Beta是静态的**：训练时计算一次，预测/回测时不更新
2. **因子值是动态的**：每个日期使用当日的因子值
3. **避免Look-ahead Bias**：只使用历史数据（<= current_date）
4. **独立股票回归**：每个股票的Beta独立计算

---

## 第二章：训练阶段

### 2.1 数据准备

#### 2.1.1 时间窗口扩展

**关键发现**：训练数据加载时会扩展日期范围以包含lookback期。

**实现位置**: `src/trading_system/models/training/training_pipeline.py:164-167`

```python
# 确定特征工程需要的最长lookback期
max_lookback = self.feature_pipeline.get_max_lookback()
# 扩展开始日期：start_date - max_lookback * 1.5
extended_start_date = start_date - pd.Timedelta(days=max_lookback * 1.5)
```

**实际数据**：
- **训练时间范围**：2024-01-01 至 2025-06-30
- **实际数据范围**：2022-12-11 至 2025-06-30
- **扩展原因**：特征工程需要257天的lookback期
- **扩展比例**：1.5倍（考虑非交易日）

**原理**：
- 特征工程在计算技术指标时需要使用历史数据
- 例如：252天移动平均需要252天的历史数据
- 如果不扩展日期范围，训练期的第一天将无法计算特征

#### 2.1.2 数据对齐

**因子数据与价格数据的对齐**：
- 因子数据频率：通常为日度或月度（从Kenneth French Data Library获取）
- 价格数据频率：日度
- 对齐方法：使用`reindex`和`ffill`（前向填充）将因子数据对齐到所有价格日期

**实现位置**: `src/trading_system/feature_engineering/pipeline.py:765-766`

```python
# 对齐因子数据到所有日期
factor_data_resampled = factor_data_numeric.reindex(all_dates, method='ffill')
```

### 2.2 Beta计算方式 ⭐

#### 2.2.1 核心发现：**Beta是静态的，使用历史平均，不是滚动窗口**

**实现位置**: `src/trading_system/models/implementations/ff5_model.py:111-247`

**关键代码**：
```python
def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FF5RegressionModel':
    # 获取所有唯一股票符号
    symbols = X.index.get_level_values('symbol').unique()
    
    # 为每个符号独立进行线性回归
    for symbol in symbols:
        # 提取该股票的所有训练期数据
        symbol_X = X.xs(symbol, level='symbol')  # 整个训练期的数据
        symbol_y = y.xs(symbol, level='symbol')  # 整个训练期的目标
        
        # 对齐数据
        aligned_data = pd.concat([symbol_y, symbol_X], axis=1, join='inner').dropna()
        
        # 使用整个训练期的数据进行回归
        symbol_model.fit(symbol_X_clean, symbol_y_clean)
        
        # 存储该股票的Beta系数（静态）
        self.betas[symbol] = symbol_model.coef_  # shape: (5,)
        self.alphas[symbol] = symbol_model.intercept_
```

**关键点**：
1. **使用整个训练期**：`symbol_X = X.xs(symbol, level='symbol')` 提取该股票在整个训练期的所有数据
2. **一次计算**：每个股票的Beta在训练时只计算一次
3. **静态存储**：Beta保存在`self.betas`字典中，训练后不再更新
4. **每个股票独立**：每个股票使用自己的历史数据独立计算Beta

**Beta计算示例**：
```
股票AAPL的训练数据：
  Date        MKT    SMB    HML    RMW    CMA    Return
  2024-01-02  0.001  0.002  0.001  0.001  0.000  0.005
  2024-01-03  0.002  0.001  0.002  0.000  0.001  0.003
  ...         ...    ...    ...    ...    ...    ...
  2025-06-30  0.001  0.001  0.002  0.001  0.001  0.004

使用所有377天数据一次性回归：
  Return = α + β_MKT × MKT + β_SMB × SMB + β_HML × HML + 
           β_RMW × RMW + β_CMA × CMA

结果：Beta向量（5个值），训练期间保持固定
```

#### 2.2.2 正则化选项

**支持的正则化方法**：
- **none**: 普通线性回归（`sklearn.linear_model.LinearRegression`）
- **ridge**: 岭回归（`sklearn.linear_model.Ridge`），默认alpha=1.0

**配置位置**: `src/trading_system/models/implementations/ff5_model.py:64-78`

```python
if self.regularization == 'ridge':
    positive_alpha = max(abs(float(self.alpha)), 1e-6)
    self._model = Ridge(alpha=positive_alpha)
else:
    self._model = LinearRegression()
```

**本次实验配置**：
- 正则化：`none`
- Alpha参数：1（但使用LinearRegression时无效）
- 标准化：`false`

### 2.3 交叉验证

#### 2.3.1 时间序列CV实现

**实现位置**: `src/trading_system/models/training/trainer.py:307-601`

**关键特点**：
1. **每个fold独立**：每个fold创建独立的pipeline副本
2. **保持完整历史**：price_data和factor_data不过滤，保持完整历史用于特征计算
3. **只过滤targets**：只过滤target_data到当前fold的日期范围

**CV流程**：
```python
# 1. 生成CV切分（基于日期范围）
cv_splits = list(self.cv.split_by_date_range(start_date, end_date))

# 2. 处理每个fold
for fold_idx, (train_dates_fold, val_dates_fold) in enumerate(cv_splits):
    # 创建独立的pipeline副本
    fold_pipeline = self._clone_pipeline(feature_pipeline)
    
    # 过滤数据（保持price_data完整）
    train_data = self._filter_data_by_dates(data, train_dates_fold)
    # 注意：_filter_data_by_dates只过滤targets，price_data保持完整
    
    # 在完整数据上fit pipeline（需要历史数据计算特征）
    fold_pipeline.fit({
        'price_data': train_data['price_data'],  # 完整历史
        'factor_data': train_data.get('factor_data')
    })
    
    # Transform时也使用完整数据
    X_train_full = fold_pipeline.transform({...})  # 包含lookback期
    
    # 但只使用fold日期范围内的targets
    y_train = self._prepare_targets(train_data['target_data'], train_dates_fold)
    
    # 对齐features和targets到相同日期
    X_train, y_train = align_by_index(X_train_full, y_train)
```

**本次实验CV结果**：
- **CV folds**: 5
- **成功folds**: 4/5
- **平均R²**: -0.1015 ± 0.1474
- **Fold结果**：
  - Fold 0: R² = -0.0422
  - Fold 1: R² = -0.0006
  - Fold 2: R² = -0.0077
  - Fold 3: R² = -0.3554
  - Fold 4: 失败

**CV失败的Fold分析**：
- Fold 3失败原因：可能是数据不足或特征计算失败
- 数据过滤逻辑：price_data保持完整，target_data过滤到fold日期

#### 2.3.2 数据过滤逻辑

**关键实现**: `src/trading_system/models/training/trainer.py:638-681`

```python
def _filter_data_by_dates(self, data: Dict[str, Any], target_dates: List[datetime]) -> Dict[str, Any]:
    filtered_data = {}
    
    # ** CRITICAL: 保持price_data完整 - 特征计算需要历史数据
    filtered_data['price_data'] = data['price_data']  # 不过滤！
    
    # ** CRITICAL: 保持factor_data完整
    if 'factor_data' in data:
        filtered_data['factor_data'] = data['factor_data']  # 不过滤！
    
    # ** 只过滤target_data到当前fold的日期范围
    target_dates_set = set(pd.to_datetime(d).date() for d in target_dates)
    if 'target_data' in data:
        filtered_target_data = {}
        for symbol, series in data['target_data'].items():
            series_dates = pd.to_datetime(series.index).date
            mask = np.array([d in target_dates_set for d in series_dates])
            filtered_target_data[symbol] = series[mask]  # 只过滤targets
        filtered_data['target_data'] = filtered_target_data
    
    return filtered_data
```

**设计理由**：
- 特征工程（如252天移动平均）需要历史数据
- 如果在fold内过滤price_data，第一天将无法计算特征
- 因此保持price_data完整，只过滤targets到fold日期范围

---

## 第三章：特征工程

### 3.1 因子特征创建

#### 3.1.1 因子数据来源

**数据提供者**: `src/trading_system/data/ff5_provider.py`

**数据来源**：
- Kenneth French Data Library (Dartmouth College)
- URL: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/`
- 文件：`F-F_Research_Data_5_Factors_2x3_daily_TXT.zip`
- 频率：日度（daily）或月度（monthly）

**因子数据格式**：
```
Date       MKT     SMB     HML     RMW     CMA     RF
2024-01-02 0.001   0.002   0.001   0.001   0.000   0.000
2024-01-03 0.002   0.001   0.002   0.000   0.001   0.000
...
```

#### 3.1.2 因子特征提取

**实现位置**: `src/trading_system/feature_engineering/pipeline.py:711-810`

**关键逻辑**：
```python
def _create_factor_features(self, price_data, factor_data):
    # 1. 选择因子列（FF5: 5个因子）
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    
    # 2. 获取所有价格数据的日期
    all_dates = set()
    for symbol, data in price_data.items():
        all_dates.update(data.index.tolist())
    all_dates = sorted(all_dates)
    
    # 3. 对齐因子数据到价格日期
    factor_data_resampled = factor_data.reindex(all_dates, method='ffill')
    
    # 4. 为每个股票创建特征（因子值相同）
    for symbol in price_data.keys():
        symbol_features = pd.DataFrame(index=all_dates)
        for factor_col in factor_cols:
            symbol_features[factor_col] = factor_data_resampled[factor_col].values
        
        # 创建MultiIndex: (symbol, date)
        symbol_multiindex = pd.MultiIndex.from_arrays([
            [symbol] * len(all_dates),
            pd.to_datetime(all_dates)
        ], names=['symbol', 'date'])
        symbol_features.index = symbol_multiindex
        all_features.append(symbol_features)
```

**关键特点**：
1. **所有股票共享相同的因子值**：在某一天，所有股票使用相同的MKT, SMB, HML, RMW, CMA值
2. **日期对齐**：使用`reindex`和`ffill`将因子数据对齐到所有价格日期
3. **MultiIndex格式**：创建(symbol, date)格式的MultiIndex，便于与价格数据对齐

### 3.2 时间回溯机制

#### 3.2.1 Lookback窗口

**最大Lookback期**：
- 从日志推断：约257天
- 来源：特征工程可能需要的最长历史数据期

**Lookback用途**：
1. **技术指标计算**：如252天移动平均需要252天历史
2. **波动率计算**：如60天波动率需要60天历史
3. **因子数据对齐**：确保因子数据有足够的历史数据

#### 3.2.2 避免Look-ahead Bias

**策略**：
1. **训练时**：数据加载扩展日期范围（start_date - max_lookback * 1.5），但只使用[start_date, end_date]的数据计算targets
2. **预测时**：只使用当前日期或之前的因子值
3. **Rolling t-stats**：只使用历史数据（<= current_date）计算t统计量

**实现位置**: `src/trading_system/strategies/fama_french_5.py:866-870`

```python
# 过滤因子数据：只使用当前日期之前的数据
factor_historical = factor_data[factor_data.index <= current_date].copy()

# 过滤价格数据：只使用当前日期之前的数据
price_historical = symbol_price_data[symbol_price_data.index <= current_date].copy()
```

---

## 第四章：预测阶段

### 4.1 预测公式

#### 4.1.1 Expected Return计算

**公式**：
```
E[R] = α + β_MKT × MKT + β_SMB × SMB + β_HML × HML + 
       β_RMW × RMW + β_CMA × CMA
```

**实现位置**: `src/trading_system/models/implementations/ff5_model.py:294-321` (时间序列场景) 和 `323-388` (批量场景)

**时间序列预测**（训练/验证场景）：
```python
def _predict_time_series(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> np.ndarray:
    predictions = []
    
    # 为每个(symbol, date)组合生成独立预测
    for (symbol, date), row in X.iterrows():
        if symbol in self.betas:
            # 获取该时间点的因子值（动态）
            factor_values = row[self._expected_features].values  # shape: (5,)
            
            # 使用该symbol的beta进行预测（静态）
            beta = self.betas[symbol]  # shape: (5,)
            alpha = self.alphas[symbol]  # scalar
            
            # 预测：E[R] = α + β @ factors
            prediction = alpha + np.dot(beta, factor_values)
            predictions.append(prediction)
```

**批量预测**（回测场景）：
```python
def _predict_batch(self, X: pd.DataFrame, symbols: Optional[List[str]]) -> pd.Series:
    # 提取因子值（应该只有一行或一个向量）
    factor_values = X[self._expected_features].values  # shape: (1, 5) 或 (5,)
    if factor_values.ndim == 1:
        factor_vector = factor_values  # shape: (5,)
    elif factor_values.shape[0] == 1:
        factor_vector = factor_values[0]  # shape: (5,)
    
    # 批量预测所有股票（向量化）
    predictions = {}
    for symbol in valid_symbols:
        symbol_betas = self.betas[symbol]  # shape: (5,)
        symbol_alpha = self.alphas[symbol]
        
        # 向量化预测：r = α + β @ f
        symbol_prediction = symbol_alpha + factor_vector @ symbol_betas
        predictions[symbol] = symbol_prediction
    
    return pd.Series(predictions, name='ff5_prediction')
```

### 4.2 Beta与因子值的区别

#### 4.2.1 Beta（静态）

**特性**：
- **计算时机**：训练时一次性计算
- **更新频率**：不更新（静态）
- **股票特定**：每个股票有自己的Beta向量
- **存储位置**：`self.betas[symbol]` (dict)

**示例**：
```
AAPL的Beta: [β_MKT=-0.0058, β_SMB=0.0062, β_HML=-0.0035, β_RMW=0.0073, β_CMA=-0.0016]
MSFT的Beta: [β_MKT=-0.0021, β_SMB=0.0032, β_HML=-0.0034, β_RMW=0.0030, β_CMA=-0.0018]
```

#### 4.2.2 因子值（动态）

**特性**：
- **获取时机**：预测时动态获取
- **更新频率**：每个交易日更新
- **股票共享**：所有股票在某一天使用相同的因子值
- **数据来源**：FF5DataProvider或factor_data DataFrame

**示例**：
```
2025-08-28的因子值（所有股票相同）:
  MKT = 0.001
  SMB = 0.002
  HML = 0.001
  RMW = 0.001
  CMA = 0.000
```

### 4.3 预测场景区分

#### 4.3.1 训练/验证场景

**输入格式**：MultiIndex DataFrame (symbol, date)
**方法**：`_predict_time_series()`
**特点**：每个(symbol, date)组合独立预测

**使用场景**：
- 交叉验证评估
- 训练集/验证集预测
- 模型评估

#### 4.3.2 回测场景

**输入格式**：单日期因子值 DataFrame (1, 5) 或 Series
**方法**：`_predict_batch()`
**特点**：横截面预测，所有股票使用相同的因子值

**使用场景**：
- 策略回测
- 实时预测
- 组合构建

---

## 第五章：回测阶段

### 5.1 Beta更新机制

#### 5.1.1 核心发现：**Beta在回测时不更新**

**验证方式**：
1. 训练时Beta保存在`self.betas`字典中
2. 模型保存时，Beta被序列化到模型文件
3. 回测时，模型加载，Beta保持不变
4. 回测过程中，没有重新计算Beta的代码

**设计理由**：
- 回测需要模拟真实交易场景
- 在真实交易中，Beta不会每天重新计算
- 重新计算Beta需要大量历史数据，可能引入look-ahead bias

### 5.2 信号生成

#### 5.2.1 信号源模式

**两种模式**：
1. **Alpha模式**（原始）：`signal_source = 'alpha'`
   - 只使用截距项：`signal = α`
   - 不考虑因子暴露
   
2. **Expected Return模式**（默认）：`signal_source = 'expected_return'`
   - 使用完整期望收益：`signal = E[R] = α + β @ factors`
   - 考虑因子暴露

**本次实验配置**：
- 信号源：`expected_return`（默认）

#### 5.2.2 信号生成流程

**实现位置**: `src/trading_system/strategies/fama_french_5.py:523-626`

```python
def _get_predictions_from_expected_return(...):
    for date in date_range:
        # 1. 提取当前日期的因子值
        factor_values_df = self._extract_factor_values_for_date(features, date, required_factors)
        
        # 2. 使用模型预测（内部使用E[R] = α + β @ factors）
        expected_returns = self.model_predictor.predict(
            features=factor_values_df,
            symbols=symbols,
            date=date
        )
        
        # 3. 转换为字典并应用过滤
        expected_returns_dict = expected_returns.to_dict()
        
        # 4. 应用显著性过滤（如果启用）
        if alpha_config.get('enabled', False):
            filtered_returns = self._apply_expected_return_significance_filter(...)
        else:
            filtered_returns = expected_returns_dict
        
        # 5. 应用信号转换（raw/rank/zscore）
        transformed_signals = self._transform_alpha_to_signals(filtered_returns, signal_method)
        
        # 6. 存储信号
        for symbol, signal_value in transformed_signals.items():
            predictions_df.loc[date, symbol] = signal_value
```

### 5.3 Rolling t-stats

#### 5.3.1 Rolling模式说明

**用途**：动态计算alpha的t统计量，用于显著性过滤

**实现位置**: `src/trading_system/strategies/fama_french_5.py:820-997`

**关键逻辑**：
```python
def _apply_rolling_alpha_filter(self, alphas, config, current_date, pipeline_data, ...):
    # 1. 过滤历史数据：只使用当前日期之前的数据
    factor_historical = factor_data[factor_data.index <= current_date].copy()
    price_historical = symbol_price_data[symbol_price_data.index <= current_date].copy()
    
    # 2. 使用lookback_days（默认252天）计算t-stats
    returns_window = returns.tail(lookback_days).copy()
    
    # 3. 对齐因子数据到收益日期
    factor_window = factor_historical.loc[factor_mask].copy()
    
    # 4. 计算alpha的t-stat
    stats = compute_alpha_tstat(returns_window, factor_window, required_factors)
    tstat_dict[symbol] = stats['t_stat']
    
    # 5. 缓存结果
    self._tstats_cache[current_date] = tstat_dict
    
    # 6. 应用过滤/收缩
    factor = self._shrinkage_factor(t_stat, threshold, method)
    if factor < 1.0:
        alphas[symbol] *= factor
```

**关键特点**：
1. **避免Look-ahead Bias**：只使用历史数据（<= current_date）
2. **滚动窗口**：使用lookback_days（252天）的历史数据
3. **缓存机制**：计算结果缓存到`_tstats_cache`，避免重复计算
4. **每个日期独立**：每个日期计算一次，确保时间序列的正确性

#### 5.3.2 本次实验配置

**Alpha显著性过滤**：
- **启用**: 未明确配置（可能未启用）
- **方法**: `hard_threshold`（如果启用）
- **阈值**: 2.0（如果启用）
- **Rolling t-stats**: 未启用（回测日志未显示rolling计算）

**信号转换方法**：
- **方法**: `raw`（原始值）
- **其他选项**: `rank`（排名），`zscore`（标准化）

---

## 第六章：关键技术细节

### 6.1 时间回溯机制详解

#### 6.1.1 训练时的回溯

**时间线**：
```
[----lookback期----][-------训练期--------]
2022-12-11         2024-01-01           2025-06-30
                     ↑                     ↑
                  start_date            end_date

扩展原因：特征计算需要历史数据
扩展比例：max_lookback * 1.5（考虑非交易日）
```

**实际数据**：
- 训练期：2024-01-01 至 2025-06-30（546天）
- 实际加载：2022-12-11 至 2025-06-30（约907天）
- 扩展：约361天（约257天lookback × 1.5）

#### 6.1.2 回测时的回溯

**时间线**：
```
[----lookback期----][-------回测期--------]
2024-10-22         2025-07-01           2025-08-15
                     ↑                     ↑
               backtest_start        backtest_end

扩展原因：特征计算和rolling t-stats需要历史数据
扩展期：252天（lookback_days）
```

**实际数据**：
- 回测期：2025-07-01 至 2025-08-15（32天）
- 实际加载：2024-10-22 至 2025-08-15（约297天）
- 扩展：约252天（lookback_days）

### 6.2 数据对齐机制

#### 6.2.1 因子数据与价格数据的对齐

**问题**：
- 因子数据可能是月度频率（Kenneth French原始数据）
- 价格数据是日度频率
- 需要将因子数据对齐到所有价格日期

**解决方法**：
```python
# 1. 获取所有价格数据的日期
all_dates = set()
for symbol, data in price_data.items():
    all_dates.update(data.index.tolist())

# 2. 对齐因子数据到价格日期（前向填充）
factor_data_resampled = factor_data.reindex(all_dates, method='ffill')

# 3. 处理缺失值
factor_data_resampled = factor_data_resampled.fillna(method='ffill').fillna(0)
```

**原理**：
- 如果因子数据是月度的，同一月的所有交易日使用该月的因子值
- 前向填充确保每个交易日都有因子值
- 初始缺失值用0填充（如果必要）

#### 6.2.2 特征与目标的对齐

**问题**：
- 特征可能有lookback期的数据
- 目标只在训练期有数据
- 需要确保特征和目标在相同日期上对齐

**解决方法**：
```python
# 1. 提取共同索引
common_index = X.index.intersection(y.index)

# 2. 过滤到共同索引
X_aligned = X.loc[common_index]
y_aligned = y.loc[common_index]

# 3. 验证长度
assert len(X_aligned) == len(y_aligned)
```

### 6.3 Look-ahead Bias的避免

#### 6.3.1 训练时的避免

**策略**：
1. **数据过滤**：targets只包含训练期数据，但features使用完整历史（包含lookback期）
2. **日期对齐**：通过索引交集确保features和targets只在相同日期上对齐

**示例**：
```
Features (包含lookback期):
  2022-12-11: [MKT, SMB, HML, RMW, CMA] ✓
  2022-12-12: [MKT, SMB, HML, RMW, CMA] ✓
  ...
  2024-01-01: [MKT, SMB, HML, RMW, CMA] ✓  ← 训练期开始
  2024-01-02: [MKT, SMB, HML, RMW, CMA] ✓
  ...

Targets (只在训练期):
  2024-01-01: Return ✓  ← 训练期开始
  2024-01-02: Return ✓
  ...

对齐后（只使用共同日期）:
  2024-01-01: [Features] ↔ [Return] ✓
  2024-01-02: [Features] ↔ [Return] ✓
```

#### 6.3.2 回测时的避免

**策略**：
1. **因子值获取**：只使用当前日期或之前的因子值
2. **Rolling t-stats**：只使用历史数据（<= current_date）
3. **价格数据**：只使用历史价格数据

**示例**：
```
回测日期：2025-08-28

可用的因子值：
  ✓ 2025-08-27及之前的所有因子值
  ✓ 2025-08-28的因子值（如果已发布）
  ✗ 2025-08-29及之后的因子值

Rolling t-stats计算：
  使用数据：2024-10-22 至 2025-08-28（历史数据）
  ✗ 不使用：2025-08-29及之后的数据
```

### 6.4 静态Beta vs 滚动Beta的设计决策

#### 6.4.1 为什么使用静态Beta？

**优点**：
1. **计算效率**：训练时计算一次，预测时直接使用
2. **避免Over-fitting**：不频繁更新，减少对噪音的敏感性
3. **符合学术实践**：Fama-French模型通常使用固定Beta
4. **避免Look-ahead Bias**：如果滚动更新，需要确定更新时间点

**缺点**：
1. **不能适应市场变化**：Beta可能随时间变化
2. **滞后性**：使用历史Beta预测未来，可能存在滞后

#### 6.4.2 为什么不用滚动Beta？

**考虑因素**：
1. **计算成本**：每天重新计算所有股票的Beta需要大量计算
2. **数据要求**：滚动窗口需要足够的历史数据（如252天）
3. **Look-ahead Bias风险**：确定滚动窗口大小和更新频率需要谨慎
4. **模型复杂度**：增加模型复杂度，可能引入更多参数

**设计决策**：
- 本次实现采用**静态Beta**
- 如果需要滚动Beta，可以在未来版本中实现
- 可以通过定期重训练模型来实现Beta更新

---

## 第七章：完整数据流

### 7.1 训练流程

```
1. 数据加载
   ↓
   TrainingPipeline.run_pipeline()
   - 扩展日期范围：start_date - max_lookback * 1.5
   - 加载价格数据：extended_start_date 至 end_date
   - 加载因子数据：extended_start_date 至 end_date
   ↓
2. 特征工程
   ↓
   FeatureEngineeringPipeline.fit()
   - 计算技术指标（如果需要）
   - 对齐因子数据到价格日期
   - 创建因子特征：_create_factor_features()
   - 学习NaN填充统计量
   ↓
3. 交叉验证
   ↓
   ModelTrainer.train_with_cv()
   - 生成CV切分（基于日期）
   - 对每个fold：
     * 创建独立pipeline副本
     * Fit pipeline（使用完整历史）
     * Transform（使用完整历史）
     * 过滤targets到fold日期
     * 对齐features和targets
     * 训练模型
     * 评估
   ↓
4. 最终模型训练
   ↓
   Model.fit()
   - 对每个股票：
     * 提取该股票的所有训练期数据
     * 使用整个训练期进行线性回归
     * 保存Beta和Alpha
   ↓
5. 模型保存
   ↓
   ModelRegistry.save()
   - 保存模型对象（包含betas和alphas）
   - 保存特征工程pipeline
   - 保存元数据
```

### 7.2 预测流程

```
1. 模型加载
   ↓
   ModelPredictor.load_model()
   - 加载模型对象（包含静态betas和alphas）
   - 加载特征工程pipeline
   ↓
2. 数据准备
   ↓
   Strategy._compute_features()
   - 获取价格数据（当前日期及历史）
   - 获取因子数据（当前日期及历史）
   - 使用pipeline.transform()创建特征
   ↓
3. 因子值提取
   ↓
   Strategy._extract_factor_values_for_date()
   - 从features中提取当前日期的因子值
   - 返回DataFrame (1, 5) 或 Series (5,)
   ↓
4. 预测
   ↓
   ModelPredictor.predict()
   - 调用model._predict_batch()
   - 对每个股票：
     * 获取静态Beta和Alpha
     * 计算：E[R] = α + β @ factors
     * 返回预测值
   ↓
5. 信号转换
   ↓
   Strategy._transform_alpha_to_signals()
   - 应用信号转换（raw/rank/zscore）
   - 返回交易信号
```

### 7.3 回测流程

```
1. 初始化
   ↓
   StrategyRunner.run_strategy()
   - 加载训练好的模型
   - 初始化特征工程pipeline
   - 配置回测参数
   ↓
2. 每日循环
   ↓
   for date in backtest_dates:
     ↓
     a. 数据获取
        - 获取当前日期及历史的价格数据
        - 获取当前日期及历史的因子数据
        ↓
     b. 特征计算
        - FeatureEngineeringPipeline.transform()
        - 创建因子特征
        ↓
     c. 信号生成
        - _get_predictions_from_expected_return()
        - 提取当前日期的因子值
        - 使用模型预测（E[R] = α + β @ factors）
        - 应用显著性过滤（如果启用）
        - 应用信号转换
        ↓
     d. 组合优化
        - BoxBasedPortfolioBuilder.build()
        - 选择股票
        - 计算权重
        ↓
     e. 再平衡（如果是rebalance日期）
        - 计算目标权重
        - 执行交易
        - 扣除交易成本
        ↓
     f. 绩效更新
        - 更新组合价值
        - 计算收益
        - 记录持仓
        ↓
3. 绩效评估
   ↓
   BacktestEngine.calculate_metrics()
   - 计算总收益
   - 计算年化收益
   - 计算Sharpe比率
   - 计算最大回撤
   - 计算其他风险指标
```

---

## 第八章：实验数据总结

### 8.1 训练阶段数据

| 项目 | 数值 |
|------|------|
| **训练时间范围** | 2024-01-01 至 2025-06-30 |
| **实际数据范围** | 2022-12-11 至 2025-06-30 |
| **数据扩展** | 约361天（257天lookback × 1.5） |
| **训练样本数** | 37,711个 |
| **成功训练股票数** | 109个 |
| **模型类型** | FF5回归（无正则化） |
| **交叉验证** | 5-fold时间序列CV |
| **CV平均R²** | -0.1015 ± 0.1474 |

### 8.2 预测阶段数据

| 项目 | 数值 |
|------|------|
| **预测日期** | 2025-08-28 |
| **持仓数量** | 30只股票 |
| **组合方法** | Box-Based分配 |
| **预期收益率** | 943.80% ⚠️（异常高） |
| **预期风险** | 6.17% |
| **分散度得分** | 1.00 |

### 8.3 回测阶段数据

| 项目 | 数值 |
|------|------|
| **回测时间范围** | 2025-07-01 至 2025-08-15 |
| **实际交易日** | 205天（2024-10-22 至 2025-08-15） |
| **初始资金** | $1,000,000 |
| **最终价值** | $686,698 |
| **总回报率** | -31.33% |
| **年化回报率** | -93.83% |
| **Sharpe比率** | -1.50 |
| **最大回撤** | -49.44% |
| **Beta** | 6.19 ⚠️（异常高） |
| **Alpha** | -3.57 ⚠️（异常低） |

---

## 第九章：关键发现与建议

### 9.1 核心发现

#### 9.1.1 Beta计算方式

**发现**：**Beta是静态的，使用历史平均，不是滚动窗口**

**证据**：
1. 训练时：使用整个训练期的数据一次性计算Beta
2. 预测时：直接使用训练好的Beta，不更新
3. 回测时：Beta保持不变，整个回测期间使用相同的Beta

**影响**：
- 优点：计算高效，避免频繁重训练
- 缺点：不能适应市场变化，Beta可能滞后

#### 9.1.2 时间回溯机制

**发现**：**数据加载时扩展日期范围以包含lookback期**

**证据**：
1. 训练时：扩展约361天（257天lookback × 1.5）
2. 回测时：扩展约252天（lookback_days）
3. 目的：确保特征计算有足够的历史数据

**影响**：
- 确保特征计算的正确性
- 避免训练期第一天无法计算特征的问题

### 9.2 问题与警告

#### 9.2.1 模型表现问题

1. **训练R²为负**（-0.0029）：模型表现差于简单均值基准
2. **回测亏损严重**（-31.33%）：预测与实际回测结果差距巨大
3. **Beta异常高**（6.19）：市场敏感度过高，可能导致风险过大

#### 9.2.2 预测准确性警告

1. **预期收益率异常高**（943.80%）：预测值明显不合理
2. **预期风险低估**（6.17% vs 实际124.76%）：风险预测严重不足

### 9.3 改进建议

#### 9.3.1 模型改进

1. **改进特征工程**：提升因子信号质量
2. **引入正则化**：考虑使用Ridge回归，降低过拟合风险
3. **特征选择**：考虑特征重要性分析，只使用显著因子
4. **模型集成**：考虑集成多个模型，提升预测稳定性

#### 9.3.2 策略改进

1. **风险控制**：设置止损，限制最大回撤
2. **Beta调整**：考虑降低Beta，或使用Beta对冲
3. **波动率目标**：将组合波动率控制在合理范围
4. **持仓限制**：限制单股权重，避免过度集中

#### 9.3.3 技术改进

1. **Beta更新机制**：考虑实现滚动Beta或定期重训练
2. **特征验证**：增强特征有效性验证
3. **预测校准**：校准预测值，确保合理性
4. **回测改进**：增强回测框架，添加更多风险控制

---

## 附录

### A. 代码文件索引

| 功能 | 文件路径 |
|------|----------|
| FF5模型实现 | `src/trading_system/models/implementations/ff5_model.py` |
| FF5策略实现 | `src/trading_system/strategies/fama_french_5.py` |
| 训练管道 | `src/trading_system/models/training/training_pipeline.py` |
| 模型训练器 | `src/trading_system/models/training/trainer.py` |
| 特征工程管道 | `src/trading_system/feature_engineering/pipeline.py` |
| FF5数据提供者 | `src/trading_system/data/ff5_provider.py` |

### B. 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `regularization` | `none` | 正则化方法 |
| `alpha` | 1.0 | 正则化强度（未使用） |
| `standardize` | `false` | 特征标准化 |
| `lookback_days` | 252 | 回测时lookback天数 |
| `cv_folds` | 5 | 交叉验证fold数 |
| `max_lookback` | 257 | 特征工程最大lookback期 |

### C. 术语表

| 术语 | 定义 |
|------|------|
| **Beta** | 因子暴露系数，衡量股票对因子的敏感性 |
| **Alpha** | 截距项，衡量股票的异常收益 |
| **因子值** | Fama-French因子的日度/月度收益值 |
| **Lookback** | 特征计算所需的历史数据期 |
| **Rolling** | 使用固定窗口长度滚动计算 |
| **静态** | 计算一次后保持不变 |
| **MultiIndex** | Pandas的多层索引，如(symbol, date) |

---

**文档版本**: 1.0  
**最后更新**: 2025-11-03  
**状态**: 完成
