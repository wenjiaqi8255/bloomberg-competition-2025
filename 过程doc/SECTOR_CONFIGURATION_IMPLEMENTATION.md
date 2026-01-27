# Sector Configuration Implementation

## 概述

本次实现让sector维度变成可配置的，支持在配置文件中设置 `sector: []` 来创建3维box（忽略sector分类），或者指定具体的sector列表来创建4维box。

## 修改内容

### 1. BoxKey类修改 (`src/trading_system/portfolio_construction/models/types.py`)

- **sector字段改为可选**: `sector: Optional[str] = None`
- **字符串表示方法更新**: 当sector为None时，只显示3维 `size_style_region`
- **from_tuple和from_string方法**: 支持3维和4维的tuple/string解析

```python
# 3维BoxKey示例
box_3d = BoxKey(size='large', style='growth', region='developed', sector=None)
# 字符串表示: "large_growth_developed"

# 4维BoxKey示例  
box_4d = BoxKey(size='large', style='growth', region='developed', sector='Technology')
# 字符串表示: "large_growth_developed_Technology"
```

### 2. BoxWeightManager修改 (`src/trading_system/portfolio_construction/box_based/box_weight_manager.py`)

#### `_generate_all_boxes` 方法
- **空sector处理**: 当sector列表为空时，生成3维box组合
- **正常sector处理**: 当sector列表有值时，生成4维box组合

#### `ConfigurableBoxWeightProvider._parse_config` 方法
- **支持3维配置**: box定义可以是3个元素（忽略sector）
- **支持4维配置**: box定义可以是4个元素（包含sector）

## 使用方法

### 1. 空sector配置（3维box）

```yaml
box_weights:
  method: "equal"
  dimensions:
    size: ["large", "mid", "small"]
    style: ["growth", "value"]
    region: ["developed", "emerging"]
    sector: []  # 空列表 = 忽略sector维度
```

**结果**: 生成 3×2×2 = 12 个3维box，每个box的sector为None

### 2. 指定sector配置（4维box）

```yaml
box_weights:
  method: "equal"
  dimensions:
    size: ["large", "mid"]
    style: ["growth", "value"]
    region: ["developed"]
    sector: ["Technology", "Healthcare", "Financials"]
```

**结果**: 生成 2×2×1×3 = 12 个4维box，每个box包含具体的sector

### 3. 混合配置（configurable weights）

```yaml
box_weights:
  method: "config"
  weights:
    - box: ["large", "growth", "developed"]  # 3维box
      weight: 0.25
    - box: ["large", "growth", "developed", "Technology"]  # 4维box
      weight: 0.25
    - box: ["mid", "value", "emerging"]  # 3维box
      weight: 0.25
    - box: ["mid", "value", "emerging", "Healthcare"]  # 4维box
      weight: 0.25
```

## 向后兼容性

- **现有配置**: 所有现有的4维配置继续正常工作
- **默认行为**: 如果不指定sector或指定为空列表，会使用默认sector列表（保持现有行为）
- **API兼容**: 所有现有的BoxKey API保持不变

## 测试验证

创建了完整的测试套件验证：
- ✅ 3维BoxKey创建和序列化
- ✅ 4维BoxKey创建和序列化  
- ✅ 空sector配置生成正确的3维box
- ✅ 正常sector配置生成正确的4维box
- ✅ configurable weights支持混合维度
- ✅ 向后兼容性验证

## 示例演示

运行演示脚本查看完整功能：

```bash
poetry run python examples/sector_configuration_demo.py
```

## 架构优势

1. **SOLID原则**: 遵循开闭原则，对扩展开放，对修改封闭
2. **KISS原则**: 保持简单，通过配置控制行为
3. **向后兼容**: 不影响现有功能
4. **灵活性**: 支持3维和4维box的混合使用
5. **可配置性**: 通过YAML配置轻松控制box维度

## 实际应用场景

1. **行业中性策略**: 使用3维box忽略sector分类
2. **行业轮动策略**: 使用4维box包含特定sector
3. **混合策略**: 部分box忽略sector，部分包含sector
4. **测试和验证**: 简化配置进行快速测试

这个实现完全满足了"在文件中写sector:[]可以变成只看其他的box，不分sector"的需求，同时保持了系统的灵活性和向后兼容性。
