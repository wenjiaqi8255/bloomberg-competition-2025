# Investment Prediction Service - MVP Implementation Plan (Updated)

## Design Philosophy

**Core Principles**:

- **DRY**: Reuse MultiModelOrchestrator patterns, BaseStrategy architecture, PortfolioBuilderFactory
- **SOLID**: Single responsibility, factory pattern, strategy pattern
- **KISS**: Simple orchestration layer, wire existing components
- **YAGNI**: Build only MVP features, defer advanced functionality
- **MVP**: Incremental delivery in 3 phases

## Architecture Overview (Updated)

```
PredictionOrchestrator (new, thin coordinator - similar to MultiModelOrchestrator)
    ├── Load trained model(s) - single/base/meta (reuse ModelRegistry)
    ├── Load Strategy (BaseStrategy subclass - FF5/ML/Meta)
    ├── Strategy.generate_signals() (existing, uses model internally)
    ├── PortfolioBuilderFactory.create_builder() (factory pattern)
    │   └── BoxBasedPortfolioBuilder or QuantitativeBuilder
    └── PredictionResultFormatter (new, format output with box details)
```

**Key Architectural Changes**:

1. **Multi-Model Support**: Handle single models, base models, AND meta-models (reference: MultiModelOrchestrator)
2. **Strategy-Driven**: Use BaseStrategy subclasses (FF5Strategy, MLStrategy, MetaStrategy) - strategies own data prep
3. **Factory Pattern**: Use PortfolioBuilderFactory.create_builder() instead of direct BoxBasedPortfolioBuilder
4. **Signal Generation**: Call strategy.generate_signals() instead of direct model prediction
5. **90% Reuse**: MultiModelOrchestrator patterns + BaseStrategy + PortfolioBuilderFactory

---

## Phase 1: Core Prediction Pipeline (MVP)

### 1.1 Create PredictionOrchestrator

**File**: `src/use_case/prediction/prediction_orchestrator.py`

**Purpose**: Thin coordinator supporting single/multi/meta models (pattern from MultiModelOrchestrator)

**Key Methods**:

```python
class PredictionOrchestrator:
    def __init__(self, config_path: str)
    def run_prediction(self) -> PredictionResult
    def _load_strategy(self) -> BaseStrategy  # Load FF5/ML/Meta strategy
    def _generate_signals(self) -> pd.DataFrame  # Via strategy.generate_signals()
    def _construct_portfolio(self, signals) -> PortfolioConstructionResult
    def _create_portfolio_builder(self) -> IPortfolioBuilder  # Factory pattern
```

**Reuse Strategy** (updated):

- **Model loading**: Handled by Strategy classes (they load their own models)
- **Strategy creation**: Use StrategyFactory (existing) like in strategy_runner.py
- **Data providers**: Reuse `_create_data_provider()` from MultiModelOrchestrator (line 123-143)
- **Signal generation**: Call `strategy.generate_signals(price_data, start_date, end_date)` (BaseStrategy line 115-174)
- **Portfolio construction**: Use `PortfolioBuilderFactory.create_builder(config)` (factory.py line 31-81)
- **Meta-model support**: Reference MetaStrategy usage in MultiModelOrchestrator (line 735-789)

**Critical Pattern - Strategy Handles Everything**:

```python
# Strategy loads its own model, prepares data, generates predictions
strategy = StrategyFactory.create_from_config(strategy_config)  # Creates FF5/ML/Meta
signals = strategy.generate_signals(price_data, prediction_date, prediction_date)

# Then portfolio construction
builder = PortfolioBuilderFactory.create_builder(portfolio_config)
request = PortfolioConstructionRequest(...)
weights = builder.build_portfolio(request)
```

### 1.2 Define PredictionResult Data Structure

**File**: `src/use_case/prediction/types.py`

**Updated to support multi-model**:

```python
@dataclass
class StockRecommendation:
    symbol: str
    weight: float
    signal_strength: float  # From strategy.generate_signals()
    box_classification: Optional[BoxKey]  # From BoxConstructionResult
    risk_score: float
    
@dataclass
class PredictionResult:
    # Core outputs
    recommendations: List[StockRecommendation]
    portfolio_weights: pd.Series
    
    # Box-based details (from BoxConstructionResult if using box_based method)
    box_allocations: Optional[Dict[str, float]]  # From BoxConstructionResult.box_coverage
    stocks_by_box: Optional[Dict[str, List[str]]]  # From BoxConstructionResult.selected_stocks
    box_construction_log: Optional[List[str]]  # From BoxConstructionResult.construction_log
    
    # Model information (support multi-model)
    strategy_type: str  # 'ff5', 'ml', 'meta', etc.
    model_id: str  # Single model ID or meta-model ID
    base_model_ids: Optional[List[str]]  # For meta-models
    model_weights: Optional[Dict[str, float]]  # For meta-models
    
    # Metadata
    prediction_date: datetime
    total_positions: int
    portfolio_method: str  # 'box_based' or 'quantitative'
    
    # Risk metrics (from portfolio construction)
    expected_return: float
    expected_risk: float
    diversification_score: float
```

### 1.3 Create Configuration File Template

**File**: `configs/prediction_config.yaml`

**Updated to support multi-model and factory pattern**:

```yaml
# Prediction configuration - supports single/multi/meta models
prediction:
  prediction_date: "2024-01-15"
  
# Strategy configuration (determines which model type)
strategy:
  type: "fama_french_5"  # Options: 'fama_french_5', 'ml', 'meta'
  name: "FF5_Prediction_Strategy"
  
  # For single model strategies (ff5, ml)
  parameters:
    model_id: "ff5_regression_20240115_120000"
    model_registry_path: "./models/"
    lookback_days: 252
    risk_free_rate: 0.02
  
  # For meta strategy (optional, only if type: 'meta')
  # meta_config:
  #   base_model_ids:
  #     - "ff5_regression_20240115_120000"
  #     - "xgboost_20240115_130000"
  #   model_weights:
  #     ff5_regression_20240115_120000: 0.6
  #     xgboost_20240115_130000: 0.4
  #   model_registry_path: "./models/"

# Data configuration - reuse from MultiModelOrchestrator pattern
data_provider:
  type: "YFinanceProvider"
  parameters:
    max_retries: 3
    retry_delay: 1.0

factor_data_provider:  # Optional, for FF5 models
  type: "FF5DataProvider"
  parameters:
    data_frequency: "daily"

# Universe configuration
universe:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - META
    - TSLA
    - NVDA
    - JPM
    - V
    - WMT

# Portfolio construction - USE FACTORY PATTERN
portfolio_construction:
  method: "box_based"  # Factory determines which builder
  
  # Box-based configuration (if method: box_based)
  stocks_per_box: 3
  min_stocks_per_box: 1
  allocation_method: "equal"
  allocation_config: {}
  
  box_weights:
    method: "config"
    weights:
      large_growth_developed_Technology: 0.15
      large_value_developed_Financials: 0.10
      mid_growth_developed_Healthcare: 0.08
      # ... more boxes
      
  classifier:
    method: "four_factor"
    size_breakpoints: [10000, 50000]  # millions
    style_method: "pb_ratio"
    cache_enabled: true
    
  box_selector:
    type: "signal_based"
  
  # Quantitative configuration (if method: quantitative)
  # optimizer:
  #   method: "mean_variance"
  #   risk_aversion: 1.0
  # covariance:
  #   lookback_days: 252
  #   method: "ledoit_wolf"

# Risk constraints
constraints:
  max_position_weight: 0.15
  min_position_weight: 0.02
  max_leverage: 1.0
```

### 1.4 Create CLI Entry Point

**File**: `src/use_case/prediction/run_prediction.py`

**Purpose**: Simple CLI similar to MultiModelOrchestrator usage

```python
def main():
    parser = argparse.ArgumentParser(
        description='Generate investment predictions from trained models'
    )
    parser.add_argument(
        '--config', 
        default='configs/prediction_config.yaml',
        help='Path to prediction configuration file'
    )
    parser.add_argument(
        '--output-dir',
        default='./prediction_results',
        help='Directory to save prediction results'
    )
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PredictionOrchestrator(args.config)
    
    # Run prediction
    result = orchestrator.run_prediction()
    
    # Format and display results
    formatter = PredictionResultFormatter()
    print(formatter.format_console_report(result))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter.save_to_json(result, output_dir / 'prediction_result.json')
    formatter.save_to_csv(result, output_dir / 'recommendations.csv')
```

---

## Phase 2: Result Formatting & Reporting

### 2.1 Create Result Formatter

**File**: `src/use_case/prediction/formatters.py`

**Purpose**: Format results with box details (from BoxConstructionResult)

```python
class PredictionResultFormatter:
    def format_console_report(self, result: PredictionResult) -> str
    def format_json_report(self, result: PredictionResult) -> Dict[str, Any]
    def save_to_csv(self, result: PredictionResult, path: Path)
    def _format_box_allocations(self, result: PredictionResult) -> str
    def _format_meta_model_info(self, result: PredictionResult) -> str  # NEW
```

**Enhanced Output with Multi-Model Support**:

```
=== Investment Recommendations for 2024-01-15 ===
Strategy: FF5Strategy (or MetaStrategy with 3 base models)
Model ID: ff5_regression_20240115_120000
Portfolio Method: box_based

[If MetaStrategy]
Base Models:
  - ff5_regression_20240115_120000 (weight: 0.40)
  - xgboost_20240115_130000 (weight: 0.35)
  - lstm_20240115_140000 (weight: 0.25)

Top 10 Stock Recommendations:
Rank  Symbol  Weight   Signal    Box Classification                      Risk
1     AAPL    12.5%    0.085     large_growth_developed_Technology       0.18
2     MSFT    11.2%    0.078     large_growth_developed_Technology       0.16
...

Box Allocations: (from BoxConstructionResult)
Box                                    Target   Actual   Selected Stocks
large_growth_developed_Technology      15.0%    23.7%    AAPL, MSFT, GOOGL
large_value_developed_Financials       10.0%    10.1%    JPM, BAC
...

Portfolio Summary:
Total Positions: 12
Expected Return: 8.2% (annualized)
Expected Risk: 15.3% (volatility)
Diversification Score: 0.82
```

### 2.2 Extract Box Details from Portfolio Builder

**Integration**: BoxConstructionResult already contains all needed info

```python
# In PredictionOrchestrator._construct_portfolio()
builder = PortfolioBuilderFactory.create_builder(portfolio_config)

# Check if builder supports detailed results
if hasattr(builder, 'build_portfolio_with_result'):
    # BoxBasedPortfolioBuilder returns BoxConstructionResult
    portfolio_result = builder.build_portfolio_with_result(request)
    weights = portfolio_result.weights
    box_details = {
        'box_coverage': portfolio_result.box_coverage,
        'selected_stocks': portfolio_result.selected_stocks,
        'target_weights': portfolio_result.target_weights,
        'construction_log': portfolio_result.construction_log
    }
else:
    # QuantitativePortfolioBuilder returns pd.Series
    weights = builder.build_portfolio(request)
    box_details = None
```

---

## Phase 3: Testing & Validation

### 3.1 Create Integration Test

**File**: `test/test_prediction_orchestrator.py`

**Test scenarios**:

```python
def test_prediction_with_single_model():
    """Test with single FF5 or ML model"""
    
def test_prediction_with_meta_model():
    """Test with meta-model ensemble"""
    
def test_box_based_portfolio_construction():
    """Verify box details are extracted"""
    
def test_quantitative_portfolio_construction():
    """Test quantitative method via factory"""
```

### 3.2 Create Example Demo Scripts

**File**: `examples/prediction_demo_single.py`

```python
# Demo: Single model prediction
orchestrator = PredictionOrchestrator("configs/prediction_ff5.yaml")
result = orchestrator.run_prediction()
print(f"Strategy: {result.strategy_type}")
print(f"Top 5 stocks: {result.recommendations[:5]}")
```

**File**: `examples/prediction_demo_meta.py`

```python
# Demo: Meta-model prediction
orchestrator = PredictionOrchestrator("configs/prediction_meta.yaml")
result = orchestrator.run_prediction()
print(f"Base models: {result.base_model_ids}")
print(f"Model weights: {result.model_weights}")
print(f"Recommendations: {result.recommendations[:5]}")
```

---

## Implementation Details (Updated)

### Key Files to Create (Minimal New Code)

1. **`src/use_case/prediction/__init__.py`** - Package init
2. **`src/use_case/prediction/prediction_orchestrator.py`** (~250 lines, orchestration)
3. **`src/use_case/prediction/types.py`** (~100 lines, data structures)
4. **`src/use_case/prediction/formatters.py`** (~200 lines, formatting)
5. **`src/use_case/prediction/run_prediction.py`** (~120 lines, CLI)
6. **`configs/prediction_config.yaml`** (~150 lines, template)
7. **`configs/prediction_meta_config.yaml`** (~170 lines, meta-model template)
8. **`test/test_prediction_orchestrator.py`** (~200 lines, tests)
9. **`examples/prediction_demo_single.py`** (~40 lines)
10. **`examples/prediction_demo_meta.py`** (~50 lines)

**Total New Code**: ~1280 lines (thin orchestration + configuration)

### Components to Reuse (No Changes)

1. **MultiModelOrchestrator patterns** - Data provider creation, meta-model handling
2. **BaseStrategy** - Strategy interface (FF5Strategy, MLStrategy, MetaStrategy)
3. **StrategyFactory** - Create strategies from config
4. **PortfolioBuilderFactory** - Create portfolio builders (factory pattern)
5. **BoxBasedPortfolioBuilder** - Returns BoxConstructionResult with details
6. **QuantitativePortfolioBuilder** - Alternative construction method
7. **ModelRegistry** - Load models (handled by strategies)
8. **PortfolioConstructionRequest** - Request structure
9. **BoxConstructionResult** - Contains all box details

### Critical Implementation Points (Updated)

1. **Strategy Creation (Reuse StrategyFactory)**:
   ```python
   # In PredictionOrchestrator._load_strategy()
   from src.trading_system.strategies.factory import StrategyFactory
   
   strategy_config = self.config['strategy']
   strategy = StrategyFactory.create_from_config(strategy_config)
   # Strategy loads its own model internally
   ```

2. **Data Provider Creation (Reuse MultiModelOrchestrator pattern)**:
   ```python
   # Exact pattern from MultiModelOrchestrator line 123-143
   def _create_data_provider(self, config: Dict[str, Any]):
       provider_type = config.get('type')
       params = config.get('parameters', {})
       if provider_type == "YFinanceProvider":
           from src.trading_system.data.yfinance_provider import YFinanceProvider
           return YFinanceProvider(**params)
   ```

3. **Signal Generation (Via Strategy)**:
   ```python
   # Strategy handles everything internally
   price_data = self.data_provider.get_data(symbols, start_date, end_date)
   signals = strategy.generate_signals(price_data, prediction_date, prediction_date)
   # signals is pd.DataFrame: dates × symbols with signal strengths
   ```

4. **Portfolio Construction (Factory Pattern)**:
   ```python
   # Use factory instead of direct instantiation
   from src.trading_system.portfolio_construction.factory import PortfolioBuilderFactory
   
   builder = PortfolioBuilderFactory.create_builder(portfolio_config)
   request = PortfolioConstructionRequest(
       date=prediction_date,
       universe=symbols,
       signals=signals.iloc[-1],  # Latest signals as Series
       price_data=price_data,
       constraints=constraints
   )
   
   # Try to get detailed result
   if hasattr(builder, 'build_portfolio_with_result'):
       result = builder.build_portfolio_with_result(request)  # BoxConstructionResult
   else:
       weights = builder.build_portfolio(request)  # pd.Series
   ```

5. **Meta-Model Support (Reference MultiModelOrchestrator)**:
   ```python
   # For MetaStrategy, config includes base models and weights
   # See MultiModelOrchestrator._create_ensemble_experiment_config (line 735-789)
   strategy_config = {
       'type': 'meta',
       'parameters': {
           'base_model_ids': ['model1', 'model2'],
           'model_weights': {'model1': 0.6, 'model2': 0.4},
           'model_registry_path': './models/'
       }
   }
   ```


---

## Validation Checklist

- [ ] Loads single model strategy (FF5, ML)
- [ ] Loads meta-model strategy with multiple base models
- [ ] Strategy generates signals correctly
- [ ] Factory creates correct portfolio builder
- [ ] Box-based builder returns detailed BoxConstructionResult
- [ ] Quantitative builder works via factory
- [ ] Box details extracted correctly
- [ ] Portfolio weights sum to 1.0
- [ ] Respects position limits
- [ ] Console output shows box allocations
- [ ] JSON output includes all metadata
- [ ] CSV export works
- [ ] Integration tests pass (single + meta)
- [ ] Demo scripts run successfully

---

## Success Metrics

**MVP Success**: User can run one command and get:

1. Stock recommendations from single or meta-model
2. Box classification for each stock (if using box_based)
3. Model weights display (for meta-models)
4. Expected portfolio return and risk
5. Construction log from portfolio builder

**Code Quality**:

- 90%+ code reuse (DRY principle)
- <1300 lines of new code
- Proper use of existing patterns (Strategy, Factory)
- All existing components untouched
- Clean separation via orchestration (SOLID)
- Simple coordination logic (KISS)
- Only essential features (YAGNI)

**Architecture Alignment**:

- Follows MultiModelOrchestrator patterns
- Uses BaseStrategy architecture correctly
- Uses PortfolioBuilderFactory (not direct class)
- Supports single, multi, and meta models
- Strategies handle their own model loading