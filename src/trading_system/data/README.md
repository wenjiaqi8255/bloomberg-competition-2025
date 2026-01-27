# Trading System Data Module

## 概述
数据模块负责提供和处理交易系统所需的所有数据，包括市场数据、因子数据等。该模块采用面向对象设计，遵循SOLID原则，提供了统一的数据访问接口。

## 架构设计

### 基类层次结构
```
BaseDataProvider (抽象基类)
├── PriceDataProvider (价格数据提供者基类)
│   └── YFinanceProvider (Yahoo Finance数据提供者)
├── FactorDataProvider (因子数据提供者基类)
│   └── FF5DataProvider (Fama-French 5因子数据提供者)
└── ClassificationProvider (分类提供者基类)
    └── StockClassifier (股票分类器)
```

### SOLID原则应用

#### 1. 单一职责原则 (Single Responsibility Principle)
- `BaseDataProvider`: 负责通用数据获取功能
- `PriceDataProvider`: 专门处理价格数据
- `FactorDataProvider`: 专门处理因子数据
- `ClassificationProvider`: 专门处理分类功能

#### 2. 开闭原则 (Open/Closed Principle)
- 基类对扩展开放，对修改封闭
- 新数据源可以通过继承基类轻松添加

#### 3. 里氏替换原则 (Liskov Substitution Principle)
- 所有子类都可以替换其基类使用
- 接口一致，行为可预测

#### 4. 接口隔离原则 (Interface Segregation Principle)
- 不同数据提供者有不同的专用接口
- 客户端只依赖需要的接口

#### 5. 依赖倒置原则 (Dependency Inversion Principle)
- 依赖抽象而非具体实现
- 通过依赖注入实现解耦

## 主要组件

### 1. 抽象基类 (`base_data_provider.py`)

#### `BaseDataProvider`
**功能**:
- 提供通用数据获取功能
- 重试机制和错误处理
- 缓存管理
- 数据验证和清洗
- 速率限制

**主要方法**:
```python
class BaseDataProvider(ABC):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 request_timeout: int = 30, cache_enabled: bool = True, 
                 rate_limit: float = 0.5)
    
    @abstractmethod
    def get_data_source(self) -> DataSource
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]
    @abstractmethod
    def _fetch_raw_data(self, *args, **kwargs) -> Any
    
    def _fetch_with_retry(self, fetch_func, *args, **kwargs) -> Optional[Any]
    def validate_data(self, data: pd.DataFrame, data_type: str = "general") -> pd.DataFrame
    def filter_by_date(self, data: pd.DataFrame, start_date: Union[str, datetime] = None, 
                      end_date: Union[str, datetime] = None) -> pd.DataFrame
    def add_data_source_metadata(self, data: pd.DataFrame) -> pd.DataFrame
```

#### `PriceDataProvider`
**功能**:
- 专门处理价格数据
- 提供价格数据验证
- 支持历史数据和实时价格

**主要方法**:
```python
class PriceDataProvider(BaseDataProvider):
    @abstractmethod
    def get_historical_data(self, symbols: Union[str, List[str]], 
                           start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None, 
                           **kwargs) -> Dict[str, pd.DataFrame]
    @abstractmethod
    def get_latest_price(self, symbols: Union[str, List[str]]) -> Dict[str, float]
    def validate_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame
```

#### `FactorDataProvider`
**功能**:
- 专门处理因子数据
- 提供因子数据验证
- 支持数据对齐功能

**主要方法**:
```python
class FactorDataProvider(BaseDataProvider):
    @abstractmethod
    def get_factor_returns(self, start_date: Union[str, datetime] = None, 
                          end_date: Union[str, datetime] = None) -> pd.DataFrame
    def validate_factor_data(self, data: pd.DataFrame) -> pd.DataFrame
    def align_with_equity_data(self, equity_data: Dict[str, pd.DataFrame], 
                              factor_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
```

#### `ClassificationProvider`
**功能**:
- 专门处理分类功能
- 提供分类接口
- 支持多种分类维度

**主要方法**:
```python
class ClassificationProvider(BaseDataProvider):
    @abstractmethod
    def classify_items(self, items: List[str], **kwargs) -> Dict[str, Any]
    @abstractmethod
    def get_classification_categories(self) -> Dict[str, List[str]]
```

### 2. 具体实现类

#### YFinance Provider (`yfinance_provider.py`)
**类名**: `YFinanceProvider`

**继承关系**: `YFinanceProvider` → `PriceDataProvider` → `BaseDataProvider`

**主要功能**:
- 从Yahoo Finance获取股票市场数据
- 支持重试机制和错误处理
- 数据验证和清洗
- 请求频率限制
- 缓存支持

**主要方法**:
```python
class YFinanceProvider(PriceDataProvider):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 request_timeout: int = 30, cache_enabled: bool = True)
    def get_historical_data(self, symbols: Union[str, List[str]], 
                           start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None, 
                           period: str = None) -> Dict[str, pd.DataFrame]
    def get_latest_price(self, symbols: Union[str, List[str]]) -> Dict[str, float]
    def get_dividends(self, symbols: Union[str, List[str]], 
                     start_date: Union[str, datetime] = None, 
                     end_date: Union[str, datetime] = None) -> Dict[str, pd.Series]
    def validate_symbol(self, symbol: str) -> bool
```

**使用示例**:
```python
from trading_system.data.yfinance_provider import YFinanceProvider

provider = YFinanceProvider(max_retries=5, retry_delay=2.0, cache_enabled=True)
data = provider.get_historical_data(['AAPL', 'MSFT'], start_date, end_date)
latest_prices = provider.get_latest_price(['AAPL', 'MSFT'])
```

#### Fama-French 5-Factor Provider (`ff5_provider.py`)
**类名**: `FF5DataProvider`

**继承关系**: `FF5DataProvider` → `FactorDataProvider` → `BaseDataProvider`

**主要功能**:
- 获取Fama-French 5因子数据
- 为因子模型提供因子数据支持
- 数据预处理和对齐
- 支持日线和月线数据

**主要方法**:
```python
class FF5DataProvider(FactorDataProvider):
    def __init__(self, data_frequency: str = "monthly", cache_dir: str = None,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 request_timeout: int = 30, cache_enabled: bool = True)
    def get_factor_returns(self, start_date: Union[str, datetime] = None, 
                          end_date: Union[str, datetime] = None) -> pd.DataFrame
    def get_risk_free_rate(self, start_date: Union[str, datetime] = None, 
                          end_date: Union[str, datetime] = None) -> pd.Series
    def get_factor_statistics(self, factor_data: pd.DataFrame = None) -> Dict
    def align_with_equity_data(self, equity_data: Dict[str, pd.DataFrame], 
                              factor_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
```

**使用示例**:
```python
from trading_system.data.ff5_provider import FF5DataProvider

provider = FF5DataProvider(data_frequency="monthly", cache_enabled=True)
factor_data = provider.get_factor_returns(start_date, end_date)
risk_free_rate = provider.get_risk_free_rate(start_date, end_date)
```

#### Stock Classifier (`stock_classifier.py`)
**类名**: `StockClassifier`

**继承关系**: `StockClassifier` → `ClassificationProvider` → `BaseDataProvider`

**主要功能**:
- 股票分类和筛选
- 基于市值、行业等维度进行分类
- 支持自定义分类规则
- IPS box-based allocation system

**主要方法**:
```python
class StockClassifier(ClassificationProvider):
    def __init__(self, yfinance_provider: YFinanceProvider = None,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 request_timeout: int = 30, cache_enabled: bool = True)
    def classify_stocks(self, symbols: List[str], 
                       price_data: Dict[str, pd.DataFrame] = None, 
                       as_of_date: datetime = None) -> Dict[str, InvestmentBox]
    def classify_stock(self, symbol: str, price_data: pd.DataFrame = None, 
                      as_of_date: datetime = None) -> Dict
    def get_box_summary(self, boxes: Dict[str, InvestmentBox]) -> Dict
    def optimize_box_structure(self, boxes: Dict[str, InvestmentBox], 
                              min_stocks_per_box: int = 2, 
                              max_boxes: int = 30) -> Dict[str, InvestmentBox]
```

**使用示例**:
```python
from trading_system.data.stock_classifier import StockClassifier
from trading_system.data.yfinance_provider import YFinanceProvider

yfinance_provider = YFinanceProvider()
classifier = StockClassifier(yfinance_provider=yfinance_provider)
boxes = classifier.classify_stocks(['AAPL', 'MSFT', 'GOOGL'])
summary = classifier.get_box_summary(boxes)
```

## 数据类型

### 输入数据
- **股票列表**: `List[str]` - 股票代码列表
- **日期范围**: `datetime` - 开始和结束日期
- **数据频率**: 支持日线、周线、月线等

### 输出数据
- **价格数据**: `pd.DataFrame` - 包含OHLCV数据
- **因子数据**: `pd.DataFrame` - Fama-French 5因子数据
- **分类结果**: `Dict[str, InvestmentBox]` - 投资组合分类结果
- **元数据**: 数据质量指标、来源信息等

## 错误处理
- API调用失败自动重试（指数退避）
- 数据验证和清洗
- 异常情况的日志记录
- 优雅降级机制
- 缓存失效处理

## 配置参数
```yaml
data:
  base:
    max_retries: 3
    retry_delay: 1.0
    request_timeout: 30
    cache_enabled: true
    rate_limit: 0.5  # 秒

  yfinance:
    rate_limit: 0.5  # 500ms between requests
    cache_ttl: 86400  # 24 hours

  ff5:
    data_frequency: "monthly"  # "daily" or "monthly"
    rate_limit: 1.0  # 1 second between requests
    cache_ttl: 86400  # 24 hours

  classifier:
    rate_limit: 0.5  # 500ms between requests
    cache_ttl: 3600  # 1 hour for classification results
```

## 依赖项
- `yfinance` - Yahoo Finance API
- `pandas` - 数据处理
- `requests` - HTTP请求
- `numpy` - 数值计算
- `abc` - 抽象基类支持

## 设计优势

### 1. 可扩展性
- 新数据源可以通过继承基类轻松添加
- 接口标准化，便于集成

### 2. 可维护性
- 代码复用，减少重复
- 统一的错误处理和日志记录
- 清晰的职责分离

### 3. 可测试性
- 依赖注入支持
- 接口抽象便于模拟测试
- 缓存机制可控制

### 4. 性能优化
- 智能缓存机制
- 速率限制避免API限制
- 数据验证减少错误处理

## 注意事项
1. YFinance有API调用频率限制，建议控制请求频率
2. Fama-French数据通常每月更新，需要注意数据时效性
3. 所有数据提供者都包含数据验证机制
4. 支持缓存机制以提高性能
5. 遵循SOLID原则，便于维护和扩展
6. 统一的错误处理和日志记录
7. 支持依赖注入，便于测试和配置