# Trading System Types Module

## 概述
类型定义模块，包含交易系统中使用的所有数据结构、枚举和类型定义。

## 主要组件

### 1. Portfolio Types (`portfolio.py`)
定义了投资组合和交易相关的核心数据结构。

#### Position (持仓)
```python
@dataclass
class Position:
    symbol: str          # 股票代码
    quantity: float      # 持仓数量
    average_cost: float  # 平均成本
    current_price: float # 当前价格
    market_value: float  # 市值
    unrealized_pnl: float # 未实现盈亏
    weight: float        # 投资组合权重 (0.0 到 1.0)

    # 属性方法
    @property
    def is_long(self) -> bool      # 是否多头
    @property
    def is_short(self) -> bool     # 是否空头
    @property
    def is_empty(self) -> bool     # 是否空仓
    @property
    def return_pct(self) -> float  # 收益率百分比

    # 方法
    def to_dict() -> Dict[str, Any]  # 转换为字典
```

#### Trade (交易)
```python
@dataclass
class Trade:
    symbol: str          # 股票代码
    side: str            # 交易方向 ('buy' 或 'sell')
    quantity: float      # 交易数量
    price: float         # 交易价格
    timestamp: datetime  # 交易时间
    commission: float    # 手续费
    trade_id: Optional[str] = None  # 交易ID

    # 属性方法
    @property
    def total_cost(self) -> float  # 总成本（含手续费）
    @property
    def is_buy(self) -> bool       # 是否买入
    @property
    def is_sell(self) -> bool      # 是否卖出

    # 方法
    def to_dict() -> Dict[str, Any]  # 转换为字典
```

#### PortfolioSnapshot (投资组合快照)
```python
@dataclass
class PortfolioSnapshot:
    timestamp: datetime    # 快照时间
    total_value: float     # 总价值
    cash_balance: float    # 现金余额
    positions: List[Position]  # 持仓列表
    daily_return: float    # 日收益率
    total_return: float    # 总收益率
    drawdown: float        # 回撤

    # 属性方法
    @property
    def equity_value(self) -> float     # 权益价值
    @property
    def cash_ratio(self) -> float       # 现金比例
    @property
    def equity_ratio(self) -> float     # 权益比例
    @property
    def positions_count(self) -> int    # 持仓数量

    # 方法
    def get_position(symbol: str) -> Optional[Position]  # 获取特定持仓
    def to_dict() -> Dict[str, Any]                      # 转换为字典
```

### 2. Enums (`enums.py`)
定义了系统中使用的枚举类型。

```python
class DataSource(Enum):
    """Data source enumeration - covers all major data providers."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    BLOOMBERG = "bloomberg"
    QUANDL = "quandl"
    KENNETH_FRENCH = "kenneth_french"
    POLYGON = "polygon"
    IEX = "iex"
    EXCEL_FILE = "excel_file"

class SignalType(Enum):
    """Trading signal types - simplified and comprehensive."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"
```

### 3. Market Data (`market_data.py`)
定义了市场数据相关的类型。

```python
@dataclass
class PriceData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None

@dataclass
class FactorData:
    date: datetime
    mkt_rf: float  # Market excess return
    smb: float    # Small minus big
    hml: float    # High minus low
    rmw: float    # Robust minus weak
    cma: float    # Conservative minus aggressive
    rf: float     # Risk-free rate
```

### 4. Signals (`signals.py`)
定义了交易信号相关的类型。

```python
@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    strength: float          # 信号强度 (0.0 到 1.0)
    timestamp: datetime
    price: float
    confidence: float = 1.0  # 置信度 (0.0 到 1.0)
    metadata: Optional[Dict[str, Any]] = None

    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY

    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL

    def is_hold(self) -> bool:
        return self.signal_type == SignalType.HOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
```

## 类型别名

为了向后兼容，定义了以下类型别名：

```python
# 向后兼容的别名
PositionList = List[Position]
TradeList = List[Trade]
PortfolioHistory = List[PortfolioSnapshot]
PriceDataFrame = PriceDict
SignalDataFrame = SignalDict
```

## 使用示例

### 创建持仓
```python
from trading_system.types import Position
from datetime import datetime

position = Position(
    symbol="AAPL",
    quantity=100,
    average_cost=150.0,
    current_price=155.0,
    market_value=15500.0,
    unrealized_pnl=500.0,
    weight=0.25
)

print(f"持仓收益率: {position.return_pct:.2%}")
print(f"是否多头: {position.is_long}")
```

### 创建交易
```python
from trading_system.types import Trade

trade = Trade(
    symbol="MSFT",
    side="buy",
    quantity=50,
    price=300.0,
    timestamp=datetime.now(),
    commission=5.0,
    trade_id="TRD_001"
)

print(f"交易总成本: ${trade.total_cost:.2f}")
print(f"是否买入: {trade.is_buy}")
```

### 创建投资组合快照
```python
from trading_system.types import PortfolioSnapshot, Position
from datetime import datetime

positions = [
    Position("AAPL", 100, 150.0, 155.0, 15500.0, 500.0, 0.6),
    Position("MSFT", 50, 300.0, 305.0, 15250.0, 250.0, 0.4)
]

snapshot = PortfolioSnapshot(
    timestamp=datetime.now(),
    total_value=30750.0,
    cash_balance=0.0,
    positions=positions,
    daily_return=0.015,
    total_return=0.125,
    drawdown=-0.02
)

print(f"现金比例: {snapshot.cash_ratio:.2%}")
print(f"权益比例: {snapshot.equity_ratio:.2%}")
print(f"持仓数量: {snapshot.positions_count}")
```

### 创建交易信号
```python
from trading_system.types import TradingSignal, SignalType
from datetime import datetime

signal = TradingSignal(
    symbol="GOOGL",
    signal_type=SignalType.BUY,
    strength=0.8,
    timestamp=datetime.now(),
    price=2500.0,
    confidence=0.75,
    metadata={"strategy": "momentum", "rsi": 30.0}
)

print(f"是否买入信号: {signal.is_buy()}")
print(f"信号强度: {signal.strength:.2f}")
```

## 数据验证

所有数据类型都包含参数验证：

### Position 验证
- 数量、成本、价格必须为正数（空仓除外）
- 权重必须在 0.0 到 1.0 之间

### Trade 验证
- 交易方向必须是 'buy' 或 'sell'
- 数量和价格必须为正数
- 手续费不能为负数

### PortfolioSnapshot 验证
- 总价值和现金余额不能为负数
- 持仓价值与现金余额之和必须等于总价值

## 序列化

所有数据类型都支持字典序列化，便于JSON存储和API传输：

```python
# 序列化
position_dict = position.to_dict()
trade_dict = trade.to_dict()
signal_dict = signal.to_dict()

# 反序列化（需要自定义实现）
position = Position(**position_dict)
```

## 注意事项

1. **时区处理**: 所有时间戳都应该是时区感知的
2. **精度处理**: 价格和数量使用适当的精度
3. **数据一致性**: 确保相关数据类型之间的一致性
4. **性能考虑**: 大量数据时考虑使用更高效的数据结构
5. **版本兼容性**: 添加新字段时考虑向后兼容性