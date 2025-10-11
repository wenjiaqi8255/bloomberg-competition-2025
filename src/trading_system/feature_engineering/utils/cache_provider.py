"""
特征缓存提供者接口

这是一个抽象接口，定义了特征缓存的标准操作。
不同的实现（本地文件、数据库、Redis等）都必须遵守这个契约。
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import pandas as pd


class FeatureCacheProvider(ABC):
    """特征缓存的抽象接口"""

    @abstractmethod
    def get(
        self,
        symbol: str,
        feature_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        获取缓存的特征数据

        Args:
            symbol: 股票代码，如 'AAPL'
            feature_name: 特征名称，如 'SMA_200'
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            包含特征值的 DataFrame，如果缓存未命中返回 None
            DataFrame 格式：
                index: DatetimeIndex (日期)
                columns: ['value'] (特征值)
        """
        pass

    @abstractmethod
    def set(
        self,
        symbol: str,
        feature_name: str,
        data: pd.DataFrame
    ) -> None:
        """
        存储特征数据到缓存

        Args:
            symbol: 股票代码
            feature_name: 特征名称
            data: 特征数据，必须包含 'value' 列
        """
        pass

    @abstractmethod
    def get_last_update(
        self,
        symbol: str,
        feature_name: str
    ) -> Optional[datetime]:
        """
        获取特征最后更新的时间

        Args:
            symbol: 股票代码
            feature_name: 特征名称

        Returns:
            最后更新时间，如果没有缓存返回 None
        """
        pass

    @abstractmethod
    def clear(self, symbol: Optional[str] = None) -> None:
        """
        清空缓存

        Args:
            symbol: 如果指定，只清空该股票的缓存；否则清空所有
        """
        pass