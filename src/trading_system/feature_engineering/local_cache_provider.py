"""
本地文件缓存实现

使用 Parquet 格式存储特征到本地文件系统。
优点：简单、快速、不需要额外依赖
缺点：无法跨机器共享
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd

from .cache_provider import FeatureCacheProvider

logger = logging.getLogger(__name__)


class LocalCacheProvider(FeatureCacheProvider):
    """使用本地 Parquet 文件的缓存实现"""

    def __init__(self, cache_dir: str = "./cache/features"):
        """
        初始化本地缓存

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalCacheProvider at {self.cache_dir}")

    def _get_cache_path(self, symbol: str, feature_name: str) -> Path:
        """获取缓存文件路径"""
        # 使用子目录按股票分组，避免单个目录文件过多
        symbol_dir = self.cache_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{feature_name}.parquet"

    def get(
        self,
        symbol: str,
        feature_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """读取缓存的特征数据"""
        cache_path = self._get_cache_path(symbol, feature_name)

        if not cache_path.exists():
            logger.debug(f"Cache MISS: {symbol}/{feature_name}")
            return None

        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(cache_path)

            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 过滤日期范围
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df[mask]

            if filtered_df.empty:
                logger.debug(f"Cache HIT but empty range: {symbol}/{feature_name}")
                return None

            logger.debug(f"Cache HIT: {symbol}/{feature_name} ({len(filtered_df)} rows)")
            return filtered_df

        except Exception as e:
            logger.warning(f"Failed to read cache for {symbol}/{feature_name}: {e}")
            return None

    def set(
        self,
        symbol: str,
        feature_name: str,
        data: pd.DataFrame
    ) -> None:
        """存储特征数据到缓存"""
        cache_path = self._get_cache_path(symbol, feature_name)

        try:
            # 确保数据有 'value' 列
            if 'value' not in data.columns:
                # 如果只有一列，重命名为 'value'
                if len(data.columns) == 1:
                    data = data.copy()
                    data.columns = ['value']
                else:
                    raise ValueError(f"Data must have 'value' column, got {data.columns}")

            # 确保索引是 DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # 写入 Parquet 文件（使用 gzip 压缩）
            data.to_parquet(cache_path, compression='gzip')
            logger.debug(f"Cached {len(data)} rows for {symbol}/{feature_name}")

        except Exception as e:
            logger.error(f"Failed to cache {symbol}/{feature_name}: {e}")

    def get_last_update(
        self,
        symbol: str,
        feature_name: str
    ) -> Optional[datetime]:
        """获取最后更新时间"""
        cache_path = self._get_cache_path(symbol, feature_name)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.index.max()
        except Exception as e:
            logger.warning(f"Failed to get last update for {symbol}/{feature_name}: {e}")
            return None

    def clear(self, symbol: Optional[str] = None) -> None:
        """清空缓存"""
        if symbol:
            # 清空特定股票的缓存
            symbol_dir = self.cache_dir / symbol
            if symbol_dir.exists():
                import shutil
                shutil.rmtree(symbol_dir)
                logger.info(f"Cleared cache for {symbol}")
        else:
            # 清空所有缓存
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all cache")