"""
股票数据持久化缓存实现

使用 Parquet 格式存储 OHLCV 数据到本地文件系统。
支持增量更新和日期范围查询。
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class StockDataCache:
    """股票数据持久化缓存，使用本地 Parquet 文件存储 OHLCV 数据"""

    def __init__(self, cache_dir: str = "./cache/stock_data", cache_version: str = "v1"):
        """
        初始化股票数据缓存

        Args:
            cache_dir: 缓存目录路径
            cache_version: 缓存版本，用于防止格式不兼容
        """
        self.cache_dir = Path(cache_dir)
        self.cache_version = cache_version
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 创建版本文件
        version_file = self.cache_dir / ".cache_version"
        if not version_file.exists():
            version_file.write_text(cache_version)
        else:
            existing_version = version_file.read_text().strip()
            if existing_version != cache_version:
                logger.warning(
                    f"Cache version mismatch: expected {cache_version}, found {existing_version}. "
                    f"Consider clearing cache."
                )

        logger.info(f"Initialized StockDataCache v{cache_version} at {self.cache_dir}")

    def _get_cache_path(self, symbol: str) -> Path:
        """获取缓存文件路径"""
        # 使用子目录按股票分组，避免单个目录文件过多
        symbol_dir = self.cache_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / "data.parquet"

    def _get_metadata_path(self, symbol: str) -> Path:
        """获取元数据文件路径"""
        symbol_dir = self.cache_dir / symbol
        return symbol_dir / "metadata.json"

    def get(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        读取缓存的股票数据

        Args:
            symbol: 股票代码
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            DataFrame with OHLCV data, or None if cache miss
        """
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            logger.debug(f"Cache MISS: {symbol}")
            return None

        try:
            # 检查缓存版本兼容性
            version_file = self.cache_dir / ".cache_version"
            if version_file.exists():
                cached_version = version_file.read_text().strip()
                if cached_version != self.cache_version:
                    logger.debug(
                        f"Cache version mismatch for {symbol}: "
                        f"{cached_version} != {self.cache_version}"
                    )
                    return None

            # 读取 Parquet 文件
            df = pd.read_parquet(cache_path)

            # 验证数据格式
            if df.empty:
                logger.debug(f"Cache HIT but empty file: {symbol}")
                return None

            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                df.index = pd.to_datetime(df.index)

            # 检查必需的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Cache format error for {symbol}: missing columns {missing_cols}")
                return None

            # 过滤日期范围
            if start_date is not None or end_date is not None:
                mask = pd.Series(True, index=df.index)
                if start_date is not None:
                    mask = mask & (df.index >= pd.to_datetime(start_date))
                if end_date is not None:
                    mask = mask & (df.index <= pd.to_datetime(end_date))
                df = df[mask]

            if df.empty:
                logger.debug(f"Cache HIT but empty range: {symbol}")
                return None

            logger.debug(f"Cache HIT: {symbol} ({len(df)} rows)")
            return df

        except Exception as e:
            logger.warning(f"Failed to read cache for {symbol}: {e}")
            # 如果读取失败，可以删除损坏的缓存文件
            try:
                cache_path.unlink(missing_ok=True)
                metadata_path = self._get_metadata_path(symbol)
                metadata_path.unlink(missing_ok=True)
                logger.debug(f"Removed corrupted cache file: {cache_path}")
            except:
                pass
            return None

    def get_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """
        获取缓存数据的日期范围

        Args:
            symbol: 股票代码

        Returns:
            (start_date, end_date) tuple or None if no cache
        """
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return None

            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                df.index = pd.to_datetime(df.index)

            return (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
        except Exception as e:
            logger.warning(f"Failed to get date range for {symbol}: {e}")
            return None

    def get_missing_date_ranges(
        self,
        symbol: str,
        requested_start: datetime,
        requested_end: datetime
    ) -> list:
        """
        获取缺失的日期范围

        Args:
            symbol: 股票代码
            requested_start: 请求的起始日期
            requested_end: 请求的结束日期

        Returns:
            List of (start, end) tuples for missing date ranges
        """
        cached_range = self.get_date_range(symbol)
        if cached_range is None:
            # 完全没有缓存
            return [(requested_start, requested_end)]

        cached_start, cached_end = cached_range
        missing_ranges = []

        # 确保日期是 datetime 类型
        if isinstance(requested_start, str):
            requested_start = pd.to_datetime(requested_start)
        if isinstance(requested_end, str):
            requested_end = pd.to_datetime(requested_end)
        if isinstance(cached_start, pd.Timestamp):
            cached_start = cached_start.to_pydatetime()
        if isinstance(cached_end, pd.Timestamp):
            cached_end = cached_end.to_pydatetime()

        # 检查前面的缺失部分
        if requested_start < cached_start:
            end_bound = cached_start - timedelta(days=1)
            if end_bound > requested_end:
                end_bound = requested_end
            if end_bound >= requested_start:
                missing_ranges.append((requested_start, end_bound))

        # 检查后面的缺失部分
        if requested_end > cached_end:
            start_bound = cached_end + timedelta(days=1)
            if start_bound < requested_start:
                start_bound = requested_start
            if start_bound <= requested_end:
                missing_ranges.append((start_bound, requested_end))

        # 如果有缺失范围，返回它们
        return missing_ranges if missing_ranges else []

    def set(self, symbol: str, data: pd.DataFrame, merge: bool = True) -> None:
        """
        存储股票数据到缓存

        Args:
            symbol: 股票代码
            data: OHLCV DataFrame
            merge: 是否与现有缓存合并（增量更新）
        """
        cache_path = self._get_cache_path(symbol)

        try:
            # 验证输入数据
            if data is None or data.empty:
                logger.warning(f"Attempted to cache empty data for {symbol}")
                return

            # 确保索引是 DatetimeIndex
            df = data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                df.index = pd.to_datetime(df.index)

            # 检查必需的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Data must have required columns {required_columns}, "
                               f"missing {missing_cols}")

            # 如果启用合并且缓存存在，合并数据
            if merge and cache_path.exists():
                try:
                    cached_df = pd.read_parquet(cache_path)
                    if not cached_df.empty:
                        if not isinstance(cached_df.index, pd.DatetimeIndex):
                            if 'Date' in cached_df.columns:
                                cached_df = cached_df.set_index('Date')
                            cached_df.index = pd.to_datetime(cached_df.index)

                        # 合并数据，保留所有列
                        all_columns = list(set(cached_df.columns) | set(df.columns))
                        df_combined = pd.concat([cached_df, df], axis=0)

                        # 按索引去重，保留最后一条（新数据优先）
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

                        # 按日期排序
                        df_combined = df_combined.sort_index()

                        # 只保留需要的列
                        df = df_combined[all_columns].copy()

                        logger.debug(
                            f"Merged cache for {symbol}: {len(cached_df)} cached rows + "
                            f"{len(data)} new rows = {len(df)} total rows"
                        )
                except Exception as e:
                    logger.warning(f"Failed to merge cache for {symbol}, overwriting: {e}")
                    # 如果合并失败，直接覆盖

            # 确保数据按日期排序
            df = df.sort_index()

            # 写入 Parquet 文件（使用 gzip 压缩）
            df.to_parquet(cache_path, compression='gzip')

            # 创建并保存元数据
            metadata = {
                'symbol': symbol,
                'cache_version': self.cache_version,
                'last_updated': datetime.now().isoformat(),
                'rows': len(df),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                },
                'columns': list(df.columns)
            }

            metadata_path = self._get_metadata_path(symbol)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(
                f"Cached {len(df)} rows for {symbol} "
                f"({metadata['date_range']['start']} to {metadata['date_range']['end']})"
            )

        except Exception as e:
            logger.error(f"Failed to cache {symbol}: {e}")
            # 如果写入失败，清理可能损坏的文件
            try:
                cache_path.unlink(missing_ok=True)
                metadata_path = self._get_metadata_path(symbol)
                metadata_path.unlink(missing_ok=True)
            except:
                pass
            raise

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        清空缓存

        Args:
            symbol: 如果指定，只清空该股票的缓存；否则清空所有
        """
        if symbol:
            # 清空特定股票的缓存
            symbol_dir = self.cache_dir / symbol
            if symbol_dir.exists():
                shutil.rmtree(symbol_dir)
                logger.info(f"Cleared cache for {symbol}")
        else:
            # 清空所有缓存
            if self.cache_dir.exists():
                # 保留版本文件
                version_file = self.cache_dir / ".cache_version"
                version_content = None
                if version_file.exists():
                    version_content = version_file.read_text()

                # 删除所有子目录
                for item in self.cache_dir.iterdir():
                    if item.is_dir() and item.name != '.cache_version':
                        shutil.rmtree(item)

                # 如果版本文件存在，重新创建
                if version_content:
                    version_file.write_text(version_content)

                logger.info("Cleared all stock data cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'cache_dir': str(self.cache_dir),
            'version': self.cache_version,
            'symbols': 0,
            'total_rows': 0,
            'total_size_mb': 0,
            'invalid_files': 0
        }

        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.startswith('.'):
                continue

            if symbol_dir.is_dir():
                cache_path = symbol_dir / "data.parquet"
                if cache_path.exists():
                    try:
                        stats['symbols'] += 1
                        stats['total_size_mb'] += cache_path.stat().st_size / (1024 * 1024)

                        # 读取文件获取行数
                        df = pd.read_parquet(cache_path)
                        stats['total_rows'] += len(df)

                        # 检查元数据版本
                        metadata_path = symbol_dir / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            if metadata.get('cache_version') != self.cache_version:
                                stats['invalid_files'] += 1
                    except Exception as e:
                        stats['invalid_files'] += 1
                        logger.debug(f"Error checking cache for {symbol_dir.name}: {e}")

        return stats

    def clear_invalid_cache(self) -> int:
        """
        清理无效的缓存文件

        Returns:
            清理的文件数量
        """
        cleared_count = 0

        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.startswith('.'):
                continue

            if symbol_dir.is_dir():
                cache_path = symbol_dir / "data.parquet"
                metadata_path = symbol_dir / "metadata.json"

                try:
                    # 检查元数据文件
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        # 检查版本
                        if metadata.get('cache_version') != self.cache_version:
                            shutil.rmtree(symbol_dir)
                            cleared_count += 1
                            logger.debug(f"Removed outdated cache: {symbol_dir.name}")
                            continue

                    # 尝试读取文件验证完整性
                    try:
                        df = pd.read_parquet(cache_path)
                        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        if df.empty or missing_cols:
                            raise ValueError("Invalid format")
                    except Exception:
                        # 文件损坏，删除
                        shutil.rmtree(symbol_dir)
                        cleared_count += 1
                        logger.debug(f"Removed corrupted cache: {symbol_dir.name}")

                except Exception as e:
                    logger.debug(f"Error checking cache file {symbol_dir.name}: {e}")
                    continue

        if cleared_count > 0:
            logger.info(f"Cleaned {cleared_count} invalid cache files")
        else:
            logger.debug("No invalid cache files found")

        return cleared_count
