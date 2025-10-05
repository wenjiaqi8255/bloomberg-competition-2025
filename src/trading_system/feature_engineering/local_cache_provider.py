"""
本地文件缓存实现

使用 Parquet 格式存储特征到本地文件系统。
优点：简单、快速、不需要额外依赖
缺点：无法跨机器共享
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from .cache_provider import FeatureCacheProvider

logger = logging.getLogger(__name__)


class LocalCacheProvider(FeatureCacheProvider):
    """使用本地 Parquet 文件的缓存实现"""

    def __init__(self, cache_dir: str = "./cache/features", cache_version: str = "v1"):
        """
        初始化本地缓存

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
                logger.warning(f"Cache version mismatch: expected {cache_version}, found {existing_version}. Consider clearing cache.")

        logger.info(f"Initialized LocalCacheProvider v{cache_version} at {self.cache_dir}")

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
            # 检查缓存版本兼容性
            version_file = cache_path.parent / ".cache_version"
            if version_file.exists():
                cached_version = version_file.read_text().strip()
                if cached_version != self.cache_version:
                    logger.debug(f"Cache version mismatch for {symbol}/{feature_name}: {cached_version} != {self.cache_version}")
                    return None

            # 读取 Parquet 文件
            df = pd.read_parquet(cache_path)

            # 验证数据格式
            if df.empty:
                logger.debug(f"Cache HIT but empty file: {symbol}/{feature_name}")
                return None

            # 确保索引是 DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 确保有 'value' 列
            if 'value' not in df.columns:
                logger.debug(f"Cache format error for {symbol}/{feature_name}: missing 'value' column")
                return None

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
            # 如果读取失败，可以删除损坏的缓存文件
            try:
                cache_path.unlink(missing_ok=True)
                logger.debug(f"Removed corrupted cache file: {cache_path}")
            except:
                pass
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
            # 验证输入数据
            if data is None or data.empty:
                logger.warning(f"Attempted to cache empty data for {symbol}/{feature_name}")
                return

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

            # 创建元数据
            metadata = {
                'symbol': symbol,
                'feature_name': feature_name,
                'cache_version': self.cache_version,
                'created_at': datetime.now().isoformat(),
                'rows': len(data),
                'date_range': f"{data.index.min().date()} to {data.index.max().date()}"
            }

            # 写入 Parquet 文件（使用 gzip 压缩）
            data.to_parquet(cache_path, compression='gzip')

            # 写入元数据文件
            metadata_path = cache_path.with_suffix('.meta.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Cached {len(data)} rows for {symbol}/{feature_name} ({metadata['date_range']})")

        except Exception as e:
            logger.error(f"Failed to cache {symbol}/{feature_name}: {e}")
            # 如果写入失败，清理可能损坏的文件
            try:
                cache_path.unlink(missing_ok=True)
                metadata_path = cache_path.with_suffix('.meta.json')
                metadata_path.unlink(missing_ok=True)
            except:
                pass

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

    def clear(self, symbol: Optional[str] = None, force: bool = False) -> None:
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
                # 重新创建版本文件
                version_file = self.cache_dir / ".cache_version"
                version_file.write_text(self.cache_version)
                logger.info("Cleared all cache")

    def clear_invalid_cache(self) -> int:
        """清理无效的缓存文件

        Returns:
            清理的文件数量
        """
        import json
        cleared_count = 0

        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.startswith('.'):
                continue

            if symbol_dir.is_dir():
                for cache_file in symbol_dir.glob("*.parquet"):
                    try:
                        # 检查元数据文件
                        metadata_file = cache_file.with_suffix('.meta.json')
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)

                            # 检查版本
                            if metadata.get('cache_version') != self.cache_version:
                                cache_file.unlink(missing_ok=True)
                                metadata_file.unlink(missing_ok=True)
                                cleared_count += 1
                                logger.debug(f"Removed outdated cache: {cache_file}")
                                continue

                        # 尝试读取文件验证完整性
                        try:
                            df = pd.read_parquet(cache_file)
                            if df.empty or 'value' not in df.columns:
                                raise ValueError("Invalid format")
                        except Exception:
                            # 文件损坏，删除
                            cache_file.unlink(missing_ok=True)
                            metadata_file.unlink(missing_ok=True)
                            cleared_count += 1
                            logger.debug(f"Removed corrupted cache: {cache_file}")

                    except Exception as e:
                        logger.debug(f"Error checking cache file {cache_file}: {e}")
                        continue

        if cleared_count > 0:
            logger.info(f"Cleaned {cleared_count} invalid cache files")
        else:
            logger.debug("No invalid cache files found")

        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        import json
        stats = {
            'cache_dir': str(self.cache_dir),
            'version': self.cache_version,
            'symbols': 0,
            'features': 0,
            'total_size_mb': 0,
            'invalid_files': 0
        }

        for symbol_dir in self.cache_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name.startswith('.'):
                continue

            if symbol_dir.is_dir():
                stats['symbols'] += 1
                for cache_file in symbol_dir.glob("*.parquet"):
                    try:
                        stats['features'] += 1
                        stats['total_size_mb'] += cache_file.stat().st_size / (1024 * 1024)

                        # 检查文件有效性
                        metadata_file = cache_file.with_suffix('.meta.json')
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            if metadata.get('cache_version') != self.cache_version:
                                stats['invalid_files'] += 1
                    except:
                        stats['invalid_files'] += 1

        return stats