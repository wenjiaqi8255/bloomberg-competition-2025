"""
阶段0：Residual Momentum 快速验证脚本

目标：验证 residual momentum 是否比当前 alpha 方法更有效

方法：
1. 手动加载factor data和stock returns
2. 对几只股票做time-series regression
3. 计算residuals
4. 计算residual momentum信号
5. 对比用alpha vs residual momentum的IC
6. 输出对比表格

使用方法：
    python validate_residual_momentum.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.trading_system.utils.alpha_stats import compute_alpha_tstat
from src.trading_system.data.ff5_provider import FF5DataProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResidualMomentumValidator:
    """
    验证Residual Momentum效果的独立脚本
    """
    
    def __init__(self, 
                 factor_data_path: str = None,
                 formation_period: int = 252,
                 skip_recent_days: int = 21,
                 forward_lookback_days: int = 21):
        """
        Args:
            factor_data_path: FF5 factor data CSV路径
            formation_period: Residual momentum formation period (days)
            skip_recent_days: 跳过最近N天（避免短期反转）
            forward_lookback_days: 计算IC时使用的forward return窗口
        """
        self.formation_period = formation_period
        self.skip_recent_days = skip_recent_days
        self.forward_lookback_days = forward_lookback_days
        
        # 加载factor data
        if factor_data_path is None:
            # 尝试多个可能的路径
            possible_paths = [
                project_root / "data" / "ff5_factors_processed.csv",
                project_root / "src" / "trading_system" / "data" / "ff5_factors_processed.csv",
            ]
            for path in possible_paths:
                if path.exists():
                    factor_data_path = str(path)
                    break
        
        if factor_data_path and os.path.exists(factor_data_path):
            logger.info(f"Loading factor data from: {factor_data_path}")
            self.factor_data = pd.read_csv(factor_data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded factor data: {self.factor_data.shape}")
        else:
            # 尝试使用FF5DataProvider
            logger.info("Trying to load factor data via FF5DataProvider...")
            try:
                # 尝试所有可能的路径
                for path in possible_paths:
                    if path.exists():
                        provider = FF5DataProvider(file_path=str(path))
                        self.factor_data = provider.get_factor_returns()
                        logger.info(f"Loaded factor data via provider: {self.factor_data.shape}")
                        break
                else:
                    # 如果没有找到文件，尝试从网络获取
                    logger.warning("No local factor data file found, trying to fetch from network...")
                    provider = FF5DataProvider()
                    self.factor_data = provider.get_factor_returns()
                    logger.info(f"Loaded factor data from network: {self.factor_data.shape}")
            except Exception as e:
                logger.error(f"Failed to load factor data: {e}")
                raise
        
        # 验证factor data
        required_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        missing = set(required_cols) - set(self.factor_data.columns)
        if missing:
            raise ValueError(f"Missing factor columns: {missing}")
        
        logger.info(f"Factor data date range: {self.factor_data.index.min()} to {self.factor_data.index.max()}")
    
    def load_stock_returns(self, symbols: List[str], 
                          start_date: datetime, 
                          end_date: datetime) -> Dict[str, pd.Series]:
        """
        加载股票收益数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Dict[symbol, returns_series]
        """
        logger.info(f"Loading stock returns for {len(symbols)} symbols...")
        
        try:
            from src.trading_system.data.yfinance_provider import YFinanceProvider
            
            provider = YFinanceProvider()
            price_data = provider.get_historical_data(symbols, start_date, end_date)
            
            returns_dict = {}
            for symbol in symbols:
                if symbol in price_data and 'Close' in price_data[symbol].columns:
                    prices = price_data[symbol]['Close']
                    returns = prices.pct_change().dropna()
                    returns_dict[symbol] = returns
            
            logger.info(f"Loaded returns for {len(returns_dict)} symbols")
            return returns_dict
            
        except Exception as e:
            logger.error(f"Failed to load stock returns: {e}")
            logger.info("Trying alternative method...")
            
            # 如果YFinanceProvider失败，尝试从CSV加载
            # 这里可以扩展支持其他数据源
            raise
    
    def fit_time_series_regression(self, 
                                   returns: pd.Series,
                                   factors: pd.DataFrame,
                                   required_factors: List[str] = None) -> Dict:
        """
        对单只股票进行time-series regression
        
        Args:
            returns: 股票收益时间序列（excess returns）
            factors: 因子数据DataFrame
            required_factors: 需要的因子列
        
        Returns:
            Dict with 'alpha', 'betas', 'residuals', 'fitted_values'
        """
        if required_factors is None:
            required_factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        
        # 对齐数据
        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) < 50:
            logger.warning(f"Insufficient common dates: {len(common_dates)}")
            return None
        
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates][required_factors]
        
        # 计算excess returns（如果returns不是excess returns）
        if 'RF' in factors.columns:
            risk_free = factors.loc[common_dates]['RF']
            returns_aligned = returns_aligned - risk_free
        
        # 去除NaN
        valid_mask = ~(returns_aligned.isna() | factors_aligned.isna().any(axis=1))
        returns_clean = returns_aligned[valid_mask]
        factors_clean = factors_aligned[valid_mask]
        
        if len(returns_clean) < 50:
            logger.warning(f"Insufficient clean data: {len(returns_clean)}")
            return None
        
        # 回归
        X = factors_clean.values
        y = returns_clean.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算fitted values和residuals
        fitted_values = model.predict(X)
        residuals = y - fitted_values
        
        # 存储为Series（保留时间索引）
        residuals_series = pd.Series(residuals, index=returns_clean.index)
        fitted_series = pd.Series(fitted_values, index=returns_clean.index)
        
        return {
            'alpha': float(model.intercept_),
            'betas': dict(zip(required_factors, model.coef_)),
            'residuals': residuals_series,
            'fitted_values': fitted_series,
            'r_squared': float(model.score(X, y)),
            'n_obs': len(returns_clean)
        }
    
    def calculate_residual_momentum(self, 
                                   residuals: pd.Series,
                                   current_date: datetime) -> float:
        """
        计算residual momentum信号
        
        Args:
            residuals: Residuals时间序列
            current_date: 当前日期（只使用<=current_date的数据）
        
        Returns:
            Standardized residual momentum
        """
        # 过滤到当前日期
        historical_residuals = residuals[residuals.index <= current_date]
        
        if len(historical_residuals) < self.formation_period + self.skip_recent_days:
            return 0.0
        
        # 跳过最近的日期
        lookback_data = historical_residuals.iloc[:-self.skip_recent_days]
        
        # 取formation period
        formation_residuals = lookback_data.iloc[-self.formation_period:]
        
        if len(formation_residuals) == 0:
            return 0.0
        
        # 计算momentum（sum）
        momentum = formation_residuals.sum()
        
        # 标准化（除以标准差）
        volatility = formation_residuals.std()
        if volatility > 0:
            standardized_momentum = momentum / volatility
        else:
            standardized_momentum = 0.0
        
        return standardized_momentum
    
    def calculate_forward_returns(self, 
                                  returns: pd.Series,
                                  date: datetime) -> float:
        """
        计算forward return（用于IC计算）
        
        Args:
            returns: 股票收益时间序列
            date: 信号日期
        
        Returns:
            Forward return (未来N天的累计收益)
        """
        # 确保date是datetime类型
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        
        # 找到date之后的数据
        future_dates = returns[returns.index > date]
        
        if len(future_dates) < self.forward_lookback_days:
            return np.nan
        
        # 取前N天的数据
        forward_data = future_dates.iloc[:self.forward_lookback_days]
        
        # 检查是否有NaN
        if forward_data.isna().any():
            return np.nan
        
        # 计算未来N天的累计收益
        try:
            forward_returns = (1 + forward_data).prod() - 1
            return float(forward_returns)
        except Exception:
            return np.nan
    
    def calculate_ic(self, 
                     signals: pd.Series,
                     forward_returns: pd.Series) -> Dict[str, float]:
        """
        计算Information Coefficient (IC)
        
        Args:
            signals: 信号Series（index是日期，values是信号值）
            forward_returns: Forward returns Series（index是日期，values是forward return）
        
        Returns:
            Dict with IC metrics
        """
        # 对齐数据
        common_dates = signals.index.intersection(forward_returns.index)
        if len(common_dates) < 10:
            return {
                'mean_ic': 0.0,
                'ic_std': 0.0,
                'ic_sharpe': 0.0,
                'positive_ic_ratio': 0.0,
                'n_obs': len(common_dates)
            }
        
        signals_aligned = signals.loc[common_dates]
        returns_aligned = forward_returns.loc[common_dates]
        
        # 去除NaN
        valid_mask = ~(signals_aligned.isna() | returns_aligned.isna())
        signals_clean = signals_aligned[valid_mask]
        returns_clean = returns_aligned[valid_mask]
        
        if len(signals_clean) < 10:
            return {
                'mean_ic': 0.0,
                'ic_std': 0.0,
                'ic_sharpe': 0.0,
                'positive_ic_ratio': 0.0,
                'n_obs': len(signals_clean)
            }
        
        # 计算IC（Pearson correlation）
        ic = signals_clean.corr(returns_clean)
        
        # 如果IC是NaN，返回0
        if pd.isna(ic):
            ic = 0.0
        
        return {
            'mean_ic': float(ic),
            'ic_std': 0.0,  # 单只股票的IC没有std
            'ic_sharpe': 0.0,
            'positive_ic_ratio': 1.0 if ic > 0 else 0.0,
            'n_obs': len(signals_clean)
        }
    
    def validate_cross_sectional(self, 
                                 symbols: List[str],
                                 start_date: datetime,
                                 end_date: datetime,
                                 train_start: datetime = None,
                                 train_end: datetime = None) -> pd.DataFrame:
        """
        横截面验证：对比alpha vs residual momentum的IC
        
        Args:
            symbols: 股票代码列表
            start_date: 验证开始日期
            end_date: 验证结束日期
            train_start: 训练开始日期（如果None，使用start_date前1年）
            train_end: 训练结束日期（如果None，使用start_date）
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("=" * 80)
        logger.info("Starting Cross-Sectional Validation")
        logger.info("=" * 80)
        
        # 设置训练期
        if train_start is None:
            train_start = start_date - timedelta(days=365)
        if train_end is None:
            train_end = start_date
        
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Validation period: {start_date} to {end_date}")
        
        # 加载股票收益数据
        all_returns = self.load_stock_returns(symbols, train_start, end_date)
        
        if len(all_returns) == 0:
            raise ValueError("No stock returns loaded")
        
        # 准备因子数据
        factor_train = self.factor_data[
            (self.factor_data.index >= train_start) & 
            (self.factor_data.index <= train_end)
        ]
        factor_val = self.factor_data[
            (self.factor_data.index >= start_date) & 
            (self.factor_data.index <= end_date)
        ]
        
        # Step 1: 训练期回归，获取alpha和residuals
        # 注意：为了计算验证期的momentum，我们需要在验证期也计算residuals
        # 但使用训练期估计的betas和alpha
        logger.info("\nStep 1: Fitting time-series regressions...")
        regression_results = {}
        
        # 首先在训练期拟合模型
        for symbol in all_returns.keys():
            returns = all_returns[symbol]
            returns_train = returns[
                (returns.index >= train_start) & 
                (returns.index <= train_end)
            ]
            
            if len(returns_train) < 50:
                logger.warning(f"Insufficient training data for {symbol}: {len(returns_train)}")
                continue
            
            result = self.fit_time_series_regression(returns_train, factor_train)
            if result is not None:
                regression_results[symbol] = result
                logger.info(f"  {symbol}: alpha={result['alpha']:.6f}, R²={result['r_squared']:.3f}, n_obs={result['n_obs']}")
        
        logger.info(f"Successfully fitted {len(regression_results)} symbols")
        
        # Step 1.5: 扩展residuals到验证期（使用训练期的betas计算验证期的residuals）
        logger.info("\nStep 1.5: Computing residuals for validation period...")
        factor_val_aligned = factor_val.copy()
        
        for symbol in regression_results.keys():
            returns = all_returns[symbol]
            returns_val = returns[
                (returns.index >= start_date) & 
                (returns.index <= end_date)
            ]
            
            if len(returns_val) == 0:
                continue
            
            # 对齐因子数据和收益数据
            common_dates = returns_val.index.intersection(factor_val_aligned.index)
            if len(common_dates) == 0:
                continue
            
            returns_aligned = returns_val.loc[common_dates]
            factors_aligned = factor_val_aligned.loc[common_dates]
            
            # 计算excess returns
            if 'RF' in factors_aligned.columns:
                risk_free = factors_aligned['RF']
                returns_excess = returns_aligned - risk_free
            else:
                returns_excess = returns_aligned
            
            # 使用训练期的betas和alpha计算fitted values
            betas = regression_results[symbol]['betas']
            alpha = regression_results[symbol]['alpha']
            
            # 确保因子顺序与betas一致
            factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            factor_values = factors_aligned[factor_cols].values
            
            # betas是字典，需要按顺序提取
            beta_array = np.array([betas[col] for col in factor_cols])
            
            # 计算fitted values: alpha + beta @ factors
            fitted_values = alpha + np.dot(factor_values, beta_array)
            fitted_series = pd.Series(fitted_values, index=common_dates)
            
            # 计算residuals
            residuals_val = returns_excess - fitted_series
            
            # 合并训练期和验证期的residuals
            residuals_train = regression_results[symbol]['residuals']
            residuals_combined = pd.concat([residuals_train, residuals_val]).sort_index()
            
            # 更新regression_results中的residuals
            regression_results[symbol]['residuals'] = residuals_combined
            
        logger.info("Extended residuals to validation period")
        
        if len(regression_results) == 0:
            raise ValueError("No successful regressions")
        
        # Step 2: 在验证期计算信号和forward returns
        logger.info("\nStep 2: Computing signals and forward returns...")
        
        # 生成rebalance dates（每月一次）
        # 使用'ME'替代已弃用的'M'
        try:
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        except:
            # 兼容旧版本pandas
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        logger.info(f"Rebalance dates: {len(rebalance_dates)}")
        if len(rebalance_dates) > 0:
            logger.info(f"  First rebalance: {rebalance_dates[0]}")
            logger.info(f"  Last rebalance: {rebalance_dates[-1]}")
        
        # 存储信号和forward returns
        alpha_signals = {}  # {date: {symbol: signal}}
        expected_return_signals = {}  # {date: {symbol: signal}} - 新增
        momentum_signals = {}  # {date: {symbol: signal}}
        forward_returns_dict = {}  # {date: {symbol: forward_return}}
        
        forward_returns_stats = {'total': 0, 'valid': 0, 'nan': 0}
        
        for date in rebalance_dates:
            alpha_signals[date] = {}
            expected_return_signals[date] = {}
            momentum_signals[date] = {}
            forward_returns_dict[date] = {}
            
            # 获取当前日期的因子值（用于计算expected return）
            if date in factor_val.index:
                current_factors = factor_val.loc[date]
            else:
                # 如果精确日期不存在，找最近的
                available_dates = factor_val.index[factor_val.index <= date]
                if len(available_dates) > 0:
                    current_factors = factor_val.loc[available_dates.max()]
                else:
                    current_factors = None
            
            for symbol in regression_results.keys():
                # Alpha信号（静态）
                alpha_signals[date][symbol] = regression_results[symbol]['alpha']
                
                # Expected Return信号（alpha + beta × factors）
                if current_factors is not None:
                    betas = regression_results[symbol]['betas']
                    alpha = regression_results[symbol]['alpha']
                    
                    # 计算expected return: alpha + beta @ factors
                    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
                    factor_values = np.array([current_factors[col] for col in factor_cols])
                    beta_array = np.array([betas[col] for col in factor_cols])
                    
                    expected_return = alpha + np.dot(beta_array, factor_values)
                    expected_return_signals[date][symbol] = expected_return
                else:
                    expected_return_signals[date][symbol] = np.nan
                
                # Residual momentum信号（动态）
                residuals = regression_results[symbol]['residuals']
                momentum_signal = self.calculate_residual_momentum(residuals, date)
                momentum_signals[date][symbol] = momentum_signal
                
                # Forward returns
                returns = all_returns[symbol]
                forward_ret = self.calculate_forward_returns(returns, date)
                forward_returns_dict[date][symbol] = forward_ret
                
                forward_returns_stats['total'] += 1
                if pd.isna(forward_ret):
                    forward_returns_stats['nan'] += 1
                else:
                    forward_returns_stats['valid'] += 1
        
        logger.info(f"Forward returns stats: {forward_returns_stats['valid']}/{forward_returns_stats['total']} valid, {forward_returns_stats['nan']} NaN")
        
        # Step 3: 计算横截面IC
        logger.info("\nStep 3: Calculating cross-sectional IC...")
        
        ic_results = []
        skipped_dates = []
        
        for date in rebalance_dates:
            # Alpha信号
            alpha_sig = pd.Series(alpha_signals[date])
            # Expected Return信号
            expected_return_sig = pd.Series(expected_return_signals[date])
            # Momentum信号
            momentum_sig = pd.Series(momentum_signals[date])
            # Forward returns
            forward_ret = pd.Series(forward_returns_dict[date])
            
            # 去除NaN
            common_symbols = alpha_sig.index.intersection(
                expected_return_sig.index
            ).intersection(momentum_sig.index).intersection(forward_ret.index)
            
            if len(common_symbols) < 5:
                skipped_dates.append((date, f"insufficient common symbols: {len(common_symbols)}"))
                continue
            
            alpha_sig_clean = alpha_sig.loc[common_symbols]
            expected_return_sig_clean = expected_return_sig.loc[common_symbols]
            momentum_sig_clean = momentum_sig.loc[common_symbols]
            forward_ret_clean = forward_ret.loc[common_symbols].dropna()
            
            if len(forward_ret_clean) < 5:
                skipped_dates.append((date, f"insufficient valid forward returns: {len(forward_ret_clean)}"))
                continue
            
            # 确保信号和forward returns对齐
            final_symbols = alpha_sig_clean.index.intersection(forward_ret_clean.index)
            if len(final_symbols) < 5:
                skipped_dates.append((date, f"insufficient aligned symbols: {len(final_symbols)}"))
                continue
            
            alpha_sig_final = alpha_sig_clean.loc[final_symbols]
            expected_return_sig_final = expected_return_sig_clean.loc[final_symbols]
            momentum_sig_final = momentum_sig_clean.loc[final_symbols]
            forward_ret_final = forward_ret_clean.loc[final_symbols]
            
            # 计算IC
            try:
                # 检查信号是否有足够的variation
                alpha_std = alpha_sig_final.std()
                expected_return_std = expected_return_sig_final.std()
                momentum_std = momentum_sig_final.std()
                
                if alpha_std == 0:
                    skipped_dates.append((date, f"alpha signal has zero variance"))
                    continue
                
                if expected_return_std == 0:
                    skipped_dates.append((date, f"expected return signal has zero variance"))
                    continue
                
                if momentum_std == 0:
                    skipped_dates.append((date, f"momentum signal has zero variance"))
                    continue
                
                # 检查是否有NaN
                if (alpha_sig_final.isna().any() or 
                    expected_return_sig_final.isna().any() or 
                    momentum_sig_final.isna().any()):
                    skipped_dates.append((date, f"signals contain NaN"))
                    continue
                
                alpha_ic = alpha_sig_final.corr(forward_ret_final)
                expected_return_ic = expected_return_sig_final.corr(forward_ret_final)
                momentum_ic = momentum_sig_final.corr(forward_ret_final)
                
                if (not pd.isna(alpha_ic) and 
                    not pd.isna(expected_return_ic) and 
                    not pd.isna(momentum_ic)):
                    ic_results.append({
                        'date': date,
                        'alpha_ic': alpha_ic,
                        'expected_return_ic': expected_return_ic,
                        'momentum_ic': momentum_ic,
                        'n_stocks': len(forward_ret_final)
                    })
                    logger.debug(f"  {date}: alpha_ic={alpha_ic:.4f}, expected_return_ic={expected_return_ic:.4f}, momentum_ic={momentum_ic:.4f}, n={len(forward_ret_final)}")
                else:
                    # 添加更详细的调试信息
                    logger.debug(f"  {date}: alpha_ic={alpha_ic}, expected_return_ic={expected_return_ic}, momentum_ic={momentum_ic}")
                    logger.debug(f"    alpha_sig: mean={alpha_sig_final.mean():.6f}, std={alpha_std:.6f}")
                    logger.debug(f"    expected_return_sig: mean={expected_return_sig_final.mean():.6f}, std={expected_return_std:.6f}")
                    logger.debug(f"    momentum_sig: mean={momentum_sig_final.mean():.6f}, std={momentum_std:.6f}")
                    skipped_dates.append((date, f"IC is NaN: alpha={alpha_ic}, expected_return={expected_return_ic}, momentum={momentum_ic}"))
            except Exception as e:
                skipped_dates.append((date, f"correlation error: {e}"))
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        if len(ic_results) == 0:
            logger.error(f"No valid IC calculations. Skipped {len(skipped_dates)} dates:")
            for date, reason in skipped_dates[:5]:  # 只显示前5个
                logger.error(f"  {date}: {reason}")
            if len(skipped_dates) > 5:
                logger.error(f"  ... and {len(skipped_dates) - 5} more")
            raise ValueError(f"No valid IC calculations. All {len(rebalance_dates)} rebalance dates were skipped.")
        
        logger.info(f"Successfully calculated IC for {len(ic_results)}/{len(rebalance_dates)} dates")
        
        ic_df = pd.DataFrame(ic_results)
        
        # Step 4: 汇总统计
        logger.info("\nStep 4: Summary Statistics")
        logger.info("=" * 80)
        
        alpha_ic_mean = ic_df['alpha_ic'].mean()
        alpha_ic_std = ic_df['alpha_ic'].std()
        alpha_ic_sharpe = alpha_ic_mean / alpha_ic_std if alpha_ic_std > 0 else 0.0
        alpha_positive_ratio = (ic_df['alpha_ic'] > 0).mean()
        
        expected_return_ic_mean = ic_df['expected_return_ic'].mean()
        expected_return_ic_std = ic_df['expected_return_ic'].std()
        expected_return_ic_sharpe = expected_return_ic_mean / expected_return_ic_std if expected_return_ic_std > 0 else 0.0
        expected_return_positive_ratio = (ic_df['expected_return_ic'] > 0).mean()
        
        momentum_ic_mean = ic_df['momentum_ic'].mean()
        momentum_ic_std = ic_df['momentum_ic'].std()
        momentum_ic_sharpe = momentum_ic_mean / momentum_ic_std if momentum_ic_std > 0 else 0.0
        momentum_positive_ratio = (ic_df['momentum_ic'] > 0).mean()
        
        # 输出对比表（三列对比）
        comparison = pd.DataFrame({
            'Metric': [
                'Mean IC',
                'IC Std',
                'IC Sharpe',
                'Positive IC Ratio',
                'N Observations'
            ],
            'Alpha (Intercept Only)': [
                f"{alpha_ic_mean:.4f}",
                f"{alpha_ic_std:.4f}",
                f"{alpha_ic_sharpe:.4f}",
                f"{alpha_positive_ratio:.2%}",
                f"{len(ic_df)}"
            ],
            'Expected Return (Alpha + Beta×Factors)': [
                f"{expected_return_ic_mean:.4f}",
                f"{expected_return_ic_std:.4f}",
                f"{expected_return_ic_sharpe:.4f}",
                f"{expected_return_positive_ratio:.2%}",
                f"{len(ic_df)}"
            ],
            'Residual Momentum': [
                f"{momentum_ic_mean:.4f}",
                f"{momentum_ic_std:.4f}",
                f"{momentum_ic_sharpe:.4f}",
                f"{momentum_positive_ratio:.2%}",
                f"{len(ic_df)}"
            ]
        })
        
        print("\n" + "=" * 100)
        print("VALIDATION RESULTS: Alpha vs Expected Return vs Residual Momentum")
        print("=" * 100)
        print(comparison.to_string(index=False))
        print("=" * 100)
        
        # 添加解释说明
        print("\n📊 Signal Explanation:")
        print("  • Alpha (Intercept Only): Uses only the regression intercept as signal")
        print("    → Ignores factor exposures, focuses on unexplained returns")
        print("  • Expected Return (Alpha + Beta×Factors): Uses full factor model prediction")
        print("    → Considers both intercept and factor loadings × current factor values")
        print("  • Residual Momentum: Uses past residuals' momentum as signal")
        print("    → Captures firm-specific momentum after controlling for factors")
        print()
        
        # 保存结果
        output_dir = project_root / "validation_results"
        output_dir.mkdir(exist_ok=True)
        
        ic_df.to_csv(output_dir / "ic_comparison.csv", index=False)
        comparison.to_csv(output_dir / "summary_comparison.csv", index=False)
        
        logger.info(f"\nResults saved to: {output_dir}")
        
        return comparison


def main():
    """主函数"""
    # 配置参数
    # 使用一些常见的股票代码（可以根据实际情况修改）
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    # 日期范围
    # 使用最近3年数据：训练2年，验证1年
    # 注意：需要留出足够的时间用于forward returns计算
    end_date = datetime.now() - timedelta(days=60)  # 使用60天前作为结束日期，确保有足够数据
    validation_start = end_date - timedelta(days=365)
    validation_end = end_date - timedelta(days=30)  # 留出30天用于forward returns
    
    # 创建验证器
    validator = ResidualMomentumValidator(
        formation_period=252,  # 12个月
        skip_recent_days=21,   # 跳过最近1个月
        forward_lookback_days=21  # 21天forward return
    )
    
    # 运行验证
    try:
        results = validator.validate_cross_sectional(
            symbols=symbols,
            start_date=validation_start,
            end_date=validation_end
        )
        
        print("\n✅ Validation completed successfully!")
        print("\n📈 Recommendation:")
        
        # 获取三个信号的IC
        alpha_ic = float(results[results['Metric'] == 'Mean IC']['Alpha (Intercept Only)'].values[0])
        expected_return_ic = float(results[results['Metric'] == 'Mean IC']['Expected Return (Alpha + Beta×Factors)'].values[0])
        momentum_ic = float(results[results['Metric'] == 'Mean IC']['Residual Momentum'].values[0])
        
        # 找出最佳信号
        best_signal = max([
            ('Alpha', alpha_ic),
            ('Expected Return', expected_return_ic),
            ('Residual Momentum', momentum_ic)
        ], key=lambda x: x[1])
        
        print(f"  → Best performing signal: {best_signal[0]} (IC = {best_signal[1]:.4f})")
        print()
        
        # 详细对比
        if momentum_ic > max(alpha_ic, expected_return_ic) + 0.01:
            print("  → Residual Momentum shows significant improvement over both Alpha and Expected Return.")
            print("    Proceed to Stage 1 implementation.")
        elif expected_return_ic > alpha_ic + 0.01:
            print("  → Expected Return (factor-aware) outperforms Alpha (intercept-only).")
            print("    This suggests factor exposures matter for prediction.")
        elif momentum_ic > max(alpha_ic, expected_return_ic):
            print("  → Residual Momentum shows marginal improvement. Consider proceeding to Stage 1.")
        else:
            print("  → Current Alpha or Expected Return method performs best.")
            print("    Residual Momentum may need parameter tuning or different universe.")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

