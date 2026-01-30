#!/usr/bin/env python3
"""
Beta异常诊断工具

深入调查Beta=83.48异常高的原因
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_beta_calculation(portfolio_returns: pd.Series, 
                             benchmark_returns: pd.Series,
                             experiment_id: str = "unknown") -> dict:
    """
    诊断Beta计算异常的原因
    
    Args:
        portfolio_returns: 组合收益率序列
        benchmark_returns: 基准收益率序列
        experiment_id: 实验ID（用于日志）
    
    Returns:
        诊断结果字典
    """
    results = {
        'experiment_id': experiment_id,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 1. 检查数据对齐
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_dates) < len(portfolio_returns) * 0.8:
        results['warnings'].append(
            f"数据对齐问题：只有{len(common_dates)}/{len(portfolio_returns)}个日期对齐"
        )
    
    if len(common_dates) < 2:
        results['issues'].append("数据对齐失败：共同日期少于2个")
        return results
    
    portfolio_aligned = portfolio_returns.loc[common_dates]
    benchmark_aligned = benchmark_returns.loc[common_dates]
    
    # 2. 检查数据单位（百分比 vs 小数）
    portfolio_abs_max = portfolio_aligned.abs().max()
    benchmark_abs_max = benchmark_aligned.abs().max()
    
    if portfolio_abs_max > 10 or benchmark_abs_max > 10:
        results['issues'].append(
            f"可能的单位问题：组合收益率最大绝对值={portfolio_abs_max:.4f}, "
            f"基准收益率最大绝对值={benchmark_abs_max:.4f} "
            f"(正常日收益率应在[-0.2, 0.2]范围内)"
        )
    
    # 3. 检查异常值
    portfolio_q99 = portfolio_aligned.quantile(0.99)
    portfolio_q01 = portfolio_aligned.quantile(0.01)
    benchmark_q99 = benchmark_aligned.quantile(0.99)
    benchmark_q01 = benchmark_aligned.quantile(0.01)
    
    if abs(portfolio_q99) > 0.2 or abs(portfolio_q01) > 0.2:
        results['warnings'].append(
            f"组合收益率异常值：99%分位数={portfolio_q99:.4f}, 1%分位数={portfolio_q01:.4f}"
        )
    
    if abs(benchmark_q99) > 0.2 or abs(benchmark_q01) > 0.2:
        results['warnings'].append(
            f"基准收益率异常值：99%分位数={benchmark_q99:.4f}, 1%分位数={benchmark_q01:.4f}"
        )
    
    # 4. 计算Beta和相关统计量
    covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
    benchmark_variance = np.var(benchmark_aligned)
    portfolio_variance = np.var(portfolio_aligned)
    
    if benchmark_variance == 0:
        results['issues'].append("基准收益率方差为0，无法计算Beta")
        return results
    
    beta = covariance / benchmark_variance
    
    # 5. 检查Beta异常的原因
    portfolio_std = np.std(portfolio_aligned)
    benchmark_std = np.std(benchmark_aligned)
    
    if portfolio_std > benchmark_std * 10:
        results['issues'].append(
            f"组合收益率波动异常高：组合标准差={portfolio_std:.6f}, "
            f"基准标准差={benchmark_std:.6f}, 比率={portfolio_std/benchmark_std:.2f}"
        )
    
    # 6. 检查相关性
    correlation = np.corrcoef(portfolio_aligned, benchmark_aligned)[0, 1]
    
    if abs(correlation) > 0.99:
        results['warnings'].append(
            f"组合与基准高度相关：相关系数={correlation:.4f} "
            f"(如果组合收益率是基准的倍数，会导致Beta异常高)"
        )
    
    # 7. 检查是否有倍数关系
    # 如果组合收益率 ≈ k × 基准收益率，则Beta ≈ k
    if abs(correlation) > 0.8:
        # 尝试线性回归：portfolio = k * benchmark + c
        from sklearn.linear_model import LinearRegression
        X = benchmark_aligned.values.reshape(-1, 1)
        y = portfolio_aligned.values
        reg = LinearRegression().fit(X, y)
        k = reg.coef_[0]
        c = reg.intercept_
        r2 = reg.score(X, y)
        
        if abs(k) > 10:
            results['issues'].append(
                f"发现倍数关系：组合收益率 ≈ {k:.2f} × 基准收益率 + {c:.4f} "
                f"(R²={r2:.4f})，这会导致Beta={k:.2f}"
            )
        
        if abs(c) > 0.01:
            results['warnings'].append(
                f"线性回归截距较大：{c:.4f} (可能表明组合收益率计算有问题)"
            )
    
    # 8. 统计信息
    results['statistics'] = {
        'n_observations': len(common_dates),
        'portfolio_mean': float(portfolio_aligned.mean()),
        'portfolio_std': float(portfolio_std),
        'portfolio_min': float(portfolio_aligned.min()),
        'portfolio_max': float(portfolio_aligned.max()),
        'benchmark_mean': float(benchmark_aligned.mean()),
        'benchmark_std': float(benchmark_std),
        'benchmark_min': float(benchmark_aligned.min()),
        'benchmark_max': float(benchmark_aligned.max()),
        'covariance': float(covariance),
        'benchmark_variance': float(benchmark_variance),
        'correlation': float(correlation),
        'beta': float(beta)
    }
    
    return results


def check_returns_calculation_method(portfolio_values: pd.Series) -> dict:
    """
    检查组合收益率计算方法是否正确
    
    Args:
        portfolio_values: 组合价值序列
    
    Returns:
        检查结果
    """
    results = {
        'method': 'unknown',
        'issues': [],
        'warnings': []
    }
    
    if len(portfolio_values) < 2:
        results['issues'].append("数据点不足")
        return results
    
    # 方法1：简单收益率 (正确方法)
    simple_returns = portfolio_values.pct_change().dropna()
    
    # 方法2：对数收益率
    log_returns = np.log(portfolio_values / portfolio_values.shift(1)).dropna()
    
    # 方法3：累计收益率（错误方法）
    cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1)
    
    # 检查是否有使用累计收益率的迹象
    if cumulative_returns.max() > 10:
        results['warnings'].append(
            f"累计收益率异常高：最大值={cumulative_returns.max():.2f} "
            f"(如果使用累计收益率计算Beta，会导致异常高的Beta)"
        )
    
    # 检查简单收益率的合理性
    if simple_returns.abs().max() > 1.0:
        results['warnings'].append(
            f"简单收益率异常：最大绝对值={simple_returns.abs().max():.4f} "
            f"(正常日收益率应在[-0.2, 0.2]范围内)"
        )
    
    results['method'] = 'simple_returns'
    results['statistics'] = {
        'simple_returns_mean': float(simple_returns.mean()),
        'simple_returns_std': float(simple_returns.std()),
        'simple_returns_min': float(simple_returns.min()),
        'simple_returns_max': float(simple_returns.max())
    }
    
    return results


def main():
    """主函数：诊断Beta异常"""
    print("=" * 80)
    print("Beta异常诊断工具")
    print("=" * 80)
    
    # 加载实际实验数据
    portfolio_returns_path = "results/ff5_regression_20251107_012512/strategy_returns.csv"
    benchmark_path = "data/universes/wls_index.csv"
    
    print(f"\n加载组合收益率数据: {portfolio_returns_path}")
    try:
        portfolio_df = pd.read_csv(portfolio_returns_path, index_col=0, parse_dates=True)
        print(f"  列名: {list(portfolio_df.columns)}")
        print(f"  数据形状: {portfolio_df.shape}")
        print(f"  日期范围: {portfolio_df.index.min()} 到 {portfolio_df.index.max()}")
        
        # 尝试找到收益率列
        if 'daily_return' in portfolio_df.columns:
            portfolio_returns = portfolio_df['daily_return']
        elif 'returns' in portfolio_df.columns:
            portfolio_returns = portfolio_df['returns']
        elif 'return' in portfolio_df.columns:
            portfolio_returns = portfolio_df['return']
        elif len(portfolio_df.columns) == 1:
            portfolio_returns = portfolio_df.iloc[:, 0]
        else:
            print(f"  警告：无法确定收益率列，使用第一列")
            portfolio_returns = portfolio_df.iloc[:, 0]
        
        print(f"  收益率统计: 均值={portfolio_returns.mean():.6f}, 标准差={portfolio_returns.std():.6f}")
        print(f"  收益率范围: [{portfolio_returns.min():.6f}, {portfolio_returns.max():.6f}]")
        
    except Exception as e:
        print(f"  错误：无法加载组合收益率数据: {e}")
        return
    
    print(f"\n加载基准数据: {benchmark_path}")
    try:
        benchmark_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        print(f"  列名: {list(benchmark_df.columns)}")
        print(f"  数据形状: {benchmark_df.shape}")
        print(f"  日期范围: {benchmark_df.index.min()} 到 {benchmark_df.index.max()}")
        
        # 尝试找到价格列（用于计算收益率）
        if 'Close' in benchmark_df.columns:
            benchmark_prices = benchmark_df['Close']
        elif 'close' in benchmark_df.columns:
            benchmark_prices = benchmark_df['close']
        elif 'Price' in benchmark_df.columns:
            benchmark_prices = benchmark_df['Price']
        elif len(benchmark_df.columns) == 1:
            benchmark_prices = benchmark_df.iloc[:, 0]
        else:
            print(f"  警告：无法确定价格列，使用第一列")
            benchmark_prices = benchmark_df.iloc[:, 0]
        
        # 计算基准收益率
        benchmark_returns = benchmark_prices.pct_change().dropna()
        print(f"  基准收益率统计: 均值={benchmark_returns.mean():.6f}, 标准差={benchmark_returns.std():.6f}")
        print(f"  基准收益率范围: [{benchmark_returns.min():.6f}, {benchmark_returns.max():.6f}]")
        
    except Exception as e:
        print(f"  错误：无法加载基准数据: {e}")
        return
    
    # 运行诊断
    print("\n" + "=" * 80)
    print("运行Beta异常诊断...")
    print("=" * 80)
    
    results = diagnose_beta_calculation(
        portfolio_returns, 
        benchmark_returns, 
        experiment_id='ff5_regression_20251107_012512'
    )
    
    # 打印结果
    print("\n诊断结果:")
    print("-" * 80)
    print(f"实验ID: {results['experiment_id']}")
    
    if results['issues']:
        print("\n⚠️  发现的问题:")
        for i, issue in enumerate(results['issues'], 1):
            print(f"  {i}. {issue}")
    
    if results['warnings']:
        print("\n⚠️  警告:")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"  {i}. {warning}")
    
    if results['statistics']:
        print("\n📊 统计信息:")
        stats = results['statistics']
        print(f"  观测数量: {stats['n_observations']}")
        print(f"  组合收益率:")
        print(f"    均值: {stats['portfolio_mean']:.6f}")
        print(f"    标准差: {stats['portfolio_std']:.6f}")
        print(f"    范围: [{stats['portfolio_min']:.6f}, {stats['portfolio_max']:.6f}]")
        print(f"  基准收益率:")
        print(f"    均值: {stats['benchmark_mean']:.6f}")
        print(f"    标准差: {stats['benchmark_std']:.6f}")
        print(f"    范围: [{stats['benchmark_min']:.6f}, {stats['benchmark_max']:.6f}]")
        print(f"  协方差: {stats['covariance']:.8f}")
        print(f"  基准方差: {stats['benchmark_variance']:.8f}")
        print(f"  相关系数: {stats['correlation']:.6f}")
        print(f"  Beta: {stats['beta']:.2f}")
        
        if abs(stats['beta']) > 10:
            print(f"\n  ⚠️  Beta异常高！正常范围应在0-2之间")
            if abs(stats['correlation']) > 0.8:
                print(f"  💡 组合与基准高度相关（{stats['correlation']:.4f}），")
                print(f"     如果组合收益率是基准的倍数，会导致Beta={stats['beta']:.2f}")
    
    # 检查组合收益率计算方法
    print("\n" + "=" * 80)
    print("检查组合收益率计算方法...")
    print("=" * 80)
    
    # 尝试从strategy_returns中找到portfolio_value
    if 'portfolio_value' in portfolio_df.columns:
        portfolio_values = portfolio_df['portfolio_value']
        calc_check = check_returns_calculation_method(portfolio_values)
        print(f"\n计算方法检查结果:")
        if calc_check['issues']:
            for issue in calc_check['issues']:
                print(f"  ⚠️  {issue}")
        if calc_check['warnings']:
            for warning in calc_check['warnings']:
                print(f"  ⚠️  {warning}")
        if 'statistics' in calc_check:
            print(f"  简单收益率统计: 均值={calc_check['statistics']['simple_returns_mean']:.6f}, "
                  f"标准差={calc_check['statistics']['simple_returns_std']:.6f}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == '__main__':
    main()

