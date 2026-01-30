#!/usr/bin/env python3
"""
检查极端收益日的详细分析脚本

分析极端收益日（如2024-12-19和2024-12-20）的持仓、价格数据等，
帮助诊断回测负收益问题。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_extreme_return_days(returns_path: str, results_dir: str = None):
    """
    检查极端收益日的详细信息
    
    Args:
        returns_path: 策略收益CSV文件路径
        results_dir: 结果目录，用于查找其他相关文件
    """
    print("=" * 80)
    print("极端收益日详细分析")
    print("=" * 80)
    
    # 读取策略收益
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns_df.columns = ['daily_return']
    returns = returns_df['daily_return']
    
    # 找出极端收益日
    extreme_days = []
    threshold = 0.05  # 5%阈值
    
    for date, ret in returns.items():
        if abs(ret) > threshold:
            extreme_days.append((date, ret))
    
    extreme_days.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n找到 {len(extreme_days)} 个极端收益日 (阈值: ±{threshold*100:.0f}%)")
    print(f"\nTop 10 极端收益日:")
    for i, (date, ret) in enumerate(extreme_days[:10], 1):
        print(f"  {i:2d}. {date.strftime('%Y-%m-%d')}: {ret*100:7.2f}%")
    
    # 分析连续极端收益日
    print(f"\n连续极端收益日分析:")
    consecutive_groups = []
    current_group = [extreme_days[0]]
    
    for i in range(1, len(extreme_days)):
        date1, ret1 = extreme_days[i-1]
        date2, ret2 = extreme_days[i]
        days_diff = (date2 - date1).days
        
        if days_diff <= 3:  # 3天内连续出现
            current_group.append((date2, ret2))
        else:
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            current_group = [(date2, ret2)]
    
    if len(current_group) > 1:
        consecutive_groups.append(current_group)
    
    for group in consecutive_groups:
        print(f"\n  连续极端收益组 ({len(group)} 天):")
        cumulative_ret = 0
        for date, ret in group:
            cumulative_ret += ret
            print(f"    {date.strftime('%Y-%m-%d')}: {ret*100:7.2f}%")
        print(f"    累积收益: {cumulative_ret*100:7.2f}%")
    
    # 检查特定日期（2024-12-19和2024-12-20）
    target_dates = [
        datetime(2024, 12, 19),
        datetime(2024, 12, 20),
        datetime(2024, 8, 15),
    ]
    
    print(f"\n" + "=" * 80)
    print("特定日期详细分析")
    print("=" * 80)
    
    for target_date in target_dates:
        target_date_str = target_date.strftime('%Y-%m-%d')
        if target_date in returns.index:
            ret = returns[target_date]
            print(f"\n📅 {target_date_str}:")
            print(f"   日收益: {ret*100:.2f}%")
            
            # 计算前后几天的收益
            date_idx = returns.index.get_loc(target_date)
            window = 5
            start_idx = max(0, date_idx - window)
            end_idx = min(len(returns), date_idx + window + 1)
            window_returns = returns.iloc[start_idx:end_idx]
            
            print(f"   前后{window}天收益:")
            for date, ret in window_returns.items():
                marker = " <-- 目标日期" if date == target_date else ""
                print(f"     {date.strftime('%Y-%m-%d')}: {ret*100:7.2f}%{marker}")
            
            # 计算累积收益
            if date_idx > 0:
                prev_returns = returns.iloc[:date_idx+1]
                cumulative = (1 + prev_returns).cumprod()
                print(f"   截至该日的累积收益: {(cumulative.iloc[-1] - 1)*100:.2f}%")
        else:
            print(f"\n⚠️  {target_date_str} 不在收益数据中")
    
    # 检查是否有backtest结果文件
    if results_dir:
        results_path = Path(results_dir)
        backtest_results_path = results_path / "backtest_results.json"
        
        if backtest_results_path.exists():
            print(f"\n" + "=" * 80)
            print("回测结果分析")
            print("=" * 80)
            
            with open(backtest_results_path, 'r') as f:
                backtest_results = json.load(f)
            
            if 'performance_metrics' in backtest_results:
                metrics = backtest_results['performance_metrics']
                print(f"\n性能指标:")
                print(f"   总收益: {metrics.get('total_return', 0)*100:.2f}%")
                print(f"   年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
                print(f"   Sharpe比率: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
                print(f"   波动率: {metrics.get('volatility', 0)*100:.2f}%")
            
            if 'trades' in backtest_results:
                trades = backtest_results['trades']
                print(f"\n交易统计:")
                print(f"   总交易数: {len(trades)}")
                
                # 分析极端收益日的交易
                for target_date in target_dates:
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    day_trades = [t for t in trades if t.get('date', '').startswith(target_date_str)]
                    if day_trades:
                        print(f"\n   {target_date_str} 的交易 ({len(day_trades)} 笔):")
                        for trade in day_trades[:10]:  # 只显示前10笔
                            print(f"     {trade.get('symbol', 'N/A'):15s} {trade.get('direction', 'N/A'):5s} "
                                  f"{trade.get('quantity', 0):8.0f} @ ${trade.get('price', 0):.2f} "
                                  f"价值: ${trade.get('value', 0):,.0f}")
    
    # 收益分布分析
    print(f"\n" + "=" * 80)
    print("收益分布统计")
    print("=" * 80)
    
    print(f"\n基本统计:")
    print(f"   平均日收益: {returns.mean()*100:.4f}%")
    print(f"   收益标准差: {returns.std()*100:.4f}%")
    print(f"   最大单日收益: {returns.max()*100:.2f}%")
    print(f"   最小单日收益: {returns.min()*100:.2f}%")
    print(f"   收益偏度: {returns.skew():.4f}")
    print(f"   收益峰度: {returns.kurtosis():.4f}")
    
    # 计算分位数
    print(f"\n收益分位数:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(returns, p)
        print(f"   {p:2d}%: {val*100:7.2f}%")
    
    # 计算累积收益
    cumulative_returns = (1 + returns).cumprod()
    print(f"\n累积收益:")
    print(f"   最终累积收益: {(cumulative_returns.iloc[-1] - 1)*100:.2f}%")
    print(f"   最高累积收益: {(cumulative_returns.max() - 1)*100:.2f}%")
    if cumulative_returns.max() > 0:
        max_dd = (cumulative_returns.max() - cumulative_returns.iloc[-1]) / cumulative_returns.max() * 100
        print(f"   最大回撤: {max_dd:.2f}%")
    
    # 建议
    print(f"\n" + "=" * 80)
    print("诊断建议")
    print("=" * 80)
    
    if returns.kurtosis() > 10:
        print(f"\n⚠️  收益峰度异常高 ({returns.kurtosis():.2f})，说明有极端异常值")
        print(f"   建议检查数据质量和组合构建逻辑")
    
    if len(consecutive_groups) > 0:
        print(f"\n⚠️  发现 {len(consecutive_groups)} 组连续极端收益日")
        print(f"   这可能表明数据问题或组合构建问题")
        print(f"   建议检查这些日期的持仓和价格数据")
    
    extreme_negative = [d for d in extreme_days if d[1] < -0.10]  # 超过-10%的损失
    if len(extreme_negative) > 0:
        print(f"\n⚠️  发现 {len(extreme_negative)} 个极端负收益日 (>-10%)")
        print(f"   建议检查这些日期的:")
        print(f"   1. 持仓集中度")
        print(f"   2. 价格数据质量")
        print(f"   3. 组合权重计算")
        print(f"   4. 交易成本计算")


def main():
    """主函数"""
    returns_path = project_root / "results" / "ff5_regression_20251107_012512" / "strategy_returns.csv"
    results_dir = project_root / "results" / "ff5_regression_20251107_012512"
    
    if not returns_path.exists():
        print(f"❌ 收益文件不存在: {returns_path}")
        return
    
    check_extreme_return_days(str(returns_path), str(results_dir))


if __name__ == "__main__":
    main()


