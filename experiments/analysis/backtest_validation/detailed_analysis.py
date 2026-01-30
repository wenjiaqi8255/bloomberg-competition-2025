#!/usr/bin/env python3
"""
详细分析极端收益日和股票重叠问题
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from trading_system.models.serving.predictor import ModelPredictor


def analyze_extreme_returns(returns_path: str):
    """分析极端收益日的详细信息"""
    print("=" * 80)
    print("极端收益日详细分析")
    print("=" * 80)
    
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns_df.columns = ['daily_return']
    returns = returns_df['daily_return']
    
    # 找出极端收益日
    extreme_days = []
    for date, ret in returns.items():
        if abs(ret) > 0.05:  # 5%阈值
            extreme_days.append((date, ret))
    
    extreme_days.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n找到 {len(extreme_days)} 个极端收益日 (>5%)")
    
    # 分析连续极端收益日
    print("\n连续极端收益日分析:")
    for i in range(len(extreme_days) - 1):
        date1, ret1 = extreme_days[i]
        date2, ret2 = extreme_days[i+1]
        
        days_diff = (date2 - date1).days
        if days_diff <= 3:  # 3天内连续出现
            print(f"  {date1.strftime('%Y-%m-%d')}: {ret1*100:7.2f}%")
            print(f"  {date2.strftime('%Y-%m-%d')}: {ret2*100:7.2f}%")
            print(f"  间隔: {days_diff} 天, 累积收益: {(ret1+ret2)*100:7.2f}%")
            print()
    
    # 分析收益分布
    print("\n收益分布统计:")
    print(f"  平均日收益: {returns.mean()*100:.4f}%")
    print(f"  收益标准差: {returns.std()*100:.4f}%")
    print(f"  最大单日收益: {returns.max()*100:.2f}%")
    print(f"  最小单日收益: {returns.min()*100:.2f}%")
    print(f"  收益偏度: {returns.skew():.4f}")
    print(f"  收益峰度: {returns.kurtosis():.4f}")
    
    # 计算累积收益
    cumulative_returns = (1 + returns).cumprod()
    print(f"\n累积收益:")
    print(f"  最终累积收益: {(cumulative_returns.iloc[-1] - 1)*100:.2f}%")
    print(f"  最高累积收益: {(cumulative_returns.max() - 1)*100:.2f}%")
    print(f"  最大回撤: {(cumulative_returns.max() - cumulative_returns.iloc[-1])/cumulative_returns.max()*100:.2f}%")
    
    return extreme_days


def analyze_universe_overlap(config_path: str, model_id: str):
    """详细分析股票重叠问题"""
    print("\n" + "=" * 80)
    print("股票重叠度详细分析")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    model_registry_path = project_root / "models"
    predictor = ModelPredictor(
        model_id=model_id,
        model_registry_path=str(model_registry_path)
    )
    model = predictor.get_current_model()
    training_symbols = set(model.get_symbol_alphas().keys())
    
    # 获取回测期股票列表
    training_setup = config.get('training_setup', {})
    training_params = training_setup.get('parameters', {})
    universe_config = training_params.get('universe', {})
    
    if universe_config.get('source') == 'csv':
        csv_path = universe_config.get('csv_path', '')
        if csv_path:
            csv_path = project_root / csv_path.replace('./', '')
            if csv_path.exists():
                universe_df = pd.read_csv(csv_path)
                
                # 尝试找到symbol列
                symbol_col = None
                for col in ['symbol', 'Symbol', 'ticker', 'Ticker', 'yfinance_ticker']:
                    if col in universe_df.columns:
                        symbol_col = col
                        break
                
                if symbol_col is None:
                    symbol_col = universe_df.columns[0]
                
                backtest_symbols = set(universe_df[symbol_col].dropna().unique())
                
                # 计算重叠
                overlap = training_symbols.intersection(backtest_symbols)
                only_training = training_symbols - backtest_symbols
                only_backtest = backtest_symbols - training_symbols
                
                print(f"\n训练期股票: {len(training_symbols)}")
                print(f"回测期股票: {len(backtest_symbols)}")
                print(f"重叠股票: {len(overlap)} ({len(overlap)/len(training_symbols)*100:.1f}%)")
                print(f"仅在训练期: {len(only_training)}")
                print(f"仅在回测期: {len(only_backtest)}")
                
                # 分析重叠股票的特征
                if overlap:
                    print(f"\n重叠股票示例 (前20个):")
                    for i, symbol in enumerate(list(overlap)[:20], 1):
                        alpha = model.get_alpha(symbol)
                        print(f"  {i:2d}. {symbol:15s} Alpha: {alpha:8.6f}")
                
                # 分析仅在训练期的股票（这些股票在回测时无法使用）
                if only_training:
                    print(f"\n仅在训练期的股票 (前20个，回测时无法使用):")
                    alphas_only_training = [(s, model.get_alpha(s)) for s in only_training]
                    alphas_only_training.sort(key=lambda x: x[1], reverse=True)
                    for i, (symbol, alpha) in enumerate(alphas_only_training[:20], 1):
                        print(f"  {i:2d}. {symbol:15s} Alpha: {alpha:8.6f}")
                
                # 分析Alpha分布
                overlap_alphas = [model.get_alpha(s) for s in overlap if s in model.get_symbol_alphas()]
                only_training_alphas = [model.get_alpha(s) for s in only_training if s in model.get_symbol_alphas()]
                
                if overlap_alphas:
                    print(f"\n重叠股票的Alpha统计:")
                    print(f"  平均值: {np.mean(overlap_alphas):.6f}")
                    print(f"  中位数: {np.median(overlap_alphas):.6f}")
                    print(f"  正Alpha比例: {sum(1 for a in overlap_alphas if a > 0)/len(overlap_alphas)*100:.1f}%")
                
                if only_training_alphas:
                    print(f"\n仅在训练期股票的Alpha统计:")
                    print(f"  平均值: {np.mean(only_training_alphas):.6f}")
                    print(f"  中位数: {np.median(only_training_alphas):.6f}")
                    print(f"  正Alpha比例: {sum(1 for a in only_training_alphas if a > 0)/len(only_training_alphas)*100:.1f}%")
                
                return {
                    'overlap': overlap,
                    'only_training': only_training,
                    'only_backtest': only_backtest,
                    'overlap_alphas': overlap_alphas,
                    'only_training_alphas': only_training_alphas
                }
    
    return None


def analyze_alpha_significance(model_id: str):
    """分析Alpha显著性过滤的影响"""
    print("\n" + "=" * 80)
    print("Alpha显著性过滤分析")
    print("=" * 80)
    
    model_registry_path = project_root / "models"
    predictor = ModelPredictor(
        model_id=model_id,
        model_registry_path=str(model_registry_path)
    )
    model = predictor.get_current_model()
    alphas = model.get_symbol_alphas()
    
    # 模拟t-statistic过滤（假设）
    # 实际中需要从alpha_tstats.csv或计算得到
    alpha_tstats_path = project_root / "alpha_tstats_ff3.csv"
    if not alpha_tstats_path.exists():
        alpha_tstats_path = project_root / "alpha_tstats.csv"
    
    if alpha_tstats_path.exists():
        tstats_df = pd.read_csv(alpha_tstats_path)
        print(f"\n找到t-statistics文件: {alpha_tstats_path}")
        
        # 分析t-statistics分布
        if 't_alpha' in tstats_df.columns:
            tstats = tstats_df['t_alpha'].dropna()
            print(f"\nt-statistics统计:")
            print(f"  总股票数: {len(tstats)}")
            print(f"  平均值: {tstats.mean():.4f}")
            print(f"  中位数: {tstats.median():.4f}")
            print(f"  标准差: {tstats.std():.4f}")
            
            # 不同阈值下的过滤效果
            for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
                significant = tstats[abs(tstats) >= threshold]
                print(f"\n  阈值 |t| >= {threshold}:")
                print(f"    显著股票数: {len(significant)} ({len(significant)/len(tstats)*100:.1f}%)")
                if len(significant) > 0:
                    print(f"    平均|t|: {abs(significant).mean():.4f}")
    else:
        print(f"\n未找到t-statistics文件")
        print(f"  查找路径: {alpha_tstats_path}")


def main():
    model_id = "ff5_regression_20251107_012512"
    config_path = project_root / "configs" / "active" / "single_experiment" / "ff5_box_based_experiment.yaml"
    returns_path = project_root / "results" / "ff5_regression_20251107_012512" / "strategy_returns.csv"
    
    # 分析极端收益
    if returns_path.exists():
        extreme_days = analyze_extreme_returns(str(returns_path))
    
    # 分析股票重叠
    if config_path.exists():
        overlap_info = analyze_universe_overlap(str(config_path), model_id)
    
    # 分析Alpha显著性
    analyze_alpha_significance(model_id)
    
    # 总结
    print("\n" + "=" * 80)
    print("问题总结")
    print("=" * 80)
    print("""
主要问题:
1. 股票重叠度低 (26.4%): 训练期250只股票，回测期9593只，只有66只重叠
   - 这意味着模型训练时看到的184只股票在回测时无法使用
   - 这些股票的Alpha信息无法被利用
   
2. 极端收益日: 2024-12-19 (+45.62%) 和 2024-12-20 (-42.47%)
   - 连续两天的极端波动，可能是数据问题或组合构建问题
   - 需要检查这两天的持仓和价格数据
   
3. Alpha分布: 65.2%正Alpha，但平均Alpha只有0.97%
   - 大部分Alpha值很小，信号强度弱
   - 34.8%的股票是负Alpha，在禁用做空的情况下会被过滤
   
建议:
1. 提高股票重叠度: 确保训练期和回测期使用相同的股票列表
2. 检查极端收益日的数据: 查看持仓、价格、权重等
3. 优化信号生成: 不要直接用Alpha值，使用rank-based或Z-score标准化
4. 调整组合构建参数: 增加stocks_per_box，降低max_position_weight
    """)


if __name__ == "__main__":
    main()


