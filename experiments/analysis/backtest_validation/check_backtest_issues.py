#!/usr/bin/env python3
"""
检查回测负收益问题的诊断脚本

执行三项检查：
1. 检查训练模型的实际Alpha分布
2. 检查极端收益日的持仓和价格数据
3. 检查训练期与回测期股票列表的重叠度
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from trading_system.models.serving.predictor import ModelPredictor
from trading_system.models.model_persistence import ModelRegistry
from trading_system.models.implementations.ff5_model import FF5RegressionModel


def check_1_alpha_distribution(model_id: str):
    """检查1: 训练模型的实际Alpha分布"""
    print("=" * 80)
    print("检查1: 训练模型的实际Alpha分布")
    print("=" * 80)
    
    try:
        # 加载模型
        model_registry_path = project_root / "models"
        predictor = ModelPredictor(
            model_id=model_id,
            model_registry_path=str(model_registry_path)
        )
        model = predictor.get_current_model()
        
        if not model:
            print(f"❌ 无法加载模型: {model_id}")
            return None
        
        if not hasattr(model, 'get_symbol_alphas'):
            print(f"❌ 模型不支持 get_symbol_alphas 方法")
            return None
        
        # 获取所有Alpha值
        alphas = model.get_symbol_alphas()
        
        if not alphas:
            print("❌ 模型中没有Alpha值")
            return None
        
        alpha_values = list(alphas.values())
        alpha_array = np.array(alpha_values)
        
        # 统计信息
        print(f"\n📊 Alpha统计信息:")
        print(f"   总股票数量: {len(alphas)}")
        print(f"   Alpha最小值: {np.min(alpha_array):.6f}")
        print(f"   Alpha最大值: {np.max(alpha_array):.6f}")
        print(f"   Alpha平均值: {np.mean(alpha_array):.6f}")
        print(f"   Alpha中位数: {np.median(alpha_array):.6f}")
        print(f"   Alpha标准差: {np.std(alpha_array):.6f}")
        
        # 正负Alpha统计
        positive_alphas = [a for a in alpha_values if a > 0]
        negative_alphas = [a for a in alpha_values if a < 0]
        zero_alphas = [a for a in alpha_values if a == 0]
        
        print(f"\n📈 Alpha符号分布:")
        print(f"   正Alpha股票: {len(positive_alphas)} ({len(positive_alphas)/len(alphas)*100:.1f}%)")
        print(f"   负Alpha股票: {len(negative_alphas)} ({len(negative_alphas)/len(alphas)*100:.1f}%)")
        print(f"   零Alpha股票: {len(zero_alphas)} ({len(zero_alphas)/len(alphas)*100:.1f}%)")
        
        if positive_alphas:
            print(f"\n   正Alpha统计:")
            print(f"     最小值: {min(positive_alphas):.6f}")
            print(f"     最大值: {max(positive_alphas):.6f}")
            print(f"     平均值: {np.mean(positive_alphas):.6f}")
        
        if negative_alphas:
            print(f"\n   负Alpha统计:")
            print(f"     最小值: {min(negative_alphas):.6f}")
            print(f"     最大值: {max(negative_alphas):.6f}")
            print(f"     平均值: {np.mean(negative_alphas):.6f}")
        
        # Top/Bottom Alpha股票
        sorted_alphas = sorted(alphas.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 10 正Alpha股票:")
        for i, (symbol, alpha) in enumerate(sorted_alphas[:10], 1):
            print(f"   {i:2d}. {symbol:15s}: {alpha:8.6f}")
        
        print(f"\n📉 Bottom 10 负Alpha股票:")
        for i, (symbol, alpha) in enumerate(sorted_alphas[-10:], 1):
            print(f"   {i:2d}. {symbol:15s}: {alpha:8.6f}")
        
        # 检查模型元数据
        if hasattr(model, 'metadata'):
            print(f"\n📋 模型训练信息:")
            print(f"   训练样本数: {model.metadata.training_samples if hasattr(model.metadata, 'training_samples') else 'N/A'}")
            print(f"   训练开始日期: {model.metadata.start_date if hasattr(model.metadata, 'start_date') else 'N/A'}")
            print(f"   训练结束日期: {model.metadata.end_date if hasattr(model.metadata, 'end_date') else 'N/A'}")
        
        return alphas
        
    except Exception as e:
        print(f"❌ 检查Alpha分布时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_2_extreme_return_days(returns_path: str, results_dir: str):
    """检查2: 极端收益日的持仓和价格数据"""
    print("\n" + "=" * 80)
    print("检查2: 极端收益日的持仓和价格数据")
    print("=" * 80)
    
    try:
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
        
        print(f"\n📊 极端收益日分析 (阈值: ±{threshold*100:.0f}%):")
        print(f"   找到 {len(extreme_days)} 个极端收益日")
        
        print(f"\n🔥 Top 10 极端收益日:")
        for i, (date, ret) in enumerate(extreme_days[:10], 1):
            print(f"   {i:2d}. {date.strftime('%Y-%m-%d')}: {ret*100:7.2f}%")
        
        # 检查是否有portfolio weights数据
        results_path = Path(results_dir)
        portfolio_weights_path = results_path / "portfolio_weights.csv"
        
        if portfolio_weights_path.exists():
            print(f"\n📈 检查极端收益日的持仓权重...")
            weights_df = pd.read_csv(portfolio_weights_path, index_col=0, parse_dates=True)
            
            for date, ret in extreme_days[:5]:  # 只检查前5个
                if date in weights_df.index:
                    weights = weights_df.loc[date]
                    non_zero_weights = weights[weights != 0].sort_values(ascending=False)
                    
                    print(f"\n   📅 {date.strftime('%Y-%m-%d')} (收益: {ret*100:.2f}%):")
                    print(f"      总持仓数: {len(non_zero_weights)}")
                    print(f"      权重总和: {weights.sum():.4f}")
                    
                    if len(non_zero_weights) > 0:
                        print(f"      Top 5 持仓:")
                        for symbol, weight in list(non_zero_weights.head().items()):
                            print(f"        {symbol:15s}: {weight*100:6.2f}%")
        else:
            print(f"\n⚠️  未找到 portfolio_weights.csv 文件")
            print(f"   路径: {portfolio_weights_path}")
        
        # 检查是否有backtest结果
        backtest_results_path = results_path / "backtest_results.json"
        if backtest_results_path.exists():
            print(f"\n📋 检查回测结果...")
            with open(backtest_results_path, 'r') as f:
                backtest_results = json.load(f)
            
            if 'trades' in backtest_results:
                trades = backtest_results['trades']
                print(f"   总交易数: {len(trades)}")
                
                # 查找极端收益日的交易
                for date, ret in extreme_days[:3]:
                    date_str = date.strftime('%Y-%m-%d')
                    day_trades = [t for t in trades if t.get('date', '').startswith(date_str)]
                    if day_trades:
                        print(f"\n   {date_str} 的交易:")
                        for trade in day_trades[:5]:
                            print(f"      {trade.get('symbol', 'N/A'):15s} {trade.get('direction', 'N/A'):5s} {trade.get('quantity', 0):8.0f} @ ${trade.get('price', 0):.2f}")
        
        return extreme_days
        
    except Exception as e:
        print(f"❌ 检查极端收益日时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_3_universe_overlap(config_path: str, model_id: str):
    """检查3: 训练期与回测期股票列表的重叠度"""
    print("\n" + "=" * 80)
    print("检查3: 训练期与回测期股票列表的重叠度")
    print("=" * 80)
    
    try:
        import yaml
        
        # 读取配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 获取训练期配置
        training_setup = config.get('training_setup', {})
        training_params = training_setup.get('parameters', {})
        train_start = training_params.get('start_date', 'N/A')
        train_end = training_params.get('end_date', 'N/A')
        
        # 获取回测期配置
        backtest_config = config.get('backtest', {})
        backtest_start = backtest_config.get('start_date', 'N/A')
        backtest_end = backtest_config.get('end_date', 'N/A')
        
        print(f"\n📅 时间范围:")
        print(f"   训练期: {train_start} 到 {train_end}")
        print(f"   回测期: {backtest_start} 到 {backtest_end}")
        
        # 加载模型，获取训练期的股票列表
        model_registry_path = project_root / "models"
        predictor = ModelPredictor(
            model_id=model_id,
            model_registry_path=str(model_registry_path)
        )
        model = predictor.get_current_model()
        
        if not model or not hasattr(model, 'get_symbol_alphas'):
            print("❌ 无法加载模型或获取股票列表")
            return None
        
        training_symbols = set(model.get_symbol_alphas().keys())
        print(f"\n📊 训练期股票数量: {len(training_symbols)}")
        
        # 获取回测期的股票列表（从配置中）
        universe_config = training_params.get('universe', {})
        if universe_config.get('source') == 'csv':
            csv_path = universe_config.get('csv_path', '')
            if csv_path:
                csv_path = project_root / csv_path.replace('./', '')
                if csv_path.exists():
                    universe_df = pd.read_csv(csv_path)
                    if 'symbol' in universe_df.columns:
                        backtest_symbols = set(universe_df['symbol'].unique())
                    elif 'Symbol' in universe_df.columns:
                        backtest_symbols = set(universe_df['Symbol'].unique())
                    else:
                        # 尝试第一列
                        backtest_symbols = set(universe_df.iloc[:, 0].unique())
                    
                    print(f"📊 回测期股票数量: {len(backtest_symbols)}")
                    
                    # 计算重叠
                    overlap = training_symbols.intersection(backtest_symbols)
                    only_training = training_symbols - backtest_symbols
                    only_backtest = backtest_symbols - training_symbols
                    
                    print(f"\n📈 重叠分析:")
                    print(f"   重叠股票数: {len(overlap)} ({len(overlap)/len(training_symbols)*100:.1f}% of 训练期)")
                    print(f"   仅在训练期: {len(only_training)}")
                    print(f"   仅在回测期: {len(only_backtest)}")
                    
                    if len(overlap) < len(training_symbols) * 0.5:
                        print(f"\n⚠️  警告: 重叠度低于50%，这可能导致信号质量问题！")
                    
                    # 显示一些示例
                    if only_training:
                        print(f"\n   仅在训练期的示例股票 (前10个):")
                        for symbol in list(only_training)[:10]:
                            print(f"      {symbol}")
                    
                    if only_backtest:
                        print(f"\n   仅在回测期的示例股票 (前10个):")
                        for symbol in list(only_backtest)[:10]:
                            print(f"      {symbol}")
                    
                    return {
                        'training_symbols': training_symbols,
                        'backtest_symbols': backtest_symbols,
                        'overlap': overlap,
                        'overlap_ratio': len(overlap) / len(training_symbols) if training_symbols else 0
                    }
                else:
                    print(f"⚠️  CSV文件不存在: {csv_path}")
            else:
                print("⚠️  配置中未指定CSV路径")
        else:
            print("⚠️  无法从配置中确定回测期股票列表")
        
        return None
        
    except Exception as e:
        print(f"❌ 检查股票重叠度时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    model_id = "ff5_regression_20251107_012512"
    config_path = project_root / "configs" / "active" / "single_experiment" / "ff5_box_based_experiment.yaml"
    returns_path = project_root / "results" / "ff5_regression_20251107_012512" / "strategy_returns.csv"
    results_dir = project_root / "results" / "ff5_regression_20251107_012512"
    
    print("🔍 FF5回测负收益问题诊断")
    print("=" * 80)
    print(f"模型ID: {model_id}")
    print(f"配置文件: {config_path}")
    print(f"收益文件: {returns_path}")
    print("=" * 80)
    
    # 检查1: Alpha分布
    alphas = check_1_alpha_distribution(model_id)
    
    # 检查2: 极端收益日
    if returns_path.exists():
        extreme_days = check_2_extreme_return_days(str(returns_path), str(results_dir))
    else:
        print(f"\n⚠️  收益文件不存在: {returns_path}")
    
    # 检查3: 股票重叠度
    if config_path.exists():
        overlap_info = check_3_universe_overlap(str(config_path), model_id)
    else:
        print(f"\n⚠️  配置文件不存在: {config_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 检查总结")
    print("=" * 80)
    
    if alphas:
        alpha_values = list(alphas.values())
        positive_ratio = sum(1 for a in alpha_values if a > 0) / len(alpha_values)
        print(f"1. Alpha分布: {len(alphas)}只股票, {positive_ratio*100:.1f}%为正Alpha")
        
        if positive_ratio < 0.3:
            print("   ⚠️  警告: 正Alpha股票比例过低，可能导致可用信号不足")
        if positive_ratio > 0.7:
            print("   ⚠️  警告: 正Alpha股票比例过高，可能存在数据问题")
    
    if overlap_info:
        overlap_ratio = overlap_info.get('overlap_ratio', 0)
        print(f"2. 股票重叠度: {overlap_ratio*100:.1f}%")
        
        if overlap_ratio < 0.5:
            print("   ⚠️  警告: 训练期和回测期股票重叠度低，模型可能无法有效预测")
    
    print("\n✅ 检查完成")


if __name__ == "__main__":
    main()


