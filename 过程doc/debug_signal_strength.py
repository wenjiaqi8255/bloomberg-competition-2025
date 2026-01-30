"""
诊断脚本：检查为什么signal_strength都是0

一次性收集所有相关信息，避免重复运行
"""
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.use_case.prediction.prediction_orchestrator import PredictionOrchestrator

def diagnose_signal_strength():
    """诊断signal_strength为0的问题"""
    
    config_path = 'configs/active/prediction/prediction_ml_xgboost_quantitative.yaml'
    
    print("="*80)
    print("信号强度诊断脚本")
    print("="*80)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Strategy type: {config['strategy']['type']}")
    print(f"   Model ID: {config['strategy']['parameters']['model_id']}")
    print(f"   Min signal strength: {config['strategy']['parameters'].get('min_signal_strength', 'N/A')}")
    print(f"   Enable normalization: {config['strategy']['parameters'].get('enable_normalization', 'N/A')}")
    print(f"   Normalization method: {config['strategy']['parameters'].get('normalization_method', 'N/A')}")
    
    # 2. 创建orchestrator并运行预测
    print("\n2. 初始化PredictionOrchestrator...")
    orchestrator = PredictionOrchestrator(config_path)
    
    # 3. 手动执行关键步骤以收集信息
    print("\n3. 加载策略...")
    strategy = orchestrator._load_strategy()
    print(f"   Strategy: {strategy.__class__.__name__}")
    print(f"   Min signal strength: {getattr(strategy, 'min_signal_strength', 'N/A')}")
    print(f"   Enable normalization: {getattr(strategy, 'enable_normalization', 'N/A')}")
    print(f"   Normalization method: {getattr(strategy, 'normalization_method', 'N/A')}")
    
    # 4. 获取预测日期和universe
    print("\n4. 获取预测日期和universe...")
    prediction_date = orchestrator._get_prediction_date()
    universe = orchestrator._get_universe()
    print(f"   Prediction date: {prediction_date}")
    print(f"   Universe size: {len(universe)}")
    print(f"   Universe (first 10): {universe[:10]}")
    
    # 5. 生成信号
    print("\n5. 生成信号...")
    signals = orchestrator._generate_signals(strategy, universe, prediction_date)
    
    print(f"\n   信号DataFrame信息:")
    print(f"   - Shape: {signals.shape}")
    print(f"   - Index (dates): {list(signals.index)}")
    print(f"   - Columns (symbols): {len(signals.columns)} symbols")
    print(f"   - Columns (first 10): {list(signals.columns[:10])}")
    
    if not signals.empty:
        latest_signals = signals.iloc[-1]
        print(f"\n   最新信号（最后一行）统计:")
        print(f"   - 总信号数: {len(latest_signals)}")
        print(f"   - 非零信号数: {(latest_signals != 0).sum()}")
        print(f"   - 零信号数: {(latest_signals == 0).sum()}")
        print(f"   - 最小值: {latest_signals.min():.10f}")
        print(f"   - 最大值: {latest_signals.max():.10f}")
        print(f"   - 均值: {latest_signals.mean():.10f}")
        print(f"   - 标准差: {latest_signals.std():.10f}")
        
        # 显示非零信号
        non_zero = latest_signals[latest_signals != 0]
        if len(non_zero) > 0:
            print(f"\n   非零信号样本 (前10个):")
            for symbol, value in non_zero.head(10).items():
                print(f"     {symbol}: {value:.10f}")
        else:
            print(f"\n   ⚠️ 所有信号都是0！")
            # 显示一些信号的原始值，即使它们是0
            print(f"\n   信号样本 (前10个，即使为0):")
            for symbol, value in latest_signals.head(10).items():
                print(f"     {symbol}: {value:.10f}")
    
    # 6. 构建组合
    print("\n6. 构建组合...")
    portfolio_result = orchestrator._construct_portfolio(signals, universe, prediction_date)
    
    # 获取权重
    if hasattr(portfolio_result, 'weights'):
        weights = portfolio_result.weights
    else:
        weights = portfolio_result
    
    print(f"   Portfolio weights:")
    print(f"   - 总仓位数: {len(weights)}")
    print(f"   - 权重总和: {weights.sum():.6f}")
    print(f"   - 最大权重: {weights.max():.6f}")
    print(f"   - 最小权重: {weights.min():.6f}")
    
    # 显示前10个权重
    print(f"\n   前10个权重:")
    sorted_weights = weights.sort_values(ascending=False).head(10)
    for symbol, weight in sorted_weights.items():
        print(f"     {symbol}: {weight:.6f}")
    
    # 7. 检查信号和权重的一致性
    print("\n7. 检查信号和权重的一致性...")
    if not signals.empty:
        latest_signals = signals.iloc[-1]
        weight_symbols = set(weights.index if hasattr(weights, 'index') else weights.keys())
        signal_symbols = set(latest_signals.index)
        
        print(f"   - 权重中的symbols数: {len(weight_symbols)}")
        print(f"   - 信号中的symbols数: {len(signal_symbols)}")
        print(f"   - 交集: {len(weight_symbols & signal_symbols)}")
        print(f"   - 仅在权重中: {len(weight_symbols - signal_symbols)}")
        print(f"   - 仅在信号中: {len(signal_symbols - weight_symbols)}")
        
        # 检查权重中的symbols是否有对应的信号
        print(f"\n   权重中的symbols的信号值:")
        for symbol in list(weight_symbols)[:10]:
            if symbol in latest_signals.index:
                signal_val = latest_signals[symbol]
                weight_val = weights[symbol] if hasattr(weights, 'index') else weights.get(symbol, 0)
                print(f"     {symbol}: signal={signal_val:.10f}, weight={weight_val:.6f}")
            else:
                print(f"     {symbol}: ⚠️ 信号中不存在, weight={weights.get(symbol, 0):.6f}")
    
    # 8. 创建推荐并检查signal_strength
    print("\n8. 创建推荐并检查signal_strength...")
    box_details = orchestrator._extract_box_details(portfolio_result)
    recommendations = orchestrator._create_recommendations(
        portfolio_result, signals, box_details, prediction_date
    )
    
    print(f"   推荐数量: {len(recommendations)}")
    
    # 统计signal_strength
    signal_strengths = [r.signal_strength for r in recommendations]
    non_zero_strengths = [s for s in signal_strengths if s != 0]
    
    print(f"\n   Signal strength统计:")
    print(f"   - 总数: {len(signal_strengths)}")
    print(f"   - 非零数: {len(non_zero_strengths)}")
    print(f"   - 零数: {len(signal_strengths) - len(non_zero_strengths)}")
    if signal_strengths:
        print(f"   - 最小值: {min(signal_strengths):.10f}")
        print(f"   - 最大值: {max(signal_strengths):.10f}")
        print(f"   - 均值: {np.mean(signal_strengths):.10f}")
    
    # 显示前10个推荐的详细信息
    print(f"\n   前10个推荐的详细信息:")
    for i, rec in enumerate(recommendations[:10], 1):
        print(f"     {i}. {rec.symbol}: weight={rec.weight:.6f}, signal_strength={rec.signal_strength:.10f}, risk_score={rec.risk_score:.6f}")
        
        # 检查原始信号
        if not signals.empty:
            latest_signals = signals.iloc[-1]
            if rec.symbol in latest_signals.index:
                original_signal = latest_signals[rec.symbol]
                print(f"        -> 原始信号值: {original_signal:.10f}")
            else:
                print(f"        -> ⚠️ 符号在信号中不存在")
    
    # 9. 诊断总结
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)
    
    if signals.empty:
        print("❌ 问题: 信号DataFrame为空")
    elif (signals.iloc[-1] != 0).sum() == 0:
        print("❌ 问题: 所有信号值都是0")
        print("   可能原因:")
        print("   1. 模型预测值都是0或接近0")
        print("   2. 归一化导致所有值变成0")
        print("   3. 过滤阈值太高，所有信号被过滤掉")
        print("   4. 模型预测本身有问题")
    else:
        non_zero_count = (signals.iloc[-1] != 0).sum()
        zero_in_recs = len([s for s in signal_strengths if s == 0])
        if zero_in_recs > 0:
            print(f"⚠️ 问题: 有{non_zero_count}个非零信号，但推荐中有{zero_in_recs}个signal_strength为0")
            print("   可能原因:")
            print("   1. 权重中的symbols与信号中的symbols不匹配")
            print("   2. 信号提取逻辑有问题（latest_signals.get()返回默认值0）")
        else:
            print("✅ 信号和signal_strength都正常")
    
    print("="*80)

if __name__ == "__main__":
    diagnose_signal_strength()

