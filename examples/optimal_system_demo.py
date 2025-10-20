#!/usr/bin/env python3
"""
Optimal System Demo - 一行代码最优系统演示
==============================================

一行代码统一最佳模型+元模型组合系统的完整演示。

Examples:
    >>> python optimal_system_demo.py --config configs/optimal_system_config.yaml
    >>> python optimal_system_demo.py --quick-test
    >>> python optimal_system_demo.py --custom-config
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_system.orchestration.optimal_system_orchestrator import (
    OptimalSystemOrchestrator, OptimalSystemConfig,
    create_optimal_system_orchestrator, quick_optimal_system
)
from trading_system.orchestration.components.optimal_model_selector import (
    create_model_selector, quick_best_model_selection
)
from trading_system.orchestration.components.optimal_metamodel_selector import (
    create_metamodel_selector, quick_optimal_metamodel
)
from trading_system.orchestration.components.system_performance_evaluator import (
    create_system_evaluator, quick_system_evaluation
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """设置日志."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_mock_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """生成模拟数据用于演示."""
    import numpy as np

    data_config = config.get('data', {})
    universe = data_config.get('universe', ['AAPL', 'GOOGL', 'MSFT'])

    # 生成时间序列数据
    train_period = data_config.get('train_period', {})
    test_period = data_config.get('test_period', {})

    # 简化的数据生成（实际应用中应使用真实数据）
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # 生成模拟价格数据
    price_data = {}
    for symbol in universe:
        np.random.seed(hash(symbol) % 2**32)  # 确保可重复性
        returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[symbol] = pd.Series(prices, index=dates)

    # 生成信号数据
    signals_data = {}
    for symbol in universe:
        signals = np.random.normal(0, 0.1, len(dates))  # 简化的信号
        signals_data[symbol] = pd.Series(signals, index=dates)

    return {
        'train_data': {
            'prices': pd.DataFrame(price_data),
            'signals': pd.DataFrame(signals_data),
            'returns': pd.DataFrame({symbol: prices.pct_change()
                                   for symbol, prices in price_data.items()})
        },
        'test_data': {
            'prices': pd.DataFrame(price_data),
            'signals': pd.DataFrame(signals_data),
            'returns': pd.DataFrame({symbol: prices.pct_change()
                                   for symbol, prices in price_data.items()})
        },
        'strategy_data': {
            'returns': pd.DataFrame(signals_data),
            'performance': {symbol: {'sharpe_ratio': np.random.uniform(0.5, 1.5),
                                   'total_return': np.random.uniform(0.1, 0.3),
                                   'volatility': np.random.uniform(0.1, 0.2)}
                          for symbol in universe}
        },
        'benchmark_data': {
            'returns': pd.Series(np.random.normal(0.0008, 0.015, len(dates)),
                               index=dates, name='SPY')
        }
    }


def demo_basic_one_line_usage():
    """演示基础一行代码用法."""
    print("🚀 演示基础一行代码用法")
    print("=" * 50)

    # 生成模拟数据
    config = load_config('configs/quick_test_config.yaml')
    data = generate_mock_data(config)

    # 一行代码创建协调器
    orchestrator = create_optimal_system_orchestrator(n_trials=5, save_results=True)

    # 一行代码运行完整系统
    model_types = ['xgboost', 'lstm']
    result = orchestrator.find_and_run_optimal_system(
        model_types, data['train_data'], data['test_data'],
        data['strategy_data'], data['benchmark_data']
    )

    # 一行代码打印结果
    print(f"✅ 系统成功运行！最佳Sharpe: {result['report']['key_metrics']['sharpe_ratio']:.3f}")
    print(f"✅ 总收益: {result['report']['key_metrics']['total_return']:.2%}")
    print(f"✅ 最大回撤: {result['report']['key_metrics']['max_drawdown']:.2%}")

    return result


def demo_quick_optimal_system():
    """演示快速最优系统一行代码."""
    print("\n🚀 演示快速最优系统一行代码")
    print("=" * 50)

    # 生成模拟数据
    config = load_config('configs/quick_test_config.yaml')
    data = generate_mock_data(config)

    # 一行代码完成整个最优系统流程
    model_types = ['xgboost', 'lstm']
    result = quick_optimal_system(
        model_types, data['train_data'], data['test_data'],
        data['strategy_data'], data['benchmark_data'], n_trials=3
    )

    # 一行代码显示结果
    print(f"✅ 快速系统完成！系统验证: {result['success']}")
    print(f"✅ 最佳模型数量: {len(result['best_models'])}")
    print(f"✅ 元模型R²: {result['best_metamodel'].get('performance', {}).get('r2', 0):.3f}")

    return result


def demo_component_level_usage():
    """演示组件级别一行代码用法."""
    print("\n🚀 演示组件级别一行代码用法")
    print("=" * 50)

    # 生成模拟数据
    config = load_config('configs/quick_test_config.yaml')
    data = generate_mock_data(config)

    # 一行代码创建模型选择器
    model_selector = create_model_selector(n_trials=3, primary_metric='sharpe_ratio')

    # 一行代码找到最佳模型
    model_types = ['xgboost', 'lstm']
    best_models = model_selector.find_best_models(
        model_types, data['train_data'], data['test_data'], data['benchmark_data']
    )
    print(f"✅ 找到 {len(best_models)} 个优化模型")

    # 一行代码创建元模型选择器
    metamodel_selector = create_metamodel_selector(n_trials=3, weight_method='equal')

    # 一行代码找到最佳元模型
    best_metamodel = metamodel_selector.find_best_metamodel(
        data['strategy_data']['returns'], data['strategy_data']['performance']
    )
    print(f"✅ 元模型权重: {list(best_metamodel.get('weights', {}).values())[:3]}...")

    # 一行代码创建系统评估器
    system_evaluator = create_system_evaluator(
        primary_metrics=['sharpe_ratio', 'total_return'],
        min_requirements={'sharpe_ratio': 0.5}
    )

    # 一行代码评估系统性能
    portfolio_returns = data['strategy_data']['returns'].mean(axis=1)
    system_performance = system_evaluator.evaluate_complete_system(
        portfolio_returns, data['strategy_data']['returns'],
        data['benchmark_data']['returns']
    )
    print(f"✅ 系统Sharpe: {system_performance['portfolio_metrics']['sharpe_ratio']:.3f}")

    return {
        'best_models': best_models,
        'best_metamodel': best_metamodel,
        'system_performance': system_performance
    }


def demo_config_driven_usage(config_path: str):
    """演示配置驱动用法."""
    print(f"\n🚀 演示配置驱动用法: {config_path}")
    print("=" * 50)

    # 加载配置
    config = load_config(config_path)

    # 生成模拟数据
    data = generate_mock_data(config)

    # 从配置创建系统
    system_config = OptimalSystemConfig(
        model_n_trials=config.get('model_selection', {}).get('n_trials', 10),
        metamodel_n_trials=config.get('metamodel_selection', {}).get('n_trials', 10),
        save_results=config.get('output', {}).get('save_results', True),
        output_directory=config.get('output', {}).get('output_directory', './results')
    )

    # 一行代码创建协调器
    orchestrator = OptimalSystemOrchestrator(system_config)

    # 一行代码运行完整系统
    model_types = config.get('model_selection', {}).get('model_types', ['xgboost'])
    result = orchestrator.find_and_run_optimal_system(
        model_types, data['train_data'], data['test_data'],
        data['strategy_data'], data['benchmark_data']
    )

    # 一行代码生成报告
    report = orchestrator.generate_complete_report(result['system_performance'])

    print(f"✅ 配置驱动系统完成！")
    print(f"✅ {report['summary']['overall_performance']}")
    print(f"✅ 风险等级: {report['summary']['risk_level']}")
    print(f"✅ 验证状态: {report['summary']['validation_status']}")

    return result


def demo_comparison_functionality():
    """演示系统对比功能."""
    print("\n🚀 演示系统对比功能")
    print("=" * 50)

    # 生成模拟数据
    config = load_config('configs/quick_test_config.yaml')
    data = generate_mock_data(config)

    # 创建多个配置
    configs = {
        'conservative': OptimalSystemConfig(
            min_sharpe_ratio=1.0, max_drawdown_threshold=-0.15
        ),
        'aggressive': OptimalSystemConfig(
            min_sharpe_ratio=0.5, max_drawdown_threshold=-0.35
        ),
        'balanced': OptimalSystemConfig(
            min_sharpe_ratio=0.8, max_drawdown_threshold=-0.25
        )
    }

    # 一行代码对比系统配置
    orchestrator = create_optimal_system_orchestrator(n_trials=2)
    comparison = orchestrator.compare_system_configurations(
        configs, ['xgboost'], data['train_data'], data['test_data']
    )

    print(f"✅ 系统对比完成！")
    print(f"✅ 最佳系统: {comparison['best_system_name']}")
    print(f"✅ 最佳Sharpe: {comparison['best_system_metrics']['sharpe_ratio']:.3f}")

    return comparison


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description='Optimal System Demo')
    parser.add_argument('--config', '-c', type=str,
                       default='configs/optimal_system_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--quick-test', action='store_true',
                       help='运行快速测试')
    parser.add_argument('--custom-config', action='store_true',
                       help='运行自定义配置演示')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    print("🎯 Optimal System Demo - 一行代码最优系统演示")
    print("=" * 60)

    try:
        # 基础一行代码演示
        demo_basic_one_line_usage()

        # 快速最优系统演示
        demo_quick_optimal_system()

        # 组件级别演示
        demo_component_level_usage()

        # 系统对比演示
        demo_comparison_functionality()

        # 配置驱动演示
        if args.custom_config:
            demo_config_driven_usage(args.config)
        elif args.quick_test:
            demo_config_driven_usage('configs/quick_test_config.yaml')
        else:
            demo_config_driven_usage(args.config)

        print("\n🎉 所有演示完成！一行代码最优系统运行成功！")

    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        raise


if __name__ == "__main__":
    main()