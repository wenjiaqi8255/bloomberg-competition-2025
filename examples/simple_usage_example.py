#!/usr/bin/env python3
"""
Simple Usage Example - 简单使用示例
=====================================

最简单的一行代码使用示例。

Usage:
    >>> python simple_usage_example.py
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_system.orchestration.optimal_system_orchestrator import (
    quick_optimal_system, create_optimal_system_orchestrator
)

def simple_one_line_example():
    """最简单的一行代码示例."""
    print("🚀 最简单的一行代码示例")
    print("=" * 40)

    # 模拟数据（实际使用中请替换为真实数据）
    model_types = ['xgboost', 'lstm']
    train_data = {'prices': None, 'signals': None}  # 您的数据
    test_data = {'prices': None, 'signals': None}   # 您的数据
    strategy_data = {'returns': None, 'performance': None}  # 您的数据
    benchmark_data = {'returns': None}  # 您的数据

    # 一行代码完成整个最优系统流程
    result = quick_optimal_system(
        model_types, train_data, test_data, strategy_data, benchmark_data, n_trials=10
    )

    # 一行代码获取结果
    print(f"✅ 系统成功！Sharpe: {result['report']['key_metrics']['sharpe_ratio']:.3f}")
    print(f"✅ 总收益: {result['report']['key_metrics']['total_return']:.2%}")
    print(f"✅ 系统有效: {result['success']}")

    return result


def step_by_step_example():
    """分步骤示例."""
    print("\n🔧 分步骤示例")
    print("=" * 40)

    # 一行代码创建协调器
    orchestrator = create_optimal_system_orchestrator(n_trials=20, save_results=True)

    # 模拟数据
    model_types = ['xgboost', 'lstm']
    train_data = {'prices': None, 'signals': None}
    test_data = {'prices': None, 'signals': None}
    strategy_data = {'returns': None, 'performance': None}
    benchmark_data = {'returns': None}

    # 一行代码找到最佳组合
    best_models, best_metamodel = orchestrator.find_optimal_combination(
        model_types, train_data, test_data, strategy_data, benchmark_data
    )
    print(f"✅ 找到 {len(best_models)} 个最佳模型")

    # 一行代码运行系统
    system_performance = orchestrator.run_optimal_system(
        best_models, best_metamodel, test_data, benchmark_data
    )
    print(f"✅ 系统Sharpe: {system_performance['system_performance']['portfolio_metrics']['sharpe_ratio']:.3f}")

    # 一行代码生成报告
    report = orchestrator.generate_complete_report(system_performance)
    print(f"✅ {report['summary']['overall_performance']}")

    return report


if __name__ == "__main__":
    # 运行简单示例
    simple_one_line_example()

    # 运行分步示例
    step_by_step_example()

    print("\n🎯 使用示例完成！")
    print("\n📖 更多使用方法请参考:")
    print("   - examples/optimal_system_demo.py")
    print("   - configs/optimal_system_config.yaml")
    print("   - src/trading_system/orchestration/optimal_system_orchestrator.py")