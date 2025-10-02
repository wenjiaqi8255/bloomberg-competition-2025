"""
Multi-Strategy Orchestration Demo
==================================

演示如何使用重构后的 SystemOrchestrator 支持多个策略的组合。

这个示例展示了三种配置：
1. 双策略配置（FF5 + ML）
2. 三策略配置（FF5 + ML + 技术指标）
3. 单策略配置（仅 ML）
"""

import sys
from pathlib import Path

# 添加 src 到 Python 路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from datetime import datetime
from typing import List

from trading_system.orchestration import SystemOrchestrator
from trading_system.orchestration.components.allocator import AllocationConfig, StrategyAllocation
from trading_system.orchestration.components.compliance import ComplianceRules, StrategyAllocationRule
from trading_system.config.system import SystemConfig
from trading_system.strategies.base_strategy import BaseStrategy


def demo_two_strategy_system():
    """演示双策略系统配置（FF5 + ML）"""
    print("\n" + "="*80)
    print("Demo 1: 双策略系统 (FF5 Core + ML Satellite)")
    print("="*80)
    
    # 注意：这里使用模拟策略类，实际使用时替换为真实策略
    # from trading_system.strategies.fama_french_5 import FamaFrench5Strategy
    # from trading_system.strategies.ml_strategy import MLStrategy
    
    # 模拟策略（实际使用时替换）
    class MockStrategy(BaseStrategy):
        def __init__(self, name: str):
            self.name = name
        
        def generate_signals(self, date: datetime):
            return []
        
        def prepare_data(self):
            return True
    
    # 创建策略
    ff5_strategy = MockStrategy(name="FF5_Core")
    ml_strategy = MockStrategy(name="ML_Satellite")
    
    # 配置资本分配 - 70% FF5 + 30% ML
    allocation_config = AllocationConfig(
        strategy_allocations=[
            StrategyAllocation(
                strategy_name="FF5_Core",
                target_weight=0.70,
                min_weight=0.65,
                max_weight=0.75,
                priority=1  # 高优先级
            ),
            StrategyAllocation(
                strategy_name="ML_Satellite",
                target_weight=0.30,
                min_weight=0.25,
                max_weight=0.35,
                priority=2
            )
        ]
    )
    
    # 可选：自定义合规规则（否则会自动从 allocation_config 生成）
    compliance_rules = ComplianceRules(
        strategy_allocation_rules=[
            StrategyAllocationRule(
                strategy_name="FF5_Core",
                min_weight=0.65,
                max_weight=0.75
            ),
            StrategyAllocationRule(
                strategy_name="ML_Satellite",
                min_weight=0.25,
                max_weight=0.35
            )
        ]
    )
    
    # 创建系统配置
    system_config = create_mock_system_config("TwoStrategySystem")
    
    # 创建系统编排器
    orchestrator = SystemOrchestrator(
        system_config=system_config,
        strategies=[ff5_strategy, ml_strategy],
        allocation_config=allocation_config,
        compliance_rules=compliance_rules
    )
    
    # 验证配置
    is_valid, issues = orchestrator.validate_system_configuration()
    print(f"\n配置验证: {'✓ 通过' if is_valid else '✗ 失败'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # 获取系统信息
    component_info = orchestrator.get_component_info()
    print(f"\n策略数量: {component_info['strategy_count']}")
    print("策略列表:")
    for name, info in component_info['strategies'].items():
        print(f"  - {name} ({info['type']})")
    
    print("\n✓ 双策略系统配置完成")
    return orchestrator


def demo_three_strategy_system():
    """演示三策略系统配置（FF5 + ML + 技术指标）"""
    print("\n" + "="*80)
    print("Demo 2: 三策略系统 (FF5 60% + ML 30% + Technical 10%)")
    print("="*80)
    
    # 模拟策略
    class MockStrategy(BaseStrategy):
        def __init__(self, name: str):
            self.name = name
        
        def generate_signals(self, date: datetime):
            return []
        
        def prepare_data(self):
            return True
    
    # 创建三个策略
    ff5_strategy = MockStrategy(name="FF5_Core")
    ml_strategy = MockStrategy(name="ML_Satellite")
    tech_strategy = MockStrategy(name="Tech_Tactical")
    
    # 配置资本分配 - 60% FF5 + 30% ML + 10% Technical
    allocation_config = AllocationConfig(
        strategy_allocations=[
            StrategyAllocation(
                strategy_name="FF5_Core",
                target_weight=0.60,
                min_weight=0.55,
                max_weight=0.65,
                priority=1
            ),
            StrategyAllocation(
                strategy_name="ML_Satellite",
                target_weight=0.30,
                min_weight=0.25,
                max_weight=0.35,
                priority=2
            ),
            StrategyAllocation(
                strategy_name="Tech_Tactical",
                target_weight=0.10,
                min_weight=0.05,
                max_weight=0.15,
                priority=3  # 最低优先级
            )
        ]
    )
    
    system_config = create_mock_system_config("ThreeStrategySystem")
    
    # 创建系统编排器（合规规则会自动生成）
    orchestrator = SystemOrchestrator(
        system_config=system_config,
        strategies=[ff5_strategy, ml_strategy, tech_strategy],
        allocation_config=allocation_config
        # compliance_rules 留空，将自动从 allocation_config 生成
    )
    
    # 验证配置
    is_valid, issues = orchestrator.validate_system_configuration()
    print(f"\n配置验证: {'✓ 通过' if is_valid else '✗ 失败'}")
    
    # 显示分配状态
    allocation_status = orchestrator.allocator.get_allocation_status()
    print("\n资本分配配置:")
    for strategy_name, config in allocation_status['config']['strategies'].items():
        print(f"  {strategy_name}:")
        print(f"    - Target: {config['target_weight']:.1%}")
        print(f"    - Range: [{config['min_weight']:.1%}, {config['max_weight']:.1%}]")
        print(f"    - Priority: {config['priority']}")
    
    print("\n✓ 三策略系统配置完成")
    return orchestrator


def demo_single_strategy_system():
    """演示单策略系统配置（仅 ML）"""
    print("\n" + "="*80)
    print("Demo 3: 单策略系统 (仅 ML - 95%)")
    print("="*80)
    
    # 模拟策略
    class MockStrategy(BaseStrategy):
        def __init__(self, name: str):
            self.name = name
        
        def generate_signals(self, date: datetime):
            return []
        
        def prepare_data(self):
            return True
    
    # 创建单个策略
    ml_strategy = MockStrategy(name="ML_Only")
    
    # 配置资本分配 - 95% ML，留 5% 现金
    allocation_config = AllocationConfig(
        strategy_allocations=[
            StrategyAllocation(
                strategy_name="ML_Only",
                target_weight=0.95,
                min_weight=0.90,
                max_weight=1.00,
                priority=1
            )
        ],
        cash_buffer_weight=0.05
    )
    
    system_config = create_mock_system_config("SingleStrategySystem")
    
    orchestrator = SystemOrchestrator(
        system_config=system_config,
        strategies=[ml_strategy],
        allocation_config=allocation_config
    )
    
    is_valid, issues = orchestrator.validate_system_configuration()
    print(f"\n配置验证: {'✓ 通过' if is_valid else '✗ 失败'}")
    
    print("\n资本分配:")
    print(f"  - ML_Only: 95%")
    print(f"  - Cash Buffer: 5%")
    
    print("\n✓ 单策略系统配置完成")
    return orchestrator


def demo_backward_compatibility():
    """演示向后兼容性 - 使用工厂方法创建传统 core-satellite 配置"""
    print("\n" + "="*80)
    print("Demo 4: 向后兼容 - 使用工厂方法创建传统配置")
    print("="*80)
    
    # 使用工厂方法创建传统 core-satellite 配置
    allocation_config = AllocationConfig.create_core_satellite(
        core_target=0.75,
        satellite_target=0.25,
        core_name="CoreStrategy",
        satellite_name="SatelliteStrategy"
    )
    
    compliance_rules = ComplianceRules.create_core_satellite(
        core_min=0.70,
        core_max=0.80,
        satellite_min=0.20,
        satellite_max=0.30,
        core_name="CoreStrategy",
        satellite_name="SatelliteStrategy"
    )
    
    print("\n使用工厂方法创建的配置:")
    print(f"  Strategies: {[a.strategy_name for a in allocation_config.strategy_allocations]}")
    print(f"  Core: {allocation_config.strategy_allocations[0].target_weight:.1%}")
    print(f"  Satellite: {allocation_config.strategy_allocations[1].target_weight:.1%}")
    
    print("\n✓ 向后兼容演示完成")


def create_mock_system_config(name: str):
    """创建模拟的系统配置"""
    from dataclasses import dataclass
    
    @dataclass
    class MockSystemConfig:
        system_name: str = name
        initial_capital: float = 1_000_000
    
    return MockSystemConfig()


def main():
    """运行所有演示"""
    print("\n" + "="*80)
    print("Multi-Strategy Orchestration 重构演示")
    print("="*80)
    print("\n这个演示展示了重构后的系统如何支持:")
    print("  1. 双策略配置")
    print("  2. 三策略配置")
    print("  3. 单策略配置")
    print("  4. 向后兼容（使用工厂方法）")
    
    try:
        # Demo 1: 双策略
        orchestrator_2 = demo_two_strategy_system()
        
        # Demo 2: 三策略
        orchestrator_3 = demo_three_strategy_system()
        
        # Demo 3: 单策略
        orchestrator_1 = demo_single_strategy_system()
        
        # Demo 4: 向后兼容
        demo_backward_compatibility()
        
        print("\n" + "="*80)
        print("✓ 所有演示完成！")
        print("="*80)
        print("\n重构总结:")
        print("  ✓ 支持任意数量策略（1个、2个、3个或更多）")
        print("  ✓ 配置清晰、类型安全")
        print("  ✓ 遵循 SOLID、KISS、YAGNI 原则")
        print("  ✓ 自动生成合规规则")
        print("  ✓ 向后兼容")
        
    except Exception as e:
        print(f"\n✗ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

