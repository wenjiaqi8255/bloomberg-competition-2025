"""
Strategy Module Usage Demo
==========================

This script demonstrates how to use the refactored strategy module
with dependency injection and factory patterns.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading_system.strategies.factory import create_strategy_from_config


def demo_dual_momentum():
    """Demo: Create and use Dual Momentum strategy"""
    print("\n" + "="*80)
    print("Demo 1: Dual Momentum Strategy")
    print("="*80)
    
    # Configuration
    config = {
        'type': 'dual_momentum',
        'name': 'DualMomentum252',
        'lookback_period': 252,
        'top_n': 5,
        'min_momentum': 0.0,
        'position_sizing': {
            'volatility_target': 0.15,
            'max_position_weight': 0.10
        }
    }
    
    # Create strategy using factory (dependencies auto-injected)
    print("\n1. Creating strategy from config...")
    strategy = create_strategy_from_config(config)
    
    print(f"✓ Strategy created: {strategy.name}")
    print(f"  Type: {strategy.__class__.__name__}")
    print(f"  Lookback: {strategy.lookback_period} days")
    print(f"  Top N: {strategy.top_n}")
    
    # Get strategy info
    print("\n2. Strategy information:")
    info = strategy.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Dual Momentum demo completed!")


def demo_fama_french():
    """Demo: Create and use Fama-French 5-Factor strategy"""
    print("\n" + "="*80)
    print("Demo 2: Fama-French 5-Factor Strategy")
    print("="*80)
    
    # Configuration
    config = {
        'type': 'fama_french',
        'name': 'FF5',
        'lookback_days': 252,
        'risk_free_rate': 0.02,
        'factor_weights': {
            'MKT': 0.30,
            'SMB': 0.15,
            'HML': 0.20,
            'RMW': 0.20,
            'CMA': 0.15
        },
        'position_sizing': {
            'volatility_target': 0.15,
            'max_position_weight': 0.10
        }
    }
    
    # Create strategy
    print("\n1. Creating strategy from config...")
    strategy = create_strategy_from_config(config)
    
    print(f"✓ Strategy created: {strategy.name}")
    print(f"  Type: {strategy.__class__.__name__}")
    print(f"  Lookback: {strategy.lookback_days} days")
    print(f"  Risk-free rate: {strategy.risk_free_rate}")
    
    # Show factor weights
    print("\n2. Factor weights:")
    for factor, weight in strategy.factor_weights.items():
        print(f"  {factor}: {weight:.2%}")
    
    print("\n✓ Fama-French demo completed!")


def demo_ml_strategy():
    """Demo: Create and use ML strategy"""
    print("\n" + "="*80)
    print("Demo 3: ML Strategy")
    print("="*80)
    
    # Configuration
    config = {
        'type': 'ml',
        'name': 'MLStrategy_v1',
        'model_id': 'demo_model',
        'min_signal_strength': 0.1,
        'position_sizing': {
            'volatility_target': 0.15,
            'max_position_weight': 0.10
        }
    }
    
    # Create strategy
    print("\n1. Creating strategy from config...")
    try:
        strategy = create_strategy_from_config(config)
        
        print(f"✓ Strategy created: {strategy.name}")
        print(f"  Type: {strategy.__class__.__name__}")
        print(f"  Min signal strength: {strategy.min_signal_strength}")
        
        # Get strategy info
        print("\n2. Strategy components:")
        info = strategy.get_info()
        print(f"  Model ID: {info.get('model_id', 'N/A')}")
        print(f"  Feature engine: Configured")
        print(f"  Position sizer: Configured")
        
    except Exception as e:
        print(f"⚠ ML strategy demo skipped: {e}")
        print("  (This is expected if ModelPredictor is not fully set up)")
    
    print("\n✓ ML strategy demo completed!")


def demo_manual_creation():
    """Demo: Manual strategy creation with explicit dependencies"""
    print("\n" + "="*80)
    print("Demo 4: Manual Strategy Creation")
    print("="*80)
    
    from src.trading_system.feature_engineering.utils.technical_features import TechnicalIndicatorCalculator
    from src.trading_system.utils.position_sizer import PositionSizer
    from src.trading_system.strategies import DualMomentumStrategy
    
    # Create dependencies manually
    print("\n1. Creating dependencies...")
    technical_calculator = TechnicalIndicatorCalculator()
    position_sizer = PositionSizer(
        volatility_target=0.15,
        max_position_weight=0.10
    )
    print("✓ Dependencies created")
    
    # Create strategy with dependency injection
    print("\n2. Injecting dependencies into strategy...")
    strategy = DualMomentumStrategy(
        name="ManualDM",
        technical_calculator=technical_calculator,
        position_sizer=position_sizer,
        lookback_period=252,
        top_n=5
    )
    
    print(f"✓ Strategy created: {strategy.name}")
    print(f"  Direct dependency injection successful!")
    
    print("\n✓ Manual creation demo completed!")


def demo_comparison():
    """Demo: Compare different creation methods"""
    print("\n" + "="*80)
    print("Demo 5: Creation Method Comparison")
    print("="*80)
    
    # Method 1: Factory with auto-injection
    print("\n1. Factory with auto-injection (RECOMMENDED):")
    config = {
        'type': 'dual_momentum',
        'name': 'DM_Auto',
        'lookback_period': 252,
        'top_n': 5,
        'position_sizing': {'volatility_target': 0.15, 'max_position_weight': 0.10}
    }
    strategy1 = create_strategy_from_config(config)
    print(f"   ✓ Created: {strategy1.name}")
    print(f"   - Dependencies: Auto-injected")
    print(f"   - Code: Minimal, config-driven")
    
    # Method 2: Manual creation
    print("\n2. Manual creation:")
    from src.trading_system.feature_engineering.utils.technical_features import TechnicalIndicatorCalculator
    from src.trading_system.utils.position_sizer import PositionSizer
    from src.trading_system.strategies import DualMomentumStrategy
    
    strategy2 = DualMomentumStrategy(
        name="DM_Manual",
        technical_calculator=TechnicalIndicatorCalculator(),
        position_sizer=PositionSizer(0.15, 0.10),
        lookback_period=252,
        top_n=5
    )
    print(f"   ✓ Created: {strategy2.name}")
    print(f"   - Dependencies: Manually created")
    print(f"   - Code: More verbose, full control")
    
    print("\n✓ Comparison demo completed!")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("STRATEGY MODULE USAGE DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows how to use the refactored strategy module.")
    print("All strategies follow dependency injection principles (SOLID).")
    
    try:
        # Run demos
        demo_dual_momentum()
        demo_fama_french()
        demo_ml_strategy()
        demo_manual_creation()
        demo_comparison()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n✅ All demos completed successfully!")
        print("\nKey takeaways:")
        print("1. Use create_strategy_from_config() for easy, config-driven setup")
        print("2. Manual creation gives you full control over dependencies")
        print("3. All strategies follow the same dependency injection pattern")
        print("4. Factory handles dependency creation automatically")
        print("5. Strategies are focused and testable (SOLID principles)")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

