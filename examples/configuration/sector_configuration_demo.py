#!/usr/bin/env python3
"""
Demo script showing how to use the new configurable sector functionality.

This demonstrates:
1. Using empty sector list to create 3-dimensional boxes (size × style × region)
2. Using specific sectors to create 4-dimensional boxes (size × style × region × sector)
3. Mixing both approaches in configurable weight method
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.trading_system.portfolio_construction.box_based.box_weight_manager import BoxWeightManager

def demo_empty_sector_configuration():
    """Demonstrate configuration with empty sector list."""
    print("=== Demo 1: Empty Sector Configuration ===")
    print("This creates 3-dimensional boxes without sector classification")
    print("Boxes will be: size × style × region\n")
    
    config = {
        'method': 'equal',
        'dimensions': {
            'size': ['large', 'mid'],
            'style': ['growth', 'value'],
            'region': ['developed', 'emerging'],
            'sector': []  # Empty sector list - ignore sector dimension
        }
    }
    
    manager = BoxWeightManager(config)
    boxes = manager.get_target_boxes()
    weights = manager.get_all_weights()
    
    print(f"Generated {len(boxes)} boxes:")
    for box, weight in weights.items():
        print(f"  {box}: {weight:.4f}")
    
    print(f"\nTotal boxes: {len(boxes)} (2 sizes × 2 styles × 2 regions)")
    print("Each box ignores sector classification - stocks are grouped only by size, style, and region.\n")

def demo_specific_sector_configuration():
    """Demonstrate configuration with specific sectors."""
    print("=== Demo 2: Specific Sector Configuration ===")
    print("This creates 4-dimensional boxes with sector classification")
    print("Boxes will be: size × style × region × sector\n")
    
    config = {
        'method': 'equal',
        'dimensions': {
            'size': ['large'],
            'style': ['growth'],
            'region': ['developed'],
            'sector': ['Technology', 'Healthcare', 'Financials']  # Specific sectors
        }
    }
    
    manager = BoxWeightManager(config)
    boxes = manager.get_target_boxes()
    weights = manager.get_all_weights()
    
    print(f"Generated {len(boxes)} boxes:")
    for box, weight in weights.items():
        print(f"  {box}: {weight:.4f}")
    
    print(f"\nTotal boxes: {len(boxes)} (1 size × 1 style × 1 region × 3 sectors)")
    print("Each box includes sector classification.\n")

def demo_mixed_configuration():
    """Demonstrate configurable weight method with mixed 3D and 4D boxes."""
    print("=== Demo 3: Mixed Configuration (Configurable Weights) ===")
    print("This shows how to manually configure weights with mixed box dimensions\n")
    
    config = {
        'method': 'config',
        'weights': [
            # 3-dimensional box (no sector)
            {
                'box': ['large', 'growth', 'developed'],
                'weight': 0.3
            },
            {
                'box': ['mid', 'value', 'emerging'],
                'weight': 0.3
            },
            # 4-dimensional boxes (with specific sectors)
            {
                'box': ['large', 'growth', 'developed', 'Technology'],
                'weight': 0.2
            },
            {
                'box': ['small', 'value', 'developed', 'Healthcare'],
                'weight': 0.2
            }
        ]
    }
    
    manager = BoxWeightManager(config)
    weights = manager.get_all_weights()
    
    print("Configured weights:")
    for box, weight in weights.items():
        sector_info = f" (sector: {box.sector})" if box.sector else " (no sector)"
        print(f"  {box}: {weight:.4f}{sector_info}")
    
    print(f"\nTotal weight: {sum(weights.values()):.4f}")
    print("This shows how you can mix 3D and 4D boxes in the same configuration.\n")

def demo_yaml_configuration_examples():
    """Show YAML configuration examples."""
    print("=== Demo 4: YAML Configuration Examples ===")
    print("Here are examples of how to configure this in YAML files:\n")
    
    print("Example 1: Empty sector (3D boxes)")
    print("""
box_weights:
  method: "equal"
  dimensions:
    size: ["large", "mid", "small"]
    style: ["growth", "value"]
    region: ["developed", "emerging"]
    sector: []  # Empty list = ignore sector dimension
""")
    
    print("Example 2: Specific sectors (4D boxes)")
    print("""
box_weights:
  method: "equal"
  dimensions:
    size: ["large", "mid"]
    style: ["growth", "value"]
    region: ["developed"]
    sector: ["Technology", "Healthcare", "Financials"]
""")
    
    print("Example 3: Manual configuration with mixed dimensions")
    print("""
box_weights:
  method: "config"
  weights:
    - box: ["large", "growth", "developed"]  # 3D box
      weight: 0.25
    - box: ["large", "growth", "developed", "Technology"]  # 4D box
      weight: 0.25
    - box: ["mid", "value", "emerging"]  # 3D box
      weight: 0.25
    - box: ["mid", "value", "emerging", "Healthcare"]  # 4D box
      weight: 0.25
""")

def main():
    """Run all demos."""
    print("🚀 Sector Configuration Demo")
    print("=" * 50)
    print()
    
    try:
        demo_empty_sector_configuration()
        demo_specific_sector_configuration()
        demo_mixed_configuration()
        demo_yaml_configuration_examples()
        
        print("✅ All demos completed successfully!")
        print("\nKey Benefits:")
        print("• Empty sector list creates 3D boxes (size × style × region)")
        print("• Specific sectors create 4D boxes (size × style × region × sector)")
        print("• You can mix both approaches in configurable weight method")
        print("• Backward compatible - existing configurations still work")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
