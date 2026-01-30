#!/usr/bin/env python3
"""
Create visualization charts for momentum feature importance analysis.
Generates publication-ready figures for defense presentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

def load_analysis_data():
    """Load feature importance analysis results."""
    with open("analysis/momentum_importance_report.json") as f:
        return json.load(f)

def create_feature_breakdown_chart(data, output_path):
    """Create horizontal bar chart of feature importance by category."""

    breakdown = data['feature_breakdown']
    df = pd.DataFrame(breakdown)

    # Sort by importance
    df = df.sort_values('total_importance', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors
    colors = []
    for cat in df['feature_type']:
        if cat == 'momentum':
            colors.append('#e74c3c')  # Red for momentum
        elif cat == 'technical':
            colors.append('#f39c12')  # Orange for trend-following
        elif cat == 'volatility':
            colors.append('#3498db')  # Blue
        elif cat == 'cross_sectional':
            colors.append('#2ecc71')  # Green
        elif cat == 'box':
            colors.append('#9b59b6')  # Purple
        else:
            colors.append('#95a5a6')  # Gray

    # Create horizontal bar chart
    bars = ax.barh(df['feature_type'], df['pct_importance'], color=colors)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, df['pct_importance'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va='center', fontsize=11, fontweight='bold')

    # Customize
    ax.set_xlabel('Feature Importance (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature Category', fontsize=13, fontweight='bold')
    ax.set_title('XGBoost Feature Importance by Category\n(Total: 84 features)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(df['pct_importance']) * 1.15)

    # Add trend-dependent annotation
    ax.axvline(x=45.8, color='red', linestyle='--', alpha=0.5, linewidth=2)
    y_position = len(df) - 1  # Top of the chart
    ax.text(45.8, y_position + 0.5,
            'Trend-dependent signals: 45.8%',
            color='red', fontsize=10, fontweight='bold',
            ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Chart saved: {output_path}")

def create_top_features_chart(data, output_path):
    """Create horizontal bar chart of top 10 features."""

    top_10 = pd.DataFrame(data['top_10_features'])
    top_10['importance_pct'] = top_10['importance'] * 100

    # Sort by importance
    top_10 = top_10.sort_values('importance_pct', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by feature type
    colors = []
    for feat in top_10['feature']:
        if 'momentum' in feat.lower() or 'exp_mom' in feat.lower():
            colors.append('#e74c3c')  # Red for momentum
        elif any(x in feat for x in ['sma', 'ema', 'macd', 'price_above']):
            colors.append('#f39c12')  # Orange for trend-following
        elif 'volatility' in feat.lower() or 'vol' in feat.lower():
            colors.append('#3498db')  # Blue for volatility
        elif 'box_' in feat.lower():
            colors.append('#9b59b6')  # Purple for box
        else:
            colors.append('#2ecc71')  # Green for others

    # Create horizontal bar chart
    bars = ax.barh(range(len(top_10)), top_10['importance_pct'], color=colors)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_10['importance_pct'])):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{pct:.2f}%", va='center', fontsize=10)

    # Customize
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance (%)', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Features by Importance\nXGBoost Model (xgboost_20251110_010814)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(top_10['importance_pct']) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Momentum'),
        Patch(facecolor='#f39c12', label='Trend-Following'),
        Patch(facecolor='#3498db', label='Volatility'),
        Patch(facecolor='#9b59b6', label='Box Classification'),
        Patch(facecolor='#2ecc71', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Chart saved: {output_path}")

def create_momentum_breakdown_pie(data, output_path):
    """Create pie chart showing trend-dependent vs independent features."""

    breakdown = {item['feature_type']: item for item in data['feature_breakdown']}

    # Calculate categories
    momentum_pct = breakdown['momentum']['pct_importance']
    technical_pct = breakdown['technical']['pct_importance']
    trend_dependent = momentum_pct + technical_pct
    trend_independent = 100 - trend_dependent

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data for pie chart
    sizes = [momentum_pct, technical_pct, trend_independent]
    labels = [f'Explicit Momentum\n({momentum_pct:.1f}%)',
              f'Trend-Following Technicals\n({technical_pct:.1f}%)',
              f'Other Features\n({trend_independent:.1f}%)']
    colors = ['#e74c3c', '#f39c12', '#3498db']
    explode = (0.05, 0.05, 0)  # Explode the momentum slices

    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})

    # Add title
    ax.set_title('Trend-Dependent Signal Exposure\nXGBoost Model Feature Importance',
                 fontsize=14, fontweight='bold', pad=20)

    # Add annotation
    ax.annotate(f'Total Trend-Dependent: {trend_dependent:.1f}%',
                xy=(0.5, -0.1), xycoords='axes fraction',
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Chart saved: {output_path}")

def main():
    """Generate all visualization charts."""

    print("=" * 80)
    print("Momentum Feature Importance - Visualization Generation")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading analysis data...")
    data = load_analysis_data()
    print(f"✅ Loaded data for {data['key_finding']['total_features']} features")

    # Create output directory
    output_dir = Path("presentation/materials/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("\n[2/4] Creating feature breakdown chart...")
    create_feature_breakdown_chart(data, output_dir / "feature_breakdown.png")

    print("\n[3/4] Creating top 10 features chart...")
    create_top_features_chart(data, output_dir / "top_10_features.png")

    print("\n[4/4] Creating momentum breakdown pie chart...")
    create_momentum_breakdown_pie(data, output_dir / "momentum_breakdown_pie.png")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Charts saved to: {output_dir}/")
    print("   - feature_breakdown.png")
    print("   - top_10_features.png")
    print("   - momentum_breakdown_pie.png")

if __name__ == "__main__":
    main()
