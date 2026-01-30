#!/usr/bin/env python3
"""
Extract and analyze feature importance from trained XGBoost model.
Calculates the percentage of importance attributed to momentum features.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
from typing import Dict, List

def load_model(model_id: str = "xgboost_20251110_010814"):
    """Load trained XGBoost model from disk."""
    model_path = Path(f"models/{model_id}/model.joblib")
    model = joblib.load(model_path)
    return model

def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features by type based on naming patterns.

    Returns:
        Dict with keys: momentum, volatility, technical, volume, cross_sectional, box, other
    """
    categories = {
        'momentum': [],
        'volatility': [],
        'technical': [],
        'volume': [],
        'cross_sectional': [],
        'box': [],
        'other': []
    }

    for feature in feature_names:
        # Momentum patterns (14 features total)
        if any(pattern in feature for pattern in [
            'momentum_', 'log_return_', 'exp_momentum', 'risk_adj_momentum'
        ]):
            categories['momentum'].append(feature)

        # Volatility patterns
        elif any(pattern in feature for pattern in [
            'volatility_', 'vol_of_vol_', 'parkinson_vol', 'gk_vol', 'range_volatility'
        ]):
            categories['volatility'].append(feature)

        # Technical indicators
        elif any(pattern in feature for pattern in [
            'sma_', 'ema_', 'macd_', 'bb_', 'rsi_', 'stochastic_', 'williams_r', 'adx', 'cci', 'mfi'
        ]):
            categories['technical'].append(feature)

        # Volume features
        elif any(pattern in feature for pattern in ['volume_', 'obv', 'vwap', 'ad_line']):
            categories['volume'].append(feature)

        # Cross-sectional features
        elif any(pattern in feature for pattern in [
            'market_cap', 'book_to_market', 'size_factor', 'value_factor',
            'country_risk', 'equity_risk', 'default_spread'
        ]):
            categories['cross_sectional'].append(feature)

        # Box features
        elif any(pattern in feature for pattern in ['size_', 'style_', 'region_', 'sector_']):
            categories['box'].append(feature)

        else:
            categories['other'].append(feature)

    return categories

def calculate_importance_by_type(model) -> pd.DataFrame:
    """
    Calculate total and percentage importance by feature type.

    Returns:
        DataFrame with columns: feature_type, total_importance, pct_importance, n_features
    """
    # Get feature importance
    importance_dict = model.get_feature_importance()

    if not importance_dict:
        raise ValueError("Model has no feature importance data")

    # Create DataFrame
    df = pd.DataFrame(list(importance_dict.items()),
                     columns=['feature', 'importance'])

    # Categorize features
    categories = categorize_features(df['feature'].tolist())

    # Map each feature to its type
    feature_to_type = {}
    for feature_type, features in categories.items():
        for feature in features:
            feature_to_type[feature] = feature_type

    df['feature_type'] = df['feature'].map(feature_to_type)

    # Group by type and calculate statistics
    summary = df.groupby('feature_type').agg({
        'importance': ['sum', 'mean', 'count']
    }).reset_index()

    summary.columns = ['feature_type', 'total_importance', 'avg_importance', 'n_features']

    # Calculate percentage
    total_importance = summary['total_importance'].sum()
    summary['pct_importance'] = (summary['total_importance'] / total_importance * 100).round(2)

    # Sort by importance
    summary = summary.sort_values('total_importance', ascending=False)

    return summary, df

def generate_report(summary: pd.DataFrame, full_df: pd.DataFrame) -> Dict:
    """Generate comprehensive report for presentation."""

    momentum_pct = summary[summary['feature_type'] == 'momentum']['pct_importance'].values[0]
    momentum_importance = summary[summary['feature_type'] == 'momentum']['total_importance'].values[0]

    report = {
        "key_finding": {
            "momentum_importance_percentage": f"{momentum_pct:.1f}%",
            "momentum_features_count": int(summary[summary['feature_type'] == 'momentum']['n_features'].values[0]),
            "total_features": int(summary['n_features'].sum())
        },
        "interpretation": {
            "english": f"Post-analysis: {momentum_pct:.1f}% of feature importance sat on momentum indicators. When momentum reversed, the entire signal architecture collapsed.",
            "technical": f"Momentum features account for {momentum_pct:.1f}% of total feature importance ({momentum_importance:.4f} gain). The model's predictions were heavily dependent on momentum continuation signals."
        },
        "feature_breakdown": summary.to_dict('records'),
        "top_10_features": full_df.nlargest(10, 'importance')[
            ['feature', 'importance']
        ].to_dict('records')
    }

    return report

def main():
    """Main execution."""
    print("=" * 80)
    print("XGBoost Feature Importance Analysis - Momentum Attribution")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model()
    print(f"✅ Loaded model with {len(model.get_feature_importance())} features")

    # Calculate importance by type
    print("\n[2/4] Calculating feature importance by type...")
    summary, full_df = calculate_importance_by_type(model)

    # Display summary
    print("\n[3/4] Feature Importance Summary")
    print("-" * 80)
    print(summary.to_string(index=False))

    # Generate report
    print("\n[4/4] Generating report...")
    report = generate_report(summary, full_df)

    # Save report
    output_path = Path("analysis/momentum_importance_report.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved: {output_path}")

    # Print key finding
    print("\n" + "=" * 80)
    print("KEY FINDING FOR PRESENTATION")
    print("=" * 80)
    print(report['interpretation']['english'])
    print("\nTechnical details:")
    print(f"- Momentum importance: {report['key_finding']['momentum_importance_percentage']}")
    print(f"- Momentum features: {report['key_finding']['momentum_features_count']}")
    print(f"- Total features: {report['key_finding']['total_features']}")

    # Print top 10 features
    print("\n" + "-" * 80)
    print("Top 10 Features:")
    print("-" * 80)
    for i, feat in enumerate(report['top_10_features'], 1):
        print(f"{i:2d}. {feat['feature']:40s} {feat['importance']:.4f}")

    return report

if __name__ == "__main__":
    main()
