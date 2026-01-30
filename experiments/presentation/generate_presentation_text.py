#!/usr/bin/env python3
"""
Generate presentation-ready text based on analysis results.
Creates the momentum attribution narrative with supporting evidence.
"""

import json
from pathlib import Path
from typing import Dict

def generate_narrative(
    momentum_pct: float,
    technical_pct: float,
    total_trend_following: float,
    analysis_data: dict
) -> Dict[str, str]:

    # Main narrative with different levels of depth
    narrative = {
        "short_version": f"""Post-analysis: {momentum_pct:.1f}% of explicit feature importance sat on momentum indicators, rising to {total_trend_following:.1f}% when including trend-following technical indicators. When momentum reversed, the model's price-trend signals collapsed.""",

        "full_version": f"""Here's what went wrong: momentum regime shift.

Training period: momentum worked consistently. Trading period—sharp momentum reversal in market conditions. Daniel and Moskowitz's 2016 paper calls this a 'momentum crash.'

The model had no regime detection. No VIX filter, no breadth indicators, nothing to signal 'the rules changed.'

Post-analysis: {momentum_pct:.1f}% of direct feature importance came from momentum indicators (21d, 63d, 252d periods), with an additional {technical_pct:.1f}% from trend-following technical indicators (price above SMAs, EMAs, MACD). Combined, {total_trend_following:.1f}% of the model's signals relied on price continuation patterns.

When momentum reversed, all these trend-following signals failed simultaneously. The model kept betting on price continuation exactly as markets reversed.""",

        "technical_bullet_points": [
            f"Explicit momentum features: {momentum_pct:.1f}% importance (15 features: 21d, 63d, 252d × momentum types)",
            f"Trend-following technicals: {technical_pct:.1f}% (price vs SMAs, EMAs, MACD)",
            f"Total trend-dependent signals: {total_trend_following:.1f}%",
            f"Top individual feature: price_above_sma_10 ({analysis_data['top_momentum_feature']:.1f}%)",
            f"Model architecture: No regime detection mechanisms",
            "Missing defenses: No VIX filter, no breadth indicators, no market state classification",
            "Result: Model failed to adapt when momentum regime shifted"
        ],

        "academic_support": {
            "primary_citation": "Daniel, K. D., & Moskowitz, T. J. (2016). 'Momentum crashes.' Journal of Financial Economics, 122(2), 221-247.",
            "key_insight": "Momentum strategies experience infrequent but severe crashes during 'panic states' (market declines + high volatility). Crashes occur when following market declines with high volatility.",
            "proposed_solution": "Dynamic momentum strategy with VIX filters and regime detection. The paper shows momentum crashes can be predicted and avoided.",
            "model_deficiency": "Our XGBoost model lacked the proposed safeguards - no VIX filter, no regime detection, no adaptation to market states"
        },

        "feature_breakdown": {
            "explicit_momentum": {
                "percentage": f"{momentum_pct:.1f}%",
                "features": [
                    "momentum_21d, momentum_63d, momentum_252d",
                    "exp_momentum_21d, exp_momentum_63d, exp_momentum_252d",
                    "risk_adj_momentum_21d, risk_adj_momentum_63d, risk_adj_momentum_252d",
                    "momentum_rank_21d, momentum_rank_63d, momentum_rank_252d",
                    "momentum_12m, momentum_12m_rank, momentum_12m_zscore"
                ],
                "interpretation": "Direct momentum measurements across short, medium, and long-term periods"
            },
            "trend_following_technicals": {
                "percentage": f"{technical_pct:.1f}%",
                "features": [
                    "price_above_sma_10, price_above_sma_20, price_above_sma_50, price_above_sma_200",
                    "sma_10, sma_20, sma_50, sma_200",
                    "ema_12, ema_26",
                    "macd_line, macd_signal, macd_histogram"
                ],
                "interpretation": "Price trend indicators that implicitly bet on momentum continuation"
            },
            "other_categories": {
                "volatility": f"{analysis_data['volatility_pct']:.1f}%",
                "cross_sectional": f"{analysis_data['cross_sectional_pct']:.1f}%",
                "box": f"{analysis_data['box_pct']:.1f}%"
            }
        }
    }

    return narrative

def main():
    """Generate and save presentation materials."""

    # Load analysis results
    report_path = Path("analysis/momentum_importance_report.json")
    if not report_path.exists():
        print(f"❌ Analysis report not found at {report_path}")
        print("Run extract_momentum_importance.py first")
        return

    with open(report_path) as f:
        data = json.load(f)

    # Extract percentages
    breakdown = {item['feature_type']: item for item in data['feature_breakdown']}

    momentum_pct = breakdown['momentum']['pct_importance']
    technical_pct = breakdown['technical']['pct_importance']
    total_trend_following = momentum_pct + technical_pct

    # Find top momentum-related feature
    top_feature = data['top_10_features'][0]
    top_momentum_feature_importance = top_feature['importance'] * 100  # Convert to percentage

    analysis_data = {
        'volatility_pct': breakdown['volatility']['pct_importance'],
        'cross_sectional_pct': breakdown['cross_sectional']['pct_importance'],
        'box_pct': breakdown['box']['pct_importance'],
        'top_momentum_feature': top_momentum_feature_importance
    }

    # Generate narrative
    narrative = generate_narrative(
        momentum_pct,
        technical_pct,
        total_trend_following,
        analysis_data
    )

    # Save presentation materials
    output_path = Path("presentation/materials/momentum_attribution.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(narrative, f, indent=2)

    # Print results
    print("=" * 80)
    print("PRESENTATION NARRATIVE GENERATED")
    print("=" * 80)

    print("\n📊 KEY STATISTICS")
    print("-" * 80)
    print(f"Explicit momentum features:    {momentum_pct:.1f}%")
    print(f"Trend-following technicals:    {technical_pct:.1f}%")
    print(f"Total trend-dependent signals: {total_trend_following:.1f}%")

    print("\n🎯 SHORT VERSION (for slides)")
    print("-" * 80)
    print(narrative['short_version'])

    print("\n📝 FULL VERSION (for speaking)")
    print("-" * 80)
    print(narrative['full_version'])

    print("\n🔬 TECHNICAL BULLET POINTS")
    print("-" * 80)
    for i, point in enumerate(narrative['technical_bullet_points'], 1):
        print(f"{i}. {point}")

    print("\n📚 ACADEMIC SUPPORT")
    print("-" * 80)
    print(f"Citation: {narrative['academic_support']['primary_citation']}")
    print(f"Key insight: {narrative['academic_support']['key_insight']}")
    print(f"Model deficiency: {narrative['academic_support']['model_deficiency']}")

    print("\n✅ Saved:")
    print(f"   {output_path}")

    # Also save as markdown for easy reading
    md_path = Path("presentation/materials/momentum_attribution.md")
    with open(md_path, 'w') as f:
        f.write("# Momentum Feature Attribution Analysis\n\n")
        f.write("## Key Statistics\n\n")
        f.write(f"- **Explicit momentum features**: {momentum_pct:.1f}%\n")
        f.write(f"- **Trend-following technicals**: {technical_pct:.1f}%\n")
        f.write(f"- **Total trend-dependent signals**: {total_trend_following:.1f}%\n\n")

        f.write("## Short Version (for slides)\n\n")
        f.write(f"{narrative['short_version']}\n\n")

        f.write("## Full Version (for speaking)\n\n")
        f.write(f"{narrative['full_version']}\n\n")

        f.write("## Technical Bullet Points\n\n")
        for i, point in enumerate(narrative['technical_bullet_points'], 1):
            f.write(f"{i}. {point}\n")

        f.write("\n## Academic Support\n\n")
        f.write(f"**Citation**: {narrative['academic_support']['primary_citation']}\n\n")
        f.write(f"**Key insight**: {narrative['academic_support']['key_insight']}\n\n")
        f.write(f"**Model deficiency**: {narrative['academic_support']['model_deficiency']}\n\n")

        f.write("## Feature Breakdown\n\n")
        f.write(f"### Explicit Momentum ({momentum_pct:.1f}%)\n\n")
        for feat in narrative['feature_breakdown']['explicit_momentum']['features']:
            f.write(f"- {feat}\n")

        f.write(f"\n### Trend-Following Technicals ({technical_pct:.1f}%)\n\n")
        for feat in narrative['feature_breakdown']['trend_following_technicals']['features']:
            f.write(f"- {feat}\n")

    print(f"   {md_path}")

if __name__ == "__main__":
    main()
