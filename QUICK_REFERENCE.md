# Quick Reference: Momentum Feature Importance for Defense

## 🎯 The Key Number: 45.8%

**What it means**: 45.8% of the XGBoost model's feature importance came from trend-dependent signals.

**Why it matters**: When momentum reversed, nearly half of all signals failed simultaneously.

**Academic support**: Daniel & Moskowitz (2016) documented "momentum crashes" causing 50-80% losses.

---

## 📊 Breakdown

- **Explicit momentum features**: 15.7% (15 features)
- **Trend-following technicals**: 30.1% (23 features)
- **Total trend-dependent**: **45.8%**

---

## 💬 What to Say (Short Version)

"Post-analysis: 15.7% of explicit feature importance sat on momentum indicators, rising to 45.8% when including trend-following technical indicators. When momentum reversed, the model's price-trend signals collapsed."

---

## 📁 Files to Reference

**Analysis**:
- `analysis/momentum_importance_report.json` - Raw data

**Presentation**:
- `presentation/materials/momentum_attribution.md` - Full narrative
- `presentation/materials/charts/` - Visualization charts (3 files)

**Documentation**:
- `DEFENSE_PRESENTATION_DATA.md` - Section: "MOMENTUM CRASH ANALYSIS"

---

## 🔬 Technical Details

**Top 5 Features**:
1. price_above_sma_10 (3.8%)
2. price_above_sma_200 (3.7%)
3. box_style_growth (3.0%)
4. box_size_large (2.6%)
5. size_factor_rank (2.2%)

**Model**: XGBoost (xgboost_20251110_010814)
**Features**: 84 total
**Training samples**: 63,394

---

## 🎓 Academic Citation

> Daniel, K. D., & Moskowitz, T. J. (2016). "Momentum crashes." Journal of Financial Economics, 122(2), 221-247.

**Key finding**: Momentum strategies experience infrequent but severe crashes during "panic states" (market declines + high volatility).

---

## ✅ Status: READY FOR DEFENSE

All analysis complete, presentation materials generated, documentation updated.
