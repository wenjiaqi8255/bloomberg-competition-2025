# Momentum Feature Attribution Analysis

## Key Statistics

- **Explicit momentum features**: 15.7%
- **Trend-following technicals**: 30.1%
- **Total trend-dependent signals**: 45.8%

## Short Version (for slides)

Post-analysis: 15.7% of explicit feature importance sat on momentum indicators, rising to 45.8% when including trend-following technical indicators. When momentum reversed, the model's price-trend signals collapsed.

## Full Version (for speaking)

Here's what went wrong: momentum regime shift.

Training period: momentum worked consistently. Trading period—sharp momentum reversal in market conditions. Daniel and Moskowitz's 2016 paper calls this a 'momentum crash.'

The model had no regime detection. No VIX filter, no breadth indicators, nothing to signal 'the rules changed.'

Post-analysis: 15.7% of direct feature importance came from momentum indicators (21d, 63d, 252d periods), with an additional 30.1% from trend-following technical indicators (price above SMAs, EMAs, MACD). Combined, 45.8% of the model's signals relied on price continuation patterns.

When momentum reversed, all these trend-following signals failed simultaneously. The model kept betting on price continuation exactly as markets reversed.

## Technical Bullet Points

1. Explicit momentum features: 15.7% importance (15 features: 21d, 63d, 252d × momentum types)
2. Trend-following technicals: 30.1% (price vs SMAs, EMAs, MACD)
3. Total trend-dependent signals: 45.8%
4. Top individual feature: price_above_sma_10 (3.8%)
5. Model architecture: No regime detection mechanisms
6. Missing defenses: No VIX filter, no breadth indicators, no market state classification
7. Result: Model failed to adapt when momentum regime shifted

## Academic Support

**Citation**: Daniel, K. D., & Moskowitz, T. J. (2016). 'Momentum crashes.' Journal of Financial Economics, 122(2), 221-247.

**Key insight**: Momentum strategies experience infrequent but severe crashes during 'panic states' (market declines + high volatility). Crashes occur when following market declines with high volatility.

**Model deficiency**: Our XGBoost model lacked the proposed safeguards - no VIX filter, no regime detection, no adaptation to market states

## Feature Breakdown

### Explicit Momentum (15.7%)

- momentum_21d, momentum_63d, momentum_252d
- exp_momentum_21d, exp_momentum_63d, exp_momentum_252d
- risk_adj_momentum_21d, risk_adj_momentum_63d, risk_adj_momentum_252d
- momentum_rank_21d, momentum_rank_63d, momentum_rank_252d
- momentum_12m, momentum_12m_rank, momentum_12m_zscore

### Trend-Following Technicals (30.1%)

- price_above_sma_10, price_above_sma_20, price_above_sma_50, price_above_sma_200
- sma_10, sma_20, sma_50, sma_200
- ema_12, ema_26
- macd_line, macd_signal, macd_histogram
