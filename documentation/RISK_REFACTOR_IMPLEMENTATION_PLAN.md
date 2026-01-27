# Risk Management & Portfolio Construction Refactoring Plan

This document outlines the detailed steps required to refactor the trading system's risk management and portfolio construction process. The goal is to align the system with the modern, 7-stage architecture described in `documentation/risk.md`.

## Guiding Principles

1.  **Separation of Concerns**: Each component has a single, well-defined responsibility.
2.  **Centralized Optimization**: Portfolio construction and risk management logic is centralized in the orchestrator, not distributed among strategies.
3.  **From Heuristics to Optimization**: Replace heuristic-based allocation (`BoxAllocator`) with a formal portfolio optimization process that uses constraints.

---

## High-Level Change Summary

| Component                       | Action                                                                   | Reason                                                                    |
| ------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `BaseStrategy`                  | **Simplify**                                                             | Strategies should only predict expected returns, not manage risk or allocate. |
| `BoxAllocator`                  | **Remove**                                                               | Replaced by optimization constraints.                                     |
| `MetaModel`                     | **Integrate**                                                            | To be used by the orchestrator to combine signals before optimization.    |
| `SystemOrchestrator`            | **Enhance**                                                              | Becomes the central hub for the new 7-stage portfolio construction process. |
| **New Optimizer Component**     | **Add**                                                                  | A new component to handle the mathematical portfolio optimization.        |
| `risk.py` (CovarianceEstimators) | **Retain & Relocate Usage**                                              | These are well-implemented and will be used by the new orchestrator flow. |

---

## Detailed Implementation Plan

### Phase 1: Decouple Strategies from Allocation [COMPLETED]

The first step is to simplify the strategies so they only produce raw signals (expected returns).

#### 1.1. Modify `BaseStrategy` (`src/trading_system/strategies/base_strategy.py`) [COMPLETED]

-   **REMOVE** the `stock_classifier` and `box_allocator` from the `__init__` method.
-   **REMOVE** the logic in `generate_signals` that calls the `stock_classifier` and `box_allocator` (lines 160-169).
-   **MODIFY** the `generate_signals` method to return the raw `predictions` DataFrame directly. This DataFrame should represent expected returns or signal strengths, not final portfolio weights.
-   **REMOVE** the `position_sizer` dependency. While not actively used for allocation in the `generate_signals` flow, its presence is misleading. The new design centralizes all sizing and risk management.

#### 1.2. Remove `BoxAllocator` (`src/trading_system/allocation/box_allocator.py`) [COMPLETED]

-   **DELETE** the file `src/trading_system/allocation/box_allocator.py`.
-   This component's logic is fundamentally replaced by optimization constraints, making it entirely obsolete.

#### 1.3. Update Strategy Configurations [COMPLETED]

-   **MODIFY** all strategy configuration YAML files to remove the `box_allocator` and `stock_classifier` sections.

### Phase 2: Refactor the `SystemOrchestrator` [COMPLETED]

The orchestrator will be rebuilt to implement the new 7-stage portfolio construction pipeline.

#### 2.1. Create a New Portfolio Optimizer Component [COMPLETED]

-   **ADD** a new file: `src/trading_system/optimization/optimizer.py`.
-   **ADD** a new class `PortfolioOptimizer` within this file.
-   This class will contain the logic for solving the optimization problem:
    -   `maximize: Sharpe Ratio = (w^T μ) / sqrt(w^T Σ w)`
    -   It will take `expected_returns` (μ) and a `covariance_matrix` (Σ) as inputs.
    -   It must be able to handle constraints:
        -   Sum of weights = 1
        -   Individual weight limits
        -   **Box constraints** (e.g., `sum of weights in Sector 'Tech' <= 0.3`).
    -   Use a library like `scipy.optimize.minimize` with the `SLSQP` method.
    -   It will need a helper function, `build_box_constraints`, as sketched out in `documentation/risk.md`.

#### 2.2. Overhaul `SystemOrchestrator` (`src/trading_system/orchestration/system_orchestrator.py`) [COMPLETED]

-   **MODIFY** the `__init__` method:
    -   **ADD** dependencies for `MetaModel`, `CovarianceEstimator` (e.g., `LedoitWolfCovarianceEstimator`), and the new `PortfolioOptimizer`.
    -   The `CapitalAllocator` might be simplified or removed, as its logic is now split between the `MetaModel` (strategy weighting) and `PortfolioOptimizer` (final asset weighting). For this refactoring, we will assume the `MetaModel` handles inter-strategy allocation.
-   **REPLACE** the `run_system` method's logic with the new 7-stage flow:

    1.  **Stage 1 (Signal Generation):**
        -   Call `self.coordinator.coordinate(date)` to get `expected_returns` from all strategies. This part remains similar.

    2.  **Stage 2 (Meta-Model Combination):**
        -   Instantiate and use the `MetaModel` to combine the signals from different strategies into a single `combined_signal` DataFrame.
        -   `combined_signal = self.meta_model.combine(strategy_signals)`

    3.  **Stage 3 (Dimensionality Reduction):**
        -   Implement logic to select the top N stocks based on the absolute values of `combined_signal`. This makes the universe manageable for covariance estimation.
        -   `reduced_universe_signals = combined_signal.abs().nlargest(200)`

    4.  **Stage 4 (Risk Model):**
        -   Use a `CovarianceEstimator` (e.g., `LedoitWolfCovarianceEstimator` from `risk.py`) to calculate the covariance matrix for the assets in the reduced universe.
        -   `cov_matrix = self.covariance_estimator.estimate(price_data, date)`

    5.  **Stage 5 (Stock Classification):**
        -   Use the existing `StockClassifier` to get box classifications for the stocks in the reduced universe.
        -   `classifications = self.stock_classifier.classify_stocks(...)`

    6.  **Stage 6 (Portfolio Optimization):**
        -   Call the new `PortfolioOptimizer` component.
        -   Pass the `reduced_universe_signals`, `cov_matrix`, `classifications`, and box limit configurations to the optimizer.
        -   `final_weights = self.portfolio_optimizer.optimize(...)`

    7.  **Stage 7 (Compliance Check):**
        -   The existing `ComplianceMonitor` can be adapted to check the `final_weights` against risk metrics (VaR, drawdown) and other rules. This stage remains conceptually similar but operates on the final, optimized portfolio.

### Phase 3: Integrate the `MetaModel` [COMPLETED]

Ensure the `MetaModel` is a core, configurable part of the system.

#### 3.1. Modify `meta_model.py` (`src/trading_system/orchestration/meta_model.py`) [COMPLETED]

-   No major changes are likely needed to the class itself, as it's already implemented.
-   Ensure its `fit` method is accessible for periodic retraining and that its `combine` method is used in the `SystemOrchestrator` as described above.

#### 3.2. Update System Configuration [COMPLETED]

-   **MODIFY** the main system configuration file (`configs/system_backtest_config.yaml` or similar) to include a `meta_model` section, specifying the combination method (e.g., `lasso`) and its parameters.

### Phase 4: Final Cleanup and Validation [COMPLETED]

-   **REVIEW** all modified components to ensure they align. [COMPLETED]
-   **DELETE** `test` files related to the `BoxAllocator` and old strategy allocation logic. [COMPLETED]
-   **ADD** new unit and integration tests for the `PortfolioOptimizer` and the new `SystemOrchestrator` workflow. [COMPLETED]
-   **UPDATE** system-level documentation to reflect the new architecture.

This structured plan ensures a methodical transition to the new, more robust risk management and portfolio construction framework.
