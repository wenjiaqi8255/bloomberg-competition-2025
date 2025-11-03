"""
Box-Based portfolio construction strategy.

Implements the Box-First methodology where stocks are first grouped
into boxes, then selected and weighted within each box according to
configured strategies.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder, IBoxSelector
from src.trading_system.portfolio_construction.models.types import PortfolioConstructionRequest, BoxConstructionResult, BoxKey
from src.trading_system.portfolio_construction.box_based.box_weight_manager import BoxWeightManager
from src.trading_system.portfolio_construction.box_based.weight_allocator import WeightAllocatorFactory
from src.trading_system.portfolio_construction.utils.adapters import ClassificationAdapter
from src.trading_system.portfolio_construction.models.exceptions import (
    PortfolioConstructionError, ClassificationError, WeightAllocationError, InvalidConfigError
)
from src.trading_system.data.stock_classifier import StockClassifier

logger = logging.getLogger(__name__)


class SignalBasedBoxSelector(IBoxSelector):
    """
    Selects stocks within a box based on signal strengths.

    Default implementation that selects the top N stocks
    by signal value within each box.
    """

    def select_stocks(self, candidate_stocks: List[str], signals: pd.Series,
                     n_stocks: int) -> List[str]:
        """
        Select top N stocks by signal strength.

        Args:
            candidate_stocks: Available stocks in the box
            signals: Signal strengths for all stocks
            n_stocks: Number of stocks to select

        Returns:
            List of selected stock symbols
        """
        if not candidate_stocks:
            logger.warning("No candidate stocks provided for selection")
            return []

        # Filter to stocks with signals
        stocks_with_signals = [s for s in candidate_stocks if s in signals]

        if not stocks_with_signals:
            logger.warning("No stocks with signals found in candidates")
            # Return first N stocks as fallback
            return candidate_stocks[:min(n_stocks, len(candidate_stocks))]

        # Sort by signal strength (descending)
        sorted_stocks = sorted(
            stocks_with_signals,
            key=lambda s: signals.get(s, 0),
            reverse=True
        )

        # Select top N
        selected = sorted_stocks[:min(n_stocks, len(sorted_stocks))]
        logger.debug(f"Selected {len(selected)} stocks from {len(candidate_stocks)} candidates")

        return selected


class BoxBasedPortfolioBuilder(IPortfolioBuilder):
    """
    Box-First portfolio construction strategy.

    Implements a systematic approach to portfolio construction:
    1. Classify stocks into boxes
    2. Select top stocks within each box
    3. Allocate box weights according to configuration
    4. Distribute weights within boxes
    """

    def __init__(self, config: Dict[str, Any], factor_data_provider=None):
        """
        Initialize Box-Based portfolio builder.

        Args:
            config: Configuration dictionary for box-based construction
            factor_data_provider: Optional factor data provider for factor model covariance
        """
        self.config = config
        self.factor_data_provider = factor_data_provider
        self._initialize_components()
        self._validate_configuration()

        logger.info(f"Initialized BoxBasedPortfolioBuilder with {len(self.box_weight_manager.get_target_boxes())} target boxes")

    def _initialize_components(self) -> None:
        """Initialize all sub-components."""
        # Stock classifier
        classifier_config = self.config.get('classifier', {})
        
        # Check if sector should be included based on box_weights configuration
        box_weights_config = self.config.get('box_weights', {})
        dimensions = box_weights_config.get('dimensions', {})
        sector_dimension = dimensions.get('sector', [])
        
        # If sector dimension is empty, don't include sector in classification
        include_sector = len(sector_dimension) > 0
        classifier_config['include_sector'] = include_sector
        
        logger.info(f"StockClassifier configured with include_sector={include_sector}")
        self.stock_classifier = StockClassifier(classifier_config)

        # Box weight manager
        self.box_weight_manager = BoxWeightManager(box_weights_config)

        # Selection parameters
        self.stocks_per_box = self.config.get('stocks_per_box', 3)
        self.min_stocks_per_box = self.config.get('min_stocks_per_box', 1)

        # Weight allocation
        allocation_method = self.config.get('allocation_method', 'equal')
        allocation_config = self.config.get('allocation_config', {}) or {}
        # Convert Pydantic model to dict if needed
        if hasattr(allocation_config, 'model_dump'):
            allocation_config = allocation_config.model_dump()
        elif not isinstance(allocation_config, dict):
            allocation_config = {}
        self.weight_allocator = WeightAllocatorFactory.create_allocator(
            allocation_method, allocation_config, factor_data_provider=self.factor_data_provider
        )

        # Box selector (can be overridden via config)
        self.box_selector = SignalBasedBoxSelector()
        
        # Centralized constraints
        self.constraints = self.config.get('constraints', {})

        # Allocation scope: 'within_box' (default, existing behavior) or 'global'
        self.allocation_scope = self.config.get('allocation_scope', 'within_box')

    def _validate_configuration(self) -> None:
        """Validate builder configuration."""
        # Validate box weight manager
        is_valid, errors = self.box_weight_manager.validate_configuration()
        if not is_valid:
            raise InvalidConfigError(
                f"Box weight configuration invalid: {errors}",
                config_section='box_weights'
            )

        # Validate numeric parameters
        if self.stocks_per_box <= 0:
            raise InvalidConfigError("stocks_per_box must be positive")

        if self.min_stocks_per_box <= 0:
            raise InvalidConfigError("min_stocks_per_box must be positive")

        if self.min_stocks_per_box > self.stocks_per_box:
            raise InvalidConfigError("min_stocks_per_box cannot exceed stocks_per_box")

        logger.info("Box-based builder configuration validation passed")

    def build_portfolio(self, request: PortfolioConstructionRequest) -> pd.Series:
        """
        Build portfolio using Box-First methodology.

        Args:
            request: Portfolio construction request

        Returns:
            Series of portfolio weights
        """
        logger.info(f"Building Box-Based portfolio for {request.date.date()}")
        construction_log = []

        try:
            # Step 1: Classify stocks into boxes
            logger.info("[Step 1/4] Classifying stocks into boxes...")
            classifications = self._classify_stocks(request.universe, request.price_data, request.date)
            construction_log.append(f"Classified {len(classifications)} stocks into boxes")

            # Step 2: Group stocks by boxes and analyze coverage
            logger.info("[Step 2/4] Analyzing box coverage...")
            box_stocks = ClassificationAdapter.group_stocks_by_box(request.universe, classifications)
            
            # Check if we have any stocks to work with
            if not box_stocks:
                raise PortfolioConstructionError("No stocks available for portfolio construction")
            
            # Analyze coverage for logging purposes
            coverage_info = self.box_weight_manager.get_coverage_info(list(box_stocks.keys()))
            construction_log.append(f"Box coverage: {coverage_info['coverage_ratio']:.1%} "
                                   f"({coverage_info['covered_boxes']}/{coverage_info['total_target_boxes']})")
            construction_log.append(f"Using {len(box_stocks)} available boxes for portfolio construction")
            
            # Continue with available boxes instead of failing on partial coverage

            # Step 3: Select stocks and allocate weights
            logger.info("[Step 3/4] Selecting stocks and allocating weights...")
            if self.allocation_scope == 'global':
                final_weights, box_results = self._construct_global_allocation(
                    box_stocks, request.signals, request.price_data, request.date
                )
            else:
                final_weights, box_results = self._construct_from_boxes(
                    box_stocks, request.signals, request.price_data, request.date
                )
            construction_log.extend(box_results['log'])

            # Step 4: Normalize final weights
            logger.info("[Step 4/4] Normalizing final weights...")
            normalized_weights = self._normalize_weights(final_weights)
            construction_log.append(f"Final portfolio: {len(normalized_weights)} positions")

            # Step 5: Apply constraints
            logger.info("[Step 5/5] Applying constraints...")
            constrained_weights = self._apply_constraints(normalized_weights)
            construction_log.append(f"Applied constraints: max_position_weight={self.constraints.get('max_position_weight', 'None')}, max_leverage={self.constraints.get('max_leverage', 'None')}")

            # Log final portfolio summary
            logger.info("🎯 FINAL PORTFOLIO SUMMARY:")
            logger.info(f"   📊 Total positions: {len(constrained_weights)}")
            logger.info(f"   💰 Total weight: {sum(constrained_weights.values()):.4f}")
            
            # Log top positions
            sorted_positions = sorted(constrained_weights.items(), key=lambda x: x[1], reverse=True)
            top_positions = sorted_positions[:10]  # Top 10 positions
            logger.info("   🏆 Top positions:")
            for stock, weight in top_positions:
                logger.info(f"      {stock}: {weight:.4f} ({weight*100:.2f}%)")

            logger.info(f"✅ Box-Based portfolio construction completed: {len(constrained_weights)} positions")
            return pd.Series(constrained_weights)

        except Exception as e:
            logger.error(f"Box-Based portfolio construction failed: {e}")
            raise PortfolioConstructionError(f"Box-Based construction failed: {e}")

    def build_portfolio_with_result(self, request: PortfolioConstructionRequest) -> BoxConstructionResult:
        """
        Build portfolio with detailed construction information.

        Args:
            request: Portfolio construction request

        Returns:
            Detailed construction result
        """
        # Get basic weights
        weights = self.build_portfolio(request)

        # Get detailed construction information
        classifications = self._classify_stocks(request.universe, request.price_data, request.date)
        box_stocks = ClassificationAdapter.group_stocks_by_box(request.universe, classifications)
        coverage_info = self.box_weight_manager.get_coverage_info(list(box_stocks.keys()))

        # Build box coverage dictionary
        box_coverage = {}
        for box_key in self.box_weight_manager.get_target_boxes():
            if box_key in box_stocks:
                box_coverage[str(box_key)] = 1.0
            else:
                box_coverage[str(box_key)] = 0.0

        return BoxConstructionResult(
            weights=weights,
            box_coverage=box_coverage,
            selected_stocks={str(k): v for k, v in box_stocks.items()},
            target_weights={str(k): v for k, v in self.box_weight_manager.get_all_weights().items()},
            construction_log=["Box-Based construction completed"]
        )

    def _classify_stocks(self, universe: List[str], price_data: Dict[str, pd.DataFrame],
                        as_of_date: datetime) -> Dict[str, BoxKey]:
        """
        Classify stocks into boxes.

        Args:
            universe: List of stocks to classify
            price_data: Price data for classification
            as_of_date: Classification date

        Returns:
            Dictionary mapping symbols to BoxKey objects
        """
        try:
            investment_boxes = self.stock_classifier.classify_stocks(
                universe, price_data, as_of_date=as_of_date
            )
            box_keys = ClassificationAdapter.convert_investment_boxes_to_box_keys(
                investment_boxes, 
                include_sector=self.stock_classifier.include_sector
            )

            logger.debug(f"Classified {len(box_keys)} stocks into boxes")
            return box_keys

        except Exception as e:
            raise ClassificationError("Stock classification failed", str(e))

    def _construct_from_boxes(self, box_stocks: Dict[BoxKey, List[str]],
                            signals: pd.Series, price_data: Dict[str, pd.DataFrame],
                            date: datetime) -> tuple[Dict[str, float], Dict[str, Any]]:
        """
        Construct portfolio by processing each box.

        Args:
            box_stocks: Dictionary mapping boxes to available stocks
            signals: Signal strengths for all stocks
            price_data: Price data for all stocks
            date: Construction date

        Returns:
            Tuple of (weights, construction_info)
        """
        final_weights = {}
        construction_log = []
        box_results = {
            'processed_boxes': 0,
            'skipped_boxes': 0,
            'total_stocks_selected': 0,
            'log': construction_log
        }

        # Use available boxes instead of target boxes
        available_boxes = list(box_stocks.keys())
        logger.info(f"Processing {len(available_boxes)} available boxes for portfolio construction")

        for box_key in available_boxes:
            try:
                box_result = self._process_single_box(
                    box_key, box_stocks.get(box_key, []), signals, price_data, date
                )

                if box_result['weights']:
                    final_weights.update(box_result['weights'])
                    box_results['processed_boxes'] += 1
                    box_results['total_stocks_selected'] += len(box_result['weights'])
                    construction_log.append(box_result['log'])
                else:
                    box_results['skipped_boxes'] += 1
                    construction_log.append(f"Skipped box {box_key}: {box_result['reason']}")

            except Exception as e:
                logger.error(f"Error processing box {box_key}: {e}")
                box_results['skipped_boxes'] += 1
                construction_log.append(f"Error in box {box_key}: {str(e)}")

        construction_log.insert(0, f"Processed {box_results['processed_boxes']} boxes, "
                                   f"skipped {box_results['skipped_boxes']}, "
                                   f"selected {box_results['total_stocks_selected']} stocks")

        return final_weights, box_results

    def _construct_global_allocation(self, box_stocks: Dict[BoxKey, List[str]],
                                     signals: pd.Series, price_data: Dict[str, pd.DataFrame],
                                     date: datetime) -> tuple[Dict[str, float], Dict[str, Any]]:
        """
        Box 仅用于筛选，汇总所有选中的股票后执行一次全局均值方差优化。
        """
        construction_log = []

        # 1) 每个 box 内按已有 selector 选股（不在 box 内分配权重）
        selected_all: List[str] = []
        for box_key, candidates in box_stocks.items():
            if len(candidates) < self.min_stocks_per_box:
                construction_log.append(f"Skipped box {box_key}: insufficient stocks")
                continue
            selected = self.box_selector.select_stocks(candidates, signals, self.stocks_per_box)
            if selected:
                selected_all.extend(selected)
                construction_log.append(f"Box {box_key}: selected {len(selected)} stocks")
            else:
                construction_log.append(f"Box {box_key}: no selection")

        # 去重，保持顺序
        if selected_all:
            seen = set()
            selected_all = [s for s in selected_all if not (s in seen or seen.add(s))]

        if len(selected_all) < 2:
            return {}, {'log': construction_log + ["Not enough stocks for global optimization"]}

        # 2) 期望收益（用 signals）
        import pandas as _pd
        expected_returns = _pd.Series({s: signals.get(s, 0.0) for s in selected_all})

        # 3) 简单协方差估计：从 price_data 计算历史收益协方差
        returns_series_list = []
        for s in selected_all:
            df = price_data.get(s)
            if df is None or df.empty:
                continue
            try:
                px = df.sort_index()
                # 兼容不同列名，优先使用 'Close'
                if 'Close' in px.columns:
                    ret = px['Close'].pct_change().dropna()
                else:
                    # 使用第一列作为价格列的保守回退
                    first_col = px.columns[0]
                    ret = px[first_col].pct_change().dropna()
                returns_series_list.append(ret.rename(s))
            except Exception:
                continue

        if not returns_series_list:
            return {}, {'log': construction_log + ["No price data for covariance"]}

        ret_df = _pd.concat(returns_series_list, axis=1).dropna()
        if ret_df.shape[1] < 2:
            return {}, {'log': construction_log + ["Insufficient assets for covariance"]}

        cov = ret_df.cov()

        # 对齐索引，避免不一致
        common = expected_returns.index.intersection(cov.index)
        if len(common) < 2:
            return {}, {'log': construction_log + ["Insufficient overlapping assets for optimization"]}
        expected_returns = expected_returns.loc[common]
        cov = cov.loc[common, common]

        # 4) 全局优化（无 box 权重约束）
        from src.trading_system.optimization.optimizer import PortfolioOptimizer
        risk_aversion = 2.0
        alloc_cfg = self.config.get('allocation_config', {}) or {}
        if isinstance(alloc_cfg, dict):
            risk_aversion = alloc_cfg.get('risk_aversion', risk_aversion)

        optimizer = PortfolioOptimizer({
            'method': 'mean_variance',
            'risk_aversion': risk_aversion,
            'enable_short_selling': False,
            'max_position_weight': self.constraints.get('max_position_weight')
        })

        constraints: List[Dict[str, Any]] = []
        optimal = optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov,
            constraints=constraints
        )

        weights = optimal.to_dict()
        return weights, {'log': construction_log + [f"Global MV optimized {len(weights)} positions"]}

    def _process_single_box(self, box_key: BoxKey, candidate_stocks: List[str],
                           signals: pd.Series, price_data: Dict[str, pd.DataFrame],
                           date: datetime) -> Dict[str, Any]:
        """
        Process a single box: select stocks and allocate weights.

        Args:
            box_key: Box to process
            candidate_stocks: Available stocks in the box
            signals: Signal strengths for all stocks
            price_data: Price data for all stocks
            date: Construction date

        Returns:
            Dictionary with processing results
        """
        # Check minimum stock requirement
        if len(candidate_stocks) < self.min_stocks_per_box:
            return {
                'weights': {},
                'reason': f"Insufficient stocks ({len(candidate_stocks)} < {self.min_stocks_per_box})"
            }

        # Select stocks within the box
        selected_stocks = self.box_selector.select_stocks(
            candidate_stocks, signals, self.stocks_per_box
        )

        # Log detailed stock selection information
        logger.info(f"📦 Box {box_key}:")
        logger.info(f"   📊 Available stocks: {len(candidate_stocks)} ({', '.join(candidate_stocks[:5])}{'...' if len(candidate_stocks) > 5 else ''})")
        logger.info(f"   🎯 Selected stocks: {len(selected_stocks)} ({', '.join(selected_stocks)})")
        
        # Log signal strengths for selected stocks
        if selected_stocks and not signals.empty:
            signal_info = []
            for stock in selected_stocks:
                signal_value = signals.get(stock, 0)
                signal_info.append(f"{stock}({signal_value:.4f})")
            logger.info(f"   📈 Signal strengths: {', '.join(signal_info)}")

        if not selected_stocks:
            logger.warning(f"   ❌ No stocks selected from box {box_key}")
            return {
                'weights': {},
                'reason': "No stocks selected from box"
            }

        # Get target weight for this box
        target_weight = self.box_weight_manager.get_target_weight(box_key)
        if target_weight <= 0:
            return {
                'weights': {},
                'reason': f"Box has zero target weight"
            }

        # Allocate weights within the box
        try:
            stock_weights = self.weight_allocator.allocate(
                selected_stocks, target_weight, signals, price_data, date
            )

            # Log detailed weight allocation
            logger.info(f"   💰 Box weight: {target_weight:.4f} ({target_weight*100:.2f}%)")
            weight_info = []
            for stock, weight in stock_weights.items():
                weight_info.append(f"{stock}({weight:.4f})")
            logger.info(f"   ⚖️  Stock weights: {', '.join(weight_info)}")

            return {
                'weights': stock_weights,
                'reason': None,
                'box_weight': target_weight,
                'selected_stocks': selected_stocks,
                'log': f"Box {box_key}: {len(selected_stocks)} stocks selected, weight {target_weight:.4f}"
            }

        except Exception as e:
            logger.error(f"Weight allocation failed for box {box_key}: {e}")
            return {
                'weights': {},
                'reason': f"Weight allocation failed: {str(e)}"
            }

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Raw weights dictionary

        Returns:
            Normalized weights dictionary
        """
        if not weights:
            return {}

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise WeightAllocationError("Total weight is non-positive", "portfolio_normalization", str(total_weight))

        normalized = {symbol: weight / total_weight for symbol, weight in weights.items()}

        # Validate normalization
        normalized_total = sum(normalized.values())
        if abs(normalized_total - 1.0) > 1e-6:
            logger.warning(f"Weight normalization result: {normalized_total:.8f} (expected 1.0)")

        return normalized

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply constraints to portfolio weights.
        
        Args:
            weights: Dictionary of stock weights
            
        Returns:
            Constrained weights dictionary
        """
        if not weights:
            return weights
            
        max_w = self.constraints.get('max_position_weight')
        max_leverage = self.constraints.get('max_leverage', 1.0)
        min_w = self.constraints.get('min_position_weight', 0.0)

        # Apply per-asset cap
        if max_w is not None:
            weights = {k: min(v, max_w) for k, v in weights.items()}
            logger.debug(f"Applied max_position_weight constraint: {max_w}")

        # Apply minimum weight threshold (zero out small positions)
        if min_w > 0:
            weights = {k: v if v >= min_w else 0.0 for k, v in weights.items()}
            logger.debug(f"Applied min_position_weight constraint: {min_w}")

        # Remove zero weights
        weights = {k: v for k, v in weights.items() if v > 0}

        if not weights:
            logger.warning("All weights were zeroed out by constraints")
            return {}

        total = sum(weights.values())
        if total <= 0:
            logger.warning("Total weight is non-positive after constraints")
            return {}

        # Apply leverage cap: if total weight > max_leverage, scale down
        if max_leverage is not None and total > max_leverage:
            scale = max_leverage / total
            weights = {k: v * scale for k, v in weights.items()}
            total = sum(weights.values())
            logger.debug(f"Applied max_leverage constraint: {max_leverage}, scaled by {scale:.4f}")

        # Normalize to sum to 1.0
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.debug(f"Final constrained weights sum: {sum(weights.values()):.6f}")
        return weights

    def get_method_name(self) -> str:
        """Get the method name."""
        return "BoxBased"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the configuration."""
        try:
            # Create a temporary builder to validate config
            temp_builder = BoxBasedPortfolioBuilder(config)
            return True
        except Exception as e:
            logger.error(f"Box-Based configuration validation failed: {e}")
            return False

    def get_construction_info(self) -> Dict[str, Any]:
        """Get detailed information about the construction method."""
        target_boxes = self.box_weight_manager.get_target_boxes()
        box_weights = self.box_weight_manager.get_all_weights()

        return {
            'method_name': self.get_method_name(),
            'method_type': self.__class__.__name__,
            'description': "Box-First portfolio construction with systematic box-based allocation",
            'configuration': {
                'stocks_per_box': self.stocks_per_box,
                'min_stocks_per_box': self.min_stocks_per_box,
                'allocation_method': self.weight_allocator.__class__.__name__,
                'box_selector': self.box_selector.__class__.__name__
            },
            'box_information': {
                'total_boxes': len(target_boxes),
                'total_box_weight': sum(box_weights.values()),
                'box_weight_manager': self.box_weight_manager.get_manager_info()
            }
        }