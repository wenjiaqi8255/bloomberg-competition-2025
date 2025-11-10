"""
Box-Based portfolio construction strategy.

Implements the Box-First methodology where stocks are first grouped
into boxes, then selected and weighted within each box according to
configured strategies.
"""

import logging
from typing import Dict, List, Any
from collections import OrderedDict
import pandas as pd
from datetime import datetime

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder, IBoxSelector
from src.trading_system.portfolio_construction.models.types import PortfolioConstructionRequest, BoxConstructionResult, BoxKey
from src.trading_system.portfolio_construction.box_based.box_weight_manager import BoxWeightManager
from src.trading_system.portfolio_construction.box_based.weight_allocator import (
    WeightAllocatorFactory, MeanVarianceAllocator
)
from src.trading_system.portfolio_construction.utils.adapters import ClassificationAdapter
from src.trading_system.portfolio_construction.utils.component_factory import ComponentFactory
from src.trading_system.portfolio_construction.utils.weight_utils import WeightUtils
from src.trading_system.portfolio_construction.models.exceptions import (
    PortfolioConstructionError, ClassificationError, WeightAllocationError, InvalidConfigError
)
from src.trading_system.portfolio_construction.box_based.services import (
    ClassificationService, StockSelectionService
)
from src.trading_system.data.stock_classifier import StockClassifier
from src.trading_system.data.offline_stock_metadata_provider import OfflineStockMetadataProvider

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
        # Stock classifier (created via factory)
        stock_classifier = ComponentFactory.create_stock_classifier(
            self.config.get('classifier', {})
        )
        
        # Classification Service
        self.classification_service = ClassificationService(
            stock_classifier,
            max_cache_size=self.config.get('classification_cache_size', 100)
        )

        # Box weight manager
        self.box_weight_manager = BoxWeightManager(self.config.get('box_weights', {}))

        # Box selector (can be overridden via config)
        box_selector = SignalBasedBoxSelector()
        
        # Stock Selection Service
        self.stock_selection_service = StockSelectionService(
            box_selector=box_selector,
            stocks_per_box=self.config.get('stocks_per_box', 3),
            min_stocks_per_box=self.config.get('min_stocks_per_box', 1)
        )

        # Weight allocation
        allocation_method = self.config.get('allocation_method', 'equal')
        allocation_config = self.config.get('allocation_config', {}) or {}
        
        if hasattr(allocation_config, 'model_dump'):
            allocation_config = allocation_config.model_dump()
        elif not isinstance(allocation_config, dict):
            allocation_config = {}
        
        # Create allocator, potentially with a covariance estimator for mean_variance
        if allocation_method == 'mean_variance':
            # MeanVarianceAllocator needs a covariance estimator, which we create via the factory
            covariance_config = self.config.get('covariance', {})
            covariance_estimator = ComponentFactory.create_covariance_estimator(
                covariance_config, self.factor_data_provider
            )
            self.weight_allocator = MeanVarianceAllocator(
                allocation_config, 
                covariance_estimator=covariance_estimator
            )
        else:
            self.weight_allocator = WeightAllocatorFactory.create_allocator(
                allocation_method, allocation_config
            )

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

        logger.info("Box-based builder configuration validation passed")

    def build_portfolio(self, request: PortfolioConstructionRequest) -> pd.Series:
        """
        Build portfolio using Box-First methodology.
        
        ✅ 简化：直接使用 request.constraints，不需要临时覆盖

        Args:
            request: Portfolio construction request

        Returns:
            Series of portfolio weights
        """
        logger.info(f"Building Box-Based portfolio for {request.date.date()}")
        
        constraints = request.constraints or self.constraints

        # Step 1: Classify stocks into boxes
        logger.info("[Step 1/3] Classifying stocks into boxes...")
        classifications = self.classification_service.classify_stocks(
            request.universe, request.price_data, request.date
        )

        # Step 2: Group stocks by boxes
        logger.info("[Step 2/3] Grouping stocks by boxes...")
        box_stocks = ClassificationAdapter.group_stocks_by_box(request.universe, classifications)
        
        if not box_stocks:
            raise PortfolioConstructionError("No stocks available for portfolio construction")

        # Step 3: Select stocks and allocate weights
        logger.info("[Step 3/3] Selecting stocks and allocating weights...")
        if self.allocation_scope == 'global':
            weights = self._construct_global_allocation(
                box_stocks, request.signals, request.price_data, request.date
            )
        else:
            weights, _ = self._construct_from_boxes(
                box_stocks, request.signals, request.price_data, request.date
            )

        # Apply constraints and normalize weights
        weights = self._apply_constraints_and_normalize(pd.Series(weights), constraints)

        # Log final portfolio summary
        logger.info("🎯 FINAL PORTFOLIO SUMMARY:")
        logger.info(f"   📊 Total positions: {len(weights)}")
        logger.info(f"   💰 Total weight: {sum(weights.values()):.4f}")
        
        # Final validation
        final_weights = pd.Series(weights)
        if not WeightUtils.validate_weights(final_weights):
            raise PortfolioConstructionError("Final weights failed validation.")

        if weights:
            sorted_positions = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            top_positions = sorted_positions[:10]
            logger.info("   🏆 Top positions:")
            for stock, weight in top_positions:
                logger.info(f"      {stock}: {weight:.4f} ({weight*100:.2f}%)")

        logger.info(f"✅ Box-Based portfolio construction completed: {len(weights)} positions")
        return pd.Series(weights)

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
        classifications = self.classification_service.classify_stocks(
            request.universe, request.price_data, request.date
        )
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

        # Use the StockSelectionService to select stocks for each box
        selected_stocks_by_box = self.stock_selection_service.select_stocks_for_boxes(
            box_stocks, signals
        )
        
        logger.info(f"Processing {len(selected_stocks_by_box)} boxes with sufficient stocks for portfolio construction")

        for box_key, selected_stocks in selected_stocks_by_box.items():
            try:
                box_result = self._process_single_box(
                    box_key, selected_stocks, signals, price_data, date
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
                                     date: datetime) -> Dict[str, float]:
        """
        ✅ 简化：复用 MeanVarianceAllocator 而不是重复实现优化逻辑
        
        Box 仅用于筛选，汇总所有选中的股票后执行一次全局均值方差优化。
        """
        # 1) 每个 box 内按已有 selector 选股
        selected_stocks_by_box = self.stock_selection_service.select_stocks_for_boxes(
            box_stocks, signals
        )
        
        selected_all: List[str] = []
        for stocks in selected_stocks_by_box.values():
            selected_all.extend(stocks)

        # 去重，保持顺序
        if selected_all:
            seen = set()
            selected_all = [s for s in selected_all if not (s in seen or seen.add(s))]

        if len(selected_all) < 2:
            logger.warning("Not enough stocks for global optimization")
            return {}

        # ✅ 简化：复用已在 _initialize_components 中创建的 weight_allocator
        if not isinstance(self.weight_allocator, MeanVarianceAllocator):
            raise PortfolioConstructionError(
                "Global allocation requires 'mean_variance' allocation method."
            )
        
        # 使用 allocator 分配权重（总权重为1.0，后续会归一化）
        weights = self.weight_allocator.allocate(
            selected_all,
            total_weight=1.0,  # 全局分配，总权重为1.0
            signals=signals,
            price_data=price_data,
            date=date
        )
        
        logger.info(f"Global MV optimized {len(weights)} positions")
        return weights

    def _process_single_box(self, box_key: BoxKey, selected_stocks: List[str],
                           signals: pd.Series, price_data: Dict[str, pd.DataFrame],
                           date: datetime) -> Dict[str, Any]:
        """
        Process a single box: allocate weights to pre-selected stocks.
        """
        # Log detailed stock selection information
        logger.info(f"📦 Box {box_key}:")
        logger.info(f"   🎯 Selected stocks: {len(selected_stocks)} ({', '.join(selected_stocks)})")
        
        # Log signal strengths for selected stocks
        if selected_stocks and not signals.empty:
            signal_info = []
            for stock in selected_stocks:
                signal_value = signals.get(stock, 0)
                signal_info.append(f"{stock}({signal_value:.4f})")
            logger.info(f"   📈 Signal strengths: {', '.join(signal_info)}")

        if not selected_stocks:
            logger.warning(f"   ❌ No stocks to allocate from box {box_key}")
            return {
                'weights': {},
                'reason': "No stocks to allocate from box"
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
                'log': f"Box {box_key}: {len(selected_stocks)} stocks allocated, weight {target_weight:.4f}"
            }

        except Exception as e:
            logger.error(f"Weight allocation failed for box {box_key}: {e}")
            return {
                'weights': {},
                'reason': f"Weight allocation failed: {str(e)}"
            }

    def _apply_constraints_and_normalize(self, weights: pd.Series, 
                                         constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply constraints and normalize weights using WeightUtils.
        
        Args:
            weights: Series of stock weights
            constraints: Constraints dictionary
            
        Returns:
            Normalized and constrained weights dictionary.
        """
        if weights.empty:
            return {}
        
        # NOTE: This is a temporary implementation. The full constraint application logic
        # will be centralized in a separate task. For now, we focus on normalization.

        # Apply min_position_weight constraint
        min_w = constraints.get('min_position_weight', 0.0)
        if min_w > 0:
            weights = weights[weights >= min_w]
            logger.debug(f"Applied min_position_weight constraint: {min_w}")

        # Apply max_leverage constraint by scaling
        max_leverage = constraints.get('max_leverage', 1.0)
        total_weight = weights.sum()
        if max_leverage is not None and total_weight > max_leverage:
            scale = max_leverage / total_weight
            weights *= scale
            logger.debug(f"Applied max_leverage constraint: {max_leverage}, scaled by {scale:.4f}")
        
        # Normalize weights using the centralized utility
        normalized_weights = WeightUtils.normalize_weights(weights)
        
        # Simple cap for max_position_weight (a more robust solution will be in ConstraintApplier)
        max_w = constraints.get('max_position_weight')
        if max_w is not None:
            if any(normalized_weights > max_w):
                 logger.warning("Post-normalization capping needed for max_position_weight. "
                                "This is a temporary solution.")
                 normalized_weights = normalized_weights.clip(upper=max_w)
                 # Re-normalize after capping
                 normalized_weights = WeightUtils.normalize_weights(normalized_weights)

        return normalized_weights.to_dict()

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
                'stocks_per_box': self.stock_selection_service.stocks_per_box,
                'min_stocks_per_box': self.stock_selection_service.min_stocks_per_box,
                'allocation_method': self.weight_allocator.__class__.__name__,
                'box_selector': self.stock_selection_service.box_selector.__class__.__name__
            },
            'box_information': {
                'total_boxes': len(target_boxes),
                'total_box_weight': sum(box_weights.values()),
                'box_weight_manager': self.box_weight_manager.get_manager_info()
            }
        }