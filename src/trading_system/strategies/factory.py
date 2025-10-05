"""
Unified Strategy Factory

This factory creates strategies using the unified architecture where
ALL strategies follow the same pattern:

    FeatureEngineeringPipeline → ModelPredictor → PositionSizer

The factory handles:
1. Creating the appropriate FeaturePipeline for each strategy type
2. Loading the correct Model for each strategy type
3. Creating the PositionSizer with specified config
4. Assembling everything into a strategy instance

Key Insight:
    The only difference between strategies is WHAT features and models they use,
    not the overall architecture!
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base_strategy import BaseStrategy
from .ml_strategy import MLStrategy
from .dual_momentum import DualMomentumStrategy
from .fama_french_5 import FamaFrench5Strategy

from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..feature_engineering.models.data_types import FeatureConfig
from ..models.serving.predictor import ModelPredictor
from ..utils.position_sizer import PositionSizer
from ..data.stock_classifier import StockClassifier
# from ..allocation.box_allocator import BoxAllocator  # File removed - functionality deprecated

logger = logging.getLogger(__name__)


class StrategyFactory:
    """
    Factory for creating strategies with the unified architecture.
    
    All strategies created by this factory follow the same pattern:
    - FeatureEngineeringPipeline for feature computation
    - ModelPredictor for predictions
    - PositionSizer for risk management
    
    The factory abstracts away the complexity of:
    - Creating the right feature pipeline config
    - Loading the right model
    - Wiring everything together
    """
    
    # Strategy class registry
    _strategy_registry = {
        'ml': MLStrategy,  # Use multi-stock version for better compatibility
        'dual_momentum': DualMomentumStrategy,
        'fama_french_5': FamaFrench5Strategy,
    }
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any], **kwargs) -> BaseStrategy:
        """
        Create a strategy from configuration with automatic component creation.
        
        Args:
            config: Strategy configuration with:
                - type: Strategy type ('ml', 'dual_momentum', 'fama_french_5')
                - name: Strategy name
                - model_id: Model identifier (e.g., 'random_forest_v1')
                - feature_config: Optional feature pipeline config
                - position_sizing: Position sizer configuration
                - Additional strategy-specific parameters
        
        Returns:
            Configured strategy instance
        
        Example:
            config = {
                'type': 'dual_momentum',
                'name': 'DM_252',
                'model_id': 'momentum_ranking_v1',
                'lookback_period': 252,
                'position_sizing': {
                    'volatility_target': 0.15,
                    'max_position_weight': 0.10
                }
            }
            
            strategy = UnifiedStrategyFactory.create_from_config(config)
        """
        strategy_type = config.get('type')
        if not strategy_type:
            raise ValueError("Configuration must include 'type' field")
        
        if strategy_type not in cls._strategy_registry:
            raise ValueError(f"Unknown strategy type: {strategy_type}. "
                           f"Available: {list(cls._strategy_registry.keys())}")
        
        name = config.get('name', f'{strategy_type}_strategy')
        model_id = config.get('model_id')
        
        if not model_id:
            raise ValueError("Configuration must include 'model_id' field")
        
        logger.info(f"Creating {strategy_type} strategy '{name}' with model '{model_id}'")
        
        # Step 1: Get or Create FeatureEngineeringPipeline
        # Check if a fitted pipeline is provided (e.g., from training)
        providers = kwargs.get('providers', {})
        logger.debug(f"StrategyFactory: Available providers: {list(providers.keys())}")
        feature_pipeline = providers.get('feature_pipeline')

        if feature_pipeline is not None:
            logger.info("✓ Using provided fitted FeatureEngineeringPipeline from training")
            logger.debug(f"✓ Feature pipeline type: {type(feature_pipeline)}")
            logger.debug(f"✓ Feature pipeline fitted: {getattr(feature_pipeline, '_is_fitted', 'Unknown')}")
        else:
            logger.warning("⚠ Creating new FeatureEngineeringPipeline from config (no fitted pipeline provided)")
            feature_pipeline = cls._create_feature_pipeline(strategy_type, config)
        
        # Step 2: Create ModelPredictor, passing along any extra context like providers
        model_predictor = cls._create_model_predictor(model_id, config, **kwargs)

        # Step 3: Create PositionSizer
        position_sizer = cls._create_position_sizer(config)
        
        # Step 4: Create Box components if configured
        stock_classifier, box_allocator = cls._create_box_components(config)

        # Step 5: Get strategy class and create instance
        strategy_class = cls._strategy_registry[strategy_type]
        
        # Extract strategy-specific parameters
        strategy_params = cls._extract_strategy_params(strategy_type, config)
        
        # Extract data providers from kwargs to pass to strategy
        # (providers dict was already extracted in Step 1)
        data_provider = providers.get('data_provider')
        factor_data_provider = providers.get('factor_data_provider')
        
        # Get universe from config or strategy_params
        universe = strategy_params.pop('universe', config.get('universe', []))

        # Create strategy with providers
        strategy = strategy_class(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            universe=universe,
            stock_classifier=stock_classifier,
            box_allocator=box_allocator,
            data_provider=data_provider,
            factor_data_provider=factor_data_provider,
            **strategy_params
        )
        
        logger.info(f"✓ Created {strategy_type} strategy '{name}' with data providers")
        return strategy
    
    @classmethod
    def _create_box_components(cls, config: Dict[str, Any]) -> (Optional[StockClassifier], Optional[Any]):
        """
        Create Box-based components if the investment framework is enabled in config.
        """
        framework_config = config.get('investment_framework', {})

        if not framework_config.get('enabled', False):
            return None, None

        logger.info("Investment framework enabled. Creating Box components.")

        # Create StockClassifier
        classification_config = framework_config.get('box_classification')
        if not classification_config:
            raise ValueError("Investment framework is enabled, but 'box_classification' config is missing.")
        stock_classifier = StockClassifier(config=classification_config)
        logger.info("✓ StockClassifier created.")

        # BoxAllocator functionality has been deprecated
        # allocation_config = framework_config.get('allocation')
        # if not allocation_config:
        #     raise ValueError("Investment framework is enabled, but 'allocation' config is missing.")
        # box_allocator = BoxAllocator(config=allocation_config)
        # logger.info("✓ BoxAllocator created.")

        return stock_classifier, None  # Return None for allocator since it's deprecated

    @classmethod
    def _create_feature_pipeline(cls, 
                                 strategy_type: str, 
                                 config: Dict[str, Any]) -> FeatureEngineeringPipeline:
        """
        Create feature pipeline based on strategy type.
        
        Different strategies need different features:
        - ML: Comprehensive features (momentum, volatility, technical, volume, etc.)
        - Dual Momentum: Only momentum features
        - Fama-French: Factor features (or proxies)
        """
        # Check if custom feature config provided
        if 'feature_config' in config:
            logger.debug("Using custom feature configuration")
            feature_config = FeatureConfig(**config['feature_config'])
            return FeatureEngineeringPipeline(feature_config)
        
        # Create default feature config based on strategy type
        logger.debug(f"Creating default feature pipeline for {strategy_type}")
        
        if strategy_type == 'ml':
            # ML strategies: comprehensive features
            feature_config = FeatureConfig(
                enabled_features=['momentum', 'volatility', 'technical', 'volume'],
                momentum_periods=[21, 63, 252],
                volatility_windows=[20, 60],
                lookback_periods=[20, 60, 252],
                include_technical=True,
                min_ic_threshold=0.02,
                min_significance=0.1
            )
            
        elif strategy_type == 'dual_momentum':
            # Dual momentum: only momentum features
            lookback = config.get('lookback_period', 252)
            feature_config = FeatureConfig(
                enabled_features=['momentum'],
                momentum_periods=[21, 63, lookback],
                include_technical=False
            )
            
        elif strategy_type in ['fama_french_5']:
            # Fama-French: NO technical features needed!
            # FF5 model uses factor data directly, not technical indicators
            logger.info("FF5 Strategy: Using minimal feature config (factors come from factor data provider)")
            feature_config = FeatureConfig(
                enabled_features=[],  # No technical features for FF5
                include_technical=False
            )
            
        else:
            # Default: balanced feature set
            feature_config = FeatureConfig()
        
        return FeatureEngineeringPipeline(feature_config)
    
    @classmethod
    def _create_model_predictor(cls,
                                model_id: str,
                                config: Dict[str, Any],
                                **kwargs) -> ModelPredictor:
        """
        Create model predictor with specified model.
        
        Three loading modes:
        1. From registry path: Load pre-trained model from disk
        2. From model path: Load model from specific path
        3. Create new instance: Create fresh model from ModelFactory (for rule-based models)
        
        Args:
            model_id: Model identifier (e.g., 'ff5_regression', 'momentum_ranking')
            config: Full configuration with:
                - model_registry_path: Path to model registry (optional)
                - model_path: Specific path to saved model (optional)
                - model_config: Configuration for creating new model (optional)
        
        Returns:
            ModelPredictor instance with loaded model
        """
        from ..models.base.model_factory import ModelFactory
        from pathlib import Path
        
        model_registry_path = config.get('model_registry_path')
        model_path = config.get('model_path')
        model_config = config.get('model_config', {})
        
        # ModelPredictor no longer takes data providers
        # Data providers are managed at the Strategy level
        predictor_kwargs = {}

        logger.info(f"Creating ModelPredictor for model '{model_id}'")
        
        # Mode 1: Load from registry path (pre-trained model)
        if model_registry_path:
            full_model_path = Path(model_registry_path) / model_id
            if full_model_path.exists():
                logger.info(f"Mode 1: Loading model from registry: {full_model_path}")
                predictor = ModelPredictor(
                    model_path=str(full_model_path),
                    model_registry_path=model_registry_path,
                    **predictor_kwargs
                )
                logger.info(f"✓ Model '{model_id}' loaded from registry")
                return predictor
            else:
                logger.warning(f"Model path {full_model_path} not found in registry, trying other modes")
        
        # Mode 2: Load from specific path (saved model)
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                logger.info(f"Mode 2: Loading model from path: {model_path}")
                predictor = ModelPredictor(
                    model_path=model_path,
                    model_registry_path=model_registry_path,
                    **predictor_kwargs
                )
                logger.info(f"✓ Model '{model_id}' loaded from path")
                return predictor
            else:
                logger.warning(f"Model path {model_path} not found, trying other modes")
        
        # Mode 3: Create new model instance from factory (for rule-based or fresh models)
        logger.info(f"Mode 3: Creating new model instance from factory")
        
        # Use the inferred type to handle versioned model_ids like 'ff5_regression_v1.2'
        model_type_to_create = cls._infer_model_type(model_id)
        
        if ModelFactory.is_registered(model_type_to_create):
            logger.info(f"Creating model '{model_type_to_create}' with config: {model_config}")
            model = ModelFactory.create(model_type_to_create, config=model_config)
            
            logger.info(f"Model '{model_type_to_create}' created (status: {model.status})")
            
            # Create predictor with direct model injection
            predictor = ModelPredictor(
                model_instance=model,
                model_registry_path=model_registry_path or './models',
                **predictor_kwargs
            )
            
            logger.info(f"✓ Model '{model_type_to_create}' created and injected into predictor")
            return predictor
        else:
            raise ValueError(
                f"Cannot load or create model for id '{model_id}' (inferred type: '{model_type_to_create}'). "
                f"Not found in registry, path, or factory. "
                f"Available model types: {list(ModelFactory.list_models().keys())}"
            )
    
    @classmethod
    def _infer_model_type(cls, model_id: str) -> str:
        """
        Infer model type from model ID.

        Handles cases like:
        - 'ff5_regression_v1' -> 'ff5_regression'
        - 'ff5_regression_20251003_001418_v1.0.0' -> 'ff5_regression'
        - 'momentum_ranking' -> 'momentum_ranking'

        Args:
            model_id: Model identifier

        Returns:
            Inferred model type
        """
        # Remove version suffix if present
        if '_v' in model_id:
            base_part = model_id.split('_v')[0]
        else:
            base_part = model_id

        # Handle case where model_name includes timestamp
        # e.g., 'ff5_regression_20251003_001418' -> 'ff5_regression'
        if '_' in base_part:
            # Try to match known model types
            known_types = ['ff5_regression', 'momentum_ranking', 'xgboost', 'lstm']
            for model_type in known_types:
                if base_part.startswith(model_type + '_'):
                    return model_type

        # Return as-is if no special handling needed
        return base_part
    
    @classmethod
    def _create_position_sizer(cls, config: Dict[str, Any]) -> PositionSizer:
        """
        Create position sizer from configuration.
        
        Args:
            config: Full configuration with position_sizing section
        
        Returns:
            PositionSizer instance
        """
        sizing_config = config.get('position_sizing', {})
        
        return PositionSizer(
            volatility_target=sizing_config.get('volatility_target', 0.15),
            max_position_weight=sizing_config.get('max_position_weight', 0.10),
            max_leverage=sizing_config.get('max_leverage', 1.0),
            min_position_weight=sizing_config.get('min_position_weight', 0.01)
        )
    
    @classmethod
    def _extract_strategy_params(cls, 
                                 strategy_type: str, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract strategy-specific parameters from config.
        
        Args:
            strategy_type: Type of strategy
            config: Full configuration
        
        Returns:
            Dictionary of strategy-specific parameters
        """
        params = {}
        
        if strategy_type == 'ml':
            params['min_signal_strength'] = config.get('min_signal_strength', 0.1)
            
        elif strategy_type == 'dual_momentum':
            params['lookback_period'] = config.get('lookback_period', 252)
            
        elif strategy_type in ['fama_french_5']:
            params['lookback_days'] = config.get('lookback_days', 252)
            params['risk_free_rate'] = config.get('risk_free_rate', 0.02)
        
        return params

    

# TODO: REQUIRED FOR FULL FUNCTIONALITY
# --------------------------------------
# 1. ModelPredictor needs to support model_id in constructor
#    OR provide a load_model(model_id) method
#
# 2. Models need to be registered in model registry:
#    - 'momentum_ranking_v1' → MomentumRankingModel
#    - 'ff5_regression_v1' → FF5RegressionModel
#    - 'random_forest_v1' → RandomForestModel
#    - etc.
#
# 3. FeatureEngineeringPipeline needs to be fittable:
#    - pipeline.fit(train_data) to learn parameters
#    - pipeline.transform(test_data) for consistent features
#
# 4. Models need to be pre-trained and saved:
#    - Train on historical data
#    - Save to model registry
#    - Load via model_id

