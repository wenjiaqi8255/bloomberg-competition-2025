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
        'fama_macbeth': MLStrategy,  # Fama-MacBeth uses ML strategy with custom features
        'fama_french_5': FamaFrench5Strategy,
        'ff5_regression': FamaFrench5Strategy,  # Map ff5_regression model to FF5 strategy
    }
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any], **kwargs) -> BaseStrategy:
        """
        Create a strategy from configuration with automatic component creation.
        
        Args:
            config: Strategy configuration with:
                - type: Strategy type ('ml', 'dual_momentum', 'fama_french_5', 'ff5_regression')
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
            # Check if config requests using fitted pipeline from model
            use_fitted_pipeline = config.get('use_fitted_pipeline', False)
            model_id = config.get('model_id')
            model_registry_path = config.get('model_registry_path', './models/')

            if use_fitted_pipeline and model_id:
                logger.info(f"✓ Loading fitted FeatureEngineeringPipeline from model '{model_id}'")
                feature_pipeline = cls._load_fitted_pipeline_from_model(
                    model_id, model_registry_path
                )
                if feature_pipeline is not None:
                    logger.info("✓ Successfully loaded fitted FeatureEngineeringPipeline from model")
                else:
                    logger.warning("⚠ Failed to load fitted pipeline from model, creating new one from config")
                    feature_pipeline = cls._create_feature_pipeline(strategy_type, config)
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
        Create feature pipeline based on strategy type with model-specific configuration support.

        Different strategies need different features:
        - ML: Comprehensive features (momentum, volatility, technical, volume, etc.)
        - Dual Momentum: Only momentum features
        - Fama-MacBeth: Cross-sectional features only
        - FF5: Factor features only (from factor data provider)

        Priority order for feature configuration:
        1. Model-specific feature_config in config (highest priority)
        2. Strategy type defaults (fallback)
        """
        # Check if custom feature config provided (highest priority)
        if 'feature_config' in config:
            logger.info(f"Using model-specific feature configuration for {strategy_type}")
            feature_config_dict = config['feature_config']

            # Ensure essential defaults for FF5 models
            if strategy_type in ['fama_french_5', 'ff5_regression']:
                # Force FF5 models to only use factor data
                feature_config_dict = {
                    'enabled_features': [],  # No technical features
                    'include_technical': False,
                    'include_cross_sectional': False,
                    'include_theoretical': False,
                    **feature_config_dict  # Allow overrides but ensure FF5 basics
                }
                logger.info(f"FF5 Strategy: Enforced factor-only config (factors come from factor data provider)")

            feature_config = FeatureConfig(**feature_config_dict)
            return FeatureEngineeringPipeline(feature_config, model_type=strategy_type)

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

        elif strategy_type == 'fama_macbeth':
            # Fama-MacBeth: cross-sectional features only
            feature_config = FeatureConfig(
                enabled_features=['cross_sectional'],
                include_cross_sectional=True,
                include_technical=False,
                include_theoretical=False,
                # Use default cross-sectional settings
                cross_sectional_features=['market_cap', 'book_to_market', 'size', 'value', 'momentum', 'volatility'],
                cross_sectional_lookback={
                    'momentum': 252,
                    'volatility': 60,
                    'ma_long': 200,
                    'ma_short': 50
                },
                winsorize_percentile=0.01
            )

        elif strategy_type in ['fama_french_5', 'ff5_regression']:
            # Fama-French: NO technical or cross-sectional features needed!
            # FF5 model uses factor data directly from factor data provider
            logger.info("FF5 Strategy: Using minimal feature config (factors come from factor data provider)")
            feature_config = FeatureConfig(
                enabled_features=[],  # No technical features for FF5
                include_cross_sectional=False,  # Explicitly disable cross-sectional
                include_technical=False,
                include_theoretical=False
            )

        else:
            # Default: balanced feature set
            feature_config = FeatureConfig()

        return FeatureEngineeringPipeline(feature_config, model_type=strategy_type)
    
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
            # Special handling for ensemble models - pass ensemble configuration
            if model_type_to_create == 'ensemble':
                # Extract ensemble-specific parameters from config['parameters']
                parameters = config.get('parameters', {})
                ensemble_config = {
                    'base_model_ids': parameters.get('base_model_ids', []),
                    'model_weights': parameters.get('model_weights', {}),
                    'model_registry_path': parameters.get('model_registry_path', './models/'),
                    'combination_method': parameters.get('combination_method', 'weighted_average')
                }
                logger.info(f"Creating ensemble model with config: {ensemble_config}")
                model = ModelFactory.create(model_type_to_create, config=ensemble_config)
            else:
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
            
        elif strategy_type in ['fama_french_5', 'ff5_regression']:
            params['lookback_days'] = config.get('lookback_days', 252)
            params['risk_free_rate'] = config.get('risk_free_rate', 0.02)
        
        return params

    @classmethod
    def _load_fitted_pipeline_from_model(cls, model_id: str, model_registry_path: str):
        """
        Load the fitted feature pipeline from a saved model.

        This method:
        1. Creates a temporary ModelPredictor to load the model
        2. Extracts the fitted feature pipeline from the model
        3. Returns the pipeline for reuse in strategy creation

        Args:
            model_id: Model ID to load
            model_registry_path: Path to model registry

        Returns:
            Fitted FeatureEngineeringPipeline or None if not found
        """
        try:
            from ..models.serving.predictor import ModelPredictor
            from ..models.model_persistence import ModelRegistry

            # Create ModelPredictor to load the model
            temp_predictor = ModelPredictor(model_registry_path=model_registry_path)
            temp_predictor.load_model(model_id)

            # Get the loaded model
            model = temp_predictor.get_current_model()
            if model is None:
                logger.error(f"Failed to load model '{model_id}'")
                return None

            # Extract fitted feature pipeline from model
            if hasattr(model, 'feature_pipeline'):
                fitted_pipeline = model.feature_pipeline
                logger.info(f"✅ Extracted fitted feature pipeline from model '{model_id}'")
                logger.info(f"✅ Pipeline type: {type(fitted_pipeline)}")
                logger.info(f"✅ Pipeline fitted: {getattr(fitted_pipeline, '_is_fitted', 'Unknown')}")
                return fitted_pipeline
            else:
                logger.warning(f"⚠ Model '{model_id}' does not have a fitted feature pipeline")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load fitted pipeline from model '{model_id}': {e}")
            return None


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

