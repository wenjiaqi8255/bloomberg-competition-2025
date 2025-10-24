"""
Prediction Orchestrator
======================

Main orchestrator for the prediction service, supporting single models,
multi-model ensembles, and meta-models. Follows the same patterns as
MultiModelOrchestrator but focused on prediction rather than training.

Key Features:
- Supports FF5, ML, and Meta strategies
- Uses StrategyFactory for strategy creation
- Uses PortfolioBuilderFactory for portfolio construction
- Extracts detailed box information from portfolio builders
- Handles both single and multi-model scenarios
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .data_types import PredictionResult, StockRecommendation
from ...trading_system.strategies.factory import StrategyFactory
from ...trading_system.portfolio_construction.factory import PortfolioBuilderFactory
from ...trading_system.portfolio_construction.models.types import (
    PortfolioConstructionRequest, 
    BoxConstructionResult,
    BoxKey
)
from ...trading_system.data.base_data_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class PredictionOrchestrator:
    """
    Orchestrates the complete prediction workflow.
    
    This orchestrator follows the same patterns as MultiModelOrchestrator
    but focuses on prediction rather than training. It supports:
    - Single model strategies (FF5, ML)
    - Multi-model ensembles
    - Meta-model strategies
    - Box-based and quantitative portfolio construction
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the prediction orchestrator.
        
        Args:
            config_path: Path to the prediction configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize data providers
        self.data_provider = self._create_data_provider(self.config.get('data_provider', {}))
        self.factor_data_provider = self._create_factor_data_provider(self.config.get('factor_data_provider', {}))
        
        # Initialize strategy and portfolio builder (lazy loading)
        self._strategy = None
        self._portfolio_builder = None
        
        logger.info(f"PredictionOrchestrator initialized with config: {config_path}")
    
    def run_prediction(self) -> PredictionResult:
        """
        Run the complete prediction workflow.
        
        Returns:
            Complete prediction result with recommendations and box details
        """
        logger.info("="*60)
        logger.info("STARTING PREDICTION WORKFLOW")
        logger.info("="*60)
        
        try:
            # Step 1: Load strategy
            logger.info("STEP 1: Loading strategy")
            strategy = self._load_strategy()
            
            # Step 2: Get prediction date and universe
            prediction_date = self._get_prediction_date()
            universe = self._get_universe()
            
            logger.info(f"Prediction date: {prediction_date}")
            logger.info(f"Universe: {len(universe)} symbols")
            
            # Step 3: Generate signals via strategy
            logger.info("STEP 2: Generating signals via strategy")
            signals = self._generate_signals(strategy, universe, prediction_date)
            
            if signals.empty:
                raise RuntimeError("Strategy generated no signals")
            
            logger.info(f"Generated signals for {len(signals.columns)} symbols")
            
            # Step 4: Construct portfolio
            logger.info("STEP 3: Constructing portfolio")
            portfolio_result = self._construct_portfolio(signals, universe, prediction_date)
            
            # Step 5: Extract box details if available
            logger.info("STEP 4: Extracting box details")
            box_details = self._extract_box_details(portfolio_result)
            
            # Step 6: Create recommendations
            logger.info("STEP 5: Creating recommendations")
            recommendations = self._create_recommendations(
                portfolio_result, signals, box_details, prediction_date
            )
            
            # Step 7: Calculate risk metrics
            logger.info("STEP 6: Calculating risk metrics")
            risk_metrics = self._calculate_risk_metrics(portfolio_result, signals)
            
            # Step 8: Create final result
            result = PredictionResult(
                recommendations=recommendations,
                portfolio_weights=portfolio_result.weights if hasattr(portfolio_result, 'weights') else portfolio_result,
                box_allocations=box_details.get('box_coverage'),
                stocks_by_box=box_details.get('selected_stocks'),
                box_construction_log=box_details.get('construction_log'),
                strategy_type=self.config['strategy']['type'],
                model_id=self._get_model_id(strategy),
                base_model_ids=self._get_base_model_ids(strategy),
                model_weights=self._get_model_weights(strategy),
                prediction_date=prediction_date,
                total_positions=len(recommendations),
                portfolio_method=self.config['portfolio_construction']['method'],
                expected_return=risk_metrics['expected_return'],
                expected_risk=risk_metrics['expected_risk'],
                diversification_score=risk_metrics['diversification_score']
            )
            
            logger.info("="*60)
            logger.info("PREDICTION WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info(f"Generated {len(recommendations)} recommendations")
            logger.info("="*60)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction workflow failed: {e}")
            raise
    
    def _create_data_provider(self, config: Dict[str, Any]) -> BaseDataProvider:
        """Create data provider from configuration (reuse MultiModelOrchestrator pattern)."""
        provider_type = config.get('type')
        params = config.get('parameters', {})
        
        if provider_type == "YFinanceProvider":
            from ...trading_system.data.yfinance_provider import YFinanceProvider
            return YFinanceProvider(**params)
        else:
            raise ValueError(f"Unsupported data provider type: {provider_type}")
    
    def _create_factor_data_provider(self, config: Dict[str, Any]) -> Optional[BaseDataProvider]:
        """Create factor data provider from configuration."""
        if not config:
            return None
            
        provider_type = config.get('type')
        params = config.get('parameters', {})
        
        if provider_type == "FF5DataProvider":
            from ...trading_system.data.ff5_provider import FF5DataProvider
            return FF5DataProvider(**params)
        else:
            logger.warning(f"Unknown factor data provider type: {provider_type}")
            return None
    
    def _load_strategy(self):
        """Load strategy using StrategyFactory (reuse existing pattern)."""
        if self._strategy is None:
            strategy_config = self.config['strategy']

            # Extract model_id from parameters if it's nested there
            if 'parameters' in strategy_config and 'model_id' in strategy_config['parameters']:
                model_id = strategy_config['parameters']['model_id']
                # Create a flattened config for StrategyFactory
                flattened_config = {
                    'type': strategy_config['type'],
                    'name': strategy_config.get('name', f"{strategy_config['type']}_strategy"),
                    'model_id': model_id,
                    **strategy_config.get('parameters', {})  # Include all parameters
                }
            else:
                # Use config as-is if model_id is already at top level
                flattened_config = strategy_config

            # Add universe to flattened config for MetaStrategy
            if strategy_config.get('type') == 'meta':
                universe = self._get_universe()
                flattened_config['universe'] = universe

            # Create providers dict with data providers
            providers = {
                'data_provider': self.data_provider,
                'factor_data_provider': self.factor_data_provider
            }

            # Create strategy using config-driven approach
            self._strategy = StrategyFactory.create_from_config(flattened_config, providers=providers)
            logger.info(f"Loaded strategy: {self._strategy.__class__.__name__}")
        return self._strategy
    
    def _get_prediction_date(self) -> datetime:
        """Get prediction date from configuration."""
        date_str = self.config.get('prediction', {}).get('prediction_date', '2024-01-15')
        if isinstance(date_str, str):
            return datetime.fromisoformat(date_str)
        return date_str
    
    def _get_universe(self) -> List[str]:
        """Get universe symbols from configuration."""
        universe_config = self.config.get('universe', {})
        
        # Check if using CSV source
        if universe_config.get('source') == 'csv':
            csv_path = universe_config.get('csv_path')
            if not csv_path:
                raise ValueError("universe.source=csv specified but csv_path is missing")
            
            filters = universe_config.get('filters', {})
            from ...trading_system.data.utils.universe_loader import load_universe_from_csv
            return load_universe_from_csv(csv_path, filters)
        
        # Fallback to inline symbols
        return universe_config.get('symbols', [])
    
    def _generate_signals(self, strategy, universe: List[str], prediction_date: datetime) -> pd.DataFrame:
        """
        Generate signals via unified orchestrator data preparation.

        ✅ REFACTORED: Following "Data preparation responsibility moves up to orchestrator" pattern.
        The orchestrator prepares ALL data, strategies only consume it.

        Args:
            strategy: Loaded strategy instance
            universe: List of symbols to predict for
            prediction_date: Date to make predictions for

        Returns:
            DataFrame with signals (dates × symbols)
        """
        # Calculate lookback period for data
        lookback_days = self.config.get('strategy', {}).get('lookback_days', 252)
        start_date = prediction_date - timedelta(days=lookback_days)

        logger.info("="*60)
        logger.info("DETAILED PREDICTION FLOW DIAGNOSIS")
        logger.info("="*60)
        logger.info(f"Strategy type: {self.config['strategy']['type']}")
        logger.info(f"Model ID: {self.config['strategy'].get('parameters', {}).get('model_id', 'N/A')}")
        logger.info(f"Universe: {universe}")
        logger.info(f"Date range: {start_date} to {prediction_date}")

        # Step 1: Get price data (orchestrator responsibility)
        logger.info("STEP 1: FETCHING PRICE DATA")
        try:
            price_data = self.data_provider.get_historical_data(
                symbols=universe,
                start_date=start_date,
                end_date=prediction_date
            )
            logger.info(f"✅ Price data fetched: {len(price_data)} symbols")
            for symbol, data in price_data.items():
                logger.info(f"  {symbol}: {data.shape} from {data.index[0]} to {data.index[-1]}")
                if not data.empty:
                    logger.info(f"    Sample close price for {symbol}: {data['Close'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch price data: {e}")
            raise

        # Step 2: Prepare complete pipeline data (orchestrator responsibility)
        logger.info("STEP 2: PREPARING COMPLETE PIPELINE DATA")
        pipeline_data = self._prepare_pipeline_data(price_data, start_date, prediction_date, strategy)

        # Step 3: Generate signals via strategy (only passing pipeline_data)
        logger.info("STEP 3: GENERATING SIGNALS VIA STRATEGY")
        logger.info(f"Calling strategy.generate_signals with:")
        logger.info(f"  - Pipeline data keys: {list(pipeline_data.keys())}")
        logger.info(f"  - Start date: {prediction_date}")
        logger.info(f"  - End date: {prediction_date}")

        try:
            signals = strategy.generate_signals(
                pipeline_data=pipeline_data,  # ✅ Only pass pipeline_data
                start_date=prediction_date,
                end_date=prediction_date
            )

            logger.info(f"✅ Signals generated: {signals.shape}")
            logger.info(f"Signals index: {signals.index}")
            logger.info(f"Signals columns: {list(signals.columns)}")

            # Check signal values
            if not signals.empty:
                logger.info("SIGNAL VALUES ANALYSIS:")
                for date_idx in signals.index:
                    date_signals = signals.loc[date_idx]
                    logger.info(f"  Date {date_idx}:")
                    for symbol in signals.columns:
                        signal_val = date_signals[symbol]
                        logger.info(f"    {symbol}: {signal_val:.6f}")

                    # Check statistics
                    non_zero_signals = date_signals[date_signals != 0]
                    logger.info(f"  Non-zero signals: {len(non_zero_signals)}")
                    if len(non_zero_signals) > 0:
                        logger.info(f"  Signal range: {non_zero_signals.min():.6f} to {non_zero_signals.max():.6f}")
                        logger.info(f"  Signal mean: {non_zero_signals.mean():.6f}")
                    break  # Just analyze first date
            else:
                logger.error("❌ No signals generated - empty DataFrame")

        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        return signals

    def _prepare_pipeline_data(self, price_data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime, strategy) -> Dict[str, Any]:
        """
        Prepare complete pipeline data including factor data if available.

        This method implements the elegant architectural pattern where the orchestrator
        prepares all necessary data (price_data + factor_data) for strategies,
        following SOLID, KISS, YAGNI, DRY principles.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            start_date: Start date for data preparation
            end_date: End date for data preparation
            strategy: Strategy instance to check for factor data provider

        Returns:
            Complete pipeline data dictionary with price_data and optionally factor_data
        """
        try:
            logger.info(f"[PredictionOrchestrator] Preparing complete pipeline data...")

            # Start with basic structure
            pipeline_data = {
                'price_data': price_data
            }

            # ✅ REFACTORED: Use orchestrator's own factor_data_provider
            # Orchestrator prepares complete data, strategies don't hold providers
            if self.factor_data_provider is not None:
                logger.info(f"[PredictionOrchestrator] Using orchestrator's factor data provider: {type(self.factor_data_provider).__name__}")
                logger.info(f"[PredictionOrchestrator] Fetching factor data...")

                try:
                    # Get symbols from price data
                    symbols = list(price_data.keys())
                    logger.info(f"[PredictionOrchestrator] Fetching factor data for {len(symbols)} symbols...")

                    # ✅ UNIFIED INTERFACE: Use standard get_data method
                    logger.info(f"[PredictionOrchestrator] Using unified get_data interface for {type(self.factor_data_provider).__name__}")
                    factor_data = self.factor_data_provider.get_data(
                        start_date=start_date,
                        end_date=end_date
                    )

                    if factor_data is not None and not factor_data.empty:
                        pipeline_data['factor_data'] = factor_data
                        logger.info(f"[PredictionOrchestrator] ✅ Factor data added: {factor_data.shape}")
                        logger.info(f"[PredictionOrchestrator] Factor data columns: {list(factor_data.columns)}")
                        logger.info(f"[PredictionOrchestrator] Factor data sample: {factor_data.head(2).to_dict()}")
                    else:
                        logger.warning(f"[PredictionOrchestrator] Factor data provider returned empty data")

                except Exception as e:
                    logger.error(f"[PredictionOrchestrator] Failed to fetch factor data: {e}")
                    logger.info(f"[PredictionOrchestrator] Continuing without factor data...")
            else:
                logger.info(f"[PredictionOrchestrator] No factor data provider available in orchestrator")

            # ✅ Cleaned up redundant backup logic since we now use orchestrator's provider directly

            logger.info(f"[PredictionOrchestrator] Pipeline data prepared with keys: {list(pipeline_data.keys())}")
            return pipeline_data

        except Exception as e:
            logger.error(f"[PredictionOrchestrator] Failed to prepare pipeline data: {e}")
            # Fallback to just price data
            return {'price_data': price_data}

    def _construct_portfolio(self, signals: pd.DataFrame, universe: List[str], prediction_date: datetime):
        """
        Construct portfolio using factory pattern.
        
        Args:
            signals: Generated signals DataFrame
            universe: List of symbols
            prediction_date: Prediction date
            
        Returns:
            Portfolio construction result (BoxConstructionResult or pd.Series)
        """
        # Create portfolio builder via factory
        portfolio_config = self.config['portfolio_construction']
        builder = PortfolioBuilderFactory.create_builder(portfolio_config)
        
        # Get latest signals as Series
        latest_signals = signals.iloc[-1] if not signals.empty else pd.Series()
        
        # Get price data for portfolio construction
        price_data = self.data_provider.get_historical_data(
            symbols=universe,
            start_date=prediction_date - timedelta(days=30),  # Shorter lookback for portfolio
            end_date=prediction_date
        )
        
        # Create portfolio construction request
        request = PortfolioConstructionRequest(
            date=prediction_date,
            universe=universe,
            signals=latest_signals,
            price_data=price_data,
            constraints=self.config.get('constraints', {})
        )
        
        # Try to get detailed result (BoxConstructionResult)
        if hasattr(builder, 'build_portfolio_with_result'):
            logger.info("Using build_portfolio_with_result for detailed output")
            result = builder.build_portfolio_with_result(request)
        else:
            logger.info("Using build_portfolio for simple output")
            result = builder.build_portfolio(request)
        
        return result
    
    def _extract_box_details(self, portfolio_result) -> Dict[str, Any]:
        """
        Extract box details from portfolio construction result.
        
        Args:
            portfolio_result: Result from portfolio builder
            
        Returns:
            Dictionary with box details or empty dict
        """
        if isinstance(portfolio_result, BoxConstructionResult):
            return {
                'box_coverage': portfolio_result.box_coverage,
                'selected_stocks': portfolio_result.selected_stocks,
                'target_weights': portfolio_result.target_weights,
                'construction_log': portfolio_result.construction_log
            }
        else:
            return {}
    
    def _create_recommendations(self, portfolio_result, signals: pd.DataFrame, 
                              box_details: Dict[str, Any], prediction_date: datetime) -> List[StockRecommendation]:
        """
        Create stock recommendations from portfolio result.
        
        Args:
            portfolio_result: Portfolio construction result
            signals: Generated signals
            box_details: Box classification details
            prediction_date: Prediction date
            
        Returns:
            List of stock recommendations
        """
        recommendations = []
        
        # Get weights
        if hasattr(portfolio_result, 'weights'):
            weights = portfolio_result.weights
        else:
            weights = portfolio_result
        
        # Get latest signals
        latest_signals = signals.iloc[-1] if not signals.empty else pd.Series()
        
        # Create recommendations
        for symbol, weight in weights.items():
            if weight > 0:  # Only include positive weights
                signal_strength = latest_signals.get(symbol, 0.0)
                
                # Get box classification if available
                box_classification = None
                if box_details.get('selected_stocks'):
                    for box_key, stocks in box_details['selected_stocks'].items():
                        if symbol in stocks:
                            try:
                                box_classification = BoxKey.from_string(box_key)
                            except ValueError:
                                logger.warning(f"Invalid box key format: {box_key}")
                            break
                
                # Calculate risk score (simplified)
                risk_score = abs(signal_strength) * 0.5  # Placeholder calculation
                
                recommendation = StockRecommendation(
                    symbol=symbol,
                    weight=weight,
                    signal_strength=signal_strength,
                    box_classification=box_classification,
                    risk_score=risk_score
                )
                recommendations.append(recommendation)
        
        # Sort by weight descending
        recommendations.sort(key=lambda x: x.weight, reverse=True)
        
        return recommendations
    
    def _calculate_risk_metrics(self, portfolio_result, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk metrics for the portfolio.
        
        Args:
            portfolio_result: Portfolio construction result
            signals: Generated signals
            
        Returns:
            Dictionary with risk metrics
        """
        # Get weights
        if hasattr(portfolio_result, 'weights'):
            weights = portfolio_result.weights
        else:
            weights = portfolio_result
        
        # Calculate expected return (simplified)
        latest_signals = signals.iloc[-1] if not signals.empty else pd.Series()
        expected_return = sum(weights.get(symbol, 0) * latest_signals.get(symbol, 0) 
                            for symbol in weights.index) * 252  # Annualized
        
        # Calculate expected risk (simplified)
        expected_risk = np.sqrt(sum(w**2 for w in weights.values)) * 0.2  # Placeholder
        
        # Calculate diversification score
        num_positions = len([w for w in weights.values if w > 0])
        diversification_score = min(num_positions / 10.0, 1.0)  # Normalize to 0-1
        
        return {
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'diversification_score': diversification_score
        }
    
    def _get_model_id(self, strategy) -> str:
        """Get model ID from strategy."""
        if hasattr(strategy, 'model_id'):
            return strategy.model_id
        elif hasattr(strategy, 'model_predictor') and hasattr(strategy.model_predictor, 'model_id'):
            return strategy.model_predictor.model_id
        else:
            return "unknown_model"
    
    def _get_base_model_ids(self, strategy) -> Optional[List[str]]:
        """Get base model IDs for meta strategies."""
        if hasattr(strategy, 'base_model_ids'):
            return strategy.base_model_ids
        return None
    
    def _get_model_weights(self, strategy) -> Optional[Dict[str, float]]:
        """Get model weights for meta strategies."""
        if hasattr(strategy, 'meta_weights'):
            return strategy.meta_weights
        elif hasattr(strategy, 'model_weights'):
            return strategy.model_weights
        return None
