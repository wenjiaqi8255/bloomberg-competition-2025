"""
Feature Engineering Debugger

This module provides a comprehensive debugging tool for isolating and testing
the feature engineering pipeline in isolation. It uses dependency injection
to mock model training and focus specifically on data quality issues.

Key Features:
- End-to-end feature pipeline testing without model training
- Detailed NaN tracking and reporting
- Data quality validation at each transformation step
- Configurable NaN handling strategies
- Visual analysis of feature computation patterns
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from ..data.yfinance_provider import YFinanceProvider
from ..data.ff5_provider import FF5DataProvider
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..feature_engineering.utils.technical_features import TechnicalIndicatorCalculator
from ..config.feature import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureEngineeringDebugger:
    """
    Comprehensive debugging tool for feature engineering pipeline.

    This tool isolates the feature engineering process from model training
    to identify data quality issues, NaN propagation patterns, and
    feature computation problems.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the debugger.

        Args:
            config_path: Optional path to feature configuration file
        """
        self.config = FeatureConfig.from_file(config_path) if config_path else FeatureConfig()
        self.data_provider = YFinanceProvider()
        self.factor_provider = FF5DataProvider()
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        self.calculator = TechnicalIndicatorCalculator()

        # Debugging state
        self.debug_results = {}
        self.nan_timeline = []
        self.data_quality_checks = []

    def debug_feature_pipeline(self,
                             symbols: List[str],
                             start_date: datetime,
                             end_date: datetime,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive debugging on the feature engineering pipeline.

        Args:
            symbols: List of stock symbols to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            save_results: Whether to save debugging results

        Returns:
            Comprehensive debugging report
        """
        logger.info(f"=== Starting Feature Engineering Pipeline Debug ===")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        debug_report = {
            'metadata': {
                'symbols': symbols,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'debug_timestamp': datetime.now().isoformat()
            },
            'data_loading': {},
            'feature_computation': {},
            'nan_analysis': {},
            'data_quality': {},
            'recommendations': []
        }

        try:
            # Step 1: Debug data loading
            logger.info("Step 1: Debugging data loading...")
            data_loading_results = self._debug_data_loading(symbols, start_date, end_date)
            debug_report['data_loading'] = data_loading_results

            # Step 2: Debug feature computation step by step
            logger.info("Step 2: Debugging feature computation...")
            feature_results = self._debug_feature_computation(data_loading_results['price_data'])
            debug_report['feature_computation'] = feature_results

            # Step 3: Analyze NaN patterns
            logger.info("Step 3: Analyzing NaN patterns...")
            nan_analysis = self._analyze_nan_patterns(feature_results)
            debug_report['nan_analysis'] = nan_analysis

            # Step 4: Data quality validation
            logger.info("Step 4: Validating data quality...")
            quality_results = self._validate_data_quality(feature_results)
            debug_report['data_quality'] = quality_results

            # Step 5: Generate recommendations
            logger.info("Step 5: Generating recommendations...")
            recommendations = self._generate_recommendations(debug_report)
            debug_report['recommendations'] = recommendations

            # Save results if requested
            if save_results:
                self._save_debug_results(debug_report, symbols, start_date, end_date)

            logger.info("=== Feature Engineering Debug Complete ===")
            return debug_report

        except Exception as e:
            logger.error(f"Debugging failed: {e}")
            debug_report['error'] = str(e)
            return debug_report

    def _debug_data_loading(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Debug the data loading process."""
        logger.info("Debugging data loading...")

        results = {
            'price_data': {},
            'factor_data': {},
            'data_quality': {}
        }

        # Load price data with detailed logging
        logger.info(f"Loading price data for {len(symbols)} symbols...")
        price_data = self.data_provider.get_historical_data(symbols, start_date, end_date)

        # Analyze price data quality
        total_price_points = 0
        missing_data_points = 0
        symbol_analysis = {}

        for symbol, df in price_data.items():
            if df is not None and not df.empty:
                total_price_points += len(df) * len(df.columns)
                missing_data_points += df.isnull().sum().sum()

                symbol_analysis[symbol] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'missing_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                    'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
                    'has_volume': 'Volume' in df.columns,
                    'volume_missing': df['Volume'].isnull().sum() if 'Volume' in df.columns else None
                }

                logger.debug(f"DEBUG: {symbol} - {len(df)} rows, {df.isnull().sum().sum()} missing values")
            else:
                symbol_analysis[symbol] = {
                    'error': 'No data available',
                    'rows': 0,
                    'columns': 0
                }

        results['price_data'] = price_data
        results['data_quality']['price_analysis'] = symbol_analysis
        results['data_quality']['total_price_points'] = total_price_points
        results['data_quality']['total_missing_price_points'] = missing_data_points
        results['data_quality']['price_data_completeness'] = (1 - missing_data_points / total_price_points) * 100 if total_price_points > 0 else 0

        # Load factor data
        try:
            logger.info("Loading factor data...")
            factor_data = self.factor_provider.get_factor_returns(start_date, end_date)
            results['factor_data'] = factor_data

            if factor_data is not None and not factor_data.empty:
                factor_missing = factor_data.isnull().sum().sum()
                factor_total = len(factor_data) * len(factor_data.columns)
                results['data_quality']['factor_analysis'] = {
                    'rows': len(factor_data),
                    'columns': len(factor_data.columns),
                    'missing_values': factor_missing,
                    'missing_percentage': (factor_missing / factor_total) * 100 if factor_total > 0 else 0,
                    'date_range': f"{factor_data.index.min().date()} to {factor_data.index.max().date()}"
                }
                logger.debug(f"DEBUG: Factor data - {len(factor_data)} rows, {factor_missing} missing values")
            else:
                results['data_quality']['factor_analysis'] = {'error': 'No factor data available'}

        except Exception as e:
            logger.warning(f"Could not load factor data: {e}")
            results['data_quality']['factor_analysis'] = {'error': str(e)}

        logger.info(f"Data loading complete - Price data completeness: {results['data_quality']['price_data_completeness']:.2f}%")
        return results

    def _debug_feature_computation(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Debug feature computation step by step."""
        logger.info("Debugging feature computation step by step...")

        results = {
            'step_by_step_analysis': {},
            'feature_categories': {},
            'nan_tracking': {},
            'final_features': None
        }

        # Use the first symbol for detailed analysis
        if not price_data:
            results['error'] = 'No price data available'
            return results

        sample_symbol = list(price_data.keys())[0]
        sample_data = price_data[sample_symbol].copy()

        logger.info(f"Using {sample_symbol} for detailed feature computation analysis")
        logger.info(f"Sample data shape: {sample_data.shape}, NaN count: {sample_data.isnull().sum().sum()}")

        # Step 1: Raw price features
        logger.info("Step 1: Computing basic price features...")
        price_features = self._debug_price_features(sample_data)
        results['step_by_step_analysis']['price_features'] = price_features

        # Step 2: Momentum features
        logger.info("Step 2: Computing momentum features...")
        momentum_features = self.calculator.compute_momentum_features(sample_data, [5, 10, 21, 63])
        results['step_by_step_analysis']['momentum_features'] = self._analyze_feature_df(momentum_features, "momentum")

        # Step 3: Volatility features
        logger.info("Step 3: Computing volatility features...")
        volatility_features = self.calculator.compute_volatility_features(sample_data, [10, 21, 63])
        results['step_by_step_analysis']['volatility_features'] = self._analyze_feature_df(volatility_features, "volatility")

        # Step 4: Technical indicators
        logger.info("Step 4: Computing technical indicators...")
        technical_features = self.calculator.compute_technical_indicators(sample_data)
        results['step_by_step_analysis']['technical_features'] = self._analyze_feature_df(technical_features, "technical")

        # Step 5: Volume features (if available)
        if 'Volume' in sample_data.columns:
            logger.info("Step 5: Computing volume features...")
            volume_features = self.calculator.compute_volume_features(sample_data)
            results['step_by_step_analysis']['volume_features'] = self._analyze_feature_df(volume_features, "volume")

        # Step 6: Combine all features
        logger.info("Step 6: Combining all features...")
        all_feature_dfs = [
            price_features,
            momentum_features,
            volatility_features,
            technical_features
        ]

        if 'Volume' in sample_data.columns:
            all_feature_dfs.append(volume_features)

        combined_features = pd.concat(all_feature_dfs, axis=1)
        results['final_features'] = self._analyze_feature_df(combined_features, "combined")

        logger.info(f"Feature computation complete - Final features: {combined_features.shape}")
        return results

    def _debug_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute basic price features with debugging."""
        features = pd.DataFrame(index=data.index)
        prices = data['Close']

        logger.debug("Computing basic price features...")

        # Price returns
        returns_1d = prices.pct_change()
        features['return_1d'] = returns_1d

        # Log returns
        log_returns = np.log(prices / prices.shift(1))
        features['log_return_1d'] = log_returns

        # Price levels
        features['price'] = prices
        features['log_price'] = np.log(prices)

        # Price ratios
        features['price_vs_ma_5d'] = prices / prices.rolling(5).mean()
        features['price_vs_ma_21d'] = prices / prices.rolling(21).mean()

        # Debug information
        nan_report = {}
        for col in features.columns:
            nan_count = features[col].isnull().sum()
            nan_report[col] = nan_count
            logger.debug(f"DEBUG: {col} has {nan_count} NaN values ({nan_count/len(features)*100:.1f}%)")

        logger.info(f"Price features - Total NaN: {sum(nan_report.values())}, Breakdown: {nan_report}")
        return features

    def _analyze_feature_df(self, features: pd.DataFrame, category: str) -> Dict[str, Any]:
        """Analyze a feature DataFrame and return detailed statistics."""
        if features.empty:
            return {'error': f'No {category} features computed'}

        total_values = len(features) * len(features.columns)
        total_nan = features.isnull().sum().sum()

        analysis = {
            'shape': features.shape,
            'total_values': total_values,
            'total_nan': total_nan,
            'nan_percentage': (total_nan / total_values) * 100 if total_values > 0 else 0,
            'columns': list(features.columns),
            'nan_per_column': {},
            'data_types': features.dtypes.to_dict(),
            'date_range': f"{features.index.min().date()} to {features.index.max().date()}"
        }

        # Analyze each column
        for col in features.columns:
            col_nan = features[col].isnull().sum()
            col_total = len(features[col])

            nan_info = {
                'count': col_nan,
                'percentage': (col_nan / col_total) * 100 if col_total > 0 else 0
            }

            # Add basic statistics for non-NaN values
            if col_nan < col_total:
                non_nan_values = features[col].dropna()
                if len(non_nan_values) > 0:
                    nan_info.update({
                        'mean': float(non_nan_values.mean()),
                        'std': float(non_nan_values.std()),
                        'min': float(non_nan_values.min()),
                        'max': float(non_nan_values.max())
                    })

            analysis['nan_per_column'][col] = nan_info

        logger.info(f"{category.title()} features - Shape: {features.shape}, NaN: {total_nan}/{total_values} ({analysis['nan_percentage']:.1f}%)")
        return analysis

    def _analyze_nan_patterns(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze NaN patterns across all feature computation steps."""
        logger.info("Analyzing NaN patterns...")

        analysis = {
            'nan_timeline': [],
            'nan_sources': {},
            'cumulative_nan_tracking': {},
            'problematic_features': [],
            'nan_clusters': {}
        }

        # Build NaN timeline
        step_order = ['price_features', 'momentum_features', 'volatility_features', 'technical_features', 'volume_features']

        cumulative_nan = 0
        for step in step_order:
            if step in feature_results['step_by_step_analysis']:
                step_data = feature_results['step_by_step_analysis'][step]
                if isinstance(step_data, dict) and 'total_nan' in step_data:
                    step_nan = step_data['total_nan']
                    cumulative_nan += step_nan

                    analysis['nan_timeline'].append({
                        'step': step,
                        'nan_added': step_nan,
                        'cumulative_nan': cumulative_nan,
                        'features_count': step_data.get('shape', (0, 0))[1] if 'shape' in step_data else 0
                    })

        # Identify problematic features (high NaN percentage)
        all_features = {}

        # Collect all feature analyses
        for step_name, step_data in feature_results['step_by_step_analysis'].items():
            if isinstance(step_data, dict) and 'nan_per_column' in step_data:
                for col_name, col_info in step_data['nan_per_column'].items():
                    full_col_name = f"{step_name}_{col_name}"
                    all_features[full_col_name] = {
                        'step': step_name,
                        'nan_percentage': col_info['percentage'],
                        'nan_count': col_info['count'],
                        'stats': {k: v for k, v in col_info.items() if k not in ['count', 'percentage']}
                    }

        # Identify problematic features (>50% NaN)
        for feature_name, feature_info in all_features.items():
            if feature_info['nan_percentage'] > 50:
                analysis['problematic_features'].append({
                    'name': feature_name,
                    'step': feature_info['step'],
                    'nan_percentage': feature_info['nan_percentage'],
                    'nan_count': feature_info['nan_count'],
                    'recommendation': self._get_nan_handling_recommendation(feature_info)
                })

        # Cluster analysis - find features with similar NaN patterns
        analysis['nan_clusters'] = self._find_nan_clusters(all_features)

        logger.info(f"NaN analysis complete - Found {len(analysis['problematic_features'])} problematic features")
        return analysis

    def _find_nan_clusters(self, all_features: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Find clusters of features with similar NaN percentages."""
        clusters = {
            'low_nan': [],      # <10% NaN
            'medium_nan': [],   # 10-30% NaN
            'high_nan': [],     # 30-50% NaN
            'extreme_nan': []   # >50% NaN
        }

        for feature_name, feature_info in all_features.items():
            nan_pct = feature_info['nan_percentage']
            if nan_pct < 10:
                clusters['low_nan'].append(feature_name)
            elif nan_pct < 30:
                clusters['medium_nan'].append(feature_name)
            elif nan_pct < 50:
                clusters['high_nan'].append(feature_name)
            else:
                clusters['extreme_nan'].append(feature_name)

        return clusters

    def _get_nan_handling_recommendation(self, feature_info: Dict) -> str:
        """Get NaN handling recommendation for a feature."""
        nan_pct = feature_info['nan_percentage']

        if nan_pct > 80:
            return "DROP_FEATURE"
        elif nan_pct > 50:
            return "FORWARD_FILL + INTERPOLATION"
        elif nan_pct > 30:
            return "FORWARD_FILL"
        elif nan_pct > 10:
            return "MEDIAN_FILL"
        else:
            return "NO_ACTION_NEEDED"

    def _validate_data_quality(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality of computed features."""
        logger.info("Validating data quality...")

        validation_results = {
            'overall_quality_score': 0,
            'quality_checks': [],
            'warnings': [],
            'errors': [],
            'statistics': {}
        }

        # Check final features
        if 'final_features' in feature_results and feature_results['final_features']:
            final_features = feature_results['final_features']

            # Overall quality score (inverse of NaN percentage)
            nan_percentage = final_features.get('nan_percentage', 100)
            validation_results['overall_quality_score'] = max(0, 100 - nan_percentage)

            # Quality checks
            validation_results['quality_checks'].append({
                'check': 'NaN Percentage',
                'status': 'PASS' if nan_percentage < 20 else 'FAIL',
                'value': nan_percentage,
                'threshold': 20,
                'description': f'Features have {nan_percentage:.1f}% NaN values'
            })

            # Feature count check
            feature_count = final_features.get('shape', [0, 0])[1]
            validation_results['quality_checks'].append({
                'check': 'Feature Count',
                'status': 'PASS' if feature_count >= 10 else 'FAIL',
                'value': feature_count,
                'threshold': 10,
                'description': f'Computed {feature_count} features'
            })

            # Data continuity check
            date_range = final_features.get('date_range', '')
            validation_results['quality_checks'].append({
                'check': 'Date Continuity',
                'status': 'PASS',
                'value': date_range,
                'description': 'Date range available'
            })

        # Check for critical issues
        if final_features.get('nan_percentage', 100) > 80:
            validation_results['errors'].append('Excessive NaN values in features')

        if final_features.get('shape', [0, 0])[1] < 5:
            validation_results['errors'].append('Insufficient number of features computed')

        # Warnings
        if 50 < final_features.get('nan_percentage', 0) <= 80:
            validation_results['warnings'].append('High NaN percentage may impact model performance')

        logger.info(f"Data quality validation complete - Score: {validation_results['overall_quality_score']:.1f}/100")
        return validation_results

    def _generate_recommendations(self, debug_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on debugging results."""
        recommendations = []

        # NaN handling recommendations
        nan_analysis = debug_report.get('nan_analysis', {})
        problematic_features = nan_analysis.get('problematic_features', [])

        if problematic_features:
            recommendations.append({
                'category': 'NaN_HANDLING',
                'priority': 'HIGH',
                'title': f'Handle {len(problematic_features)} features with high NaN values',
                'description': 'Several features have excessive NaN values that need to be addressed',
                'actions': [
                    'Implement forward filling for time-series features',
                    'Use interpolation for gap filling',
                    'Consider median filling for extreme values',
                    'Drop features with >80% NaN values'
                ],
                'affected_features': [f['name'] for f in problematic_features[:5]]  # Show first 5
            })

        # Data quality recommendations
        quality_results = debug_report.get('data_quality', {})
        quality_score = quality_results.get('overall_quality_score', 0)

        if quality_score < 50:
            recommendations.append({
                'category': 'DATA_QUALITY',
                'priority': 'HIGH',
                'title': f'Improve data quality (current score: {quality_score:.1f}/100)',
                'description': 'Overall data quality is below acceptable threshold',
                'actions': [
                    'Verify data source quality',
                    'Check for missing trading days',
                    'Validate data preprocessing steps',
                    'Review lookback period configurations'
                ]
            })

        # Feature engineering recommendations
        feature_results = debug_report.get('feature_computation', {})
        if 'step_by_step_analysis' in feature_results:
            recommendations.append({
                'category': 'FEATURE_ENGINEERING',
                'priority': 'MEDIUM',
                'title': 'Optimize feature computation order',
                'description': 'Some feature computation steps may be optimized to reduce NaN propagation',
                'actions': [
                    'Compute features in order of increasing lookback period',
                    'Implement cascading NaN handling',
                    'Add intermediate validation checkpoints',
                    'Consider alternative indicator calculations'
                ]
            })

        # Configuration recommendations
        data_loading = debug_report.get('data_loading', {})
        price_completeness = data_loading.get('data_quality', {}).get('price_data_completeness', 100)

        if price_completeness < 95:
            recommendations.append({
                'category': 'CONFIGURATION',
                'priority': 'MEDIUM',
                'title': f'Improve price data completeness ({price_completeness:.1f}%)',
                'description': 'Some price data is missing, consider adjusting configuration',
                'actions': [
                    'Extend data loading window',
                    'Add more data sources as fallback',
                    'Adjust lookback buffer size',
                    'Review symbol selection criteria'
                ]
            })

        if not recommendations:
            recommendations.append({
                'category': 'GENERAL',
                'priority': 'LOW',
                'title': 'Feature engineering pipeline looks healthy',
                'description': 'No major issues detected in the feature engineering process',
                'actions': [
                    'Continue monitoring for data quality issues',
                    'Periodically re-run this debugging analysis'
                ]
            })

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    def _save_debug_results(self, debug_report: Dict[str, Any], symbols: List[str], start_date: datetime, end_date: datetime):
        """Save debugging results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_debug_{symbols[0]}_{timestamp}.json"
        filepath = Path("results") / filename

        # Ensure results directory exists
        filepath.parent.mkdir(exist_ok=True)

        try:
            with open(filepath, 'w') as f:
                json.dump(debug_report, f, indent=2, default=str)
            logger.info(f"Debug results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save debug results: {e}")

    def test_nan_handling_strategies(self, sample_data: pd.DataFrame) -> Dict[str, Any]:
        """Test different NaN handling strategies on sample data."""
        logger.info("Testing NaN handling strategies...")

        strategies = {
            'NO_HANDLING': lambda df: df,
            'DROP_NAN': lambda df: df.dropna(),
            'FORWARD_FILL': lambda df: df.fillna(method='ffill'),
            'BACKWARD_FILL': lambda df: df.fillna(method='bfill'),
            'MEDIAN_FILL': lambda df: df.fillna(df.median()),
            'INTERPOLATE_LINEAR': lambda df: df.interpolate(method='linear'),
            'INTERPOLATE_TIME': lambda df: df.interpolate(method='time'),
            'COMBINED_STRATEGY': self._combined_nan_strategy
        }

        results = {}

        for strategy_name, strategy_func in strategies.items():
            logger.debug(f"Testing {strategy_name} strategy...")

            try:
                # Apply strategy to sample data
                processed_data = strategy_func(sample_data.copy())

                # Calculate effectiveness metrics
                original_nan = sample_data.isnull().sum().sum()
                processed_nan = processed_data.isnull().sum().sum()
                reduction_percentage = ((original_nan - processed_nan) / original_nan * 100) if original_nan > 0 else 100

                # Calculate data preservation
                original_rows = len(sample_data)
                processed_rows = len(processed_data)
                preservation_rate = processed_rows / original_rows * 100

                results[strategy_name] = {
                    'original_nan': original_nan,
                    'processed_nan': processed_nan,
                    'nan_reduction_percentage': reduction_percentage,
                    'data_preservation_rate': preservation_rate,
                    'final_shape': processed_data.shape,
                    'effectiveness_score': (reduction_percentage * 0.7 + preservation_rate * 0.3)
                }

                logger.debug(f"{strategy_name}: {reduction_percentage:.1f}% NaN reduction, {preservation_rate:.1f}% data preserved")

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                results[strategy_name] = {
                    'error': str(e),
                    'effectiveness_score': 0
                }

        # Sort strategies by effectiveness
        sorted_strategies = sorted(results.items(), key=lambda x: x[1].get('effectiveness_score', 0), reverse=True)

        logger.info(f"NaN handling strategies tested - Best: {sorted_strategies[0][0]} ({sorted_strategies[0][1].get('effectiveness_score', 0):.1f} score)")
        return dict(sorted_strategies)

    def _combined_nan_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combined NaN handling strategy."""
        # Step 1: Forward fill
        df_filled = df.fillna(method='ffill')

        # Step 2: Backward fill for leading NaNs
        df_filled = df_filled.fillna(method='bfill')

        # Step 3: Interpolate remaining gaps
        df_filled = df_filled.interpolate(method='linear')

        # Step 4: Median fill for any remaining NaNs
        df_filled = df_filled.fillna(df_filled.median())

        return df_filled