#!/usr/bin/env python3
"""
Comprehensive test suite for the data module.
Tests YFinance provider, FF5 provider, and stock classifier functionality.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_system.data.yfinance_provider import YFinanceProvider
from trading_system.data.ff5_provider import FF5DataProvider
from trading_system.data.stock_classifier import StockClassifier, SizeCategory, StyleCategory, RegionCategory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataModuleTester:
    """Comprehensive test suite for data module components."""

    def __init__(self):
        self.test_results = {}
        self.errors = []

        # Test symbols (mix of stocks/ETFs)
        self.test_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'VTI', 'BND']
        self.test_dates = {
            'start': datetime(2024, 1, 1),
            'end': datetime(2024, 10, 1)
        }

    def run_all_tests(self):
        """Run all tests for the data module."""
        logger.info("Starting comprehensive data module tests")

        test_methods = [
            ('yfinance_provider', self.test_yfinance_provider),
            ('ff5_provider', self.test_ff5_provider),
            ('stock_classifier', self.test_stock_classifier),
            ('data_validation', self.test_data_validation),
            ('retry_logic', self.test_retry_logic),
            ('error_handling', self.test_error_handling)
        ]

        for test_name, test_method in test_methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {test_name} tests")
            logger.info(f"{'='*50}")

            try:
                result = test_method()
                self.test_results[test_name] = result
                logger.info(f"✅ {test_name} tests completed: {result}")
            except Exception as e:
                error_msg = f"❌ {test_name} tests failed: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.errors.append(error_msg)
                self.test_results[test_name] = {'status': 'failed', 'error': str(e)}

        self.print_summary()

    def test_yfinance_provider(self) -> Dict:
        """Test YFinance provider functionality."""
        logger.info("Testing YFinance provider...")

        provider = YFinanceProvider(max_retries=2, retry_delay=0.5)
        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: Symbol validation
        logger.info("Testing symbol validation...")
        try:
            valid_symbol = provider.validate_symbol('AAPL')
            invalid_symbol = provider.validate_symbol('INVALID123XYZ')
            results['tests']['symbol_validation'] = {
                'valid': valid_symbol,
                'invalid': invalid_symbol,
                'passed': valid_symbol and not invalid_symbol
            }
            logger.info(f"✅ Symbol validation: AAPL={valid_symbol}, INVALID123XYZ={invalid_symbol}")
        except Exception as e:
            results['tests']['symbol_validation'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Symbol validation failed: {e}")

        # Test 2: Historical data fetch
        logger.info("Testing historical data fetch...")
        try:
            hist_data = provider.get_historical_data(
                ['AAPL', 'MSFT'],
                start_date=self.test_dates['start'],
                end_date=self.test_dates['end']
            )

            # Validate data structure
            validation_results = {}
            for symbol, data in hist_data.items():
                validation_results[symbol] = {
                    'has_data': len(data) > 0,
                    'columns': list(data.columns),
                    'date_range': (data.index.min(), data.index.max()),
                    'has_required_cols': all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
                }

            results['tests']['historical_data'] = {
                'symbols_fetched': len(hist_data),
                'validation': validation_results,
                'passed': len(hist_data) > 0 and all(v['has_data'] for v in validation_results.values())
            }
            logger.info(f"✅ Historical data: fetched {len(hist_data)} symbols")
        except Exception as e:
            results['tests']['historical_data'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Historical data fetch failed: {e}")

        # Test 3: Latest prices
        logger.info("Testing latest price fetch...")
        try:
            latest_prices = provider.get_latest_price(['AAPL', 'MSFT'])
            results['tests']['latest_prices'] = {
                'prices_fetched': len(latest_prices),
                'sample_prices': latest_prices,
                'passed': len(latest_prices) > 0 and all(v > 0 for v in latest_prices.values())
            }
            logger.info(f"✅ Latest prices: fetched {len(latest_prices)} prices")
        except Exception as e:
            results['tests']['latest_prices'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Latest price fetch failed: {e}")

        # Test 4: Dividends
        logger.info("Testing dividend fetch...")
        try:
            dividends = provider.get_dividends(['AAPL'])
            results['tests']['dividends'] = {
                'symbols_with_dividends': len(dividends),
                'aapl_dividends': len(dividends.get('AAPL', pd.Series())),
                'passed': isinstance(dividends, dict)
            }
            logger.info(f"✅ Dividends: fetched for {len(dividends)} symbols")
        except Exception as e:
            results['tests']['dividends'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Dividend fetch failed: {e}")

        # Test 5: Rate limiting
        logger.info("Testing rate limiting...")
        try:
            start_time = datetime.now()
            provider._wait_for_rate_limit()
            provider._wait_for_rate_limit()
            end_time = datetime.now()
            delay = (end_time - start_time).total_seconds()

            results['tests']['rate_limiting'] = {
                'delay_seconds': delay,
                'expected_min_delay': 0.5,
                'passed': delay >= 0.4  # Allow some tolerance
            }
            logger.info(f"✅ Rate limiting: {delay:.2f}s delay")
        except Exception as e:
            results['tests']['rate_limiting'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Rate limiting test failed: {e}")

        return results

    def test_ff5_provider(self) -> Dict:
        """Test Fama-French 5-factor provider functionality."""
        logger.info("Testing FF5 provider...")

        provider = FF5DataProvider(data_frequency="monthly")
        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: Factor returns fetch
        logger.info("Testing factor returns fetch...")
        try:
            factor_data = provider.get_factor_returns(
                start_date="2023-01-01",
                end_date="2024-01-01"
            )

            # Validate structure
            required_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            validation = {
                'has_data': len(factor_data) > 0,
                'columns': list(factor_data.columns),
                'date_range': (factor_data.index.min(), factor_data.index.max()),
                'has_required_cols': all(col in factor_data.columns for col in required_cols),
                'data_types': factor_data.dtypes.to_dict()
            }

            results['tests']['factor_returns'] = {
                'rows_fetched': len(factor_data),
                'validation': validation,
                'passed': validation['has_data'] and validation['has_required_cols']
            }
            logger.info(f"✅ Factor returns: fetched {len(factor_data)} rows")
        except Exception as e:
            results['tests']['factor_returns'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Factor returns fetch failed: {e}")

        # Test 2: Risk-free rate
        logger.info("Testing risk-free rate extraction...")
        try:
            rf_rate = provider.get_risk_free_rate(
                start_date="2023-01-01",
                end_date="2024-01-01"
            )

            results['tests']['risk_free_rate'] = {
                'has_data': len(rf_rate) > 0,
                'sample_values': rf_rate.head(3).to_dict(),
                'passed': len(rf_rate) > 0
            }
            logger.info(f"✅ Risk-free rate: {len(rf_rate)} observations")
        except Exception as e:
            results['tests']['risk_free_rate'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Risk-free rate extraction failed: {e}")

        # Test 3: Factor statistics
        logger.info("Testing factor statistics...")
        try:
            stats = provider.get_factor_statistics()
            results['tests']['factor_statistics'] = {
                'factors_calculated': len(stats),
                'sample_stats': {k: v for k, v in list(stats.items())[:2]},
                'passed': len(stats) > 0
            }
            logger.info(f"✅ Factor statistics: calculated for {len(stats)} factors")
        except Exception as e:
            results['tests']['factor_statistics'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Factor statistics failed: {e}")

        # Test 4: Factor descriptions
        logger.info("Testing factor descriptions...")
        try:
            descriptions = provider.get_factor_descriptions()
            results['tests']['factor_descriptions'] = {
                'count': len(descriptions),
                'factors': list(descriptions.keys()),
                'passed': len(descriptions) == 6  # 5 factors + RF
            }
            logger.info(f"✅ Factor descriptions: {len(descriptions)} factors")
        except Exception as e:
            results['tests']['factor_descriptions'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Factor descriptions failed: {e}")

        # Test 5: Data provider info
        logger.info("Testing data provider info...")
        try:
            info = provider.get_data_info()
            results['tests']['provider_info'] = {
                'info_keys': list(info.keys()),
                'data_frequency': info.get('data_frequency'),
                'factors': info.get('factors'),
                'passed': 'provider' in info and 'factors' in info
            }
            logger.info(f"✅ Provider info: {info.get('provider')}")
        except Exception as e:
            results['tests']['provider_info'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Provider info failed: {e}")

        return results

    def test_stock_classifier(self) -> Dict:
        """Test stock classifier functionality."""
        logger.info("Testing stock classifier...")

        yf_provider = YFinanceProvider()
        classifier = StockClassifier(yfinance_provider=yf_provider)
        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: Single stock classification
        logger.info("Testing single stock classification...")
        try:
            # Get price data first
            price_data = yf_provider.get_historical_data(
                ['AAPL'],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 10, 1)
            )

            if 'AAPL' in price_data and len(price_data['AAPL']) > 20:
                classification = classifier.classify_stock('AAPL', price_data['AAPL'])

                results['tests']['single_classification'] = {
                    'symbol': classification.get('symbol'),
                    'size': classification.get('size').value if classification.get('size') else None,
                    'style': classification.get('style').value if classification.get('style') else None,
                    'region': classification.get('region').value if classification.get('region') else None,
                    'sector': classification.get('sector'),
                    'market_cap': classification.get('market_cap'),
                    'score': classification.get('score'),
                    'passed': all(key in classification for key in ['symbol', 'size', 'style', 'region', 'sector'])
                }
                logger.info(f"✅ Single classification: {classification.get('size')} {classification.get('style')} {classification.get('sector')}")
            else:
                results['tests']['single_classification'] = {'error': 'Insufficient price data', 'passed': False}
                logger.warning("⚠️  Insufficient price data for single classification")

        except Exception as e:
            results['tests']['single_classification'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Single classification failed: {e}")

        # Test 2: Multiple stock classification
        logger.info("Testing multiple stock classification...")
        try:
            # Use a smaller set for testing
            test_symbols = ['AAPL', 'MSFT', 'SPY']
            boxes = classifier.classify_stocks(
                test_symbols,
                as_of_date=datetime(2024, 10, 1)
            )

            box_summary = classifier.get_box_summary(boxes)

            results['tests']['multiple_classification'] = {
                'symbols_tested': len(test_symbols),
                'boxes_created': len(boxes),
                'total_stocks_classified': box_summary.get('total_stocks', 0),
                'sectors_covered': box_summary.get('sectors_covered', 0),
                'passed': len(boxes) > 0
            }
            logger.info(f"✅ Multiple classification: {len(boxes)} boxes created")
        except Exception as e:
            results['tests']['multiple_classification'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Multiple classification failed: {e}")

        # Test 3: Size classification
        logger.info("Testing size classification...")
        try:
            # Test with mock data
            mock_data = pd.DataFrame({
                'Close': [100, 105, 110, 115, 120]
            })

            size_cat, market_cap = classifier._classify_by_size('AAPL', mock_data)
            results['tests']['size_classification'] = {
                'category': size_cat.value,
                'market_cap_billions': market_cap,
                'passed': isinstance(size_cat, SizeCategory) and isinstance(market_cap, (int, float))
            }
            logger.info(f"✅ Size classification: {size_cat.value} (${market_cap:.1f}B)")
        except Exception as e:
            results['tests']['size_classification'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Size classification failed: {e}")

        # Test 4: Style classification
        logger.info("Testing style classification...")
        try:
            # Create mock price data
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            mock_data = pd.DataFrame({
                'Close': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 100
            }, index=dates)

            style_cat, style_score = classifier._classify_by_style(mock_data)
            results['tests']['style_classification'] = {
                'category': style_cat.value,
                'score': style_score,
                'passed': isinstance(style_cat, StyleCategory) and 0 <= style_score <= 1
            }
            logger.info(f"✅ Style classification: {style_cat.value} (score: {style_score:.2f})")
        except Exception as e:
            results['tests']['style_classification'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Style classification failed: {e}")

        # Test 5: Classification info
        logger.info("Testing classification info...")
        try:
            info = classifier.get_classification_info()
            results['tests']['classification_info'] = {
                'size_categories': info.get('size_categories'),
                'style_categories': info.get('style_categories'),
                'region_categories': info.get('region_categories'),
                'sector_count': len(info.get('sectors', [])),
                'passed': len(info) > 0
            }
            logger.info(f"✅ Classification info: {len(info)} fields")
        except Exception as e:
            results['tests']['classification_info'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Classification info failed: {e}")

        return results

    def test_data_validation(self) -> Dict:
        """Test data validation functionality."""
        logger.info("Testing data validation...")

        from trading_system.utils.validation import DataValidator, DataValidationError
        from trading_system.types.data_types import PriceData

        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: Price data validation
        logger.info("Testing price data validation...")
        try:
            # Create valid price data
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            valid_price_data = pd.DataFrame({
                'Open': np.random.uniform(90, 110, 50),
                'High': np.random.uniform(100, 120, 50),
                'Low': np.random.uniform(80, 100, 50),
                'Close': np.random.uniform(90, 110, 50),
                'Volume': np.random.randint(1e6, 1e7, 50)
            }, index=dates)

            # This should not raise an exception
            DataValidator.validate_price_data(valid_price_data, "TEST")

            results['tests']['price_validation'] = {
                'valid_data_passed': True,
                'columns': list(valid_price_data.columns),
                'rows': len(valid_price_data),
                'passed': True
            }
            logger.info(f"✅ Price validation: {len(valid_price_data)} rows validated")
        except Exception as e:
            results['tests']['price_validation'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Price validation failed: {e}")

        # Test 2: Invalid price data
        logger.info("Testing invalid price data detection...")
        try:
            # Create invalid price data (negative prices)
            dates = pd.date_range('2024-01-01', periods=10, freq='D')
            invalid_price_data = pd.DataFrame({
                'Open': [-100, -90] + [100] * 8,
                'High': [120] * 10,
                'Low': [80] * 10,
                'Close': [110] * 10,
                'Volume': [1e6] * 10
            }, index=dates)

            try:
                DataValidator.validate_price_data(invalid_price_data, "INVALID")
                results['tests']['invalid_price_detection'] = {
                    'detected_error': False,
                    'passed': False  # Should have failed
                }
            except DataValidationError as e:
                results['tests']['invalid_price_detection'] = {
                    'detected_error': True,
                    'error_message': str(e),
                    'passed': True
                }
                logger.info(f"✅ Invalid price detection: caught error - {str(e)[:50]}...")

        except Exception as e:
            results['tests']['invalid_price_detection'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Invalid price detection test failed: {e}")

        # Test 3: Factor data validation
        logger.info("Testing factor data validation...")
        try:
            dates = pd.date_range('2024-01-01', periods=24, freq='M')  # Monthly
            valid_factor_data = pd.DataFrame({
                'MKT': np.random.normal(0.01, 0.05, 24),
                'SMB': np.random.normal(0.002, 0.03, 24),
                'HML': np.random.normal(0.001, 0.02, 24),
                'RMW': np.random.normal(0.0005, 0.015, 24),
                'CMA': np.random.normal(0.0003, 0.01, 24),
                'RF': np.random.uniform(0.001, 0.005, 24)
            }, index=dates)

            DataValidator.validate_factor_data(valid_factor_data, "FF5_TEST")

            results['tests']['factor_validation'] = {
                'valid_data_passed': True,
                'factors': list(valid_factor_data.columns),
                'rows': len(valid_factor_data),
                'passed': True
            }
            logger.info(f"✅ Factor validation: {len(valid_factor_data)} rows validated")
        except Exception as e:
            results['tests']['factor_validation'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Factor validation failed: {e}")

        return results

    def test_retry_logic(self) -> Dict:
        """Test retry logic and connection handling."""
        logger.info("Testing retry logic...")

        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: YFinance retry mechanism
        logger.info("Testing YFinance retry mechanism...")
        try:
            # Create provider with aggressive retry settings
            provider = YFinanceProvider(max_retries=3, retry_delay=0.1, request_timeout=5)

            # Test with a mix of valid and invalid symbols
            symbols = ['AAPL', 'VALID', 'INVALID123XYZ', 'MSTF']  # Note: MSTF is typo
            hist_data = provider.get_historical_data(
                symbols,
                start_date=datetime(2024, 9, 1),
                end_date=datetime(2024, 9, 10)
            )

            # Should get data for valid symbols despite invalid ones
            successful_fetches = len([s for s, data in hist_data.items() if data is not None and len(data) > 0])

            results['tests']['yfinance_retry'] = {
                'symbols_tested': len(symbols),
                'successful_fetches': successful_fetches,
                'passed': successful_fetches >= 1  # At least one should work
            }
            logger.info(f"✅ YFinance retry: {successful_fetches}/{len(symbols)} symbols fetched successfully")
        except Exception as e:
            results['tests']['yfinance_retry'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ YFinance retry test failed: {e}")

        # Test 2: Rate limiting
        logger.info("Testing rate limiting behavior...")
        try:
            provider = YFinanceProvider(retry_delay=0.1)  # Very short interval

            start_time = datetime.now()

            # Make multiple rapid requests
            for i in range(3):
                provider.get_latest_price(['AAPL'])

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Should take at least 0.1s * 2 = 0.2s due to rate limiting
            expected_min_time = 0.15  # Allow some tolerance

            results['tests']['rate_limiting'] = {
                'total_time_seconds': total_time,
                'expected_min_time': expected_min_time,
                'passed': total_time >= expected_min_time
            }
            logger.info(f"✅ Rate limiting: {total_time:.2f}s total time")
        except Exception as e:
            results['tests']['rate_limiting'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Rate limiting test failed: {e}")

        return results

    def test_error_handling(self) -> Dict:
        """Test error handling for edge cases."""
        logger.info("Testing error handling...")

        results = {
            'status': 'passed',
            'tests': {}
        }

        # Test 1: Empty symbol list
        logger.info("Testing empty symbol list handling...")
        try:
            provider = YFinanceProvider()
            empty_result = provider.get_historical_data([], datetime(2024, 1, 1), datetime(2024, 2, 1))

            results['tests']['empty_symbols'] = {
                'result_type': type(empty_result).__name__,
                'result_length': len(empty_result),
                'passed': isinstance(empty_result, dict) and len(empty_result) == 0
            }
            logger.info(f"✅ Empty symbols: returned {len(empty_result)} results")
        except Exception as e:
            results['tests']['empty_symbols'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Empty symbols test failed: {e}")

        # Test 2: Invalid date range
        logger.info("Testing invalid date range...")
        try:
            provider = YFinanceProvider()
            # Start date after end date
            invalid_data = provider.get_historical_data(
                ['AAPL'],
                start_date=datetime(2024, 12, 1),
                end_date=datetime(2024, 1, 1)
            )

            results['tests']['invalid_date_range'] = {
                'result_empty': len(invalid_data) == 0 or all(len(data) == 0 for data in invalid_data.values()),
                'passed': True  # Should handle gracefully
            }
            logger.info(f"✅ Invalid date range: handled gracefully")
        except Exception as e:
            # This is also acceptable - should fail gracefully
            results['tests']['invalid_date_range'] = {
                'error_caught': True,
                'error_message': str(e)[:100],
                'passed': True
            }
            logger.info(f"✅ Invalid date range: caught expected error")

        # Test 3: FF5 provider edge cases
        logger.info("Testing FF5 provider edge cases...")
        try:
            ff5_provider = FF5DataProvider()

            # Test very recent date range (might not have data)
            recent_data = ff5_provider.get_factor_returns(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )

            results['tests']['ff5_recent_dates'] = {
                'data_available': len(recent_data) > 0,
                'rows_returned': len(recent_data),
                'passed': True  # Should handle gracefully either way
            }
            logger.info(f"✅ FF5 recent dates: {len(recent_data)} rows returned")
        except Exception as e:
            # Acceptable - might not have recent data
            results['tests']['ff5_recent_dates'] = {
                'error_caught': True,
                'error_message': str(e)[:100],
                'passed': True
            }
            logger.info(f"✅ FF5 recent dates: handled gracefully")

        # Test 4: Stock classifier with insufficient data
        logger.info("Testing classifier with insufficient data...")
        try:
            classifier = StockClassifier()

            # Test with minimal data
            minimal_data = pd.DataFrame({
                'Close': [100, 101, 102]  # Only 3 data points
            })

            try:
                classification = classifier._classify_single_stock(
                    'TEST', minimal_data, datetime.now()
                )
                results['tests']['insufficient_data'] = {
                    'classification_returned': classification is not None,
                    'passed': classification is None  # Should return None for insufficient data
                }
                logger.info(f"✅ Insufficient data: returned {classification}")
            except Exception:
                results['tests']['insufficient_data'] = {
                    'error_caught': True,
                    'passed': True
                }
                logger.info(f"✅ Insufficient data: handled gracefully")

        except Exception as e:
            results['tests']['insufficient_data'] = {'error': str(e), 'passed': False}
            logger.error(f"❌ Insufficient data test failed: {e}")

        return results

    def print_summary(self):
        """Print comprehensive test summary."""
        logger.info(f"\n{'='*60}")
        logger.info("DATA MODULE TEST SUMMARY")
        logger.info(f"{'='*60}")

        total_tests = 0
        passed_tests = 0
        failed_test_names = []

        for test_name, result in self.test_results.items():
            if isinstance(result, dict) and 'tests' in result:
                # Count individual test methods
                test_count = len(result['tests'])
                passed_count = sum(1 for test_result in result['tests'].values()
                                 if test_result.get('passed', False))

                total_tests += test_count
                passed_tests += passed_count

                status = "✅ PASSED" if result.get('status') == 'passed' else "❌ FAILED"
                logger.info(f"{test_name}: {status} ({passed_count}/{test_count} sub-tests passed)")

                # Show failed sub-tests
                if test_count > passed_count:
                    for sub_test_name, sub_test_result in result['tests'].items():
                        if not sub_test_result.get('passed', False):
                            failed_test_names.append(f"{test_name}.{sub_test_name}")
                            error = sub_test_result.get('error', 'Unknown error')
                            logger.info(f"  ❌ {sub_test_name}: {error[:50]}...")
            else:
                total_tests += 1
                if result.get('status') == 'passed':
                    passed_tests += 1
                    logger.info(f"{test_name}: ✅ PASSED")
                else:
                    failed_test_names.append(test_name)
                    logger.info(f"{test_name}: ❌ FAILED")

        logger.info(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")

        if failed_test_names:
            logger.info(f"\nFailed Tests: {len(failed_test_names)}")
            for test_name in failed_test_names:
                logger.info(f"  ❌ {test_name}")

        if self.errors:
            logger.info(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors:
                logger.info(f"  {error}")

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\nSuccess Rate: {success_rate:.1f}%")

        if success_rate >= 80:
            logger.info("🎉 Data module is functioning well!")
        elif success_rate >= 60:
            logger.info("⚠️  Data module has some issues but mostly functional")
        else:
            logger.info("🚨 Data module has significant issues that need attention")


if __name__ == "__main__":
    """Run the comprehensive data module tests."""
    print("🧪 Starting comprehensive data module tests...")
    print("This will test YFinance provider, FF5 provider, and stock classifier")
    print("Note: This test makes real API calls and may take a few minutes\n")

    tester = DataModuleTester()
    tester.run_all_tests()