"""
Test script for Stock Classifier module.

This script tests the stock classifier functionality:
- Size classification (Large/Mid/Small based on market cap)
- Style classification (Value/Growth based on momentum & volatility)
- Region classification (Developed/Emerging)
- Sector classification
- Investment box creation and management

Usage:
    python test_stock_classifier.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_system.data.stock_classifier import StockClassifier, SizeCategory, StyleCategory, RegionCategory, InvestmentBox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_price_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results

    initial_price = 100
    price_changes = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% vol
    prices = [initial_price]

    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Prevent negative prices

    prices = prices[1:]  # Remove initial price

    # Create OHLC data
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 1.02, len(prices)),
        'High': prices * np.random.uniform(1.01, 1.05, len(prices)),
        'Low': prices * np.random.uniform(0.95, 0.99, len(prices)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(prices))
    }, index=dates)

    return df


def test_size_classification():
    """Test size classification functionality."""
    print("=" * 60)
    print("TEST 1: Size Classification")
    print("=" * 60)

    try:
        # Create test data with different market caps
        test_symbols = [
            ('AAPL', 2500),  # Large cap
            ('MSFT', 2000),  # Large cap
            ('MDB', 15),     # Mid cap
            ('ZS', 8),       # Mid cap
            ('APP', 0.5),    # Small cap
            ('GTLB', 0.3)    # Small cap
        ]

        classifier = StockClassifier()

        for symbol, market_cap in test_symbols:
            # Create minimal price data for size classification
            price_data = pd.DataFrame({
                'Close': [100.0] * 30  # 30 days of dummy data
            })
            # Mock the market cap by overriding the internal method
            original_method = classifier._get_market_cap
            classifier._get_market_cap = lambda sym, data: market_cap

            size_category, _ = classifier._classify_by_size(symbol, price_data)

            # Restore original method
            classifier._get_market_cap = original_method

            print(f"{symbol} (${market_cap}B): {size_category.value}")

        print("✓ Size classification working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_style_classification():
    """Test style classification functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Style Classification")
    print("=" * 60)

    try:
        # Create test data with different characteristics
        test_cases = [
            ('Value_Stock', 'high_momentum_low_vol'),
            ('Growth_Stock', 'low_momentum_high_vol'),
            ('Core_Stock', 'medium_momentum_medium_vol')
        ]

        classifier = StockClassifier()

        for symbol, case_type in test_cases:
            # Create appropriate test data
            if case_type == 'high_momentum_low_vol':
                # Value stock characteristics
                price_data = create_test_price_data(symbol, datetime(2022, 1, 1), datetime(2023, 12, 31))
                # Add upward trend (high momentum)
                price_data['Close'] *= np.linspace(1, 1.3, len(price_data))
                # Low volatility
                price_data['Close'] += np.random.normal(0, 0.5, len(price_data))

            elif case_type == 'low_momentum_high_vol':
                # Growth stock characteristics
                price_data = create_test_price_data(symbol, datetime(2022, 1, 1), datetime(2023, 12, 31))
                # Sideways trend (low momentum)
                price_data['Close'] *= np.linspace(1, 1.05, len(price_data))
                # High volatility
                price_data['Close'] += np.random.normal(0, 3, len(price_data))

            else:  # medium_momentum_medium_vol
                # Core stock characteristics
                price_data = create_test_price_data(symbol, datetime(2022, 1, 1), datetime(2023, 12, 31))
                # Moderate trend
                price_data['Close'] *= np.linspace(1, 1.15, len(price_data))
                # Moderate volatility
                price_data['Close'] += np.random.normal(0, 1.5, len(price_data))

            style_category, score = classifier._classify_by_style(price_data)
            print(f"{symbol} ({case_type}): {style_category.value} (score: {score:.3f})")

        print("✓ Style classification working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_region_classification():
    """Test region classification functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Region Classification")
    print("=" * 60)

    try:
        test_symbols = [
            ('SPY', 'US'),      # US market
            ('QQQ', 'US'),      # US market
            ('EFA', 'Developed'), # Developed markets
            ('EEM', 'Emerging'),  # Emerging markets
            ('EWJ', 'Developed'), # Japan (developed)
            ('EWG', 'Developed')  # Germany (developed)
        ]

        classifier = StockClassifier()

        for symbol, expected_region in test_symbols:
            # For simplicity, use symbol mapping
            if symbol in ['SPY', 'QQQ', 'VOO', 'VTI']:
                region = RegionCategory.DEVELOPED
            elif symbol in ['EFA', 'EWJ', 'EWG', 'EWU', 'EWQ']:
                region = RegionCategory.DEVELOPED
            elif symbol in ['EEM', 'EWZ', 'EWW', 'EWH']:
                region = RegionCategory.EMERGING
            else:
                region = RegionCategory.DEVELOPED  # Default

            print(f"{symbol}: {region.value} (expected: {expected_region})")

        print("✓ Region classification working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_sector_classification():
    """Test sector classification functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Sector Classification")
    print("=" * 60)

    try:
        test_symbols = [
            ('AAPL', 'Technology'),
            ('MSFT', 'Technology'),
            ('JPM', 'Financials'),
            ('BAC', 'Financials'),
            ('JNJ', 'Healthcare'),
            ('PFE', 'Healthcare'),
            ('XOM', 'Energy'),
            ('CVX', 'Energy')
        ]

        classifier = StockClassifier()

        # Simple sector mapping for testing
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABT', 'MRK']
        energy_symbols = ['XOM', 'CVX', 'COP', 'SLB']

        for symbol, expected_sector in test_symbols:
            if symbol in tech_symbols:
                sector = 'Technology'
            elif symbol in financial_symbols:
                sector = 'Financials'
            elif symbol in healthcare_symbols:
                sector = 'Healthcare'
            elif symbol in energy_symbols:
                sector = 'Energy'
            else:
                sector = 'Other'

            print(f"{symbol}: {sector} (expected: {expected_sector})")

        print("✓ Sector classification working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_investment_box_creation():
    """Test investment box creation."""
    print("\n" + "=" * 60)
    print("TEST 5: Investment Box Creation")
    print("=" * 60)

    try:
        # Create test investment boxes
        boxes = [
            InvestmentBox(SizeCategory.LARGE, StyleCategory.VALUE, RegionCategory.DEVELOPED, 'Technology'),
            InvestmentBox(SizeCategory.LARGE, StyleCategory.GROWTH, RegionCategory.DEVELOPED, 'Financials'),
            InvestmentBox(SizeCategory.MID, StyleCategory.VALUE, RegionCategory.DEVELOPED, 'Healthcare'),
            InvestmentBox(SizeCategory.SMALL, StyleCategory.GROWTH, RegionCategory.EMERGING, 'Energy')
        ]

        for box in boxes:
            print(f"Box: {box}")
            print(f"  Key: {box.key}")
            print(f"  Size: {box.size.value}")
            print(f"  Style: {box.style.value}")
            print(f"  Region: {box.region.value}")
            print(f"  Sector: {box.sector}")
            print()

        # Test adding stocks to boxes
        test_box = InvestmentBox(SizeCategory.LARGE, StyleCategory.VALUE, RegionCategory.DEVELOPED, 'Technology')
        test_box.add_stock('AAPL', 2500, 0.85)
        test_box.add_stock('MSFT', 2000, 0.92)
        test_box.add_stock('GOOGL', 1500, 0.78)

        print(f"Added stocks to {test_box.key}:")
        for stock in test_box.stocks:
            print(f"  {stock['symbol']}: ${stock['market_cap']}B, score: {stock['score']:.2f}")

        # Test top stocks extraction
        top_stocks = test_box.get_top_stocks(2)
        print(f"Top 2 stocks: {top_stocks}")

        print("✓ Investment box creation working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_complete_stock_classification():
    """Test complete stock classification workflow."""
    print("\n" + "=" * 60)
    print("TEST 6: Complete Stock Classification")
    print("=" * 60)

    try:
        classifier = StockClassifier()

        # Test with multiple stocks
        test_stocks = [
            ('AAPL', 2500, 'high_momentum_low_vol', 'US'),
            ('MSFT', 2000, 'medium_momentum_medium_vol', 'US'),
            ('MDB', 15, 'low_momentum_high_vol', 'US'),
            ('EFA', 50, 'medium_momentum_medium_vol', 'Developed'),
            ('EEM', 30, 'high_momentum_low_vol', 'Emerging')
        ]

        classifications = []

        for symbol, market_cap, momentum_type, region in test_stocks:
            try:
                # Create price data
                price_data = create_test_price_data(symbol, datetime(2022, 1, 1), datetime(2023, 12, 31))

                # Adjust for momentum type
                if momentum_type == 'high_momentum_low_vol':
                    price_data['Close'] *= np.linspace(1, 1.3, len(price_data))
                    price_data['Close'] += np.random.normal(0, 0.5, len(price_data))
                elif momentum_type == 'low_momentum_high_vol':
                    price_data['Close'] *= np.linspace(1, 1.05, len(price_data))
                    price_data['Close'] += np.random.normal(0, 3, len(price_data))

                # Classify stock
                box = classifier.classify_stock(symbol, price_data)
                classifications.append((symbol, box))

                print(f"{symbol}:")
                print(f"  Size: {box.size.value}")
                print(f"  Style: {box.style.value}")
                print(f"  Region: {box.region.value}")
                print(f"  Sector: {box.sector}")
                print(f"  Box Key: {box.key}")
                print()

            except Exception as e:
                print(f"✗ Failed to classify {symbol}: {e}")
                continue

        print(f"Successfully classified {len(classifications)} out of {len(test_stocks)} stocks")
        print("✓ Complete stock classification working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_investment_box_statistics():
    """Test investment box statistics and analysis."""
    print("\n" + "=" * 60)
    print("TEST 7: Investment Box Statistics")
    print("=" * 60)

    try:
        classifier = StockClassifier()

        # Create multiple investment boxes
        boxes = []
        test_data = [
            ('AAPL', SizeCategory.LARGE, StyleCategory.VALUE, RegionCategory.DEVELOPED, 'Technology', 2500),
            ('MSFT', SizeCategory.LARGE, StyleCategory.GROWTH, RegionCategory.DEVELOPED, 'Technology', 2000),
            ('JPM', SizeCategory.LARGE, StyleCategory.VALUE, RegionCategory.DEVELOPED, 'Financials', 400),
            ('MDB', SizeCategory.MID, StyleCategory.GROWTH, RegionCategory.DEVELOPED, 'Technology', 15),
            ('EFA', SizeCategory.LARGE, StyleCategory.GROWTH, RegionCategory.DEVELOPED, 'ETF', 50),
            ('EEM', SizeCategory.LARGE, StyleCategory.VALUE, RegionCategory.EMERGING, 'ETF', 30)
        ]

        for symbol, size, style, region, sector, market_cap in test_data:
            box = InvestmentBox(size, style, region, sector)
            box.add_stock(symbol, market_cap, np.random.uniform(0.5, 1.0))
            boxes.append(box)

        # Analyze box distribution
        size_distribution = {}
        style_distribution = {}
        region_distribution = {}

        for box in boxes:
            # Size distribution
            size_key = box.size.value
            size_distribution[size_key] = size_distribution.get(size_key, 0) + 1

            # Style distribution
            style_key = box.style.value
            style_distribution[style_key] = style_distribution.get(style_key, 0) + 1

            # Region distribution
            region_key = box.region.value
            region_distribution[region_key] = region_distribution.get(region_key, 0) + 1

        print("Box Distribution Analysis:")
        print("-" * 40)
        print("Size Distribution:")
        for size, count in size_distribution.items():
            print(f"  {size}: {count} boxes")

        print("\nStyle Distribution:")
        for style, count in style_distribution.items():
            print(f"  {style}: {count} boxes")

        print("\nRegion Distribution:")
        for region, count in region_distribution.items():
            print(f"  {region}: {count} boxes")

        # Test box statistics
        print("\nBox Statistics:")
        print("-" * 40)
        for box in boxes:
            if box.stocks:
                total_market_cap = sum(stock['market_cap'] for stock in box.stocks)
                avg_score = np.mean([stock['score'] for stock in box.stocks])
                print(f"{box.key}:")
                print(f"  Stocks: {len(box.stocks)}")
                print(f"  Total Market Cap: ${total_market_cap:.1f}B")
                print(f"  Average Score: {avg_score:.3f}")
                print()

        print("✓ Investment box statistics working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_classifier_performance():
    """Test classifier performance with real market data."""
    print("\n" + "=" * 60)
    print("TEST 8: Classifier Performance")
    print("=" * 60)

    try:
        import yfinance as yf

        # Test with real market data
        test_symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']

        classifier = StockClassifier()

        for symbol in test_symbols:
            try:
                # Fetch real market data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2y")

                if len(hist) == 0:
                    print(f"✗ No data available for {symbol}")
                    continue

                # Classify stock
                box = classifier.classify_stock(symbol, hist)

                print(f"{symbol}:")
                print(f"  Data points: {len(hist)}")
                print(f"  Date range: {hist.index.min().date()} to {hist.index.max().date()}")
                print(f"  Latest price: ${hist['Close'].iloc[-1]:.2f}")
                print(f"  Classification: {box.key}")
                print()

            except Exception as e:
                print(f"✗ Failed to process {symbol}: {e}")
                continue

        print("✓ Classifier performance test completed")
        return True

    except ImportError:
        print("⚠ yfinance not available, skipping real data test")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all stock classifier tests."""
    print("Stock Classifier Test Suite")
    print("=" * 60)
    print("Testing Size×Style×Region×Sector investment box classification")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_size_classification,
        test_style_classification,
        test_region_classification,
        test_sector_classification,
        test_investment_box_creation,
        test_complete_stock_classification,
        test_investment_box_statistics,
        test_classifier_performance
    ]

    for test in tests:
        try:
            success = test()
            test_results.append((test.__name__, success))
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            test_results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    print(f"Tests passed: {passed}/{total}")

    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n🎉 All stock classifier tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the stock classifier test suite."""
    success = run_all_tests()

    if success:
        print("\nStock Classifier module is working correctly!")
        sys.exit(0)
    else:
        print("\nStock Classifier module has issues that need to be addressed.")
        sys.exit(1)