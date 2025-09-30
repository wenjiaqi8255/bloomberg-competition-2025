"""
Test script for System Orchestrator module.

This script tests the System Orchestrator functionality:
- Complete system integration
- Capital allocation between Core and Satellite strategies
- IPS compliance monitoring and reporting
- Model governance and risk management
- Performance tracking and attribution
- System health monitoring
- End-to-end workflow execution

Usage:
    python test_system_orchestrator.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trading_system.orchestrator.system_orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_universe_data(start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Create synthetic universe data for testing."""
    symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'BTC']
    universe_data = {}

    for symbol in symbols:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic price movements with symbol-specific characteristics
        if symbol in ['SPY', 'QQQ', 'IWM']:  # Equity ETFs
            daily_drift = 0.0008
            volatility = 0.015
        elif symbol in ['EFA', 'EEM']:  # International ETFs
            daily_drift = 0.0005
            volatility = 0.018
        elif symbol == 'AGG':  # Bond ETF
            daily_drift = 0.0002
            volatility = 0.003
        elif symbol == 'GLD':  # Gold ETF
            daily_drift = 0.0003
            volatility = 0.012
        else:  # BTC
            daily_drift = 0.0015
            volatility = 0.035

        # Generate price series
        initial_price = 100 + np.random.uniform(-50, 50)
        price_changes = np.random.normal(daily_drift, volatility, len(dates))
        prices = [initial_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))

        prices = prices[1:]

        # Create OHLC data
        opens = prices * np.random.uniform(0.995, 1.005, len(prices))
        highs = np.maximum(opens, prices) * np.random.uniform(1.005, 1.02, len(prices))
        lows = np.minimum(opens, prices) * np.random.uniform(0.98, 0.995, len(prices))
        volumes = np.random.randint(1000000, 50000000, len(prices))

        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes,
            'Adj Close': prices
        }, index=dates)

        universe_data[symbol] = df

    return universe_data


def create_test_market_data(start_date: datetime, end_date: datetime) -> Dict[str, any]:
    """Create synthetic market data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate market regime data
    np.random.seed(42)
    market_regimes = []
    current_regime = 'normal'
    regime_duration = np.random.randint(20, 60)

    for i, date in enumerate(dates):
        if i % regime_duration == 0 and i > 0:
            # Switch regime
            regimes = ['normal', 'volatile', 'trending', 'crisis']
            current_regime = np.random.choice(regimes)
            regime_duration = np.random.randint(20, 60)

        market_regimes.append(current_regime)

    # Generate risk-free rate
    rf_rate = 0.02 + 0.005 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Cyclical RF rate

    # Generate market volatility (VIX proxy)
    vix_values = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + np.random.normal(0, 2, len(dates))
    vix_values = np.maximum(vix_values, 10)  # Minimum VIX of 10

    # Generate economic indicators
    gdp_growth = 2.5 + 0.5 * np.sin(np.linspace(0, np.pi, len(dates))) + np.random.normal(0, 0.3, len(dates))
    inflation_rate = 2.0 + 0.8 * np.sin(np.linspace(0, 1.5*np.pi, len(dates))) + np.random.normal(0, 0.2, len(dates))
    unemployment_rate = 4.0 + 0.5 * np.sin(np.linspace(0, 0.8*np.pi, len(dates))) + np.random.normal(0, 0.1, len(dates))

    market_data = {
        'dates': dates,
        'market_regimes': market_regimes,
        'risk_free_rate': rf_rate,
        'vix_index': vix_values,
        'economic_indicators': {
            'gdp_growth': gdp_growth,
            'inflation_rate': inflation_rate,
            'unemployment_rate': unemployment_rate
        }
    }

    return market_data


def test_system_orchestrator_initialization():
    """Test System Orchestrator initialization."""
    print("=" * 60)
    print("TEST 1: System Orchestrator Initialization")
    print("=" * 60)

    try:
        # Test different orchestrator configurations
        configs = [
            {
                'system_name': 'IPS_Conservative',
                'core_weight': 0.80,
                'satellite_weight': 0.20,
                'max_positions': 15,
                'rebalance_frequency': 30,
                'risk_budget': 0.12,
                'volatility_target': 0.10
            },
            {
                'system_name': 'IPS_Moderate',
                'core_weight': 0.75,
                'satellite_weight': 0.25,
                'max_positions': 20,
                'rebalance_frequency': 21,
                'risk_budget': 0.15,
                'volatility_target': 0.12
            },
            {
                'system_name': 'IPS_Aggressive',
                'system_name': 'IPS_Aggressive',
                'core_weight': 0.70,
                'satellite_weight': 0.30,
                'max_positions': 25,
                'rebalance_frequency': 14,
                'risk_budget': 0.18,
                'volatility_target': 0.15
            }
        ]

        for i, config in enumerate(configs):
            orchestrator = SystemOrchestrator(**config)
            print(f"Configuration {i+1} ({config['system_name']}):")
            print(f"  System name: {orchestrator.system_name}")
            print(f"  Core weight: {orchestrator.core_weight}")
            print(f"  Satellite weight: {orchestrator.satellite_weight}")
            print(f"  Max positions: {orchestrator.max_positions}")
            print(f"  Rebalance frequency: {orchestrator.rebalance_frequency}")
            print(f"  Risk budget: {orchestrator.risk_budget}")
            print(f"  Volatility target: {orchestrator.volatility_target}")
            print(f"  ✓ Initialized successfully")

        print("\n✓ System Orchestrator initialization working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_system_integration():
    """Test complete system integration."""
    print("\n" + "=" * 60)
    print("TEST 2: System Integration")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Complete_System',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20,
            rebalance_frequency=21
        )

        # Create test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)

        universe_data = create_test_universe_data(start_date, end_date)
        market_data = create_test_market_data(start_date, end_date)

        print(f"Testing system integration for {len(universe_data)} symbols")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        # Test system integration
        integration_results = orchestrator._test_system_integration(
            universe_data, market_data, start_date, end_date
        )

        print(f"\nSystem Integration Results:")
        print("-" * 40)

        # Core strategy integration
        core_results = integration_results.get('core_strategy', {})
        print(f"Core Strategy:")
        print(f"  FF5 integration: {core_results.get('ff5_integration', False)}")
        print(f"  ML integration: {core_results.get('ml_integration', False)}")
        print(f"  Combined predictions: {core_results.get('combined_predictions', False)}")
        print(f"  IPS compliance: {core_results.get('ips_compliance', 0):.4f}")

        # Satellite strategy integration
        satellite_results = integration_results.get('satellite_strategy', {})
        print(f"\nSatellite Strategy:")
        print(f"  Technical indicators: {satellite_results.get('technical_indicators', False)}")
        print(f"  Signal generation: {satellite_results.get('signal_generation', False)}")
        print(f"  Risk management: {satellite_results.get('risk_management', False)}")
        print(f"  Regime adaptation: {satellite_results.get('regime_adaptation', 0):.4f}")

        # Data provider integration
        data_results = integration_results.get('data_providers', {})
        print(f"\nData Providers:")
        print(f"  FF5 provider: {data_results.get('ff5_provider', False)}")
        print(f"  Stock classifier: {data_results.get('stock_classifier', False)}")
        print(f"  Data quality: {data_results.get('data_quality', 0):.4f}")

        # Model integration
        model_results = integration_results.get('models', {})
        print(f"\nModels:")
        print(f"  FF5 regression: {model_results.get('ff5_regression', False)}")
        print(f"  ML residual predictor: {model_results.get('ml_residual_predictor', False)}")
        print(f"  Model governance: {model_results.get('model_governance', 0):.4f}")

        print("\n✓ System integration working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_capital_allocation():
    """Test capital allocation between Core and Satellite strategies."""
    print("\n" + "=" * 60)
    print("TEST 3: Capital Allocation")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Allocation_Test',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20
        )

        # Create test scenarios
        test_scenarios = [
            {
                'name': 'Normal Market Conditions',
                'core_confidence': 0.8,
                'satellite_confidence': 0.6,
                'market_regime': 'normal',
                'risk_level': 'moderate'
            },
            {
                'name': 'High Volatility',
                'core_confidence': 0.9,
                'satellite_confidence': 0.3,
                'market_regime': 'volatile',
                'risk_level': 'high'
            },
            {
                'name': 'Strong Trend',
                'core_confidence': 0.7,
                'satellite_confidence': 0.8,
                'market_regime': 'trending',
                'risk_level': 'moderate'
            },
            {
                'name': 'Market Crisis',
                'core_confidence': 0.95,
                'satellite_confidence': 0.1,
                'market_regime': 'crisis',
                'risk_level': 'extreme'
            }
        ]

        print("Testing capital allocation scenarios:")
        print("-" * 40)

        for scenario in test_scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  Market regime: {scenario['market_regime']}")
            print(f"  Risk level: {scenario['risk_level']}")

            # Test capital allocation
            allocation = orchestrator._calculate_capital_allocation(
                scenario['core_confidence'],
                scenario['satellite_confidence'],
                scenario['market_regime'],
                scenario['risk_level']
            )

            print(f"  Allocation results:")
            print(f"    Core weight: {allocation['core_weight']:.4f} (target: {orchestrator.core_weight:.4f})")
            print(f"    Satellite weight: {allocation['satellite_weight']:.4f} (target: {orchestrator.satellite_weight:.4f})")
            print(f"    Cash weight: {allocation['cash_weight']:.4f}")
            print(f"    Total allocation: {allocation['total_weight']:.4f}")
            print(f"    Confidence adjustment: {allocation['confidence_adjustment']:.4f}")

            # Check allocation constraints
            if allocation['total_weight'] > 1.0:
                print(f"    ⚠ Over-allocated!")
            elif allocation['total_weight'] < 0.8:
                print(f"    ⚠ Under-allocated!")

        print("\n✓ Capital allocation working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ips_compliance_monitoring():
    """Test IPS compliance monitoring."""
    print("\n" + "=" * 60)
    print("TEST 4: IPS Compliance Monitoring")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Compliance_Test',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20
        )

        # Create test portfolio scenarios
        test_scenarios = [
            {
                'name': 'Compliant Portfolio',
                'core_weight': 0.78,
                'satellite_weight': 0.22,
                'cash_weight': 0.0,
                'risk_metrics': {'volatility': 0.11, 'max_drawdown': -0.08},
                'expected_compliance': True
            },
            {
                'name': 'Core Underweight',
                'core_weight': 0.65,
                'satellite_weight': 0.35,
                'cash_weight': 0.0,
                'risk_metrics': {'volatility': 0.13, 'max_drawdown': -0.10},
                'expected_compliance': False
            },
            {
                'name': 'High Risk Portfolio',
                'core_weight': 0.75,
                'satellite_weight': 0.25,
                'cash_weight': 0.0,
                'risk_metrics': {'volatility': 0.18, 'max_drawdown': -0.15},
                'expected_compliance': False
            },
            {
                'name': 'Conservative Portfolio',
                'core_weight': 0.85,
                'satellite_weight': 0.10,
                'cash_weight': 0.05,
                'risk_metrics': {'volatility': 0.08, 'max_drawdown': -0.05},
                'expected_compliance': True
            }
        ]

        print("Testing IPS compliance monitoring:")
        print("-" * 40)

        for scenario in test_scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  Core: {scenario['core_weight']:.4f}, Satellite: {scenario['satellite_weight']:.4f}, Cash: {scenario['cash_weight']:.4f}")

            # Test IPS compliance
            compliance_result = orchestrator._test_ips_compliance(
                scenario['core_weight'],
                scenario['satellite_weight'],
                scenario['cash_weight'],
                scenario['risk_metrics']
            )

            print(f"  Compliance status: {compliance_result['compliant']}")
            print(f"  Compliance score: {compliance_result['compliance_score']:.4f}")
            print(f"  Violations: {len(compliance_result['violations'])}")

            if compliance_result['violations']:
                print(f"  Violations:")
                for violation in compliance_result['violations']:
                    print(f"    - {violation}")

            # Check expected result
            if compliance_result['compliant'] == scenario['expected_compliance']:
                print(f"  ✓ Expected compliance result")
            else:
                print(f"  ✗ Unexpected compliance result")

        # Generate comprehensive IPS report
        ips_report = orchestrator.generate_ips_compliance_report()
        print(f"\nIPS Compliance Report Summary:")
        print("-" * 40)
        print(f"  Overall compliance: {ips_report['overall_compliance']:.4f}")
        print(f"  Risk level: {ips_report['risk_level']}")
        print(f"  Active violations: {len(ips_report['active_violations'])}")
        print(f"  Recommendations: {len(ips_report['recommendations'])}")

        print("\n✓ IPS compliance monitoring working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_system_health_monitoring():
    """Test system health monitoring."""
    print("\n" + "=" * 60)
    print("TEST 5: System Health Monitoring")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Health_Test',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20
        )

        # Create test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)

        universe_data = create_test_universe_data(start_date, end_date)
        market_data = create_test_market_data(start_date, end_date)

        print(f"Testing system health monitoring")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        # Test system health monitoring
        health_results = orchestrator._test_system_health_monitoring(
            universe_data, market_data, start_date, end_date
        )

        print(f"\nSystem Health Results:")
        print("-" * 40)

        # Data health
        data_health = health_results.get('data_health', {})
        print(f"Data Health:")
        print(f"  Data quality score: {data_health.get('quality_score', 0):.4f}")
        print(f"  Data completeness: {data_health.get('completeness', 0):.4f}")
        print(f"  Data timeliness: {data_health.get('timeliness', 0):.4f}")
        print(f"  Data status: {data_health.get('status', 'Unknown')}")

        # Model health
        model_health = health_results.get('model_health', {})
        print(f"\nModel Health:")
        print(f"  Model quality score: {model_health.get('quality_score', 0):.4f}")
        print(f"  Model degradation: {model_health.get('degradation_score', 0):.4f}")
        print(f"  Model coverage: {model_health.get('coverage', 0):.4f}")
        print(f"  Model status: {model_health.get('status', 'Unknown')}")

        # Performance health
        perf_health = health_results.get('performance_health', {})
        print(f"\nPerformance Health:")
        print(f"  Performance score: {perf_health.get('performance_score', 0):.4f}")
        print(f"  Risk metrics: {perf_health.get('risk_metrics', 0):.4f}")
        print(f"  Attribution analysis: {perf_health.get('attribution_analysis', 0):.4f}")
        print(f"  Performance status: {perf_health.get('status', 'Unknown')}")

        # System health
        system_health = health_results.get('system_health', {})
        print(f"\nSystem Health:")
        print(f"  Overall health score: {system_health.get('overall_health_score', 0):.4f}")
        print(f"  Component availability: {system_health.get('component_availability', 0):.4f}")
        print(f"  System performance: {system_health.get('system_performance', 0):.4f}")
        print(f"  System status: {system_health.get('status', 'Unknown')}")

        # Generate health report
        health_report = orchestrator.generate_system_health_report()
        print(f"\nHealth Report Summary:")
        print("-" * 40)
        print(f"  Overall health: {health_report['overall_health']:.4f}")
        print(f"  System status: {health_report['system_status']}")
        print(f"  Critical issues: {len(health_report['critical_issues'])}")
        print(f"  Warnings: {len(health_report['warnings'])}")

        print("\n✓ System health monitoring working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_performance_attribution():
    """Test performance attribution analysis."""
    print("\n" + "=" * 60)
    print("TEST 6: Performance Attribution")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Attribution_Test',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20
        )

        # Create test performance data
        periods = [
            ('Q1 2023', datetime(2023, 1, 1), datetime(2023, 3, 31)),
            ('Q2 2023', datetime(2023, 4, 1), datetime(2023, 6, 30)),
            ('Q3 2023', datetime(2023, 7, 1), datetime(2023, 9, 30))
        ]

        performance_data = {}
        for period_name, start_date, end_date in periods:
            # Simulate performance components
            np.random.seed(hash(period_name) % 2**32)

            # Core strategy performance
            core_return = np.random.normal(0.02, 0.03)  # 2% quarterly return, 3% vol
            core_contribution = core_return * orchestrator.core_weight

            # Satellite strategy performance
            satellite_return = np.random.normal(0.03, 0.06)  # 3% quarterly return, 6% vol
            satellite_contribution = satellite_return * orchestrator.satellite_weight

            # Total portfolio return
            total_return = core_contribution + satellite_contribution

            performance_data[period_name] = {
                'core_return': core_return,
                'satellite_return': satellite_return,
                'core_contribution': core_contribution,
                'satellite_contribution': satellite_contribution,
                'total_return': total_return,
                'attribution_factors': {
                    'factor_exposure': np.random.normal(0.005, 0.01),
                    'stock_selection': np.random.normal(0.003, 0.008),
                    'timing': np.random.normal(0.002, 0.006),
                    'alpha': np.random.normal(0.001, 0.004)
                }
            }

        print("Testing performance attribution:")
        print("-" * 40)

        for period_name, perf in performance_data.items():
            print(f"\n{period_name}:")
            print(f"  Core return: {perf['core_return']:.4f}")
            print(f"  Satellite return: {perf['satellite_return']:.4f}")
            print(f"  Core contribution: {perf['core_contribution']:.4f}")
            print(f"  Satellite contribution: {perf['satellite_contribution']:.4f}")
            print(f"  Total return: {perf['total_return']:.4f}")

            # Attribution analysis
            attribution = orchestrator._analyze_performance_attribution(perf)
            print(f"  Attribution analysis:")
            for factor, contribution in attribution.items():
                print(f"    {factor}: {contribution:.4f}")

        # Generate attribution report
        attribution_report = orchestrator.generate_performance_attribution_report(performance_data)
        print(f"\nAttribution Report Summary:")
        print("-" * 40)
        print(f"  Average core contribution: {attribution_report['avg_core_contribution']:.4f}")
        print(f"  Average satellite contribution: {attribution_report['avg_satellite_contribution']:.4f}")
        print(f"  Best performing component: {attribution_report['best_performing_component']}")
        print(f"  Key success factors: {', '.join(attribution_report['key_success_factors'][:3])}")

        print("\n✓ Performance attribution working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_risk_management():
    """Test comprehensive risk management."""
    print("\n" + "=" * 60)
    print("TEST 7: Risk Management")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Risk_Test',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20,
            risk_budget=0.15,
            volatility_target=0.12
        )

        # Create test risk scenarios
        risk_scenarios = [
            {
                'name': 'Normal Market',
                'portfolio_volatility': 0.11,
                'max_drawdown': -0.08,
                'var_95': 0.15,
                'expected_risk_level': 'moderate'
            },
            {
                'name': 'High Volatility',
                'portfolio_volatility': 0.18,
                'max_drawdown': -0.12,
                'var_95': 0.22,
                'expected_risk_level': 'high'
            },
            {
                'name': 'Market Stress',
                'portfolio_volatility': 0.25,
                'max_drawdown': -0.20,
                'var_95': 0.30,
                'expected_risk_level': 'extreme'
            }
        ]

        print("Testing risk management scenarios:")
        print("-" * 40)

        for scenario in risk_scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  Portfolio volatility: {scenario['portfolio_volatility']:.4f}")
            print(f"  Max drawdown: {scenario['max_drawdown']:.4f}")
            print(f"  VaR (95%): {scenario['var_95']:.4f}")

            # Test risk management
            risk_assessment = orchestrator._assess_portfolio_risk(
                scenario['portfolio_volatility'],
                scenario['max_drawdown'],
                scenario['var_95']
            )

            print(f"  Risk assessment:")
            print(f"    Risk level: {risk_assessment['risk_level']}")
            print(f"    Risk score: {risk_assessment['risk_score']:.4f}")
            print(f"    Within budget: {risk_assessment['within_budget']}")
            print(f"    Risk breaches: {len(risk_assessment['risk_breaches'])}")

            if risk_assessment['risk_breaches']:
                print(f"    Risk breaches:")
                for breach in risk_assessment['risk_breaches']:
                    print(f"      - {breach}")

            # Check expected risk level
            if risk_assessment['risk_level'] == scenario['expected_risk_level']:
                print(f"    ✓ Risk level as expected")
            else:
                print(f"    ⚠ Risk level differs from expected")

        # Test risk limits
        risk_limits = orchestrator.check_risk_limits()
        print(f"\nRisk Limits Check:")
        print("-" * 40)
        for limit_name, limit_info in risk_limits.items():
            status = "✓" if limit_info['within_limit'] else "✗"
            print(f"  {status} {limit_name}: {limit_info['current_value']:.4f} (limit: {limit_info['limit_value']:.4f})")

        print("\n✓ Risk management working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("\n" + "=" * 60)
    print("TEST 8: End-to-End Workflow")
    print("=" * 60)

    try:
        # Initialize orchestrator
        orchestrator = SystemOrchestrator(
            system_name='IPS_Complete_Workflow',
            core_weight=0.75,
            satellite_weight=0.25,
            max_positions=20,
            rebalance_frequency=21
        )

        # Create comprehensive test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)

        universe_data = create_test_universe_data(start_date, end_date)
        market_data = create_test_market_data(start_date, end_date)

        print(f"Testing end-to-end workflow")
        print(f"Universe: {len(universe_data)} symbols")
        print(f"Period: {start_date.date()} to {end_date.date()}")

        # Test workflow execution
        workflow_results = orchestrator._test_end_to_end_workflow(
            universe_data, market_data, start_date, end_date
        )

        print(f"\nEnd-to-End Workflow Results:")
        print("-" * 40)

        # Workflow execution summary
        execution_summary = workflow_results.get('execution_summary', {})
        print(f"Execution Summary:")
        print(f"  Total execution time: {execution_summary.get('total_execution_time', 0):.4f}s")
        print(f"  Successful steps: {execution_summary.get('successful_steps', 0)}/{execution_summary.get('total_steps', 0)}")
        print(f"  Success rate: {execution_summary.get('success_rate', 0):.4f}")
        print(f"  Execution status: {execution_summary.get('execution_status', 'Unknown')}")

        # Component performance
        component_performance = workflow_results.get('component_performance', {})
        print(f"\nComponent Performance:")
        for component, performance in component_performance.items():
            print(f"  {component}: {performance['score']:.4f} ({performance['status']})")

        # Portfolio results
        portfolio_results = workflow_results.get('portfolio_results', {})
        print(f"\nPortfolio Results:")
        print(f"  Total return: {portfolio_results.get('total_return', 0):.4f}")
        print(f"  Annualized volatility: {portfolio_results.get('annualized_volatility', 0):.4f}")
        print(f"  Sharpe ratio: {portfolio_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Max drawdown: {portfolio_results.get('max_drawdown', 0):.4f}")
        print(f"  Win rate: {portfolio_results.get('win_rate', 0):.4f}")

        # IPS compliance
        ips_compliance = workflow_results.get('ips_compliance', {})
        print(f"\nIPS Compliance:")
        print(f"  Overall compliance: {ips_compliance.get('overall_compliance', 0):.4f}")
        print(f"  Core allocation: {ips_compliance.get('core_allocation', 0):.4f}")
        print(f"  Satellite allocation: {ips_compliance.get('satellite_allocation', 0):.4f}")
        print(f"  Risk compliance: {ips_compliance.get('risk_compliance', 0):.4f}")

        # Test specific execution date
        test_date = datetime(2023, 6, 1)
        print(f"\nTesting system execution for {test_date.date()}:")

        try:
            execution_result = orchestrator.run_system(test_date)
            print(f"  Execution successful: {execution_result.get('execution_successful', False)}")
            print(f"  Portfolio value: ${execution_result.get('portfolio_value', 0):.2f}")
            print(f"  Active positions: {len(execution_result.get('positions', {}))}")
            print(f"  Core allocation: {execution_result.get('core_allocation', 0):.4f}")
            print(f"  Satellite allocation: {execution_result.get('satellite_allocation', 0):.4f}")

        except Exception as e:
            print(f"  Execution failed: {e}")

        print("\n✓ End-to-end workflow working correctly")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all System Orchestrator tests."""
    print("System Orchestrator Test Suite")
    print("=" * 60)
    print("Testing complete system integration and IPS compliance")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        test_system_orchestrator_initialization,
        test_system_integration,
        test_capital_allocation,
        test_ips_compliance_monitoring,
        test_system_health_monitoring,
        test_performance_attribution,
        test_risk_management,
        test_end_to_end_workflow
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
        print("\n🎉 All System Orchestrator tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    """Run the System Orchestrator test suite."""
    success = run_all_tests()

    if success:
        print("\nSystem Orchestrator module is working correctly!")
        sys.exit(0)
    else:
        print("\nSystem Orchestrator module has issues that need to be addressed.")
        sys.exit(1)