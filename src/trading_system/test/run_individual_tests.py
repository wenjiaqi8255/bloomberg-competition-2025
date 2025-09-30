"""
Individual Test Runner for Week 3 IPS Core System

This script runs individual test scripts for each module to make input/output
more predictable before running the complete system integration.

Usage:
    python run_individual_tests.py

Available test scripts:
    - test_ff5_provider.py: FF5 Data Provider module
    - test_stock_classifier.py: Stock Classifier module
    - test_ff5_regression.py: FF5 Regression Engine module
    - test_residual_predictor.py: ML Residual Predictor module
    - test_core_ffml_strategy.py: Core FFML Strategy module
    - test_satellite_strategy.py: Satellite Strategy module
    - test_system_orchestrator.py: System Orchestrator module
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_test_script(script_name: str, description: str) -> dict:
    """Run a single test script and return results."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("=" * 80)

    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        return {
            'script_name': script_name,
            'success': False,
            'error': f"Script not found: {script_path}",
            'execution_time': 0
        }

    try:
        start_time = time.time()

        # Run the test script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Display output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        # Determine success
        success = result.returncode == 0

        return {
            'script_name': script_name,
            'success': success,
            'return_code': result.returncode,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            'script_name': script_name,
            'success': False,
            'error': "Test timed out after 5 minutes",
            'execution_time': 300
        }
    except Exception as e:
        return {
            'script_name': script_name,
            'success': False,
            'error': str(e),
            'execution_time': 0
        }


def run_all_individual_tests():
    """Run all individual test scripts."""
    print("Week 3 IPS Core System - Individual Module Testing")
    print("=" * 80)
    print("Running individual test scripts for each module")
    print("This makes input/output more predictable before system integration")
    print("=" * 80)

    # Define test scripts in order of dependencies
    test_scripts = [
        {
            'script': 'test_ff5_provider.py',
            'description': 'FF5 Data Provider - Kenneth French library integration'
        },
        {
            'script': 'test_stock_classifier.py',
            'description': 'Stock Classifier - Size×Style×Region×Sector classification'
        },
        {
            'script': 'test_ff5_regression.py',
            'description': 'FF5 Regression Engine - Rolling window beta estimation'
        },
        {
            'script': 'test_residual_predictor.py',
            'description': 'ML Residual Predictor - Ensemble methods and model governance'
        },
        {
            'script': 'test_core_ffml_strategy.py',
            'description': 'Core FFML Strategy - FF5 + ML residual prediction (Method A)'
        },
        {
            'script': 'test_satellite_strategy.py',
            'description': 'Satellite Strategy - Technical indicators and risk management'
        },
        {
            'script': 'test_system_orchestrator.py',
            'description': 'System Orchestrator - Complete system integration and IPS compliance'
        }
    ]

    test_results = []
    total_start_time = time.time()

    # Run each test script
    for test_info in test_scripts:
        result = run_test_script(test_info['script'], test_info['description'])
        test_results.append(result)

        # Small delay between tests
        time.sleep(2)

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # Generate summary report
    print("\n" + "=" * 80)
    print("INDIVIDUAL TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in test_results if result['success'])
    total = len(test_results)

    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    print("\nDetailed Results:")
    print("-" * 60)

    for result in test_results:
        script_name = result['script_name']
        success = result['success']
        execution_time = result.get('execution_time', 0)

        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {script_name} ({execution_time:.2f}s)")

        if not success:
            error = result.get('error', result.get('stderr', 'Unknown error'))
            return_code = result.get('return_code', 'N/A')
            print(f"     Error: {error}")
            print(f"     Return code: {return_code}")

    # Provide recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if passed == total:
        print("🎉 All individual tests passed!")
        print("✓ Ready for complete system integration")
        print("✓ All modules are working correctly")
        print("✓ Input/output behavior is predictable")
        print("\nNext steps:")
        print("1. Run the complete system integration")
        print("2. Test with real market data")
        print("3. Validate IPS compliance")
        print("4. Monitor model performance")

        return True

    else:
        failed_tests = [result['script_name'] for result in test_results if not result['success']]

        print(f"⚠ {total - passed} tests failed")
        print("⚠ Fix failed modules before proceeding to system integration")
        print("\nFailed modules:")
        for i, test_name in enumerate(failed_tests, 1):
            print(f"  {i}. {test_name}")

        print("\nTroubleshooting steps:")
        print("1. Check error messages above for specific issues")
        print("2. Verify dependencies are installed correctly")
        print("3. Check data availability and connectivity")
        print("4. Review module configurations")
        print("5. Run individual tests again after fixes")

        return False


def main():
    """Main function to run individual tests."""
    try:
        success = run_all_individual_tests()

        if success:
            print("\n🎉 Individual module testing completed successfully!")
            print("Ready for complete system integration.")
            sys.exit(0)
        else:
            print("\n⚠ Some modules failed testing. Please fix issues before proceeding.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """Run the individual test suite."""
    main()