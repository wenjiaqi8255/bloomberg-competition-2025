"""
Single Model Prediction Demo
============================

Demonstrates how to use the prediction service with a single trained model.
Shows basic usage patterns and result interpretation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from use_case.prediction.prediction_orchestrator import PredictionOrchestrator
from use_case.prediction.formatters import PredictionResultFormatter


def main():
    """Run single model prediction demo."""
    print("="*60)
    print("SINGLE MODEL PREDICTION DEMO")
    print("="*60)
    
    # Configuration file path
    config_path = "configs/prediction_config.yaml"
    
    try:
        # Initialize orchestrator
        print(f"Loading configuration from: {config_path}")
        orchestrator = PredictionOrchestrator(config_path)
        
        # Run prediction
        print("Running prediction workflow...")
        result = orchestrator.run_prediction()
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # Basic information
        print(f"Strategy Type: {result.strategy_type}")
        print(f"Model ID: {result.model_id}")
        print(f"Portfolio Method: {result.portfolio_method}")
        print(f"Total Positions: {result.total_positions}")
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Expected Risk: {result.expected_risk:.2%}")
        print(f"Diversification Score: {result.diversification_score:.2f}")
        
        # Top recommendations
        print(f"\nTop 5 Stock Recommendations:")
        print("-" * 40)
        for i, rec in enumerate(result.top_recommendations(5), 1):
            box_info = str(rec.box_classification) if rec.box_classification else "N/A"
            print(f"{i}. {rec.symbol:<8} {rec.weight:6.1%} "
                  f"(signal: {rec.signal_strength:6.3f}, box: {box_info})")
        
        # Box allocations (if available)
        if result.box_allocations:
            print(f"\nBox Allocations:")
            print("-" * 20)
            for box_key, weight in result.box_allocations.items():
                stocks = result.stocks_by_box.get(box_key, []) if result.stocks_by_box else []
                stocks_str = ", ".join(stocks[:3]) + ("..." if len(stocks) > 3 else "")
                print(f"{box_key}: {weight:.1%} ({stocks_str})")
        
        # Save results
        output_dir = Path("prediction_demo_results")
        output_dir.mkdir(exist_ok=True)
        
        formatter = PredictionResultFormatter()
        
        # Save JSON
        json_path = output_dir / "single_model_results.json"
        formatter.save_to_json(result, json_path)
        print(f"\nResults saved to: {json_path}")
        
        # Save CSV
        csv_path = output_dir / "single_model_recommendations.csv"
        formatter.save_to_csv(result, csv_path)
        print(f"CSV saved to: {csv_path}")
        
        # Save summary
        summary_path = output_dir / "single_model_summary.txt"
        formatter.save_summary_report(result, summary_path)
        print(f"Summary saved to: {summary_path}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

