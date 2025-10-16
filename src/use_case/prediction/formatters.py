"""
Prediction Result Formatters
============================

Provides formatting utilities for prediction results, supporting multiple
output formats including console, JSON, and CSV with detailed box information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from .data_types import PredictionResult, StockRecommendation

logger = logging.getLogger(__name__)


class PredictionResultFormatter:
    """
    Formats prediction results for different output formats.
    
    Supports console display, JSON export, CSV export, and summary reports
    with detailed box information and multi-model support.
    """
    
    def format_console_report(self, result: PredictionResult) -> str:
        """
        Format prediction result for console display.
        
        Args:
            result: Prediction result to format
            
        Returns:
            Formatted string for console output
        """
        lines = []
        
        # Header
        lines.append(f"Strategy: {result.strategy_type.upper()}")
        lines.append(f"Model ID: {result.model_id}")
        lines.append(f"Portfolio Method: {result.portfolio_method}")
        lines.append(f"Prediction Date: {result.prediction_date.strftime('%Y-%m-%d')}")
        lines.append("")
        
        # Meta-model information
        if result.is_meta_model:
            lines.append("META-MODEL INFORMATION:")
            lines.append("-" * 30)
            lines.append(f"Base Models: {len(result.base_model_ids)}")
            for model_id, weight in result.model_weights.items():
                lines.append(f"  - {model_id}: {weight:.1%}")
            lines.append("")
        
        # Top recommendations
        lines.append("TOP STOCK RECOMMENDATIONS:")
        lines.append("-" * 50)
        lines.append(f"{'Rank':<4} {'Symbol':<8} {'Weight':<8} {'Signal':<8} {'Box Classification':<35} {'Risk':<6}")
        lines.append("-" * 50)
        
        for i, rec in enumerate(result.top_recommendations(10), 1):
            box_str = str(rec.box_classification) if rec.box_classification else "N/A"
            lines.append(
                f"{i:<4} {rec.symbol:<8} {rec.weight:<8.1%} {rec.signal_strength:<8.3f} "
                f"{box_str:<35} {rec.risk_score:<6.2f}"
            )
        lines.append("")
        
        # Box allocations (if available)
        if result.box_allocations:
            lines.append("BOX ALLOCATIONS:")
            lines.append("-" * 60)
            lines.append(f"{'Box Classification':<35} {'Target':<8} {'Actual':<8} {'Stocks':<15}")
            lines.append("-" * 60)
            
            for box_key, actual_weight in result.box_allocations.items():
                target_weight = result.box_allocations.get(box_key, 0.0)  # Placeholder
                stocks = result.stocks_by_box.get(box_key, []) if result.stocks_by_box else []
                stocks_str = ", ".join(stocks[:3]) + ("..." if len(stocks) > 3 else "")
                
                lines.append(
                    f"{box_key:<35} {target_weight:<8.1%} {actual_weight:<8.1%} {stocks_str:<15}"
                )
            lines.append("")
        
        # Portfolio summary
        lines.append("PORTFOLIO SUMMARY:")
        lines.append("-" * 25)
        lines.append(f"Total Positions: {result.total_positions}")
        lines.append(f"Expected Return: {result.expected_return:.2%} (annualized)")
        lines.append(f"Expected Risk: {result.expected_risk:.2%} (volatility)")
        lines.append(f"Diversification Score: {result.diversification_score:.2f}")
        
        # Construction log (if available)
        if result.box_construction_log:
            lines.append("")
            lines.append("CONSTRUCTION LOG:")
            lines.append("-" * 20)
            for log_entry in result.box_construction_log[-5:]:  # Show last 5 entries
                lines.append(f"  {log_entry}")
        
        return "\n".join(lines)
    
    def format_json_report(self, result: PredictionResult) -> Dict[str, Any]:
        """
        Format prediction result as JSON-serializable dictionary.
        
        Args:
            result: Prediction result to format
            
        Returns:
            Dictionary ready for JSON serialization
        """
        # Convert recommendations to dictionaries
        recommendations = []
        for rec in result.recommendations:
            rec_dict = {
                'symbol': rec.symbol,
                'weight': rec.weight,
                'signal_strength': rec.signal_strength,
                'risk_score': rec.risk_score,
                'box_classification': str(rec.box_classification) if rec.box_classification else None
            }
            recommendations.append(rec_dict)
        
        # Convert portfolio weights to dictionary
        portfolio_weights = result.portfolio_weights.to_dict() if hasattr(result.portfolio_weights, 'to_dict') else {}
        
        # Build result dictionary
        json_result = {
            'metadata': {
                'strategy_type': result.strategy_type,
                'model_id': result.model_id,
                'prediction_date': result.prediction_date.isoformat(),
                'total_positions': result.total_positions,
                'portfolio_method': result.portfolio_method,
                'is_meta_model': result.is_meta_model
            },
            'model_info': {
                'base_model_ids': result.base_model_ids,
                'model_weights': result.model_weights
            },
            'recommendations': recommendations,
            'portfolio_weights': portfolio_weights,
            'box_details': {
                'box_allocations': result.box_allocations,
                'stocks_by_box': result.stocks_by_box,
                'construction_log': result.box_construction_log
            },
            'risk_metrics': {
                'expected_return': result.expected_return,
                'expected_risk': result.expected_risk,
                'diversification_score': result.diversification_score
            },
            'summary': result.to_summary_dict()
        }
        
        return json_result
    
    def save_to_json(self, result: PredictionResult, path: Path) -> None:
        """
        Save prediction result to JSON file.
        
        Args:
            result: Prediction result to save
            path: Path to save JSON file
        """
        json_data = self.format_json_report(result)
        
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to {path}")
    
    def save_to_csv(self, result: PredictionResult, path: Path) -> None:
        """
        Save recommendations to CSV file.
        
        Args:
            result: Prediction result to save
            path: Path to save CSV file
        """
        # Prepare data for CSV
        csv_data = []
        for rec in result.recommendations:
            row = {
                'symbol': rec.symbol,
                'weight': rec.weight,
                'signal_strength': rec.signal_strength,
                'risk_score': rec.risk_score,
                'box_classification': str(rec.box_classification) if rec.box_classification else None,
                'box_size': rec.box_classification.size if rec.box_classification else None,
                'box_style': rec.box_classification.style if rec.box_classification else None,
                'box_region': rec.box_classification.region if rec.box_classification else None,
                'box_sector': rec.box_classification.sector if rec.box_classification else None
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(path, index=False)
        
        logger.info(f"CSV results saved to {path}")
    
    def save_summary_report(self, result: PredictionResult, path: Path) -> None:
        """
        Save human-readable summary report to text file.
        
        Args:
            result: Prediction result to save
            path: Path to save summary file
        """
        with open(path, 'w') as f:
            f.write("INVESTMENT PREDICTION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Strategy: {result.strategy_type}\n")
            f.write(f"Model ID: {result.model_id}\n")
            f.write(f"Prediction Date: {result.prediction_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Portfolio Method: {result.portfolio_method}\n\n")
            
            if result.is_meta_model:
                f.write("META-MODEL DETAILS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Base Models: {len(result.base_model_ids)}\n")
                for model_id, weight in result.model_weights.items():
                    f.write(f"  - {model_id}: {weight:.1%}\n")
                f.write("\n")
            
            f.write("TOP 10 RECOMMENDATIONS:\n")
            f.write("-" * 25 + "\n")
            for i, rec in enumerate(result.top_recommendations(10), 1):
                f.write(f"{i:2d}. {rec.symbol:<8} {rec.weight:6.1%} "
                       f"(signal: {rec.signal_strength:6.3f}, risk: {rec.risk_score:.2f})\n")
            
            f.write(f"\nPORTFOLIO METRICS:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total Positions: {result.total_positions}\n")
            f.write(f"Expected Return: {result.expected_return:.2%}\n")
            f.write(f"Expected Risk: {result.expected_risk:.2%}\n")
            f.write(f"Diversification Score: {result.diversification_score:.2f}\n")
            
            if result.box_allocations:
                f.write(f"\nBOX ALLOCATIONS:\n")
                f.write("-" * 16 + "\n")
                for box_key, weight in result.box_allocations.items():
                    f.write(f"{box_key}: {weight:.1%}\n")
        
        logger.info(f"Summary report saved to {path}")
    
    def _format_box_allocations(self, result: PredictionResult) -> str:
        """
        Format box allocations section for console output.
        
        Args:
            result: Prediction result
            
        Returns:
            Formatted box allocations string
        """
        if not result.box_allocations:
            return "No box allocation information available."
        
        lines = []
        lines.append("Box Allocations:")
        lines.append("-" * 20)
        
        for box_key, weight in result.box_allocations.items():
            stocks = result.stocks_by_box.get(box_key, []) if result.stocks_by_box else []
            stocks_str = ", ".join(stocks[:3]) + ("..." if len(stocks) > 3 else "")
            lines.append(f"{box_key}: {weight:.1%} ({stocks_str})")
        
        return "\n".join(lines)
    
    def _format_meta_model_info(self, result: PredictionResult) -> str:
        """
        Format meta-model information for console output.
        
        Args:
            result: Prediction result
            
        Returns:
            Formatted meta-model info string
        """
        if not result.is_meta_model:
            return ""
        
        lines = []
        lines.append("Meta-Model Information:")
        lines.append("-" * 25)
        lines.append(f"Base Models: {len(result.base_model_ids)}")
        
        for model_id, weight in result.model_weights.items():
            lines.append(f"  - {model_id}: {weight:.1%}")
        
        return "\n".join(lines)
