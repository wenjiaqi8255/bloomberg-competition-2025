"""
Monitoring Dashboard for Model Monitoring Visualization

This module provides dashboard generation capabilities for model monitoring,
including performance charts, drift visualization, and alert timelines.
"""

import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import numpy as np

from .monitor import ModelMonitor, ModelHealthStatus

logger = logging.getLogger(__name__)


@dataclass
class DashboardChart:
    """Single chart component for dashboard."""
    title: str
    figure: go.Figure
    chart_type: str  # 'performance', 'drift', 'timeline', 'metrics'
    description: str


@dataclass
class Dashboard:
    """Complete monitoring dashboard."""
    model_id: str
    title: str
    charts: List[DashboardChart]
    summary_metrics: Dict[str, Any]
    generated_at: datetime

    def to_html(self) -> str:
        """Convert dashboard to HTML format."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Dashboard - {self.model_id}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }}
                .chart {{ margin-bottom: 30px; }}
                .metrics {{ display: flex; justify-content: space-around; }}
                .metric {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Monitoring Dashboard</h1>
                <h2>{self.model_id}</h2>
                <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <h3>Summary Metrics</h3>
                <div class="metrics">
        """

        for key, value in self.summary_metrics.items():
            html_content += f'<div class="metric"><strong>{key}:</strong> {value}</div>'

        html_content += """
                </div>
            </div>
        """

        for i, chart in enumerate(self.charts):
            chart_html = chart.figure.to_html(include_plotlyjs=False, div_id=f"chart_{i}")
            html_content += f"""
            <div class="chart">
                <h3>{chart.title}</h3>
                <p>{chart.description}</p>
                {chart_html}
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        return html_content


class MonitoringDashboard:
    """
    Factory class for creating monitoring dashboards.

    This class generates various visualizations for model monitoring
    including performance trends, drift analysis, and alert timelines.
    """

    def __init__(self):
        """Initialize dashboard factory."""
        pass

    def create_dashboard(self, monitor: ModelMonitor) -> Dashboard:
        """
        Create a comprehensive monitoring dashboard.

        Args:
            monitor: ModelMonitor instance with monitoring data

        Returns:
            Dashboard object with all visualizations
        """
        try:
            # Get data from monitor
            metrics_history = monitor.get_metrics_history()
            current_metrics = monitor.get_current_metrics()

            charts = []

            # 1. Performance trend chart
            if metrics_history:
                performance_chart = self._create_performance_chart(metrics_history)
                charts.append(performance_chart)

            # 2. Health status timeline
            health_chart = self._create_health_timeline(monitor)
            charts.append(health_chart)

            # 3. Prediction volume chart
            volume_chart = self._create_prediction_volume_chart(monitor)
            charts.append(volume_chart)

            # 4. Metrics summary chart
            if metrics_history:
                metrics_chart = self._create_metrics_summary_chart(metrics_history)
                charts.append(metrics_chart)

            # Create dashboard
            dashboard = Dashboard(
                model_id=monitor.model_id,
                title=f"Model Monitoring Dashboard - {monitor.model_id}",
                charts=charts,
                summary_metrics=current_metrics,
                generated_at=datetime.now()
            )

            logger.info(f"Created dashboard for model {monitor.model_id}")
            return dashboard

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            # Return minimal dashboard with error info
            return Dashboard(
                model_id=monitor.model_id,
                title=f"Model Monitoring Dashboard - {monitor.model_id}",
                charts=[],
                summary_metrics={"error": str(e)},
                generated_at=datetime.now()
            )

    def _create_performance_chart(self, metrics_history: List[Dict[str, Any]]) -> DashboardChart:
        """Create performance trend chart."""
        try:
            # Extract data
            timestamps = [m['timestamp'] for m in metrics_history]

            # Create subplot for multiple metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score', 'RMSE', 'Correlation', 'Sample Count'),
                vertical_spacing=0.1
            )

            # R² Score
            r2_values = [m['metrics'].get('r2', 0) for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=r2_values, name='R²', line=dict(color='blue')),
                row=1, col=1
            )

            # RMSE
            rmse_values = [m['metrics'].get('rmse', 0) for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=rmse_values, name='RMSE', line=dict(color='red')),
                row=1, col=2
            )

            # Correlation
            corr_values = [m['metrics'].get('correlation', 0) for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=corr_values, name='Correlation', line=dict(color='green')),
                row=2, col=1
            )

            # Sample Count
            sample_counts = [m['sample_count'] for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=sample_counts, name='Samples', line=dict(color='purple')),
                row=2, col=2
            )

            fig.update_layout(
                title="Performance Metrics Over Time",
                showlegend=False,
                height=600
            )

            return DashboardChart(
                title="Performance Trends",
                figure=fig,
                chart_type="performance",
                description="Historical performance metrics showing model behavior over time"
            )

        except Exception as e:
            logger.error(f"Failed to create performance chart: {e}")
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", xref="paper", yref="paper")
            return DashboardChart(
                title="Performance Trends",
                figure=fig,
                chart_type="performance",
                description="Error occurred while creating this chart"
            )

    def _create_health_timeline(self, monitor: ModelMonitor) -> DashboardChart:
        """Create health status timeline chart."""
        try:
            # Get health status history (simplified - using current status)
            health_status = monitor.health_status

            # Create a simple status indicator
            fig = go.Figure()

            # Add current status as a gauge
            status_values = {'healthy': 0, 'warning': 1, 'critical': 2, 'error': 3, 'degraded': 4}
            current_value = status_values.get(health_status.status, 0)

            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = current_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Current Status: {health_status.status.upper()}"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, 4]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 4], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ))

            fig.update_layout(
                title="Model Health Status",
                height=400
            )

            return DashboardChart(
                title="Health Status",
                figure=fig,
                chart_type="timeline",
                description=f"Current model health: {health_status.status}. Issues: {len(health_status.issues)}"
            )

        except Exception as e:
            logger.error(f"Failed to create health timeline: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", xref="paper", yref="paper")
            return DashboardChart(
                title="Health Status",
                figure=fig,
                chart_type="timeline",
                description="Error occurred while creating this chart"
            )

    def _create_prediction_volume_chart(self, monitor: ModelMonitor) -> DashboardChart:
        """Create prediction volume chart."""
        try:
            # Get recent predictions
            recent_predictions = [r for r in monitor.prediction_log
                                if r.timestamp > datetime.now() - timedelta(days=7)]

            if not recent_predictions:
                fig = go.Figure()
                fig.add_annotation(text="No prediction data available", xref="paper", yref="paper")
                return DashboardChart(
                    title="Prediction Volume",
                    figure=fig,
                    chart_type="metrics",
                    description="No prediction data available for the last 7 days"
                )

            # Group by day
            daily_counts = {}
            for record in recent_predictions:
                date = record.timestamp.date()
                daily_counts[date] = daily_counts.get(date, 0) + 1

            # Create bar chart
            dates = sorted(daily_counts.keys())
            counts = [daily_counts[date] for date in dates]

            fig = go.Figure(data=[
                go.Bar(x=dates, y=counts, name='Daily Predictions')
            ])

            fig.update_layout(
                title="Prediction Volume (Last 7 Days)",
                xaxis_title="Date",
                yaxis_title="Number of Predictions",
                height=400
            )

            return DashboardChart(
                title="Prediction Volume",
                figure=fig,
                chart_type="metrics",
                description=f"Total predictions in last 7 days: {sum(counts)}"
            )

        except Exception as e:
            logger.error(f"Failed to create prediction volume chart: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", xref="paper", yref="paper")
            return DashboardChart(
                title="Prediction Volume",
                figure=fig,
                chart_type="metrics",
                description="Error occurred while creating this chart"
            )

    def _create_metrics_summary_chart(self, metrics_history: List[Dict[str, Any]]) -> DashboardChart:
        """Create metrics summary chart."""
        try:
            # Get latest metrics
            if not metrics_history:
                fig = go.Figure()
                fig.add_annotation(text="No metrics history available", xref="paper", yref="paper")
                return DashboardChart(
                    title="Metrics Summary",
                    figure=fig,
                    chart_type="metrics",
                    description="No metrics history available"
                )

            latest_metrics = metrics_history[-1]['metrics']

            # Create a summary table
            metric_names = []
            metric_values = []

            for key, value in latest_metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.append(key.upper())
                    metric_values.append(round(value, 4))

            fig = go.Figure(data=[go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='lightblue',
                           align='left'),
                cells=dict(values=[metric_names, metric_values],
                          fill_color='lightgray',
                          align='left'))
            ])

            fig.update_layout(
                title="Latest Performance Metrics",
                height=400
            )

            return DashboardChart(
                title="Metrics Summary",
                figure=fig,
                chart_type="metrics",
                description="Latest performance metrics from model evaluation"
            )

        except Exception as e:
            logger.error(f"Failed to create metrics summary: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", xref="paper", yref="paper")
            return DashboardChart(
                title="Metrics Summary",
                figure=fig,
                chart_type="metrics",
                description="Error occurred while creating this chart"
            )

    def save_dashboard(self, dashboard: Dashboard, filepath: str) -> str:
        """
        Save dashboard to HTML file.

        Args:
            dashboard: Dashboard object to save
            filepath: Path to save the HTML file

        Returns:
            Path to saved file
        """
        try:
            html_content = dashboard.to_html()

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Dashboard saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")
            raise