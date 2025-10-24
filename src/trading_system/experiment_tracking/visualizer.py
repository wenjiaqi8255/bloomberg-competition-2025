"""
Experiment visualization components.

This module provides visualization utilities that are independent of
any specific tracking backend. Visualizations can be used with
WandB, MLflow, or saved as files.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import visualization libraries, make them optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, some visualizations will be disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure as mpl_figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    mpl_figure = None
    logger.warning("Matplotlib not available, some visualizations will be disabled")


class ExperimentVisualizer:
    """
    Creates visualizations for experiment tracking.

    This class is independent of any tracking backend and can
    generate visualizations that can be:
    - Logged to WandB
    - Saved as files
    - Returned as figure objects
    - Converted to different formats
    """

    def __init__(self, backend: str = "plotly"):
        """
        Initialize visualizer.

        Args:
            backend: Visualization backend ("plotly", "matplotlib", "auto")
        """
        self.backend = backend
        self._validate_backend()

    def _validate_backend(self) -> None:
        """Validate that the requested backend is available."""
        if self.backend == "auto":
            if PLOTLY_AVAILABLE:
                self.backend = "plotly"
            elif MATPLOTLIB_AVAILABLE:
                self.backend = "matplotlib"
            else:
                logger.warning("No visualization libraries available, visualizations will be disabled")
                self.backend = None
        elif self.backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            self.backend = "matplotlib" if MATPLOTLIB_AVAILABLE else None
        elif self.backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, falling back to plotly")
            self.backend = "plotly" if PLOTLY_AVAILABLE else None

        if self.backend is None:
            logger.warning("No visualization backend available")

    def create_training_curve(self, metrics_history: Dict[str, List[float]],
                           title: str = "Training Progress") -> Optional[Any]:
        """
        Create training curve visualization.

        Args:
            metrics_history: Dictionary of metric names to lists of values
            title: Chart title

        Returns:
            Figure object or None if visualization fails
        """
        if self.backend is None:
            return None

        try:
            if self.backend == "plotly":
                return self._create_plotly_training_curve(metrics_history, title)
            elif self.backend == "matplotlib":
                return self._create_matplotlib_training_curve(metrics_history, title)
        except Exception as e:
            logger.error(f"Failed to create training curve: {e}")
            return None

    def _create_plotly_training_curve(self, metrics_history: Dict[str, List[float]],
                                    title: str) -> go.Figure:
        """Create training curve using Plotly."""
        fig = go.Figure()

        for metric_name, values in metrics_history.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Step/Epoch",
            yaxis_title="Metric Value",
            template="plotly_white",
            hovermode='x unified'
        )

        return fig

    def _create_matplotlib_training_curve(self, metrics_history: Dict[str, List[float]],
                                        title: str) -> Any:
        """Create training curve using Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_name, values in metrics_history.items():
            ax.plot(range(len(values)), values, marker='o', label=metric_name, linewidth=2)

        ax.set_xlabel("Step/Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_feature_importance(self, importance_data: Union[Dict[str, float], pd.Series],
                               top_n: int = 20, title: str = "Feature Importance") -> Optional[Any]:
        """
        Create feature importance visualization.

        Args:
            importance_data: Feature names and importance scores
            top_n: Number of top features to show
            title: Chart title

        Returns:
            Figure object or None if visualization fails
        """
        if self.backend is None:
            return None

        try:
            # Convert to pandas Series if needed
            if isinstance(importance_data, dict):
                importance_series = pd.Series(importance_data)
            else:
                importance_series = importance_data

            # Sort and get top features
            top_features = importance_series.sort_values(ascending=False).head(top_n)

            if self.backend == "plotly":
                return self._create_plotly_feature_importance(top_features, title)
            elif self.backend == "matplotlib":
                return self._create_matplotlib_feature_importance(top_features, title)

        except Exception as e:
            logger.error(f"Failed to create feature importance chart: {e}")
            return None

    def _create_plotly_feature_importance(self, importance_series: pd.Series,
                                        title: str) -> go.Figure:
        """Create feature importance chart using Plotly."""
        fig = go.Figure(data=[
            go.Bar(
                x=importance_series.values,
                y=importance_series.index,
                orientation='h',
                marker_color='skyblue'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=max(400, len(importance_series) * 25)
        )

        return fig

    def _create_matplotlib_feature_importance(self, importance_series: pd.Series,
                                           title: str) -> Any:
        """Create feature importance chart using Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_series) * 0.3)))

        # Create horizontal bar plot
        ax.barh(range(len(importance_series)), importance_series.values, color='skyblue')
        ax.set_yticks(range(len(importance_series)))
        ax.set_yticklabels(importance_series.index)
        ax.set_xlabel("Importance")
        ax.set_title(title)

        # Invert y-axis to show most important at top
        ax.invert_yaxis()
        plt.tight_layout()

        return fig

    def create_portfolio_performance(self, portfolio_data: pd.DataFrame,
                                  benchmark_data: Optional[pd.DataFrame] = None,
                                  title: str = "Portfolio Performance") -> Optional[Any]:
        """
        Create portfolio performance visualization.

        Args:
            portfolio_data: Portfolio value data (index: dates, column: portfolio_value)
            benchmark_data: Optional benchmark data
            title: Chart title

        Returns:
            Figure object or None if visualization fails
        """
        if self.backend is None:
            return None

        try:
            if self.backend == "plotly":
                return self._create_plotly_portfolio_performance(portfolio_data, benchmark_data, title)
            elif self.backend == "matplotlib":
                return self._create_matplotlib_portfolio_performance(portfolio_data, benchmark_data, title)

        except Exception as e:
            logger.error(f"Failed to create portfolio performance chart: {e}")
            return None

    def _create_plotly_portfolio_performance(self, portfolio_data: pd.DataFrame,
                                           benchmark_data: Optional[pd.DataFrame],
                                           title: str) -> go.Figure:
        """Create portfolio performance chart using Plotly."""
        fig = go.Figure()

        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data.iloc[:, 0] if len(portfolio_data.columns) == 1 else portfolio_data['portfolio_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))

        # Benchmark comparison
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to portfolio starting value
            portfolio_start = portfolio_data.iloc[0, 0] if len(portfolio_data.columns) == 1 else portfolio_data['portfolio_value'].iloc[0]
            benchmark_start = benchmark_data.iloc[0, 0]

            normalized_benchmark = benchmark_data.iloc[:, 0] * (portfolio_start / benchmark_start)

            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=normalized_benchmark,
                mode='lines',
                name='Benchmark (Normalized)',
                line=dict(color='gray', width=1, dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            template="plotly_white",
            hovermode='x unified'
        )

        return fig

    def _create_matplotlib_portfolio_performance(self, portfolio_data: pd.DataFrame,
                                              benchmark_data: Optional[pd.DataFrame],
                                              title: str) -> Any:
        """Create portfolio performance chart using Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Portfolio performance
        portfolio_values = portfolio_data.iloc[:, 0] if len(portfolio_data.columns) == 1 else portfolio_data['portfolio_value']
        ax.plot(portfolio_data.index, portfolio_values, label='Portfolio', color='blue', linewidth=2)

        # Benchmark comparison
        if benchmark_data is not None and not benchmark_data.empty:
            portfolio_start = portfolio_values.iloc[0]
            benchmark_start = benchmark_data.iloc[0, 0]
            normalized_benchmark = benchmark_data.iloc[:, 0] * (portfolio_start / benchmark_start)

            ax.plot(benchmark_data.index, normalized_benchmark,
                   label='Benchmark (Normalized)', color='gray', linewidth=1, linestyle='--')

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def create_drawdown_chart(self, portfolio_data: pd.DataFrame,
                            title: str = "Portfolio Drawdown") -> Optional[Any]:
        """
        Create drawdown visualization.

        Args:
            portfolio_data: Portfolio value data
            title: Chart title

        Returns:
            Figure object or None if visualization fails
        """
        if self.backend is None:
            return None

        try:
            # Calculate drawdown
            portfolio_values = portfolio_data.iloc[:, 0] if len(portfolio_data.columns) == 1 else portfolio_data['portfolio_value']
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max

            if self.backend == "plotly":
                return self._create_plotly_drawdown_chart(drawdown, title)
            elif self.backend == "matplotlib":
                return self._create_matplotlib_drawdown_chart(drawdown, title)

        except Exception as e:
            logger.error(f"Failed to create drawdown chart: {e}")
            return None

    def _create_plotly_drawdown_chart(self, drawdown: pd.Series, title: str) -> go.Figure:
        """Create drawdown chart using Plotly."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red', width=2),
            fillcolor='rgba(255,0,0,0.3)'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            yaxis_tickformat='.1f%%'
        )

        return fig

    def _create_matplotlib_drawdown_chart(self, drawdown: pd.Series, title: str) -> Any:
        """Create drawdown chart using Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(drawdown.index, drawdown * 100, 0,
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown * 100, color='red', linewidth=2)

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                 title: str = "Correlation Matrix") -> Optional[Any]:
        """
        Create correlation heatmap visualization.

        Args:
            correlation_matrix: Correlation matrix
            title: Chart title

        Returns:
            Figure object or None if visualization fails
        """
        if self.backend is None:
            return None

        try:
            if self.backend == "plotly":
                return self._create_plotly_correlation_heatmap(correlation_matrix, title)
            elif self.backend == "matplotlib":
                return self._create_matplotlib_correlation_heatmap(correlation_matrix, title)

        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {e}")
            return None

    def _create_plotly_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                         title: str) -> go.Figure:
        """Create correlation heatmap using Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            template="plotly_white",
            width=max(600, len(correlation_matrix.columns) * 50),
            height=max(400, len(correlation_matrix.index) * 50)
        )

        return fig

    def _create_matplotlib_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                            title: str) -> Any:
        """Create correlation heatmap using Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(correlation_matrix.index)

        # Add text annotations
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        plt.tight_layout()
        return fig

    def save_figure(self, figure: Any, filepath: str, format: str = "png", dpi: int = 300) -> bool:
        """
        Save figure to file.

        Args:
            figure: Figure object to save
            filepath: Output file path
            format: Output format ("png", "pdf", "svg", "html")
            dpi: Resolution for raster formats

        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(figure, 'write_image'):  # Plotly figure
                if format.lower() == "html":
                    figure.write_html(filepath)
                else:
                    figure.write_image(filepath, format=format, width=1200, height=800, scale=dpi/100)
            elif hasattr(figure, 'savefig'):  # Matplotlib figure
                figure.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
            else:
                logger.error(f"Unsupported figure type: {type(figure)}")
                return False

            logger.info(f"Figure saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
            return False

    def figure_to_image_bytes(self, figure: Any, format: str = "png", dpi: int = 300) -> Optional[bytes]:
        """
        Convert figure to image bytes.

        Args:
            figure: Figure object to convert
            format: Output format ("png", "pdf", "svg")
            dpi: Resolution for raster formats

        Returns:
            Image bytes or None if conversion fails
        """
        try:
            if hasattr(figure, 'to_image'):  # Plotly figure
                return figure.to_image(format=format, width=1200, height=800, scale=dpi/100)
            elif hasattr(figure, 'savefig'):  # Matplotlib figure
                from io import BytesIO
                buffer = BytesIO()
                figure.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
                buffer.seek(0)
                return buffer.read()
            else:
                logger.error(f"Unsupported figure type: {type(figure)}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert figure to bytes: {e}")
            return None