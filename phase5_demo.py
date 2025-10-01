#!/usr/bin/env python3
"""
Phase 5 Monitoring Enhancements Demo

This script demonstrates the monitoring enhancements implemented in Phase 5,
including experiment tracking integration, performance degradation alerts,
and dashboard generation.
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from src.trading_system.models.serving.monitor import ModelMonitor
    from src.trading_system.models.serving.dashboard import MonitoringDashboard
    from src.trading_system.utils.experiment_tracking.interface import NullExperimentTracker
    from src.trading_system.models.base.base_model import BaseModel

    class DemoModel(BaseModel):
        """Demo model for monitoring."""

        def __init__(self):
            self.model_id = "demo_model"
            self.model_type = "demo"
            self.config = {"demo": True}
            self.is_trained = True
            self.status = "trained"
            self.metadata = {"demo": True}

        def predict(self, X):
            # Simple linear prediction with some noise
            if hasattr(X, 'values'):
                return X.values[:, 0] * 0.1 + np.random.normal(0, 0.1, len(X))
            return np.array(X) * 0.1 + np.random.normal(0, 0.1, len(X))

        def fit(self, X, y):
            return self

        def get_feature_importance(self):
            return {"feature_1": 0.6, "feature_2": 0.4}

    def demo_monitoring():
        """Demonstrate Phase 5 monitoring enhancements."""
        print("=== Phase 5 Monitoring Enhancements Demo ===\n")

        # 1. Initialize model monitor with tracking (using null tracker for demo)
        print("1. Initializing ModelMonitor with experiment tracking...")
        tracker = NullExperimentTracker()
        monitor = ModelMonitor("demo_model", config={
            'performance_window': 7,
            'degradation_threshold': 0.2,
            'min_samples': 5
        }, tracker=tracker)
        print(f"   Monitor initialized for model: {monitor.model_id}")
        print(f"   Tracking run ID: {tracker._run_count - 1}")
        print()

        # 2. Log some predictions
        print("2. Logging predictions...")
        model = DemoModel()

        # Log initial good predictions
        for i in range(15):
            features = {"feature_1": np.random.normal(0, 1), "feature_2": np.random.normal(0, 1)}
            prediction = model.predict([list(features.values())])[0]
            actual = prediction + np.random.normal(0, 0.05)  # Small error
            monitor.log_prediction(features, prediction, actual=actual)

        print(f"   Logged {len(monitor.prediction_log)} predictions")
        print()

        # 3. Check model health
        print("3. Checking model health...")
        health_status = monitor.get_health_status(model)
        print(f"   Health status: {health_status.status}")
        print(f"   Issues: {health_status.issues}")
        print(f"   Recommendations: {health_status.recommendations}")
        print()

        # 4. Simulate performance degradation
        print("4. Simulating performance degradation...")

        # Set baseline metrics
        monitor.baseline_metrics = {"r2": 0.8, "correlation": 0.9}

        # Log poor predictions
        for i in range(10):
            features = {"feature_1": np.random.normal(0, 1), "feature_2": np.random.normal(0, 1)}
            prediction = model.predict([list(features.values())])[0]
            actual = -prediction + np.random.normal(0, 0.5)  # Poor correlation
            monitor.log_prediction(features, prediction, actual=actual)

        print("   Logged 10 poor predictions to simulate degradation")
        print()

        # 5. Create dashboard
        print("5. Creating monitoring dashboard...")
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(monitor)

        print(f"   Dashboard created with {len(dashboard.charts)} charts:")
        for chart in dashboard.charts:
            print(f"     - {chart.title} ({chart.chart_type})")
        print()

        # 6. Save dashboard
        print("6. Saving dashboard to HTML...")
        dashboard_path = "monitoring_dashboard_demo.html"
        dashboard_factory.save_dashboard(dashboard, dashboard_path)
        print(f"   Dashboard saved to: {dashboard_path}")
        print()

        # 7. Stop monitoring
        print("7. Stopping monitoring and finalizing tracking...")
        monitor.stop_monitoring()
        print("   Monitoring stopped and run finalized")
        print()

        # 8. Summary
        print("=== Demo Summary ===")
        print(f"Total predictions logged: {len(monitor.prediction_log)}")
        print(f"Final health status: {health_status.status}")
        print(f"Charts generated: {len(dashboard.charts)}")
        print(f"Dashboard file: {dashboard_path}")
        print("\nPhase 5 monitoring enhancements demo completed successfully!")

    if __name__ == "__main__":
        demo_monitoring()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you have installed the required dependencies.")
    sys.exit(1)