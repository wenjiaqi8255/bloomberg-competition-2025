"""
Tests for experiment configuration classes.

This module tests the ExperimentConfig and related configuration classes,
ensuring they properly validate and handle experiment configuration.
"""

import json
import pytest
from dataclasses import asdict

from src.trading_system.utils.experiment_tracking.config import (
    ExperimentConfig,
    OptimizationConfig,
    MonitoringConfig,
    create_training_config,
    create_optimization_config,
    create_backtest_config,
    create_monitoring_config,
    RUN_TYPES,
    ALERT_LEVELS
)


class TestExperimentConfig:
    """Test cases for ExperimentConfig class."""

    def test_minimal_config(self):
        """Test creating a minimal valid configuration."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        assert config.project_name == "test_project"
        assert config.experiment_name == "test_experiment"
        assert config.run_type == "training"
        assert config.tags == []
        assert config.hyperparameters == {}
        assert config.metadata == {}
        assert config.group is None

    def test_full_config(self):
        """Test creating a full configuration with all fields."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            group="test_group",
            tags=["tag1", "tag2"],
            entity="test_entity",
            hyperparameters={"param1": "value1"},
            metadata={"meta1": "data1"},
            notes="Test notes",
            run_id="test_run_123",
            resume="must",
            data_info={"source": "test"},
            model_info={"type": "xgboost"}
        )

        assert config.entity == "test_entity"
        assert config.tags == ["tag1", "tag2"]
        assert config.hyperparameters == {"param1": "value1"}
        assert config.metadata == {"meta1": "data1"}
        assert config.notes == "Test notes"
        assert config.run_id == "test_run_123"
        assert config.resume == "must"

    def test_invalid_run_type(self):
        """Test that invalid run types raise ValueError."""
        with pytest.raises(ValueError, match="run_type must be one of"):
            ExperimentConfig(
                project_name="test_project",
                experiment_name="test_experiment",
                run_type="invalid_type"
            )

    def test_invalid_resume(self):
        """Test that invalid resume values raise ValueError."""
        with pytest.raises(ValueError, match="resume must be one of"):
            ExperimentConfig(
                project_name="test_project",
                experiment_name="test_experiment",
                run_type="training",
                resume="invalid"
            )

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValueError."""
        # Missing project_name
        with pytest.raises(ValueError, match="project_name is required"):
            ExperimentConfig(
                project_name="",
                experiment_name="test_experiment",
                run_type="training"
            )

        # Missing experiment_name
        with pytest.raises(ValueError, match="experiment_name is required"):
            ExperimentConfig(
                project_name="test_project",
                experiment_name="",
                run_type="training"
            )

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            tags=["tag1"],
            hyperparameters={"param1": "value1"}
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["project_name"] == "test_project"
        assert config_dict["tags"] == ["tag1"]
        assert config_dict["hyperparameters"] == {"param1": "value1"}

    def test_to_json(self):
        """Test converting configuration to JSON."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        json_str = config.to_json()
        parsed = json.loads(json_str)

        assert parsed["project_name"] == "test_project"
        assert parsed["experiment_name"] == "test_experiment"
        assert parsed["run_type"] == "training"

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "project_name": "test_project",
            "experiment_name": "test_experiment",
            "run_type": "training",
            "tags": ["tag1", "tag2"],
            "hyperparameters": {"param1": "value1"}
        }

        config = ExperimentConfig.from_dict(config_dict)
        assert config.project_name == "test_project"
        assert config.experiment_name == "test_experiment"
        assert config.tags == ["tag1", "tag2"]
        assert config.hyperparameters == {"param1": "value1"}

    def test_from_json(self):
        """Test creating configuration from JSON."""
        json_str = json.dumps({
            "project_name": "test_project",
            "experiment_name": "test_experiment",
            "run_type": "training"
        })

        config = ExperimentConfig.from_json(json_str)
        assert config.project_name == "test_project"
        assert config.experiment_name == "test_experiment"
        assert config.run_type == "training"

    def test_copy(self):
        """Test copying configuration with updates."""
        original = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        copied = original.copy(experiment_name="new_experiment")
        assert copied.project_name == "test_project"
        assert copied.experiment_name == "new_experiment"
        assert copied.run_type == "training"

        # Original should be unchanged
        assert original.experiment_name == "test_experiment"

    def test_tag_operations(self):
        """Test adding and removing tags."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            tags=["tag1"]
        )

        # Add new tag
        config.add_tag("tag2")
        assert "tag2" in config.tags
        assert len(config.tags) == 2

        # Add existing tag (should not duplicate)
        config.add_tag("tag1")
        assert config.tags.count("tag1") == 1

        # Remove tag
        config.remove_tag("tag1")
        assert "tag1" not in config.tags
        assert "tag2" in config.tags

    def test_add_hyperparameter(self):
        """Test adding hyperparameters."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        config.add_hyperparameter("learning_rate", 0.01)
        assert config.hyperparameters["learning_rate"] == 0.01

        # Overwrite existing parameter
        config.add_hyperparameter("learning_rate", 0.02)
        assert config.hyperparameters["learning_rate"] == 0.02

    def test_add_metadata(self):
        """Test adding metadata."""
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        config.add_metadata("author", "test_user")
        assert config.metadata["author"] == "test_user"

    def test_created_at_timestamp(self):
        """Test that created_at is automatically set."""
        import time
        from datetime import datetime

        before = datetime.now()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )
        after = datetime.now()

        # Should be between before and after
        created_at = datetime.fromisoformat(config.created_at)
        assert before <= created_at <= after


class TestOptimizationConfig:
    """Test cases for OptimizationConfig class."""

    def test_default_config(self):
        """Test creating default optimization configuration."""
        config = OptimizationConfig()

        assert config.n_trials == 100
        assert config.timeout is None
        assert config.direction == "maximize"
        assert config.enable_pruning is True
        assert config.pruning_warmup_steps == 10
        assert config.pruning_interval == 1
        assert config.search_space == {}
        assert config.n_jobs == 1

    def test_full_config(self):
        """Test creating full optimization configuration."""
        config = OptimizationConfig(
            n_trials=200,
            timeout=3600,
            direction="minimize",
            enable_pruning=False,
            pruning_warmup_steps=20,
            pruning_interval=2,
            search_space={"param1": [1, 2, 3]},
            n_jobs=4
        )

        assert config.n_trials == 200
        assert config.timeout == 3600
        assert config.direction == "minimize"
        assert config.enable_pruning is False
        assert config.pruning_warmup_steps == 20
        assert config.pruning_interval == 2
        assert config.search_space == {"param1": [1, 2, 3]}
        assert config.n_jobs == 4

    def test_invalid_direction(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            OptimizationConfig(direction="invalid")

    def test_negative_n_trials(self):
        """Test that negative n_trials raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            OptimizationConfig(n_trials=-1)

    def test_negative_n_jobs(self):
        """Test that negative n_jobs raises ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be positive"):
            OptimizationConfig(n_jobs=0)

    def test_negative_pruning_steps(self):
        """Test that negative pruning warmup steps raises ValueError."""
        with pytest.raises(ValueError, match="pruning_warmup_steps must be non-negative"):
            OptimizationConfig(pruning_warmup_steps=-1)


class TestMonitoringConfig:
    """Test cases for MonitoringConfig class."""

    def test_default_config(self):
        """Test creating default monitoring configuration."""
        config = MonitoringConfig()

        assert config.check_interval == 3600
        assert config.alert_threshold == 0.05
        assert config.baseline_window == 252
        assert config.evaluation_window == 21
        assert config.metrics_to_monitor == ["ic", "sharpe", "max_drawdown"]
        assert config.alert_cooldown == 86400
        assert config.alert_recipients == []

    def test_full_config(self):
        """Test creating full monitoring configuration."""
        config = MonitoringConfig(
            check_interval=1800,
            alert_threshold=0.1,
            baseline_window=500,
            evaluation_window=42,
            metrics_to_monitor=["ic", "rmse"],
            alert_cooldown=43200,
            alert_recipients=["user@example.com"]
        )

        assert config.check_interval == 1800
        assert config.alert_threshold == 0.1
        assert config.baseline_window == 500
        assert config.evaluation_window == 42
        assert config.metrics_to_monitor == ["ic", "rmse"]
        assert config.alert_cooldown == 43200
        assert config.alert_recipients == ["user@example.com"]

    def test_invalid_threshold(self):
        """Test that invalid alert threshold raises ValueError."""
        with pytest.raises(ValueError, match="alert_threshold must be between 0 and 1"):
            MonitoringConfig(alert_threshold=1.5)

    def test_negative_intervals(self):
        """Test that negative intervals raise ValueError."""
        with pytest.raises(ValueError, match="check_interval must be positive"):
            MonitoringConfig(check_interval=-1)

        with pytest.raises(ValueError, match="baseline_window must be positive"):
            MonitoringConfig(baseline_window=0)

        with pytest.raises(ValueError, match="evaluation_window must be positive"):
            MonitoringConfig(evaluation_window=0)


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_training_config(self):
        """Test creating training configuration."""
        config = create_training_config(
            project_name="test_project",
            model_type="xgboost",
            hyperparameters={"n_estimators": 100}
        )

        assert config.project_name == "test_project"
        assert config.run_type == "training"
        assert "xgboost_training_" in config.experiment_name
        assert config.hyperparameters == {"n_estimators": 100}
        assert config.model_info == {"model_type": "xgboost"}

    def test_create_optimization_config(self):
        """Test creating optimization configuration."""
        config = create_optimization_config(
            project_name="test_project",
            model_type="xgboost",
            search_space={"n_estimators": [50, 100, 200]},
            n_trials=50
        )

        assert config.project_name == "test_project"
        assert config.run_type == "optimization"
        assert "xgboost_optimization_" in config.experiment_name
        assert config.metadata["search_space"] == {"n_estimators": [50, 100, 200]}
        assert config.hyperparameters["n_trials"] == 50

    def test_create_backtest_config(self):
        """Test creating backtest configuration."""
        strategy_config = {"lookback": 20, "rebalance_freq": "monthly"}
        config = create_backtest_config(
            project_name="test_project",
            strategy_name="dual_momentum",
            strategy_config=strategy_config
        )

        assert config.project_name == "test_project"
        assert config.run_type == "backtest"
        assert "dual_momentum_backtest_" in config.experiment_name
        assert config.hyperparameters == strategy_config
        assert config.model_info == {"strategy_name": "dual_momentum"}

    def test_create_monitoring_config(self):
        """Test creating monitoring configuration."""
        monitoring_config = MonitoringConfig(check_interval=1800)
        config = create_monitoring_config(
            project_name="test_project",
            model_id="model_123",
            monitoring_config=monitoring_config
        )

        assert config.project_name == "test_project"
        assert config.run_type == "monitoring"
        assert "model_123_monitoring_" in config.experiment_name
        assert config.model_info == {"model_id": "model_123"}
        assert "check_interval" in config.hyperparameters


class TestConstants:
    """Test cases for module constants."""

    def test_run_types(self):
        """Test that RUN_TYPES contains expected values."""
        assert "training" in RUN_TYPES
        assert "evaluation" in RUN_TYPES
        assert "optimization" in RUN_TYPES
        assert "backtest" in RUN_TYPES
        assert "monitoring" in RUN_TYPES

    def test_alert_levels(self):
        """Test that ALERT_LEVELS contains expected values."""
        assert "info" in ALERT_LEVELS
        assert "warning" in ALERT_LEVELS
        assert "error" in ALERT_LEVELS