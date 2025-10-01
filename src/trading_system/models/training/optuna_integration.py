"""
Optuna Integration Module

This module provides comprehensive integration with Optuna for hyperparameter
optimization, including advanced samplers, pruners, and optimization strategies.

Key Features:
- Advanced Optuna sampler configurations
- Custom pruner implementations
- Study management and persistence
- Multi-objective optimization
- Parallel optimization support
- Study analysis and visualization
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import logging
import time
import pickle
from pathlib import Path
import json

import pandas as pd
import numpy as np

# Optuna imports (optional)
try:
    import optuna
    from optuna.trial import Trial, TrialState
    from optuna.study import Study, StudyDirection
    from optuna.pruners import BasePruner, MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    from optuna.samplers import BaseSampler, TPESampler, RandomSampler, CmaEsSampler, GridSampler
    from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
    from optuna.storages import RDBStorage, JournalStorage, JournalFileStorage
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    Trial = None
    Study = None
    logger = logging.getLogger(__name__)
    logger.warning("Optuna not available. Install with: pip install optuna")
else:
    logger = logging.getLogger(__name__)

from .hyperparameter_optimizer import HyperparameterConfig, SearchSpace, OptimizationResult
from ..base.base_model import BaseModel


@dataclass
class OptunaConfig:
    """Optuna-specific configuration."""
    # Study storage
    storage_url: Optional[str] = None  # Database URL for persistence
    study_name: str = "hyperparameter_optimization"
    load_if_exists: bool = True

    # Parallel optimization
    n_jobs: int = 1  # Number of parallel jobs
    direction: str = "maximize"  # "maximize", "minimize", or list for multi-objective

    # Advanced sampler settings
    sampler_type: str = "tpe"  # "tpe", "random", "cmaes", "grid", "nsgaii", "motpe"
    sampler_config: Dict[str, Any] = field(default_factory=dict)

    # Advanced pruner settings
    pruner_type: str = "median"  # "median", "hyperband", "successive_halving", "custom"
    pruner_config: Dict[str, Any] = field(default_factory=dict)

    # Study management
    save_study: bool = True
    study_dir: str = "./optuna_studies"

    # Visualization and analysis
    generate_plots: bool = True
    analysis_dir: str = "./optuna_analysis"


class CustomPruner(BasePruner):
    """Custom pruner implementation for trading models."""

    def __init__(self,
                 patience: int = 10,
                 min_improvement: float = 0.01,
                 min_trials: int = 5,
                 warmup_steps: int = 3):
        """
        Initialize custom pruner.

        Args:
            patience: Number of trials to wait for improvement
            min_improvement: Minimum relative improvement required
            min_trials: Minimum number of trials before pruning starts
            warmup_steps: Number of steps before pruning begins
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_trials = min_trials
        self.warmup_steps = warmup_steps
        self._best_scores: List[float] = []
        self._trial_count = 0

    def prune(self, study: Study, trial: Trial) -> bool:
        """
        Determine if trial should be pruned.

        Args:
            study: Optuna study
            trial: Current trial

        Returns:
            True if trial should be pruned
        """
        self._trial_count += 1

        # Don't prune in early trials
        if self._trial_count < self.min_trials:
            return False

        # Get intermediate values
        intermediate_values = trial.intermediate_values
        if not intermediate_values:
            return False

        # Get latest intermediate value
        latest_step = max(intermediate_values.keys())
        latest_value = intermediate_values[latest_step]

        # Don't prune during warmup
        if latest_step < self.warmup_steps:
            return False

        # Track best scores
        self._best_scores.append(latest_value)
        if len(self._best_scores) > self.patience:
            self._best_scores.pop(0)

        # Check for improvement
        if len(self._best_scores) >= self.patience:
            best_so_far = max(self._best_scores)
            current_best = self._best_scores[-1]

            relative_improvement = (current_best - best_so_far) / max(abs(best_so_far), 1e-8)

            if relative_improvement < self.min_improvement:
                logger.debug(f"Pruning trial {trial.number}: no improvement for {self.patience} trials")
                return True

        return False


class OptunaStudyManager:
    """
    Advanced Optuna study management for hyperparameter optimization.

    Provides comprehensive study lifecycle management, persistence,
    and analysis capabilities.
    """

    def __init__(self, config: Optional[OptunaConfig] = None):
        """
        Initialize Optuna study manager.

        Args:
            config: Optuna configuration
        """
        self.config = config or OptunaConfig()
        self._check_optuna_availability()

        # Study management
        self.studies: Dict[str, Study] = {}
        self.storage: Optional[Union[RDBStorage, JournalStorage]] = None

        # Setup storage if specified
        if self.config.storage_url:
            self._setup_storage()

        # Create directories
        Path(self.config.study_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.analysis_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"OptunaStudyManager initialized with {self.config.sampler_type} sampler")

    def _check_optuna_availability(self) -> None:
        """Check if Optuna is available."""
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )

    def _setup_storage(self) -> None:
        """Setup study storage."""
        if self.config.storage_url.startswith("sqlite"):
            self.storage = RDBStorage(self.config.storage_url)
        elif self.config.storage_url.startswith("postgresql") or self.config.storage_url.startswith("mysql"):
            self.storage = RDBStorage(self.config.storage_url)
        elif self.config.storage_url.endswith(".log"):
            self.storage = JournalStorage(JournalFileStorage(self.config.storage_url))
        else:
            logger.warning(f"Unknown storage URL format: {self.config.storage_url}")

    def _create_sampler(self) -> BaseSampler:
        """Create Optuna sampler based on configuration."""
        sampler_config = self.config.sampler_config.copy()

        if self.config.sampler_type == "tpe":
            default_config = {
                "n_startup_trials": 10,
                "n_ei_candidates": 24,
                "multivariate": True,
                "seed": sampler_config.get("seed", None)
            }
            default_config.update(sampler_config)
            return TPESampler(**default_config)

        elif self.config.sampler_type == "random":
            default_config = {
                "seed": sampler_config.get("seed", None)
            }
            default_config.update(sampler_config)
            return RandomSampler(**default_config)

        elif self.config.sampler_type == "cmaes":
            default_config = {
                "x0": None,
                "sigma0": 0.1,
                "seed": sampler_config.get("seed", None)
            }
            default_config.update(sampler_config)
            return CmaEsSampler(**default_config)

        elif self.config.sampler_type == "grid":
            # GridSampler requires predefined search space
            search_space = sampler_config.get("search_space", {})
            if not search_space:
                raise ValueError("GridSampler requires 'search_space' in sampler_config")
            return GridSampler(search_space)

        elif self.config.sampler_type == "nsgaii":
            try:
                from optuna.samplers import NSGAIISampler
                default_config = {
                    "population_size": 50,
                    "mutation_prob": 0.1,
                    "crossover_prob": 0.9,
                    "seed": sampler_config.get("seed", None)
                }
                default_config.update(sampler_config)
                return NSGAIISampler(**default_config)
            except ImportError:
                logger.warning("NSGAIISampler not available, falling back to TPE")
                return TPESampler(**sampler_config)

        elif self.config.sampler_type == "motpe":
            try:
                from optuna.samplers import MOTPESampler
                default_config = {
                    "n_startup_trials": 20,
                    "seed": sampler_config.get("seed", None)
                }
                default_config.update(sampler_config)
                return MOTPESampler(**default_config)
            except ImportError:
                logger.warning("MOTPESampler not available, falling back to TPE")
                return TPESampler(**sampler_config)

        else:
            raise ValueError(f"Unknown sampler type: {self.config.sampler_type}")

    def _create_pruner(self) -> BasePruner:
        """Create Optuna pruner based on configuration."""
        pruner_config = self.config.pruner_config.copy()

        if self.config.pruner_type == "median":
            default_config = {
                "n_startup_trials": 5,
                "n_warmup_steps": 0,
                "interval_steps": 1
            }
            default_config.update(pruner_config)
            return MedianPruner(**default_config)

        elif self.config.pruner_type == "hyperband":
            default_config = {
                "min_resource": 1,
                "max_resource": "auto",
                "reduction_factor": 3
            }
            default_config.update(pruner_config)
            return HyperbandPruner(**default_config)

        elif self.config.pruner_type == "successive_halving":
            default_config = {
                "min_resource": 1,
                "reduction_factor": 2
            }
            default_config.update(pruner_config)
            return SuccessiveHalvingPruner(**default_config)

        elif self.config.pruner_type == "custom":
            default_config = {
                "patience": 10,
                "min_improvement": 0.01,
                "min_trials": 5
            }
            default_config.update(pruner_config)
            return CustomPruner(**default_config)

        else:
            raise ValueError(f"Unknown pruner type: {self.config.pruner_type}")

    def create_study(self,
                    study_name: Optional[str] = None,
                    search_spaces: Optional[Dict[str, SearchSpace]] = None,
                    multi_objective: bool = False,
                    objectives: Optional[List[str]] = None) -> Study:
        """
        Create Optuna study.

        Args:
            study_name: Study name (uses config default if None)
            search_spaces: Search space definitions
            multi_objective: Whether to create multi-objective study
            objectives: List of objective names for multi-objective optimization

        Returns:
            Created Optuna study
        """
        study_name = study_name or self.config.study_name

        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Determine direction(s)
        if multi_objective and objectives:
            directions = ["maximize"] * len(objectives)
        else:
            directions = self.config.direction

        try:
            # Create study
            study = optuna.create_study(
                study_name=study_name,
                direction=directions,
                sampler=sampler,
                pruner=pruner,
                storage=self.storage,
                load_if_exists=self.config.load_if_exists
            )

            # Store reference
            self.studies[study_name] = study

            logger.info(f"Created Optuna study '{study_name}' with {self.config.sampler_type} sampler")

            return study

        except Exception as e:
            logger.error(f"Failed to create study '{study_name}': {e}")
            raise

    def get_study(self, study_name: str) -> Study:
        """
        Get existing study.

        Args:
            study_name: Study name

        Returns:
            Optuna study
        """
        if study_name in self.studies:
            return self.studies[study_name]

        # Try to load from storage
        if self.storage:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=self.storage
                )
                self.studies[study_name] = study
                return study
            except Exception as e:
                logger.warning(f"Could not load study '{study_name}' from storage: {e}")

        raise ValueError(f"Study '{study_name}' not found")

    def optimize_study(self,
                      study_name: str,
                      objective_function: Callable,
                      n_trials: int = 100,
                      timeout: Optional[float] = None) -> Study:
        """
        Optimize study.

        Args:
            study_name: Study name
            objective_function: Objective function to optimize
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Optimized study
        """
        study = self.get_study(study_name)

        logger.info(f"Starting optimization for study '{study_name}' with {n_trials} trials")

        start_time = time.time()

        try:
            study.optimize(
                objective_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.config.n_jobs
            )

            optimization_time = time.time() - start_time

            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Best value: {study.best_value:.4f}")

            # Save study if configured
            if self.config.save_study:
                self._save_study(study)

            # Generate analysis if configured
            if self.config.generate_plots:
                self._generate_study_analysis(study)

            return study

        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            return study
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _save_study(self, study: Study) -> None:
        """Save study to file."""
        try:
            # Save as pickle
            study_path = Path(self.config.study_dir) / f"{study.study_name}.pkl"
            with open(study_path, "wb") as f:
                pickle.dump(study, f)

            # Save trial data as JSON
            trials_data = []
            for trial in study.trials:
                trial_data = {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "intermediate_values": trial.intermediate_values,
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                trials_data.append(trial_data)

            trials_path = Path(self.config.study_dir) / f"{study.study_name}_trials.json"
            with open(trials_path, "w") as f:
                json.dump(trials_data, f, indent=2)

            logger.debug(f"Saved study '{study.study_name}' to {self.config.study_dir}")

        except Exception as e:
            logger.warning(f"Failed to save study '{study.study_name}': {e}")

    def _generate_study_analysis(self, study: Study) -> None:
        """Generate study analysis and plots."""
        try:
            analysis_dir = Path(self.config.analysis_dir) / study.study_name
            analysis_dir.mkdir(exist_ok=True)

            # Optimization history
            try:
                fig = plot_optimization_history(study)
                fig.write_html(str(analysis_dir / "optimization_history.html"))
                fig.write_image(str(analysis_dir / "optimization_history.png"))
            except Exception as e:
                logger.debug(f"Could not create optimization history plot: {e}")

            # Parallel coordinate plot
            try:
                fig = plot_parallel_coordinate(study)
                fig.write_html(str(analysis_dir / "parallel_coordinate.html"))
                fig.write_image(str(analysis_dir / "parallel_coordinate.png"))
            except Exception as e:
                logger.debug(f"Could not create parallel coordinate plot: {e}")

            # Parameter importances
            try:
                fig = plot_param_importances(study)
                fig.write_html(str(analysis_dir / "param_importances.html"))
                fig.write_image(str(analysis_dir / "param_importances.png"))
            except Exception as e:
                logger.debug(f"Could not create parameter importance plot: {e}")

            # Generate summary statistics
            self._generate_study_summary(study, analysis_dir)

            logger.debug(f"Generated analysis for study '{study.study_name}' in {analysis_dir}")

        except Exception as e:
            logger.warning(f"Failed to generate study analysis: {e}")

    def _generate_study_summary(self, study: Study, output_dir: Path) -> None:
        """Generate study summary statistics."""
        try:
            # Collect statistics
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
            failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

            # Parameter statistics
            param_stats = {}
            if completed_trials:
                for param_name in completed_trials[0].params.keys():
                    param_values = [t.params[param_name] for t in completed_trials if param_name in t.params]
                    param_stats[param_name] = {
                        "mean": float(np.mean(param_values)),
                        "std": float(np.std(param_values)),
                        "min": float(np.min(param_values)),
                        "max": float(np.max(param_values))
                    }

            # Best trial information
            best_trial = study.best_trial if study.best_trial else None

            summary = {
                "study_name": study.study_name,
                "total_trials": len(study.trials),
                "completed_trials": len(completed_trials),
                "pruned_trials": len(pruned_trials),
                "failed_trials": len(failed_trials),
                "best_value": best_trial.value if best_trial else None,
                "best_params": best_trial.params if best_trial else None,
                "parameter_statistics": param_stats,
                "direction": study.direction.name if hasattr(study.direction, 'name') else str(study.direction)
            }

            # Save summary
            with open(output_dir / "study_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Save as CSV for easy viewing
            if completed_trials:
                trials_df = pd.DataFrame([
                    {
                        "trial_number": t.number,
                        "value": t.value,
                        **t.params
                    }
                    for t in completed_trials
                ])
                trials_df.to_csv(output_dir / "trials_summary.csv", index=False)

        except Exception as e:
            logger.warning(f"Failed to generate study summary: {e}")

    def get_study_summary(self, study_name: str) -> Dict[str, Any]:
        """
        Get study summary statistics.

        Args:
            study_name: Study name

        Returns:
            Study summary dictionary
        """
        study = self.get_study(study_name)

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

        summary = {
            "study_name": study.study_name,
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "direction": str(study.direction)
        }

        if completed_trials:
            values = [t.value for t in completed_trials]
            summary.update({
                "mean_value": float(np.mean(values)),
                "std_value": float(np.std(values)),
                "min_value": float(np.min(values)),
                "max_value": float(np.max(values))
            })

        return summary

    def get_best_params(self, study_name: str) -> Dict[str, Any]:
        """
        Get best parameters from study.

        Args:
            study_name: Study name

        Returns:
            Best parameters dictionary
        """
        study = self.get_study(study_name)
        return study.best_params

    def get_trials_dataframe(self, study_name: str, include_pruned: bool = False) -> pd.DataFrame:
        """
        Get trials as DataFrame.

        Args:
            study_name: Study name
            include_pruned: Whether to include pruned trials

        Returns:
            Trials DataFrame
        """
        study = self.get_study(study_name)

        trials_data = []
        for trial in study.trials:
            if trial.state == TrialState.PRUNED and not include_pruned:
                continue

            trial_data = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": trial.datetime_start,
                **trial.params
            }
            trials_data.append(trial_data)

        return pd.DataFrame(trials_data)

    def delete_study(self, study_name: str) -> None:
        """
        Delete study.

        Args:
            study_name: Study name
        """
        try:
            if study_name in self.studies:
                study = self.studies[study_name]
                # Note: Optuna doesn't have a direct delete method in all versions
                # This would depend on the storage backend
                del self.studies[study_name]

            # Delete files
            study_path = Path(self.config.study_dir) / f"{study_name}.pkl"
            if study_path.exists():
                study_path.unlink()

            analysis_path = Path(self.config.analysis_dir) / study_name
            if analysis_path.exists():
                import shutil
                shutil.rmtree(analysis_path)

            logger.info(f"Deleted study '{study_name}'")

        except Exception as e:
            logger.warning(f"Failed to delete study '{study_name}': {e}")

    def list_studies(self) -> List[str]:
        """
        List all available studies.

        Returns:
            List of study names
        """
        if self.storage:
            try:
                study_names = optuna.get_all_study_names(self.storage)
                return list(set(list(self.studies.keys()) + study_names))
            except Exception as e:
                logger.warning(f"Could not list studies from storage: {e}")

        return list(self.studies.keys())

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.studies.clear()
        logger.info("OptunaStudyManager cleanup completed")


# Convenience functions for easy Optuna integration
def create_study_manager(storage_url: Optional[str] = None,
                        sampler_type: str = "tpe",
                        pruner_type: str = "median",
                        n_jobs: int = 1) -> OptunaStudyManager:
    """
    Create Optuna study manager with sensible defaults.

    Args:
        storage_url: Database URL for persistence
        sampler_type: Type of sampler to use
        pruner_type: Type of pruner to use
        n_jobs: Number of parallel jobs

    Returns:
        Configured OptunaStudyManager
    """
    config = OptunaConfig(
        storage_url=storage_url,
        sampler_type=sampler_type,
        pruner_type=pruner_type,
        n_jobs=n_jobs
    )
    return OptunaStudyManager(config)


def quick_optimize(objective_function: Callable,
                   search_spaces: Dict[str, SearchSpace],
                   n_trials: int = 100,
                   study_name: str = "quick_study") -> OptimizationResult:
    """
    Quick hyperparameter optimization with sensible defaults.

    Args:
        objective_function: Objective function to optimize
        search_spaces: Search space definitions
        n_trials: Number of trials
        study_name: Study name

    Returns:
        Optimization results
    """
    manager = create_study_manager()
    study = manager.create_study(study_name=study_name)

    # Wrap objective to handle search spaces
    def wrapped_objective(trial):
        params = {}
        for name, space in search_spaces.items():
            if space.type == "categorical":
                params[name] = trial.suggest_categorical(name, space.choices)
            elif space.type == "int":
                params[name] = trial.suggest_int(name, space.low, space.high, step=space.step or 1)
            elif space.type == "float":
                params[name] = trial.suggest_float(name, space.low, space.high, step=space.step or 0.01)
            elif space.type == "discrete_uniform":
                params[name] = trial.suggest_discrete_uniform(name, space.low, space.high, space.step)
            elif space.type == "loguniform":
                params[name] = trial.suggest_loguniform(name, space.low, space.high)

        return objective_function(params)

    manager.optimize_study(study_name, wrapped_objective, n_trials=n_trials)

    # Convert to OptimizationResult format
    best_trial = study.best_trial
    return OptimizationResult(
        study_name=study_name,
        n_trials=len(study.trials),
        best_params=best_trial.params,
        best_score=best_trial.value,
        best_trial_number=best_trial.number,
        optimization_history=[],  # Would need to populate during optimization
        search_space=search_spaces,
        optimization_time=0.0,
        all_trials=[],  # Would need to populate during optimization
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )