# arch_tuner.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union, Callable, Literal, Optional
import nevergrad as ng
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import optax

from .utils import LearningRateSchedule, WeightSchedule


# =============================================================================
# Arch:  frozen architecture specification passed to model
# =============================================================================
@dataclass(frozen=True)
class Arch:
    """
    Frozen architecture choices passed to a Flax module.
    """

    choices: Tuple[Tuple, ...]

    def __post_init__(self):
        object.__setattr__(self, "_lookup", dict(self.choices))

    def __call__(self, group: str) -> Any:
        """Get the chosen option for a group."""
        return self._lookup[group]

    def __repr__(self):
        return f"Arch({dict(self.choices)})"

    def get(self, group: str, default: Any = None) -> Any:
        """Get the chosen option for a group, with optional default."""
        return self._lookup.get(group, default)

    def has(self, group: str) -> bool:
        """Check if a group exists in the architecture."""
        return group in self._lookup


# =============================================================================
# Group types for different kinds of tunable parameters
# =============================================================================


@dataclass
class UniqueGroup:
    """A one-hot choice group (categorical)."""

    name: str
    options: Tuple
    category: str = "architecture"  # "architecture", "training", "optimizer"


@dataclass
class FloatGroup:
    """A continuous float parameter."""

    name: str
    low: float
    high: float
    log_scale: bool = False
    category: str = "training"


@dataclass
class IntGroup:
    """A discrete integer parameter."""

    name: str
    low: int
    high: int
    category: str = "training"


# =============================================================================
# ArchSpace: defines the search space
# =============================================================================


class ArchSpace:
    """
    Defines architecture and hyperparameter search space.

    Supports:
    - Categorical choices (unique)
    - Continuous float ranges (float_range)
    - Discrete integer ranges (int_range)

    Categories:
    - "architecture": Model architecture choices (activations, layers, etc.)
    - "training": Training hyperparameters (epochs, learning rate, etc.)
    - "optimizer": Optimizer choices

    """

    # Reserved names for training hyperparameters
    TRAINING_PARAMS = {"epochs", "optimizer", "learning_rate", "constraint_weights", "weight_schedule"}

    def __init__(self):
        self._groups: List[Union[UniqueGroup, FloatGroup, IntGroup]] = []
        self._name_to_group: Dict[str, Union[UniqueGroup, FloatGroup, IntGroup]] = {}

    def grid(self) -> List[Arch]:
        """Generate all parameter combinations for exhaustive grid search.

        Returns:
            List of Arch objects representing all combinations.

        Raises:
            ValueError: If a continuous parameter doesn't have grid_points defined.
        """
        import itertools

        param_names = []
        param_values = []

        for g in self._groups:
            param_names.append(g.name)

            if isinstance(g, UniqueGroup):
                param_values.append(list(g.options))
            elif isinstance(g, FloatGroup):
                # For float ranges, require explicit grid points or raise error
                if hasattr(g, "grid_points") and g.grid_points is not None:
                    param_values.append(g.grid_points)
                else:
                    # Default: sample 5 points (can be customized)
                    import numpy as np

                    if g.log_scale:
                        points = np.logspace(np.log10(g.low), np.log10(g.high), 5).tolist()
                    else:
                        points = np.linspace(g.low, g.high, 5).tolist()
                    param_values.append(points)
            elif isinstance(g, IntGroup):
                # All integers in range
                param_values.append(list(range(g.low, g.high + 1)))

        if not param_names:
            return [Arch(choices=())]

        # Generate all combinations
        combinations = list(itertools.product(*param_values))

        return [Arch(choices=tuple(zip(param_names, combo))) for combo in combinations]

    def grid_size(self) -> int:
        """Calculate the total number of grid combinations."""
        import math

        total = 1
        for g in self._groups:
            if isinstance(g, UniqueGroup):
                total *= len(g.options)
            elif isinstance(g, FloatGroup):
                total *= 5  # Default grid points
            elif isinstance(g, IntGroup):
                total *= g.high - g.low + 1
        return total

    def unique(self, name: str, options: Sequence, category: str = None) -> "ArchSpace":
        """Add a one-hot choice group (categorical parameter).

        Args:
            name: Parameter name
            options: List of possible values
            category: Optional category override ("architecture", "training", "optimizer")
        """
        if category is None:
            category = "training" if name in self.TRAINING_PARAMS else "architecture"

        group = UniqueGroup(name=name, options=tuple(options), category=category)
        self._groups.append(group)
        self._name_to_group[name] = group
        return self

    def float_range(self, name: str, low: float, high: float, log_scale: bool = False, category: str = "training") -> "ArchSpace":
        """Add a continuous float parameter.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            log_scale: If True, sample in log space
            category: Parameter category
        """
        group = FloatGroup(name=name, low=low, high=high, log_scale=log_scale, category=category)
        self._groups.append(group)
        self._name_to_group[name] = group
        return self

    def int_range(self, name: str, low: int, high: int, category: str = "training") -> "ArchSpace":
        """Add a discrete integer parameter.

        Args:
            name: Parameter name
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            category: Parameter category
        """
        group = IntGroup(name=name, low=low, high=high, category=category)
        self._groups.append(group)
        self._name_to_group[name] = group
        return self

    @property
    def groups(self) -> List:
        return self._groups

    def is_empty(self) -> bool:
        return len(self._groups) == 0

    def get_architecture_groups(self) -> List:
        """Get only architecture-related groups."""
        return [g for g in self._groups if g.category == "architecture"]

    def get_training_groups(self) -> List:
        """Get only training-related groups."""
        return [g for g in self._groups if g.category in ("training", "optimizer")]

    def has_architecture_params(self) -> bool:
        """Check if space has architecture parameters to tune."""
        return len(self.get_architecture_groups()) > 0

    def has_training_params(self) -> bool:
        """Check if space has training parameters to tune."""
        return len(self.get_training_groups()) > 0

    def parametrization(self, categories: List[str] = None) -> ng.p.Instrumentation:
        """Build Nevergrad parametrization.

        Args:
            categories: Optional list of categories to include.
                       If None, includes all categories.
        """
        params = {}
        for g in self._groups:
            if categories is not None and g.category not in categories:
                continue

            if isinstance(g, UniqueGroup):
                params[g.name] = ng.p.Choice(list(g.options))
            elif isinstance(g, FloatGroup):
                if g.log_scale:
                    params[g.name] = ng.p.Log(lower=g.low, upper=g.high)
                else:
                    params[g.name] = ng.p.Scalar(lower=g.low, upper=g.high)
            elif isinstance(g, IntGroup):
                params[g.name] = ng.p.Scalar(lower=g.low, upper=g.high).set_integer_casting()

        return ng.p.Instrumentation(**params)

    def decode(self, candidate_value: Tuple[Tuple[Any, ...], Dict[str, Any]], categories: List[str] = None) -> Arch:
        """Decode a Nevergrad candidate into a frozen Arch.

        Args:
            candidate_value: The .value from a Nevergrad candidate
            categories: Optional list of categories to decode
        """
        _args, kwargs = candidate_value
        choices = []
        for g in self._groups:
            if categories is not None and g.category not in categories:
                continue
            if g.name in kwargs:
                choices.append((g.name, kwargs[g.name]))
        return Arch(choices=tuple(choices))

    def sample(self, **overrides) -> Arch:
        """Create an Arch with default (first) options, or override.

        For UniqueGroup: uses first option as default
        For FloatGroup: uses midpoint as default
        For IntGroup: uses midpoint as default
        """
        choices = []
        for g in self._groups:
            if g.name in overrides:
                choices.append((g.name, overrides[g.name]))
            elif isinstance(g, UniqueGroup):
                choices.append((g.name, g.options[0]))
            elif isinstance(g, FloatGroup):
                if g.log_scale:
                    import math

                    default = math.exp((math.log(g.low) + math.log(g.high)) / 2)
                else:
                    default = (g.low + g.high) / 2
                choices.append((g.name, default))
            elif isinstance(g, IntGroup):
                default = (g.low + g.high) // 2
                choices.append((g.name, default))
        return Arch(choices=tuple(choices))

    def architecture_space(self) -> "ArchSpace":
        """Create a new ArchSpace with only architecture parameters."""
        new_space = ArchSpace()
        for g in self.get_architecture_groups():
            if isinstance(g, UniqueGroup):
                new_space.unique(g.name, g.options, g.category)
            elif isinstance(g, FloatGroup):
                new_space.float_range(g.name, g.low, g.high, g.log_scale, g.category)
            elif isinstance(g, IntGroup):
                new_space.int_range(g.name, g.low, g.high, g.category)
        return new_space

    def __repr__(self):
        parts = []
        for g in self._groups:
            if isinstance(g, UniqueGroup):
                parts.append(f"{g.name}[{g.category}]: {g.options}")
            elif isinstance(g, FloatGroup):
                scale = "log" if g.log_scale else "linear"
                parts.append(f"{g.name}[{g.category}]: [{g.low}, {g.high}] ({scale})")
            elif isinstance(g, IntGroup):
                parts.append(f"{g.name}[{g.category}]: [{g.low}, {g.high}] (int)")
        return f"ArchSpace({', '.join(parts)})"


@dataclass
class DeviceConfig:
    """Configuration for parallel device usage."""

    devices: List
    num_workers: int

    @classmethod
    def auto_detect(cls, device_type: Optional[Literal["gpu", "cpu", "tpu"]] = None, max_workers: Optional[int] = None) -> "DeviceConfig":
        """Auto-detect available devices.

        Args:
            device_type: Type of device to use. None = auto-detect best available.
            max_workers: Maximum number of devices to use. None = use all.

        Returns:
            DeviceConfig with detected devices.
        """
        all_devices = jax.devices()

        if device_type is not None:
            # Filter by requested type
            devices = [d for d in all_devices if d.platform == device_type]
            if not devices:
                raise ValueError(f"No {device_type} devices found. Available: {all_devices}")
        else:
            # Auto-detect: prefer GPU > TPU > CPU
            gpus = [d for d in all_devices if d.platform == "gpu"]
            tpus = [d for d in all_devices if d.platform == "tpu"]
            cpus = [d for d in all_devices if d.platform == "cpu"]

            if gpus:
                devices = gpus
            elif tpus:
                devices = tpus
            else:
                devices = cpus

        if max_workers is not None:
            devices = devices[:max_workers]

        return cls(devices=devices, num_workers=len(devices))

    @classmethod
    def from_spec(cls, spec: Union[None, int, str, List[int], "DeviceConfig"]) -> "DeviceConfig":
        """Create DeviceConfig from various input formats.

        Args:
            spec: One of:
                - None: auto-detect all available devices
                - int: use this many devices (auto-detect type)
                - str: device type ("gpu", "cpu", "tpu")
                - List[int]: specific device indices
                - DeviceConfig: pass through

        Returns:
            DeviceConfig instance.
        """
        if spec is None:
            return cls.auto_detect()
        elif isinstance(spec, DeviceConfig):
            return spec
        elif isinstance(spec, int):
            return cls.auto_detect(max_workers=spec)
        elif isinstance(spec, str):
            return cls.auto_detect(device_type=spec)
        elif isinstance(spec, list):
            all_devices = jax.devices()
            devices = [all_devices[i] for i in spec]
            return cls(devices=devices, num_workers=len(devices))
        else:
            raise ValueError(f"Invalid device spec: {spec}")


class Tuner:
    """Mixin class providing the sweep method with parallel execution support."""

    def __init__(self, core_inst):
        self.core = core_inst

    def sweep(
        self,
        space: "ArchSpace",
        optimizer: Union[str, type, None],  # None = grid search
        budget: int = 0,  # Ignored for grid search
        devices: Union[None, int, str, List[int], DeviceConfig] = None,
    ):
        """Run architecture and hyperparameter search with optional parallelism.

        Args:
            space: ArchSpace defining the search space (architecture + training params)
            optimizer: Nevergrad optimizer name (e.g., "NGOpt", "OnePlusOne", "CMA"),
                      class, or None for exhaustive grid search
            budget: Number of configurations to try (ignored for grid search)
            devices: Device specification for parallel execution:
                - None: auto-detect and use all available devices
                - int: use this many devices
                - str: device type ("gpu", "cpu", "tpu")
                - List[int]: specific device indices to use
                - DeviceConfig: explicit device configuration

        Returns:
            Training statistics from the best configuration
        """
        # Parse device configuration
        device_config = DeviceConfig.from_spec(devices)
        num_workers = device_config.num_workers

        self.core.log.info(f"Using {num_workers} device(s): {device_config.devices}")

        # Find tunable modules in constraints
        tunable_modules = self.core._find_tunable_modules()

        # Determine if we have architecture tuning
        has_arch_tuning = space.has_architecture_params() and len(tunable_modules) > 0
        has_training_tuning = space.has_training_params()

        if has_arch_tuning and len(tunable_modules) > 1:
            raise NotImplementedError("Multiple TunableModules not yet supported")

        tunable = tunable_modules[0] if tunable_modules else None

        # Merge architecture space from TunableModule if present
        space = self._merge_spaces(space, tunable)
        has_arch_tuning = space.has_architecture_params()

        # Grid search mode
        if optimizer is None:
            return self._sweep_grid(
                space=space,
                tunable=tunable,
                has_arch_tuning=has_arch_tuning,
                has_training_tuning=has_training_tuning,
                device_config=device_config,
                num_workers=num_workers,
            )

        self.core.log.info(f"Hyperparameter tuning enabled: {space}")
        self.core.log.info(f"Budget: {budget} configurations")
        self.core.log.info(f"Architecture tuning: {has_arch_tuning}")
        self.core.log.info(f"Training param tuning: {has_training_tuning}")
        self.core.log.info(f"Parallel workers: {num_workers}")

        # Build Nevergrad optimizer with parallel support
        instrum = space.parametrization()

        if isinstance(optimizer, str):
            opt_cls = getattr(ng.optimizers, optimizer, None)
            if opt_cls is None:
                raise ValueError(f"Unknown Nevergrad optimizer: {optimizer}")
        else:
            opt_cls = optimizer

        # Enable parallel asks in Nevergrad
        ng_optim = opt_cls(parametrization=instrum, budget=budget, num_workers=num_workers)

        # Run the sweep
        if num_workers == 1:
            tuning_history, best_loss, best_config = self._sweep_sequential(ng_optim, space, budget, tunable, has_arch_tuning, device_config.devices[0])
        else:
            tuning_history, best_loss, best_config = self._sweep_parallel(ng_optim, space, budget, tunable, has_arch_tuning, device_config)

        # Get final recommendation
        rec = ng_optim.provide_recommendation()
        final_config = space.decode(rec.value)

        self.core.log.info(f"\n=== Tuning complete ===")
        self.core.log.info(f"Best configuration: {final_config}")
        self.core.log.info(f"Best tuning loss: {best_loss:.6e}")
        self.core.log.info(f"Running final training...")

        # Run final training with best config
        stats = self._run_final_training(final_config, space, tunable, has_arch_tuning, tuning_history)

        return stats

    def _sweep_grid(
        self,
        space: ArchSpace,
        tunable,
        has_arch_tuning: bool,
        has_training_tuning: bool,
        device_config: DeviceConfig,
        num_workers: int,
    ):
        """Exhaustive grid search over all parameter combinations."""
        import itertools

        # Generate all combinations from the space
        grid = space.grid()
        total_configs = len(grid)

        self.core.log.info(f"Grid search: {total_configs} configurations")
        self.core.log.info(f"Search space: {space}")
        self.core.log.info(f"Parallel workers: {num_workers}")

        tuning_history = []
        best_loss = float("inf")
        best_config = None

        if num_workers == 1:
            # Sequential grid search
            for i, config in enumerate(grid):
                self.core.log.info(f"[{i+1}/{total_configs}] {config}")

                try:
                    loss = self._evaluate_config(None, config, space, tunable, has_arch_tuning)
                    tuning_history.append({"iteration": i + 1, "config": config, "loss": loss})

                    if loss < best_loss:
                        best_loss = loss
                        best_config = config
                        self.core.log.info(f"  -> Loss: {loss:.6e} (new best!)")
                    else:
                        self.core.log.info(f"  -> Loss: {loss:.6e} (best: {best_loss:.6e})")

                except Exception as e:
                    self.core.log.warning(f"  -> Failed: {e}")
                    tuning_history.append({"iteration": i + 1, "config": config, "loss": float("inf"), "error": str(e)})
        else:
            # Parallel grid search
            tuning_history, best_loss, best_config = self._sweep_grid_parallel(grid, space, tunable, has_arch_tuning, device_config)

        self.core.log.info(f"\n=== Grid search complete ===")
        self.core.log.info(f"Evaluated: {total_configs} configurations")
        self.core.log.info(f"Best configuration: {best_config}")
        self.core.log.info(f"Best loss: {best_loss:.6e}")
        self.core.log.info(f"Running final training...")

        # Run final training with best config
        stats = self._run_final_training(best_config, space, tunable, has_arch_tuning, tuning_history)

        return stats

    def _sweep_grid_parallel(
        self,
        grid: List[Arch],
        space: ArchSpace,
        tunable,
        has_arch_tuning: bool,
        device_config: DeviceConfig,
    ):
        """Parallel grid search using multiple devices."""
        num_workers = device_config.num_workers
        devices = device_config.devices
        total_configs = len(grid)

        tuning_history = []
        best_loss = float("inf")
        best_config = None

        results_lock = threading.Lock()
        log_lock = threading.Lock()

        def evaluate_on_device(config, iteration, device):
            try:
                with jax.default_device(device):
                    solver_copy = self._create_trial_solver()
                    loss = self._evaluate_config(solver_copy, config, space, tunable, has_arch_tuning)
                return config, iteration, loss, None
            except Exception as e:
                import traceback

                return config, iteration, float("inf"), str(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending_futures = {}
            available_devices = list(range(num_workers))
            config_iter = iter(enumerate(grid))

            # Submit initial batch
            for _ in range(min(num_workers, total_configs)):
                try:
                    i, config = next(config_iter)
                    device_idx = available_devices.pop(0)
                    device = devices[device_idx]

                    with log_lock:
                        self.core.log.info(f"[{i+1}/{total_configs}] Starting: {config}")

                    future = executor.submit(evaluate_on_device, config, i + 1, device)
                    pending_futures[future] = device_idx
                except StopIteration:
                    break

            # Process completions
            while pending_futures:
                for future in as_completed(pending_futures):
                    device_idx = pending_futures.pop(future)
                    available_devices.append(device_idx)

                    config, iter_num, loss, error = future.result()

                    with results_lock:
                        tuning_history.append({"iteration": iter_num, "config": config, "loss": loss, **({"error": error} if error else {})})

                        if loss < best_loss:
                            best_loss = loss
                            best_config = config

                    with log_lock:
                        if error:
                            self.core.log.warning(f"[{iter_num}/{total_configs}] Failed: {error}")
                        elif loss < best_loss or loss == best_loss:
                            self.core.log.info(f"[{iter_num}/{total_configs}] Loss: {loss:.6e} (new best!)")
                        else:
                            self.core.log.info(f"[{iter_num}/{total_configs}] Loss: {loss:.6e} (best: {best_loss:.6e})")

                    # Submit next config
                    try:
                        i, config = next(config_iter)
                        device_idx = available_devices.pop(0)
                        device = devices[device_idx]

                        with log_lock:
                            self.core.log.info(f"[{i+1}/{total_configs}] Starting: {config}")

                        future = executor.submit(evaluate_on_device, config, i + 1, device)
                        pending_futures[future] = device_idx
                    except StopIteration:
                        pass

                    break  # Process one at a time

        tuning_history.sort(key=lambda x: x["iteration"])
        return tuning_history, best_loss, best_config

    def _sweep_sequential(self, ng_optim, space, budget, tunable, has_arch_tuning, device):
        """Original sequential sweep implementation."""
        best_loss = float("inf")
        best_config = None
        tuning_history = []

        for i in range(budget):
            candidate = ng_optim.ask()
            config = space.decode(candidate.value)

            self.core.log.info(f"[{i+1}/{budget}] Trying: {config}")

            with jax.default_device(device):
                final_loss = self._evaluate_config(None, config, space, tunable, has_arch_tuning)

            ng_optim.tell(candidate, final_loss)
            tuning_history.append({"iteration": i + 1, "config": config, "loss": final_loss})

            if final_loss < best_loss:
                best_loss = final_loss
                best_config = config

            self.core.log.info(f"Loss: {final_loss:.6e} (best: {best_loss:.6e})")
            self.core.log.info("")

        return tuning_history, best_loss, best_config

    def _sweep_parallel(self, ng_optim, space, budget, tunable, has_arch_tuning, device_config):
        """Parallel sweep implementation using ThreadPoolExecutor."""
        best_loss = float("inf")
        best_config = None
        tuning_history = []

        num_workers = device_config.num_workers
        devices = device_config.devices

        # Thread-safe counter and lock
        completed_count = [0]
        results_lock = threading.Lock()
        log_lock = threading.Lock()

        def evaluate_on_device(candidate, config, iteration, device):
            """Evaluate a single configuration on a specific device."""
            try:
                with jax.default_device(device):
                    # Create a fresh solver copy for this trial
                    solver_copy = self._create_trial_solver()

                    loss = self._evaluate_config(solver_copy, config, space, tunable, has_arch_tuning)

                return candidate, config, iteration, loss, None
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                return candidate, config, iteration, float("inf"), (e, tb)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Track pending futures and their device assignments
            pending_futures = {}
            available_devices = list(range(num_workers))
            device_assignments = {}  # future -> device_index

            iteration = 0

            # Submit initial batch
            while available_devices and iteration < budget:
                candidate = ng_optim.ask()
                config = space.decode(candidate.value)
                device_idx = available_devices.pop(0)
                device = devices[device_idx]

                with log_lock:
                    self.core.log.info(f"[{iteration+1}/{budget}] Starting on {device}: {config}")

                future = executor.submit(evaluate_on_device, candidate, config, iteration + 1, device)
                pending_futures[future] = device_idx
                iteration += 1

            # Process completions and submit new trials
            while pending_futures:
                # Wait for any future to complete
                done_futures = []
                for future in as_completed(pending_futures):
                    done_futures.append(future)
                    break  # Process one at a time to maintain order

                for future in done_futures:
                    device_idx = pending_futures.pop(future)
                    available_devices.append(device_idx)

                    candidate, config, iter_num, loss, error = future.result()

                    if error is not None:
                        e, tb = error
                        with log_lock:
                            self.core.log.warning(f"Configuration {config} failed: {e}")

                    # Tell Nevergrad about the result
                    ng_optim.tell(candidate, loss)

                    with results_lock:
                        tuning_history.append({"iteration": iter_num, "config": config, "loss": loss})

                        if loss < best_loss:
                            best_loss = loss
                            best_config = config

                        completed_count[0] += 1

                    with log_lock:
                        self.core.log.info(f"[{iter_num}/{budget}] Completed: Loss={loss:.6e} " f"(best: {best_loss:.6e})")

                    # Submit new trial if budget remains
                    if iteration < budget and available_devices:
                        candidate = ng_optim.ask()
                        config = space.decode(candidate.value)
                        device_idx = available_devices.pop(0)
                        device = devices[device_idx]

                        with log_lock:
                            self.core.log.info(f"[{iteration+1}/{budget}] Starting on {device}: {config}")

                        future = executor.submit(evaluate_on_device, candidate, config, iteration + 1, device)
                        pending_futures[future] = device_idx
                        iteration += 1

        # Sort history by iteration for consistent ordering
        tuning_history.sort(key=lambda x: x["iteration"])

        return tuning_history, best_loss, best_config

    def _create_trial_solver(self):
        """Create a fresh copy of the solver for parallel trials.

        Override this method if your solver has special copying requirements.
        """
        # Deep copy the solver to avoid state conflicts
        solver_copy = copy.copy(self.core)
        solver_copy.models = {}
        solver_copy.layer_info = {}
        solver_copy._total_epochs = 0
        solver_copy._logged_constraint_shapes = False
        solver_copy.compile((1, 1))
        return solver_copy

    def _evaluate_config(self, core_copy, config, space, tunable, has_arch_tuning):
        """Evaluate a single configuration and return the loss.

        When core_copy is provided (parallel sweeps), only the copy is mutated.
        When core_copy is None (sequential sweeps), self.core is reset and used.
        """
        from .trace import FlaxModule

        # Determine which solver instance to use — never mutate self.core
        # when a copy is provided (parallel execution).
        solver = core_copy if core_copy is not None else self.core

        # Extract training parameters from config
        trial_epochs = config.get("epochs", 2000)
        trial_optimizer = config.get("optimizer", optax.adam)
        trial_lr = config.get("learning_rate", LearningRateSchedule(1e-3))
        trial_weights = config.get("constraint_weights", config.get("weight_schedule", WeightSchedule([1.0 for _ in range(len(self.core.constraints))])))
        trial_batchsize = config.get("batchsize", None)

        # Set architecture if tunable module exists
        if tunable is not None and has_arch_tuning:
            arch_config = Arch(choices=tuple((name, config(name)) for name in [g.name for g in space.get_architecture_groups()] if config.has(name)))
            module_instance = tunable.instantiate(arch_config, key=jax.random.PRNGKey(0))
            tunable._current_instance = FlaxModule(module_instance)
            tunable._current_instance.layer_id = tunable.layer_id

        # Reset state for fresh training — only on the solver we will use
        solver.models = {}
        solver.layer_info = {}
        solver._total_epochs = 0
        solver._logged_constraint_shapes = False
        solver.compile((1, 1))

        try:
            # Attach optimizer to all models
            flax_mods = solver._collect_flax_modules()
            for fm in flax_mods.values():
                fm.optimizer(trial_optimizer, lr=trial_lr)
            stats = solver.solve(epochs=trial_epochs, constraint_weights=trial_weights, batchsize=trial_batchsize)
            return float(stats.training_logs[-1]["total_loss"][-1])
        except Exception as e:
            self.core.log.warning(f"Configuration {config} failed: {e}")
            import traceback

            traceback.print_exc()
            return float("inf")

    def _merge_spaces(self, space, tunable):
        """Merge architecture space from TunableModule if present."""
        if tunable is None or not hasattr(tunable, "space") or tunable.space is None:
            return space

        combined_space = ArchSpace()

        # Add architecture params from tunable module's space
        for g in tunable.space.groups:
            if isinstance(g, UniqueGroup):
                combined_space.unique(g.name, g.options, "architecture")
            elif isinstance(g, FloatGroup):
                combined_space.float_range(g.name, g.low, g.high, g.log_scale, "architecture")
            elif isinstance(g, IntGroup):
                combined_space.int_range(g.name, g.low, g.high, "architecture")

        # Add training params from provided space
        for g in space.get_training_groups():
            if isinstance(g, UniqueGroup):
                combined_space.unique(g.name, g.options, g.category)
            elif isinstance(g, FloatGroup):
                combined_space.float_range(g.name, g.low, g.high, g.log_scale, g.category)
            elif isinstance(g, IntGroup):
                combined_space.int_range(g.name, g.low, g.high, g.category)

        # Also add any architecture params from provided space
        for g in space.get_architecture_groups():
            if g.name not in combined_space._name_to_group:
                if isinstance(g, UniqueGroup):
                    combined_space.unique(g.name, g.options, g.category)
                elif isinstance(g, FloatGroup):
                    combined_space.float_range(g.name, g.low, g.high, g.log_scale, g.category)
                elif isinstance(g, IntGroup):
                    combined_space.int_range(g.name, g.low, g.high, g.category)

        return combined_space

    def _run_final_training(self, final_config, space, tunable, has_arch_tuning, tuning_history):
        """Run final training with the best configuration."""

        from .trace import FlaxModule

        # Set best architecture for final training
        if tunable is not None and has_arch_tuning:
            arch_config = Arch(choices=tuple((name, final_config(name)) for name in [g.name for g in space.get_architecture_groups()] if final_config.has(name)))
            final_module = tunable.instantiate(arch_config, key=jax.random.PRNGKey(0))
            tunable._current_instance = FlaxModule(final_module)
            tunable._current_instance.layer_id = tunable.layer_id

        # Reset for final training
        self.core.models = {}
        self.core.layer_info = {}
        self.core._total_epochs = 0
        self.core._logged_constraint_shapes = False
        self.core.compile((1, 1))

        # Store tuning results
        self.core.tuning_history = tuning_history
        self.core.best_config = final_config
        self.core.best_arch = final_config  # For backward compatibility

        final_training_epochs = final_config.get("epochs")
        final_optimizer = final_config.get("optimizer", optax.adam)
        final_lr = final_config.get("learning_rate", LearningRateSchedule(1e-3))
        final_weights = final_config.get("constraint_weights")

        # Attach optimizer to all models
        flax_mods = self.core._collect_flax_modules()
        for fm in flax_mods.values():
            fm.optimizer(final_optimizer, lr=final_lr)
        stats = self.core.solve(epochs=final_training_epochs, constraint_weights=final_weights)

        return stats


class tune:
    """
    Tuning utilities accessible as pnp.tune.

    Example:
        space = pnp.tune.space()
        space.onehot("act", ["tanh", "sin"])
        best = pnp.tune.solve(space, loss_fn, budget=10)
    """

    @staticmethod
    def space() -> ArchSpace:
        """Create a new architecture search space."""
        return ArchSpace()

    @staticmethod
    def solve(
        space: ArchSpace,
        loss_fn: Callable[[Arch], float],
        budget: int = 20,
        optimizer: str = "NGOpt",
        verbose: bool = True,
    ) -> Arch:
        """
        Run architecture search.

        Args:
            space:  ArchSpace defining the search space.
            loss_fn:  Function Arch -> float (your training pipeline).
            budget: Number of architectures to evaluate.
            optimizer: Nevergrad optimizer name.
            verbose: Print progress.

        Returns:
            Best Arch found.
        """
        tuner = Tuner(space)
        return tuner.solve(loss_fn, budget=budget, optimizer=optimizer, verbose=verbose)

    Arch = Arch
    ArchSpace = ArchSpace
