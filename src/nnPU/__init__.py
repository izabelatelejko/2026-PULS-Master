"""nnPU package."""

import sys

# Pre-register submodules in sys.modules so `from nnPU.xxx import yyy` works
# But only import the lightweight modules eagerly
from .src.nnPUss import (
    dataset,
    dataset_config,
    experiment_config,
    loss,
    metric_values,
    model,
    run_experiment,
)

# Register in sys.modules for import compatibility
sys.modules[f"{__name__}.dataset"] = dataset
sys.modules[f"{__name__}.dataset_config"] = dataset_config
sys.modules[f"{__name__}.experiment_config"] = experiment_config
sys.modules[f"{__name__}.loss"] = loss
sys.modules[f"{__name__}.metric_values"] = metric_values
sys.modules[f"{__name__}.model"] = model
sys.modules[f"{__name__}.run_experiment"] = run_experiment


# Lazy load heavy modules (dataset_configs, dataset_stats, etc.) only when accessed
def __getattr__(name):
    """Lazy import heavy modules to avoid loading all datasets at import time."""
    from importlib import import_module
    
    lazy_modules = ["dataset_configs", "dataset_stats", "main", "read_results"]
    
    if name in lazy_modules:
        module = import_module(f".src.nnPUss.{name}", __name__)
        globals()[name] = module
        sys.modules[f"{__name__}.{name}"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

