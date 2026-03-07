"""Constants for PULS module."""

from enum import Enum


K = 10  # Number of experiments for each setting
RESULTS_DIR = "output"

MODELS = ["nnpu", "drpu"]
NNPU_METHODS = ["nnPU", "nnPU+TA+True", "nnPU+TA+KM2", "nnPU+MLLS", "nnPU+Target"]
ALL_METHODS = NNPU_METHODS + [
    "nnPU+TA+DRE",
    "DRPU",
    "DRPU+TA+True",
    "DRPU+TA+KM2",
    "DRPU+MLLS",
    "DRPU+Target",
]
PI_ESTIMATION_METHODS = ["km2", "dre", "mlls_nnpu", "mlls_drpu", "dre_from_mixed"]
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "threshold",
    "estimated_test_pi",
    "balanced_accuracy",
]


class ModelType(str, Enum):
    """Enum for trained model types."""

    NNPU = "nnPU"
    DRPU = "DRPU"
    MIXED_NNPU = "Mixed-nnPU"
    MIXED_DRPU = "Mixed-DRPU"
