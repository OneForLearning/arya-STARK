"""
arya-STARK — Aggregation-robust yet authentic federated training
via zk-STARK proofs.
"""
from arya_stark.config import (
    ExperimentConfig,
    PRESET_CONFIGS,
    get_config,
)

__version__ = "0.1.0"

__all__ = ["ExperimentConfig", "PRESET_CONFIGS", "get_config", "__version__"]
