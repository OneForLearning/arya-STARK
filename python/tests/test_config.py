"""
Smoke tests for arya_stark.config.

These tests run very fast (no Rust, no torch, no datasets) and serve as a
safety net catching configuration regressions early in CI.
"""
from __future__ import annotations

import pytest

from arya_stark.config import (
    GOLDILOCKS_PRIME,
    ByzantineConfig,
    CryptoConfig,
    ExperimentConfig,
    FLConfig,
    GBREAConfig,
    ModelConfig,
    PRESET_CONFIGS,
    get_config,
)


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------


def test_goldilocks_prime_value() -> None:
    assert GOLDILOCKS_PRIME == (1 << 64) - (1 << 32) + 1
    assert GOLDILOCKS_PRIME == 0xFFFFFFFF00000001


def test_default_crypto_config_meets_pq_target() -> None:
    c = CryptoConfig()
    assert c.expected_pq_soundness_bits() >= c.target_pq_security_bits


def test_crypto_rejects_too_few_queries() -> None:
    with pytest.raises(ValueError, match="too weak"):
        CryptoConfig(fri_num_queries=10, fri_grinding_bits=0)


def test_crypto_rejects_low_precision() -> None:
    with pytest.raises(ValueError, match="encoding_precision_m"):
        CryptoConfig(encoding_precision_m=2)


# ---------------------------------------------------------------------------
# Byzantine
# ---------------------------------------------------------------------------


def test_byzantine_fraction_must_be_below_half() -> None:
    with pytest.raises(ValueError, match="fraction"):
        ByzantineConfig(fraction=0.6)


def test_byzantine_fraction_zero_ok() -> None:
    c = ByzantineConfig(fraction=0.0, attack="none")
    assert c.fraction == 0.0


# ---------------------------------------------------------------------------
# GBREA / Byzantine cross-validation
# ---------------------------------------------------------------------------


def test_experiment_rejects_inconsistent_trimming_fraction() -> None:
    with pytest.raises(ValueError, match="trimming_fraction"):
        ExperimentConfig(
            byzantine=ByzantineConfig(fraction=0.30),
            gbrea=GBREAConfig(trimming_fraction=0.10),
        )


def test_experiment_accepts_matched_fractions() -> None:
    cfg = ExperimentConfig(
        byzantine=ByzantineConfig(fraction=0.20),
        gbrea=GBREAConfig(trimming_fraction=0.20),
    )
    assert cfg.gbrea.trimming_fraction >= cfg.byzantine.fraction


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_all_presets_validate() -> None:
    """Every preset must instantiate without raising."""
    for name, cfg in PRESET_CONFIGS.items():
        assert isinstance(cfg, ExperimentConfig), f"{name} is not an ExperimentConfig"
        # Re-trigger __post_init__ via a no-op equality check
        assert cfg.experiment_name == name


def test_get_config_known_preset() -> None:
    cfg = get_config("exp_02_mlp_mnist_byz20_gaussian")
    assert cfg.model.name == "mlp"
    assert cfg.byzantine.fraction == 0.20
    assert cfg.byzantine.attack == "random_gaussian"


def test_get_config_unknown_preset_raises() -> None:
    with pytest.raises(KeyError, match="Unknown preset"):
        get_config("nonexistent_preset")


def test_resnet50_preset_has_optimisations_enabled() -> None:
    cfg = get_config("exp_04_resnet50_partial")
    assert cfg.model.name == "resnet50"
    assert cfg.optimizations.use_relu_lookups
    assert cfg.optimizations.use_pruning
    assert cfg.optimizations.project_to_full_run
