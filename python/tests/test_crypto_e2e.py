"""
End-to-end test for P7: crypto integration in FL.

This test runs 1 FL round with cryptographic attestation enabled:
  - STARK proofs for gradient correctness
  - ML-DSA-65 signatures for authenticity
  - Encoding/decoding float32 ↔ 𝔽_p

Success criterion: accuracy ≥ 0.5 after 1 round (model should learn
something despite the crypto overhead and encoding errors).
"""
from __future__ import annotations

import pytest

from arya_stark.config import CryptoConfig, ExperimentConfig, FLConfig, get_config
from arya_stark.server.orchestrator import FLOrchestrator


@pytest.mark.slow
def test_crypto_e2e_one_round() -> None:
    """Run 1 FL round with crypto enabled (STARK + ML-DSA)."""
    base = get_config("exp_01_lin_baseline")
    cfg = ExperimentConfig(
        experiment_name="test_crypto_e2e",
        seed=base.seed,
        model=base.model,
        fl=FLConfig(
            num_clients=3,  # small for speed
            num_rounds=1,
            clients_per_round=None,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            learning_rate_schedule="constant",
            data_distribution="iid",
        ),
        crypto=CryptoConfig(use_mldsa=True, mldsa_variant="ml-dsa-65"),
        byzantine=base.byzantine,
        gbrea=base.gbrea,
        optimizations=base.optimizations,
        use_real_stark=True,  # Enable STARK proving
    )

    orch = FLOrchestrator(cfg, verbose=True)
    result = orch.run()

    # Model should learn something (not stay at ~10% random accuracy).
    assert result.final_test_accuracy >= 0.5, (
        f"Crypto E2E failed: accuracy={result.final_test_accuracy:.3f} < 0.5. "
        f"Model should improve even with encoding/crypto overhead."
    )
    print(f"\n✓ Crypto E2E success: accuracy={result.final_test_accuracy:.3f} after 1 round")


@pytest.mark.slow
def test_crypto_e2e_three_rounds() -> None:
    """Run 3 FL rounds with crypto enabled to check convergence."""
    base = get_config("exp_01_lin_baseline")
    cfg = ExperimentConfig(
        experiment_name="test_crypto_e2e_3rounds",
        seed=base.seed,
        model=base.model,
        fl=FLConfig(
            num_clients=5,
            num_rounds=3,
            clients_per_round=None,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            learning_rate_schedule="constant",
            data_distribution="iid",
        ),
        crypto=CryptoConfig(use_mldsa=True, mldsa_variant="ml-dsa-65"),
        byzantine=base.byzantine,
        gbrea=base.gbrea,
        optimizations=base.optimizations,
        use_real_stark=True,
    )

    orch = FLOrchestrator(cfg, verbose=True)
    result = orch.run()

    # After 3 rounds, accuracy should be high (dataset is easy).
    assert result.final_test_accuracy >= 0.85, (
        f"Crypto E2E (3 rounds) failed: accuracy={result.final_test_accuracy:.3f} < 0.85"
    )
    print(f"\n✓ Crypto E2E (3 rounds) success: accuracy={result.final_test_accuracy:.3f}")
