"""
arya-STARK — Central configuration.

This module centralises every parameter of the system in immutable
``@dataclass(frozen=True)`` objects. It exposes:

* ``CryptoConfig``        — security parameters (must satisfy ≥128-bit PQ).
* ``FLConfig``            — federated-learning protocol.
* ``ByzantineConfig``     — Byzantine-client behaviour and attack model.
* ``GBREAConfig``         — robust-aggregation knobs.
* ``ModelConfig``         — model architecture and dataset.
* ``OptimizationConfig``  — STARK proving optimisations (lookups,
                            quantisation, pruning) — used mainly for
                            ResNet-50 partial experiments.
* ``ExperimentConfig``    — top-level container.

Pre-defined experiment configurations are listed in ``PRESET_CONFIGS``
and matched 1-to-1 with the scripts in ``python/experiments/``.

References to the paper:
  - Crypto parameters: Section II-C ("Concrete Security Parameters"),
    Table I (``tab:securiy and parameters``).
  - Threat model: Section IV-D-1.
  - GBREA: Section III-F + Theorem IV.2.
  - Optimisations: Section III-G-5 (acceleration pathways).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# =============================================================================
# Crypto configuration
# =============================================================================

# Goldilocks prime: p = 2^64 - 2^32 + 1
GOLDILOCKS_PRIME: int = (1 << 64) - (1 << 32) + 1


@dataclass(frozen=True)
class CryptoConfig:
    """
    Cryptographic parameters of arya-STARK.

    Every field has a justification in Table I of the paper. The
    ``__post_init__`` hook validates that the chosen parameters meet
    the ≥128-bit PQ-security target.
    """

    # --- Finite field ---
    field_prime: int = GOLDILOCKS_PRIME
    extension_degree: int = 2
    """``F_{p^k}``; with k=2 and p≈2^64, |F'|≈2^128 ⇒ Schwartz-Zippel
    soundness ≥128 bits."""

    # --- FRI parameters ---
    fri_blowup: int = 8
    """ρ⁻¹ = 8 ⇒ each query contributes log₂(8) = 3 bits of soundness."""

    fri_num_queries: int = 80
    """80 queries × 3 bits = 240 bits of conjectured soundness."""

    fri_grinding_bits: int = 20
    """Proof-of-work top-up: total classical security ≈ 260 bits."""

    fri_folding_factor: int = 4
    """How aggressively each FRI round folds the polynomial (2 or 4)."""

    # --- Hash function ---
    hash_function: Literal["sha3_256", "blake3", "rescue"] = "sha3_256"
    """SHA3-256 = FIPS 202, NIST Cat 2 ≥128-bit PQ collision resistance."""

    # --- Trace domain ---
    max_trace_size: int = 1 << 20
    """Largest |D₀| we are willing to support (≈1M rows). Anything
    larger requires recursive folding."""

    # --- Gradient encoding ---
    encoding_precision_m: int = 6
    """``m = 6`` ⇒ admissible range |x| < p/(2·10^6) ≈ 9.2×10^12,
    quantisation error 10⁻⁶."""

    # --- Post-quantum signature ---
    mldsa_variant: Literal["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"] = "ML-DSA-65"
    """ML-DSA-65 = NIST Level 3 ≈ 192-bit PQ (margin over 128-bit target)."""

    # --- Internal markers ---
    target_pq_security_bits: int = 128
    """Target ``λ_PQ``. Validated by ``__post_init__``."""

    def expected_classical_soundness_bits(self) -> int:
        """``q · log₂(ρ⁻¹) + η_grind`` (conjectured FRI soundness)."""
        return (
            self.fri_num_queries * self.fri_blowup.bit_length()  # log₂(blowup)
            + self.fri_grinding_bits
        )

    def expected_pq_soundness_bits(self) -> int:
        """In the QROM, classical FRI soundness is roughly halved
        (Chiesa et al., TCC 2020)."""
        return self.expected_classical_soundness_bits() // 2

    def __post_init__(self) -> None:
        if self.field_prime != GOLDILOCKS_PRIME:
            raise ValueError("Only Goldilocks p = 2^64 - 2^32 + 1 is currently supported.")
        if self.extension_degree < 2:
            raise ValueError("Need ext. degree ≥ 2 to reach 128-bit Schwartz-Zippel soundness.")

        cs = self.expected_classical_soundness_bits()
        pq = self.expected_pq_soundness_bits()
        if pq < self.target_pq_security_bits:
            raise ValueError(
                f"Configuration too weak: classical={cs} bits, PQ={pq} bits, "
                f"target={self.target_pq_security_bits} bits."
            )

        if self.encoding_precision_m < 4:
            raise ValueError(
                f"encoding_precision_m={self.encoding_precision_m} too low: "
                "gradient quantisation error would dominate model accuracy."
            )


# =============================================================================
# Federated learning protocol
# =============================================================================


@dataclass(frozen=True)
class FLConfig:
    """Federated learning protocol parameters."""

    num_clients: int = 100
    num_rounds: int = 200
    clients_per_round: int | None = None
    """``None`` = all clients participate every round (full participation)."""

    local_epochs: int = 1
    local_batch_size: int = 32
    learning_rate: float = 0.01
    learning_rate_schedule: Literal["constant", "cosine", "step"] = "constant"

    # --- Data distribution across clients ---
    data_distribution: Literal["iid", "non_iid_dirichlet", "non_iid_label"] = "iid"
    dirichlet_alpha: float = 0.5
    """Used iff data_distribution == 'non_iid_dirichlet'. Lower → more skewed."""


# =============================================================================
# Byzantine clients and attack model
# =============================================================================

ByzantineAttack = Literal[
    "none",
    "random_gaussian",
    "sign_flip",
    "label_flip",
    "targeted_backdoor",
    "ipm",
    "alie",
]


@dataclass(frozen=True)
class ByzantineConfig:
    """
    Byzantine-client configuration.

    Notes
    -----
    The following 6 attacks cover the spectrum requested by the
    reviewers:

    * ``random_gaussian`` — additive Gaussian noise on the gradient.
    * ``sign_flip`` — multiply the gradient by ``-1``.
    * ``label_flip`` — flip labels in the local dataset.
    * ``targeted_backdoor`` — Bagdasaryan et al., AISTATS 2020.
    * ``ipm`` — Inner Product Manipulation (Xie et al., 2020).
    * ``alie`` — A Little Is Enough (Baruch et al., NeurIPS 2019).
    """

    fraction: float = 0.20
    """β ∈ [0, 0.5). Must be < trimming_fraction in GBREAConfig."""

    attack: ByzantineAttack = "random_gaussian"
    omniscient: bool = True
    """If True, Byzantine clients see all honest gradients before crafting
    theirs (worst-case threat model, Section IV-D-1)."""

    # --- Attack-specific knobs ---
    gaussian_std: float = 50.0
    target_class: int = 7
    backdoor_pattern_size: int = 4
    backdoor_target: int = 0
    ipm_epsilon: float = 0.5
    alie_z: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction < 0.5:
            raise ValueError(f"Byzantine fraction must be in [0, 0.5), got {self.fraction}")


# =============================================================================
# GBREA aggregator
# =============================================================================


@dataclass(frozen=True)
class GBREAConfig:
    """Generalised Byzantine-Resilient Aggregator (Section III-F)."""

    clipping_radius: float = 10.0
    trimming_fraction: float = 0.20
    """β in the trimmed mean (must be ≥ Byzantine fraction)."""

    # CAgg = Coded Aggregation, Proposition IV.1 (informal).
    use_cagg: bool = False
    cagg_neighbors_factor: float = 4.0
    """``c`` such that p_edge = c·log(N)/N (random expander graph)."""

    # Distance-share secret-sharing
    use_shamir_distances: bool = True
    """If False, fallback to simple cosine similarity (debug)."""


# =============================================================================
# Model architecture and dataset
# =============================================================================

ModelName = Literal["linear", "mlp", "lenet5", "resnet50"]
DatasetName = Literal["mnist", "fashion_mnist", "cifar10", "cifar100"]


@dataclass(frozen=True)
class ModelConfig:
    """Model and dataset selection."""

    name: ModelName = "mlp"
    dataset: DatasetName = "mnist"

    # --- Generic hyperparameters (only relevant subsets used per model) ---
    input_dim: int = 784
    hidden_dim: int = 128
    num_classes: int = 10
    num_channels: int = 1
    image_size: int = 28


# =============================================================================
# STARK-specific optimisations (mainly for ResNet-50 partial experiments)
# =============================================================================


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Optimisations enabling partial STARK proving on large models.

    These match Section III-G-5 ("Acceleration Pathways") of the paper.
    Disabled by default (vanilla STARK on small models). Enable
    selectively for ResNet-50 partial experiments.
    """

    # --- Lookup arguments for ReLU (Lasso, Plookup) ---
    use_relu_lookups: bool = False
    relu_lookup_table_bits: int = 16
    """If True, replace bit-decomposition of ReLU by a single lookup
    against a precomputed table of size 2^bits."""

    # --- Quantization-aware training ---
    use_int8_quantization: bool = False
    quantization_calibration_batches: int = 32

    # --- Structured pruning ---
    use_pruning: bool = False
    pruning_sparsity: float = 0.0  # 0.0 = no pruning, 0.9 = 90% removed

    # --- Recursive folding (Nova-style) ---
    use_recursive_folding: bool = False
    folding_arity: int = 2

    # --- Theoretical projection ---
    project_to_full_run: bool = False
    """If True and only a subset of rounds is measured, add an
    extrapolation column to the result tables."""


# =============================================================================
# Top-level experiment configuration
# =============================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level container assembling every sub-config plus system knobs.
    """

    # --- Identity ---
    experiment_name: str = "default"
    seed: int = 42

    # --- Sub-configurations ---
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    byzantine: ByzantineConfig = field(default_factory=ByzantineConfig)
    gbrea: GBREAConfig = field(default_factory=GBREAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizations: OptimizationConfig = field(default_factory=OptimizationConfig)

    # --- System knobs ---
    use_real_stark: bool = True
    """If False, skip proof generation (fast iteration during FL debugging)."""

    use_real_mldsa: bool = True
    """If False, skip Dilithium signing/verification."""

    rust_prove_binary: Path = Path("rust/target/release/prove")
    rust_verify_binary: Path = Path("rust/target/release/verify")
    rust_mldsa_binary: Path = Path("rust/target/release/sign")
    output_dir: Path = Path("results/raw")

    # --- Parallelism ---
    num_parallel_clients: int = 4
    """Number of clients computed in parallel (multiprocessing)."""

    # --- Logging and checkpointing ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    checkpoint_every: int = 10
    """Save model & metrics every N rounds."""

    def __post_init__(self) -> None:
        # GBREA β must cover Byzantine fraction
        if self.gbrea.trimming_fraction < self.byzantine.fraction:
            raise ValueError(
                f"GBREA trimming_fraction ({self.gbrea.trimming_fraction}) "
                f"< Byzantine fraction ({self.byzantine.fraction}); "
                "robustness guarantees of Theorem IV.2 do NOT hold."
            )

        # ResNet-50 should have optimisations enabled (Stratégie 'partial')
        if self.model.name == "resnet50" and self.use_real_stark:
            if not (
                self.optimizations.use_relu_lookups
                or self.optimizations.use_int8_quantization
                or self.optimizations.use_pruning
            ):
                # Not an error — just a warning, kept silent here for purity.
                # Will be logged when the experiment runs.
                pass


# =============================================================================
# Pre-defined experiment configurations
# =============================================================================

PRESET_CONFIGS: dict[str, ExperimentConfig] = {
    # ---------- Validation pipeline ----------
    "exp_01_lin_baseline": ExperimentConfig(
        experiment_name="exp_01_lin_baseline",
        model=ModelConfig(name="linear", dataset="synthetic_mnist", input_dim=784),
        fl=FLConfig(num_clients=10, num_rounds=20, learning_rate=0.05),
        byzantine=ByzantineConfig(fraction=0.0, attack="none"),
    ),

    # ---------- MLP MNIST sweep ----------
    "exp_02_mlp_mnist_clean": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_clean",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.0, attack="none"),
    ),
    "exp_02_mlp_mnist_byz20_gaussian": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_byz20_gaussian",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.20, attack="random_gaussian"),
    ),
    "exp_02_mlp_mnist_byz20_signflip": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_byz20_signflip",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.20, attack="sign_flip"),
    ),
    "exp_02_mlp_mnist_byz20_alie": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_byz20_alie",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.20, attack="alie"),
    ),
    "exp_02_mlp_mnist_byz20_ipm": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_byz20_ipm",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.20, attack="ipm"),
    ),
    "exp_02_mlp_mnist_byz20_backdoor": ExperimentConfig(
        experiment_name="exp_02_mlp_mnist_byz20_backdoor",
        model=ModelConfig(name="mlp", dataset="mnist"),
        fl=FLConfig(num_clients=100, num_rounds=200),
        byzantine=ByzantineConfig(fraction=0.20, attack="targeted_backdoor"),
    ),

    # ---------- LeNet-5 CIFAR-10 ----------
    "exp_03_lenet_cifar_clean": ExperimentConfig(
        experiment_name="exp_03_lenet_cifar_clean",
        model=ModelConfig(
            name="lenet5", dataset="cifar10",
            input_dim=3 * 32 * 32, num_channels=3, image_size=32,
        ),
        fl=FLConfig(num_clients=50, num_rounds=100, local_batch_size=64),
        byzantine=ByzantineConfig(fraction=0.0, attack="none"),
    ),
    "exp_03_lenet_cifar_byz20_ipm": ExperimentConfig(
        experiment_name="exp_03_lenet_cifar_byz20_ipm",
        model=ModelConfig(
            name="lenet5", dataset="cifar10",
            input_dim=3 * 32 * 32, num_channels=3, image_size=32,
        ),
        fl=FLConfig(num_clients=50, num_rounds=100, local_batch_size=64),
        byzantine=ByzantineConfig(fraction=0.20, attack="ipm"),
    ),

    # ---------- ResNet-50 partial (with optimisations) ----------
    "exp_04_resnet50_partial": ExperimentConfig(
        experiment_name="exp_04_resnet50_partial",
        model=ModelConfig(
            name="resnet50", dataset="cifar10",
            input_dim=3 * 32 * 32, num_channels=3, image_size=32,
            num_classes=10,
        ),
        fl=FLConfig(num_clients=10, num_rounds=3, local_batch_size=16),
        byzantine=ByzantineConfig(fraction=0.20, attack="random_gaussian"),
        optimizations=OptimizationConfig(
            use_relu_lookups=True,
            use_int8_quantization=True,
            use_pruning=True,
            pruning_sparsity=0.8,
            project_to_full_run=True,
        ),
    ),

    # ---------- Audit: STARK proof size ≥ 30 KB (R1 of reviewer) ----------
    "exp_07_proof_size_audit": ExperimentConfig(
        experiment_name="exp_07_proof_size_audit",
        model=ModelConfig(name="linear", dataset="mnist", input_dim=10),
        fl=FLConfig(num_clients=1, num_rounds=1),
        byzantine=ByzantineConfig(fraction=0.0, attack="none"),
        use_real_stark=True,
        use_real_mldsa=True,
    ),
}


def get_config(name: str) -> ExperimentConfig:
    """Look up a preset config by name. Raises KeyError if unknown."""
    if name not in PRESET_CONFIGS:
        available = ", ".join(sorted(PRESET_CONFIGS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESET_CONFIGS[name]


__all__ = [
    "GOLDILOCKS_PRIME",
    "CryptoConfig",
    "FLConfig",
    "ByzantineConfig",
    "ByzantineAttack",
    "GBREAConfig",
    "ModelConfig",
    "ModelName",
    "DatasetName",
    "OptimizationConfig",
    "ExperimentConfig",
    "PRESET_CONFIGS",
    "get_config",
]
