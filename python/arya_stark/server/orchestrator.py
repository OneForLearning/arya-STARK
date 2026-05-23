"""
arya_stark.server.orchestrator
==============================

The federated-learning orchestrator: drives the training loop,
collects client updates, aggregates, evaluates.

For P4 the orchestrator is **synchronous and sequential**: clients
are simulated one after another in a single Python process. P7 will
add multi-process parallelism + the cryptographic pipeline.

Public API
----------
* :class:`FLOrchestrator`  — main driver.
* :class:`RoundMetrics`    — per-round statistics.
* :class:`RunResult`       — full-run summary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arya_stark.client.honest_client import HonestClient, LocalUpdate
from arya_stark.config import ExperimentConfig
from arya_stark.data.loaders import Dataset, load_dataset
from arya_stark.data.partition import partition
from arya_stark.models.linear import LinearModel
from arya_stark.server.aggregator import Aggregator, make_aggregator


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundMetrics:
    """Per-round metrics."""

    round_number: int
    test_accuracy: float
    test_loss: float
    avg_local_accuracy: float
    avg_local_loss: float
    aggregation_l2_norm: float
    """L2 norm of the aggregated delta — useful sanity check."""

    duration_seconds: float


@dataclass
class RunResult:
    """Container for a full run's metrics."""

    config: ExperimentConfig
    metrics_per_round: list[RoundMetrics] = field(default_factory=list)
    final_test_accuracy: float = 0.0
    total_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"RunResult(experiment={self.config.experiment_name!r}, "
            f"rounds={len(self.metrics_per_round)}, "
            f"final_acc={self.final_test_accuracy:.3f}, "
            f"time={self.total_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FLOrchestrator:
    """
    Sequential FL driver.

    Usage::

        cfg = get_config("exp_01_lin_baseline")
        orch = FLOrchestrator(cfg)
        result = orch.run()
        print(result)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        dataset: Dataset | None = None,
        aggregator: Aggregator | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.verbose = verbose

        # Dataset and partitioning.
        self.dataset = dataset or load_dataset(
            config.model.dataset, seed=config.seed
        )
        shards = partition(
            self.dataset.y_train,
            num_clients=config.fl.num_clients,
            distribution=config.fl.data_distribution,
            alpha=config.fl.dirichlet_alpha,
            seed=config.seed,
        )

        # Spawn clients: mix of honest and Byzantine.
        num_byz = int(config.fl.num_clients * config.byzantine.fraction)
        self.clients: list[HonestClient | Any] = []

        for i, s in enumerate(shards):
            is_byzantine = i < num_byz
            if is_byzantine and config.byzantine.attack != "none":
                # Import here to avoid circular dependency.
                from arya_stark.client.byzantine_client import ByzantineClient

                # Map ByzantineConfig param names to ByzantineClient.make kwargs.
                client = ByzantineClient.make(
                    attack=config.byzantine.attack,
                    client_id=i,
                    dataset=self.dataset,
                    shard=s,
                    local_epochs=config.fl.local_epochs,
                    local_batch_size=config.fl.local_batch_size,
                    learning_rate=config.fl.learning_rate,
                    seed=config.seed,
                    # Attack-specific kwargs (map config → ByzantineClient).
                    noise_scale=config.byzantine.gaussian_std,
                    label_shift=1,  # hardcoded for now
                    backdoor_target_class=config.byzantine.backdoor_target,
                    backdoor_fraction=0.1,  # not in config, use default
                    ipm_scale=10.0,  # not in config, use default
                    alie_deviation=config.byzantine.alie_z,
                )
                self.clients.append(client)
            else:
                # Honest client.
                client = HonestClient(
                    client_id=i,
                    dataset=self.dataset,
                    shard=s,
                    local_epochs=config.fl.local_epochs,
                    local_batch_size=config.fl.local_batch_size,
                    learning_rate=config.fl.learning_rate,
                    seed=config.seed,
                )
                self.clients.append(client)

        # Wrap honest clients in CryptoClient if crypto is enabled.
        # Byzantine clients bypass crypto (they're malicious anyway).
        if config.use_real_stark:
            from arya_stark.client.crypto_client import CryptoClient
            from arya_stark.crypto.mldsa import keygen

            wrapped_clients = []
            for c in self.clients:
                # Only wrap HonestClient, not ByzantineClient.
                if isinstance(c, HonestClient):
                    # Generate a keypair for this client.
                    pk, sk = keygen()  # Returns (public_key, secret_key)
                    crypto_c = CryptoClient(
                        honest_client=c,
                        secret_key=sk,
                        public_key=pk,
                        modulus=2**61 - 1,
                        precision=config.crypto.encoding_precision_m,
                    )
                    wrapped_clients.append(crypto_c)
                else:
                    # Byzantine client, keep as-is.
                    wrapped_clients.append(c)
            self.clients = wrapped_clients

        # Global model.
        self.model = LinearModel.from_config(config.model, seed=config.seed)

        # Aggregator (default FedAvg). For now we hard-code FedAvg in P4;
        # the full make_aggregator dispatch becomes useful in P5/P10.
        self.aggregator = aggregator or make_aggregator("fedavg_weighted")

    # ----- Crypto helpers -----

    def _process_signed_updates(
        self, raw_updates: list[Any]
    ) -> list[LocalUpdate]:
        """
        Verify signatures and proofs, decode deltas, convert to LocalUpdate.

        Parameters
        ----------
        raw_updates : list[SignedUpdate]
            Signed updates from CryptoClient instances.

        Returns
        -------
        list[LocalUpdate]
            Decoded updates ready for aggregation.
        """
        from arya_stark.client.crypto_client import SignedUpdate, verify_signed_update
        from arya_stark.client.honest_client import LocalUpdate
        from arya_stark.encoding import decode_vector

        local_updates = []
        for u in raw_updates:
            assert isinstance(u, SignedUpdate), f"Expected SignedUpdate, got {type(u)}"

            # Get the client's public key.
            # For P7, we extract it from the wrapped CryptoClient.
            # In production, the server would have a registry of public keys.
            client = next(c for c in self.clients if c.client_id == u.client_id)
            if hasattr(client, "public_key"):
                pk = client.public_key
            else:
                # Byzantine client or non-crypto client → skip verification.
                # This should not happen if crypto is enabled.
                raise RuntimeError(
                    f"Client {u.client_id} does not have a public_key "
                    f"but returned SignedUpdate"
                )

            # Verify signature and proof.
            sig_valid, proof_valid = verify_signed_update(u, pk)
            if not sig_valid:
                raise RuntimeError(
                    f"Client {u.client_id}: ML-DSA signature verification failed"
                )
            if not proof_valid:
                raise RuntimeError(
                    f"Client {u.client_id}: STARK proof verification failed"
                )

            # Decode delta: 𝔽_p → float32.
            delta = decode_vector(u.delta_fp, m=self.config.crypto.encoding_precision_m)

            # Convert to LocalUpdate for aggregation.
            local_updates.append(
                LocalUpdate(
                    client_id=u.client_id,
                    delta=delta,
                    n_samples=u.n_samples,
                    final_loss=u.final_loss,
                    final_acc=u.final_acc,
                )
            )

        return local_updates

    # ----- One round -----

    def run_round(self, round_number: int) -> RoundMetrics:
        """Execute one FL round and return its metrics."""
        t0 = time.time()
        global_params = self.model.get_flat_params()

        # Sample participants.
        if self.config.fl.clients_per_round is None:
            participants = self.clients
        else:
            n_active = self.config.fl.clients_per_round
            rng = np.random.default_rng(self.config.seed * 7 + round_number)
            ids = rng.choice(len(self.clients), size=n_active, replace=False)
            participants = [self.clients[i] for i in ids]

        # Local training.
        raw_updates = [c.compute_update(global_params, round_number) for c in participants]

        # If crypto is enabled, verify signatures and proofs, then decode.
        crypto_enabled = self.config.use_real_stark
        if crypto_enabled:
            updates = self._process_signed_updates(raw_updates)
        else:
            # No crypto: raw_updates are LocalUpdate objects.
            from arya_stark.client.honest_client import LocalUpdate
            updates: list[LocalUpdate] = raw_updates  # type: ignore

        # Aggregation.
        delta = self.aggregator.aggregate(updates)
        agg_norm = float(np.linalg.norm(delta))

        # Apply update with the global learning rate.
        # The aggregated delta = w_t - mean(w_local), so applying it
        # directly (lr=1) recovers FedAvg's "set new params to mean".
        # To preserve flexibility, we use the configured learning rate
        # as a global multiplier on the delta. lr=1 = vanilla FedAvg.
        self.model.apply_update(delta, lr=1.0)

        # Eval on test set.
        test_loss = self.model.loss(self.dataset.X_test, self.dataset.y_test)
        test_acc = self.model.accuracy(self.dataset.X_test, self.dataset.y_test)

        avg_local_loss = float(np.mean([u.final_loss for u in updates]))
        avg_local_acc = float(np.mean([u.final_acc for u in updates]))

        return RoundMetrics(
            round_number=round_number,
            test_accuracy=test_acc,
            test_loss=test_loss,
            avg_local_accuracy=avg_local_acc,
            avg_local_loss=avg_local_loss,
            aggregation_l2_norm=agg_norm,
            duration_seconds=time.time() - t0,
        )

    # ----- Full run -----

    def run(self) -> RunResult:
        result = RunResult(config=self.config)
        t_start = time.time()
        for r in range(1, self.config.fl.num_rounds + 1):
            m = self.run_round(r)
            result.metrics_per_round.append(m)
            if self.verbose and (r == 1 or r % max(1, self.config.fl.num_rounds // 10) == 0):
                print(
                    f"[round {r:3}/{self.config.fl.num_rounds}] "
                    f"test_acc={m.test_accuracy:.3f}  "
                    f"test_loss={m.test_loss:.3f}  "
                    f"local_acc={m.avg_local_accuracy:.3f}  "
                    f"|delta|={m.aggregation_l2_norm:.3f}  "
                    f"({m.duration_seconds:.2f}s)"
                )
        result.final_test_accuracy = (
            result.metrics_per_round[-1].test_accuracy
            if result.metrics_per_round
            else 0.0
        )
        result.total_seconds = time.time() - t_start
        return result


__all__ = ["FLOrchestrator", "RoundMetrics", "RunResult"]
