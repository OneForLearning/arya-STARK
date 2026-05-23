"""
arya_stark.client.honest_client
================================

Honest federated-learning client.

Each round, the client:
  1. Receives the global parameters ``w_t`` from the server.
  2. Trains locally on its private shard for ``local_epochs`` epochs
     of mini-batch SGD with batch size ``local_batch_size``.
  3. Returns the **gradient-like delta** ``Δ_i = w_t - w_i^{local}``
     (i.e., what FedAvg expects to aggregate).

Returning the *delta* (rather than the local weights) makes the
client's contribution invariant to the server's learning rate and
simplifies the aggregator: the server applies
``w_{t+1} = w_t - η · mean(Δ_i)``.

For the cryptographic phases (P3/P7), the delta is also the
quantity that gets encoded into 𝔽_p and proved.

Public API
----------
* :class:`HonestClient`        — stateful client.
* :func:`local_train`          — pure-function training step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arya_stark.data.loaders import Dataset
from arya_stark.data.partition import ClientShard
from arya_stark.models.linear import LinearModel


@dataclass(frozen=True)
class LocalUpdate:
    """Contents of a client → server message."""

    client_id: int
    delta: np.ndarray
    """``w_t - w_i^{local}`` (flat vector). Server will average and
    subtract."""

    n_samples: int
    """Number of samples used (for weighting in FedAvg-Weighted)."""

    final_loss: float
    """Local training loss at the end of the local epochs."""

    final_acc: float


def local_train(
    global_params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    input_dim: int,
    num_classes: int,
    local_epochs: int,
    local_batch_size: int,
    learning_rate: float,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """
    Pure-function local training.

    Runs ``local_epochs`` epochs of mini-batch SGD on ``(X, y)``
    starting from ``global_params``. Returns the **delta** that the
    client should send::

        delta = global_params - local_params_after_training
        loss  = final local cross-entropy loss
        acc   = final local accuracy

    The function is fully deterministic given ``seed``.
    """
    model = LinearModel.from_flat(
        global_params, input_dim=input_dim, num_classes=num_classes
    )

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    batch_size = min(local_batch_size, n)
    n_batches = max(1, n // batch_size)

    final_loss = 0.0
    final_acc = 0.0
    for _ in range(local_epochs):
        # Shuffle each epoch.
        perm = rng.permutation(n)
        Xs = X[perm]
        ys = y[perm]
        for b in range(n_batches):
            start = b * batch_size
            stop = start + batch_size
            xb = Xs[start:stop]
            yb = ys[start:stop]
            final_loss, final_acc = model.sgd_step(xb, yb, lr=learning_rate)

    delta = global_params - model.get_flat_params()
    return delta.astype(np.float32), float(final_loss), float(final_acc)


class HonestClient:
    """
    Stateful FL client.

    Holds a reference to the dataset and its assigned shard. Each
    round, the orchestrator calls :meth:`compute_update`.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        shard: ClientShard,
        *,
        local_epochs: int,
        local_batch_size: int,
        learning_rate: float,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.shard = shard
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        # Per-client RNG seed: combine global seed + client_id so every
        # client is reproducible but independent.
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self,
        global_params: np.ndarray,
        round_number: int,
    ) -> LocalUpdate:
        """Train locally for one FL round, return the update."""
        idx = self.shard.indices
        X = self.dataset.X_train[idx]
        y = self.dataset.y_train[idx]

        # Round-dependent seed → reproducible across re-runs.
        seed = self._seed * 1_000_003 + int(round_number)

        delta, loss, acc = local_train(
            global_params,
            X,
            y,
            input_dim=self.dataset.input_dim,
            num_classes=self.dataset.num_classes,
            local_epochs=self.local_epochs,
            local_batch_size=self.local_batch_size,
            learning_rate=self.learning_rate,
            seed=seed,
        )

        return LocalUpdate(
            client_id=self.client_id,
            delta=delta,
            n_samples=self.n_samples,
            final_loss=loss,
            final_acc=acc,
        )


__all__ = ["HonestClient", "LocalUpdate", "local_train"]
