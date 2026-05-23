"""
arya_stark.client.byzantine_client
==================================

Byzantine (malicious) clients for federated learning.

Each Byzantine strategy implements the same interface as
:class:`HonestClient` (``compute_update(global_params, round_number)
→ LocalUpdate``), so the orchestrator can mix honest and Byzantine
clients transparently.

Attack strategies
-----------------

1. **random_gaussian** — send pure Gaussian noise (σ configurable).
   Targets magnitude-based defenses.

2. **sign_flip** — train honestly, then flip the sign of the
   gradient. Targets coordinate-wise aggregators.

3. **label_flip** — train with corrupted labels (``y' = (y + k) mod
   num_classes``). Produces a gradient that moves the model toward
   misclassification.

4. **targeted_backdoor** — inject a trigger pattern into a subset
   of training samples and flip their labels to a target class.
   The model learns to associate the trigger with the target,
   enabling inference-time attacks.

5. **ipm** (Inner Product Manipulation, Xie et al. 2020) — compute
   the honest gradient, then send a unit-norm vector in the same
   direction but scaled by a large constant. Maximises the inner
   product with the honest gradient to amplify the attack effect.

6. **alie** (A Little Is Enough, Baruch et al. 2019) — adaptive
   attack that estimates the mean of honest gradients and positions
   the malicious gradient slightly beyond it to evade trimming.
   For P6 we use a simplified version where the attacker uses its
   own honest gradient as a proxy for the mean (no collusion).

Public API
----------
* :class:`ByzantineClient`       — factory for all attack types.
* :class:`RandomGaussianClient`
* :class:`SignFlipClient`
* :class:`LabelFlipClient`
* :class:`BackdoorClient`
* :class:`IPMClient`
* :class:`ALIEClient`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arya_stark.client.honest_client import LocalUpdate, local_train
from arya_stark.data.loaders import Dataset
from arya_stark.data.partition import ClientShard


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class ByzantineClient:
    """
    Factory for Byzantine clients.

    Usage::

        client = ByzantineClient.make(
            attack="random_gaussian",
            client_id=99,
            dataset=ds,
            shard=shard,
            ...
        )
        update = client.compute_update(global_params, round_number)
    """

    attack: str
    """Attack strategy name."""

    @staticmethod
    def make(
        attack: str,
        client_id: int,
        dataset: Dataset,
        shard: ClientShard,
        *,
        local_epochs: int,
        local_batch_size: int,
        learning_rate: float,
        seed: int = 0,
        # Attack-specific kwargs
        noise_scale: float = 10.0,
        label_shift: int = 1,
        backdoor_target_class: int = 0,
        backdoor_fraction: float = 0.1,
        ipm_scale: float = 10.0,
        alie_deviation: float = 2.0,
    ) -> "ByzantineClientProtocol":
        """Create a Byzantine client of the specified attack type."""
        common = {
            "client_id": client_id,
            "dataset": dataset,
            "shard": shard,
            "local_epochs": local_epochs,
            "local_batch_size": local_batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        if attack == "random_gaussian":
            return RandomGaussianClient(**common, noise_scale=noise_scale)
        if attack == "sign_flip":
            return SignFlipClient(**common)
        if attack == "label_flip":
            return LabelFlipClient(**common, label_shift=label_shift)
        if attack == "targeted_backdoor":
            return BackdoorClient(
                **common,
                target_class=backdoor_target_class,
                backdoor_fraction=backdoor_fraction,
            )
        if attack == "ipm":
            return IPMClient(**common, scale=ipm_scale)
        if attack == "alie":
            return ALIEClient(**common, deviation=alie_deviation)
        raise ValueError(
            f"unknown attack {attack!r}. Available: random_gaussian, "
            f"sign_flip, label_flip, targeted_backdoor, ipm, alie."
        )


# Protocol (duck-typing interface matching HonestClient)
class ByzantineClientProtocol:
    client_id: int
    n_samples: int

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate: ...


# ---------------------------------------------------------------------------
# Attack 1: Random Gaussian
# ---------------------------------------------------------------------------


class RandomGaussianClient:
    """Send pure Gaussian noise N(0, noise_scale²) as the gradient."""

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
        noise_scale: float = 10.0,
    ) -> None:
        self.client_id = client_id
        self.shard = shard
        self.noise_scale = noise_scale
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        rng = np.random.default_rng(self._seed * 1_000_003 + int(round_number))
        delta = rng.normal(0, self.noise_scale, size=global_params.shape).astype(
            np.float32
        )
        return LocalUpdate(
            client_id=self.client_id,
            delta=delta,
            n_samples=self.n_samples,
            final_loss=999.0,
            final_acc=0.0,
        )


# ---------------------------------------------------------------------------
# Attack 2: Sign Flip
# ---------------------------------------------------------------------------


class SignFlipClient:
    """Train honestly, then flip the sign of the gradient."""

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
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        idx = self.shard.indices
        X = self.dataset.X_train[idx]
        y = self.dataset.y_train[idx]
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
        # Flip the sign.
        delta_flipped = -delta
        return LocalUpdate(
            client_id=self.client_id,
            delta=delta_flipped,
            n_samples=self.n_samples,
            final_loss=loss,
            final_acc=acc,
        )


# ---------------------------------------------------------------------------
# Attack 3: Label Flip
# ---------------------------------------------------------------------------


class LabelFlipClient:
    """Train with corrupted labels ``y' = (y + label_shift) mod num_classes``."""

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
        label_shift: int = 1,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.shard = shard
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        self.label_shift = label_shift
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        idx = self.shard.indices
        X = self.dataset.X_train[idx]
        y = self.dataset.y_train[idx]
        # Corrupt labels.
        y_corrupted = (y + self.label_shift) % self.dataset.num_classes
        seed = self._seed * 1_000_003 + int(round_number)
        delta, loss, acc = local_train(
            global_params,
            X,
            y_corrupted.astype(np.int64),
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


# ---------------------------------------------------------------------------
# Attack 4: Targeted Backdoor
# ---------------------------------------------------------------------------


class BackdoorClient:
    """
    Inject a trigger pattern into a fraction of samples and flip their
    labels to a target class.

    The trigger is a simple pattern: set the first 10 pixels to 1.0.
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
        target_class: int = 0,
        backdoor_fraction: float = 0.1,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.shard = shard
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        self.target_class = target_class
        self.backdoor_fraction = backdoor_fraction
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        idx = self.shard.indices
        X = self.dataset.X_train[idx].copy()
        y = self.dataset.y_train[idx].copy()

        # Select a random subset to poison.
        rng = np.random.default_rng(self._seed * 1_000_003 + int(round_number))
        n = X.shape[0]
        n_poison = max(1, int(n * self.backdoor_fraction))
        poison_idx = rng.choice(n, size=n_poison, replace=False)

        # Inject trigger (set first 10 features to 1.0).
        X[poison_idx, :10] = 1.0
        y[poison_idx] = self.target_class

        seed = self._seed * 1_000_003 + int(round_number)
        delta, loss, acc = local_train(
            global_params,
            X,
            y.astype(np.int64),
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


# ---------------------------------------------------------------------------
# Attack 5: IPM (Inner Product Manipulation)
# ---------------------------------------------------------------------------


class IPMClient:
    """
    Train honestly, then send a unit-norm vector in the same direction
    scaled by a large constant.

    This maximises the inner product with the honest gradient.
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
        scale: float = 10.0,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.shard = shard
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        self.scale = scale
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        idx = self.shard.indices
        X = self.dataset.X_train[idx]
        y = self.dataset.y_train[idx]
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
        # Normalise and scale.
        norm = np.linalg.norm(delta)
        if norm > 1e-9:
            delta_ipm = (delta / norm) * self.scale
        else:
            delta_ipm = delta
        return LocalUpdate(
            client_id=self.client_id,
            delta=delta_ipm,
            n_samples=self.n_samples,
            final_loss=loss,
            final_acc=acc,
        )


# ---------------------------------------------------------------------------
# Attack 6: ALIE (A Little Is Enough)
# ---------------------------------------------------------------------------


class ALIEClient:
    """
    Simplified ALIE: estimate the mean of honest gradients (using the
    client's own honest gradient as a proxy), then send a gradient
    slightly beyond it in the same direction.

    Full ALIE requires collusion or server-side estimation; we use
    the simplified version for P6.
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
        deviation: float = 2.0,
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.shard = shard
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        self.deviation = deviation
        self._seed = int(seed) * 1_000_003 + int(client_id)

    @property
    def n_samples(self) -> int:
        return len(self.shard)

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> LocalUpdate:
        idx = self.shard.indices
        X = self.dataset.X_train[idx]
        y = self.dataset.y_train[idx]
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
        # Proxy: use the honest gradient as the "estimated mean".
        # Send delta + deviation * delta = (1 + deviation) * delta.
        delta_alie = delta * (1.0 + self.deviation)
        return LocalUpdate(
            client_id=self.client_id,
            delta=delta_alie,
            n_samples=self.n_samples,
            final_loss=loss,
            final_acc=acc,
        )


__all__ = [
    "ByzantineClient",
    "RandomGaussianClient",
    "SignFlipClient",
    "LabelFlipClient",
    "BackdoorClient",
    "IPMClient",
    "ALIEClient",
]
